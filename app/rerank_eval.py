#!/usr/bin/env python3
"""Rerank existing eval files after the fact.

eval_retrieval.py stores every question's full retrieved list — including the
chunk texts — so a reranker can be applied to a past evaluation without
re-running retrieval, indexing, or QA generation. For each base eval file and
provider this reads the stored top-k lists, reranks them, recomputes gold
ranks and aggregates with eval_retrieval's own scoring code, and writes the
standard twin file next to the base:

  eval_<dt>_<stem>.json  ->  eval_<dt>_<stem>_rerank_<provider>.json

The twin is bit-for-bit equivalent to what a live --rerank run would produce
(same inputs, same math). Any pipeline_runs/pipeline_*.json that references
the base eval gains a matching twin experiment row (metrics, rerank provider,
updated num_experiments and best_by_mrr) so the viewer's results table shows
it. Twins that already exist are skipped, so re-running is harmless.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import setup_logging  # noqa: E402
from rerank import (  # noqa: E402
    DEFAULT_RERANK_MODELS,
    RERANK_PROVIDERS,
    rerank_results,
)
from eval_retrieval import make_record, print_aggregates, summarize  # noqa: E402

log = logging.getLogger(__name__)


def results_from_record(rec):
    """Rebuild the retriever result list a stored question record came from."""
    return [
        {"rank": r["rank"], "score": r["score"], "similarity": r["similarity"],
         "chunk": {"chunk_index": r["chunk_index"], "text": r.get("text") or ""}}
        for r in rec["retrieved"]
    ]


def rerank_eval_file(path, provider, model):
    """Write the rerank twin of one base eval file; return its path, or None
    if the file cannot or need not be reranked (already a twin, twin exists)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        sys.exit(f"ERROR: could not read {path}: {e}")
    meta = data.get("metadata", {})
    if meta.get("rerank"):
        log.info(f"  {path.name}: already a rerank eval, skipping")
        return None
    out_path = path.with_name(f"{path.stem}_rerank_{provider}.json")
    if out_path.is_file():
        log.info(f"  {path.name}: {out_path.name} already exists, skipping")
        return out_path

    questions = data.get("questions", [])
    ks = meta.get("ks")
    if not questions or not ks:
        sys.exit(f"ERROR: {path} has no questions/ks metadata; not an eval file?")

    records = []
    for i, rec in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] reranking ...", end="\r", flush=True)
        q = {"chunk_index": rec["chunk_index"], "question_type": rec["question_type"],
             "question": rec["question"], "answer": rec.get("answer")}
        started = time.perf_counter()
        reranked = rerank_results(rec["question"], results_from_record(rec),
                                  provider=provider, model=model)
        rr_elapsed = time.perf_counter() - started
        record = make_record(q, reranked, rec["retrieval_seconds"] + rr_elapsed)
        record["rerank_seconds"] = rr_elapsed
        records.append(record)
    print()

    aggregates = summarize(records, ks)
    out = {
        "metadata": {
            **meta,
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "rerank": provider,
            "rerank_model": model,
            "derived_from": path.name,
        },
        "aggregates": aggregates,
        "questions": records,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"  with {provider} rerank ({model}):")
    print_aggregates(aggregates, ks)
    log.info(f"    -> {out_path}")
    return out_path


def update_pipeline_runs(base_path, twin_path, provider):
    """Insert a twin experiment row after the base row in every pipeline run
    results file that references the base eval file."""
    try:
        twin = json.loads(twin_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    overall = twin.get("aggregates", {}).get("overall", {})

    runs_dir = base_path.parent.parent / "pipeline_runs"
    if not runs_dir.is_dir():
        return
    for f in sorted(runs_dir.glob("pipeline_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows = data.get("experiments", [])
        base_idx = next((i for i, r in enumerate(rows)
                         if Path(r.get("eval_file", "")).name == base_path.name), None)
        if base_idx is None:
            continue
        if any(Path(r.get("eval_file", "")).name == twin_path.name for r in rows):
            continue  # twin row already recorded
        base_row = rows[base_idx]
        twin_row = {
            **base_row,
            "experiment_id": base_row["experiment_id"] + f"_rerank_{provider}",
            "eval_file": Path(base_row["eval_file"]).with_name(twin_path.name).as_posix(),
            "rerank": provider,
            "metrics": overall,
        }
        rows.insert(base_idx + 1, twin_row)
        data["metadata"]["num_experiments"] = len(rows)
        data["best_by_mrr"] = max(rows, key=lambda r: (r.get("metrics") or {}).get("mrr") or 0)
        f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"    added {twin_row['experiment_id']} to {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Rerank existing eval files from their stored retrieved lists.")
    parser.add_argument("--eval", required=True,
                        help="Base eval file path(s), comma-separated")
    parser.add_argument("--provider", required=True,
                        choices=RERANK_PROVIDERS + ("both",),
                        help="Rerank provider ('both' runs every provider)")
    parser.add_argument("--rerank-model",
                        help="Rerank model override (only with a single provider; "
                             "defaults per provider: "
                             + ", ".join(f"{p}: {m}" for p, m in DEFAULT_RERANK_MODELS.items()))
    args = parser.parse_args()

    setup_logging("rerank_eval")

    providers = list(RERANK_PROVIDERS) if args.provider == "both" else [args.provider]
    if args.rerank_model and len(providers) > 1:
        sys.exit("ERROR: --rerank-model cannot be combined with --provider both")

    paths = []
    for token in args.eval.split(","):
        p = Path(token.strip())
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            sys.exit(f"ERROR: eval file not found: {token.strip()}")
        paths.append(p)

    # local first so a missing Cohere key can't block the keyless provider
    for provider in sorted(providers, key=lambda p: p != "local"):
        model = args.rerank_model or DEFAULT_RERANK_MODELS[provider]
        for path in paths:
            log.info(f"\nReranking {path.name} with {provider} ({model}) ...")
            twin = rerank_eval_file(path, provider, model)
            if twin is not None:
                update_pipeline_runs(path, twin, provider)

    log.info(f"\nDone: {len(paths)} eval file(s) x {len(providers)} provider(s)")


if __name__ == "__main__":
    main()
