#!/usr/bin/env python3
"""Evaluate retrieval quality of stored indexes against generated QA pairs.

Reads a qa_*.json produced by generate_qa.py (each question is grounded in
exactly one source chunk), runs every question as a query against one or more
stored indexes (milvus / chromadb from embed_chunks.py, BM25 pickles from
index_bm25.py), and scores the ranked results against each question's
originating chunk_index.

An index is only comparable if it was built from the same chunk run the QA
file was generated from; this is checked against the chunk_run recorded in
each index's metadata (--force overrides). --db all selects every index whose
chunk run matches the QA file.

Questions for vector DBs are embedded in one batched API call per embedding
model; BM25 needs no API. Retrieval runs once per question at --topk and
metrics are computed at every cutoff in --ks from the same ranked list:

  Recall@K, Precision@K, NDCG@K, MRR, MAP

Each question has a single gold chunk, so per question Recall@K is 0/1
(hit rate), Precision@K = Recall@K / K, MAP == MRR, and IDCG == 1.

With --rerank cohere, each question's top-k results are additionally reordered
by the Cohere rerank API and scored again. Metrics are taken from the ranked
list right before reranking, so one retrieval run yields both without-rerank
and with-rerank numbers (the latter in a separate eval_*_rerank.json).

Outputs one eval_<dt>_<index>.json per index under <title>/evaluations/ with
per-question records and aggregate metrics (overall and per question_type),
plus an eval_summary_<dt>.json comparison when several indexes are evaluated.
"""

import argparse
import json
import os
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (Path(os.environ["PDF_DATA_DIR"]) if os.environ.get("PDF_DATA_DIR")
            else ROOT / "data")  # per-user override set by the web viewer
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import setup_logging  # noqa: E402
from rerank import (  # noqa: E402
    DEFAULT_RERANK_MODEL,
    RERANK_PROVIDERS,
    rerank_results,
)
from retriever_topk import (  # noqa: E402
    DEFAULT_ALPHA,
    EXTRACTORS,
    SEARCHERS,
    embed_query,
    scan_dbs,
    search_hybrid,
)

log = logging.getLogger(__name__)

DEFAULT_TOPK = 10
DEFAULT_KS = "1,3,5,10"

RELEVANCE_NOTE = ("single gold chunk per question: recall@k is 0/1 per question, "
                  "precision@k = recall@k / k, map == mrr, idcg == 1")


# ----------------------------------------------------------------- qa files


def scan_qa_files():
    """Return one dict per qa_*.json found inside a chunk run dir."""
    qa_files = []
    for extractor in EXTRACTORS:
        base = DATA_DIR / extractor
        if not base.is_dir():
            continue
        for p in sorted(base.glob("*/*/*_chunk_*/qa_*.json")):
            try:
                items = json.loads(p.read_text(encoding="utf-8")).get("items", [])
                num_pairs = sum(len(it.get("qa_pairs", [])) for it in items)
            except (json.JSONDecodeError, OSError):
                num_pairs = None
            qa_files.append({
                "path": p,
                "rel": p.relative_to(ROOT).as_posix(),
                "title_dir": p.parent.parent,
                "num_pairs": num_pairs,
            })
    return qa_files


def choose_qa(qa_files, preselect=None):
    if not qa_files:
        sys.exit("ERROR: no qa_*.json files found under data/*/*/*/*_chunk_*/; "
                 "run generate_qa.py first")

    # newest by the qa_<YYYYMMDD_HHMMSS>_ filename stamp is the default choice
    default = max(range(len(qa_files)), key=lambda i: qa_files[i]["path"].name) + 1

    log.info("\nAvailable QA files:")
    for i, qa in enumerate(qa_files, 1):
        pairs = "?" if qa["num_pairs"] is None else qa["num_pairs"]
        mark = "  (latest)" if i == default else ""
        log.info(f"  [{i}] {qa['rel']}  ({pairs} QA pairs){mark}")

    if preselect is not None:
        choice = preselect
        log.info(f"\nQA file (from --qa): {choice}")
    else:
        choice = input(f"\nChoose a QA file (number or path, Enter for [{default}]): ").strip()
        if not choice:
            choice = str(default)

    if choice.isdigit() and 1 <= int(choice) <= len(qa_files):
        return qa_files[int(choice) - 1]
    norm = choice.rstrip("/")
    for qa in qa_files:
        if norm in (qa["rel"], str(qa["path"])):
            return qa
    sys.exit(f"ERROR: '{choice}' is not a valid QA file choice")


def load_qa(qa):
    """Load a QA file; return (metadata, questions) with one flat entry per question."""
    try:
        data = json.loads(qa["path"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        sys.exit(f"ERROR: could not read {qa['rel']}: {e}")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        sys.exit(f"ERROR: {qa['rel']} has no items")

    questions = []
    for item in items:
        if not isinstance(item.get("chunk_index"), int):
            sys.exit(f"ERROR: {qa['rel']} has an item without an integer chunk_index")
        for pair in item.get("qa_pairs", []):
            if not isinstance(pair.get("question"), str) or not pair["question"].strip():
                sys.exit(f"ERROR: {qa['rel']} chunk {item['chunk_index']} has an empty question")
            questions.append({
                "chunk_index": item["chunk_index"],
                "question_type": pair.get("question_type"),
                "question": pair["question"],
                "answer": pair.get("answer"),
            })
    if not questions:
        sys.exit(f"ERROR: {qa['rel']} has no questions")
    log.info(f"Loaded {len(questions)} questions over {len(items)} chunks from {qa['rel']}")
    return data.get("metadata", {}), questions


# ------------------------------------------------------------ index selection


def index_chunk_run(db):
    """Return the chunk run an index was built from, or None if unrecorded."""
    if db["type"] == "chromadb":
        try:
            import chromadb
        except ImportError:
            sys.exit("ERROR: chromadb not installed. Run: pip install chromadb")
        collection = chromadb.PersistentClient(path=str(db["path"])).get_collection("chunks")
        return (collection.metadata or {}).get("chunk_run")
    sidecar = db["path"].with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))["metadata"]["chunk_run"]
    except (json.JSONDecodeError, KeyError):
        return None


def choose_eval_dbs(dbs, qa_chunk_run, preselect=None, force=False):
    """Pick the indexes to evaluate; only chunk-run matches are allowed unless --force."""
    if not dbs:
        sys.exit("ERROR: no indexes found under data/*/*/*/{embedding_databases,bm25}; "
                 "run embed_chunks.py or index_bm25.py first")

    log.info("\nAvailable indexes ([x] = built from this QA file's chunk run):")
    type_labels = {"bm25": "BM-25", "milvus": "Milvus", "chromadb": "ChromaDB"}
    for i, db in enumerate(dbs, 1):
        mark = "x" if db["chunk_run"] == qa_chunk_run else " "
        label = type_labels.get(db["type"], db["type"])
        log.info(f"  [{i}] [{mark}] {label:<8} {db['rel']}")

    if preselect is not None:
        choice = preselect
        log.info(f"\nIndex(es) (from --db): {choice}")
    else:
        choice = input("\nChoose index(es) (numbers/paths, comma-separated, or 'all' "
                       "for every matching index): ").strip()

    if choice.strip().lower() == "all":
        picked = [db for db in dbs if db["chunk_run"] == qa_chunk_run]
        skipped = len(dbs) - len(picked)
        if skipped:
            log.info(f"Skipping {skipped} index(es) built from a different chunk run")
        if not picked:
            sys.exit(f"ERROR: no index was built from chunk run '{qa_chunk_run}'")
        return picked

    picked = []
    for token in choice.split(","):
        token = token.strip()
        if token.isdigit() and 1 <= int(token) <= len(dbs):
            db = dbs[int(token) - 1]
        else:
            db = find_db(dbs, token)
        if db not in picked:
            picked.append(db)

    for db in picked:
        check_chunk_run(db, qa_chunk_run, force)
    return picked


def check_chunk_run(db, qa_chunk_run, force):
    """Refuse (or warn with --force) when an index's chunk run doesn't match the QA file's."""
    if db["chunk_run"] != qa_chunk_run:
        msg = (f"index {db['rel']} was built from chunk run "
               f"'{db['chunk_run']}' but the QA file is from '{qa_chunk_run}'; "
               "chunk_index values are not comparable")
        if force:
            log.warning(f"WARNING: {msg} (--force)")
        else:
            sys.exit(f"ERROR: {msg} (pass --force to evaluate anyway)")


def find_db(dbs, token):
    norm = token.rstrip("/")
    if norm.endswith(".json"):  # accept the bm25 sidecar as an alias for the pkl
        norm = norm[:-len(".json")] + ".pkl"
    for db in dbs:
        if norm in (db["rel"], str(db["path"])):
            return db
    sys.exit(f"ERROR: '{token}' is not a valid index choice")


def resolve_hybrids(dbs, arg, qa_chunk_run, alphas, force=False):
    """Turn --hybrid into a list of synthetic hybrid 'db' entries, one per
    (pair, alpha) so several vector weights can be evaluated in one run.

    arg is 'auto' (pair every matching vector index with every matching bm25
    index) or comma-separated 'VECTOR_PATH+BM25_PATH' pairs."""
    pairs = []
    if arg.strip().lower() == "auto":
        vecs = [db for db in dbs
                if db["type"] in ("milvus", "chromadb") and db["chunk_run"] == qa_chunk_run]
        bms = [db for db in dbs if db["type"] == "bm25" and db["chunk_run"] == qa_chunk_run]
        if not vecs or not bms:
            sys.exit(f"ERROR: --hybrid auto needs at least one vector index and one bm25 "
                     f"index built from chunk run '{qa_chunk_run}' "
                     f"(found {len(vecs)} vector, {len(bms)} bm25)")
        pairs = [(v, b) for v in vecs for b in bms]
    else:
        for token in arg.split(","):
            token = token.strip()
            if token.count("+") != 1:
                sys.exit(f"ERROR: --hybrid pair must look like VECTOR_PATH+BM25_PATH, got '{token}'")
            vec_tok, bm_tok = token.split("+")
            vec, bm = find_db(dbs, vec_tok.strip()), find_db(dbs, bm_tok.strip())
            if vec["type"] not in ("milvus", "chromadb"):
                sys.exit(f"ERROR: --hybrid left side must be a vector index, got {vec['rel']}")
            if bm["type"] != "bm25":
                sys.exit(f"ERROR: --hybrid right side must be a bm25 index, got {bm['rel']}")
            for db in (vec, bm):
                check_chunk_run(db, qa_chunk_run, force)
            pairs.append((vec, bm))

    hybrids = []
    for vec, bm in pairs:
        for alpha in alphas:
            hybrids.append({
                "type": "hybrid",
                "vec": vec,
                "bm25": bm,
                "alpha": alpha,
                "rel": f"hybrid(alpha={alpha:g}) {vec['rel']} + {bm['rel']}",
                "stem": f"hybrid_a{alpha:g}_{vec['path'].stem}__{bm['path'].stem}",
                "chunk_run": vec["chunk_run"],
            })
    return hybrids


def parse_ks(arg, topk):
    ks = []
    for token in arg.split(","):
        token = token.strip()
        try:
            k = int(token)
        except ValueError:
            sys.exit(f"ERROR: --ks must be comma-separated integers, got '{token}'")
        if k <= 0:
            sys.exit("ERROR: every --ks cutoff must be > 0")
        if k > topk:
            log.warning(f"WARNING: dropping cutoff {k} > --topk {topk}")
        elif k not in ks:
            ks.append(k)
    if topk not in ks:
        ks.append(topk)
    return sorted(ks)


# ------------------------------------------------------------------- scoring


def gold_rank(results, gold):
    """Rank (1-based) of the gold chunk in the results, or None if not retrieved."""
    for r in results:
        if r["chunk"].get("chunk_index") == gold:
            return r["rank"]
    return None


def aggregate(records, ks):
    """Metrics over per-question records (gold_rank None = miss). Single gold
    chunk per query."""
    n = len(records)
    hits = [r["gold_rank"] for r in records if r["gold_rank"] is not None]
    out = {
        "num_questions": n,
        "mrr": sum(1.0 / r for r in hits) / n,
        "map": sum(1.0 / r for r in hits) / n,
        "avg_retrieval_time": sum(r["retrieval_seconds"] for r in records) / n,
    }
    for k in ks:
        khits = [r for r in hits if r <= k]
        out[f"recall@{k}"] = len(khits) / n
        out[f"precision@{k}"] = len(khits) / (n * k)
        out[f"ndcg@{k}"] = sum(1.0 / math.log2(r + 1) for r in khits) / n
    return out


def make_record(q, results, elapsed):
    return {
        "chunk_index": q["chunk_index"],
        "question_type": q["question_type"],
        "question": q["question"],
        "answer": q["answer"],
        "gold_rank": gold_rank(results, q["chunk_index"]),
        "retrieval_seconds": elapsed,
        "retrieved": [
            {"rank": r["rank"], "chunk_index": r["chunk"].get("chunk_index"),
             "score": r["score"], "similarity": r["similarity"],
             **({"pre_rerank_rank": r["pre_rerank_rank"]} if "pre_rerank_rank" in r else {}),
             "text": r["chunk"].get("text")}
            for r in results
        ],
    }


def summarize(records, ks):
    aggregates = {"overall": aggregate(records, ks)}
    by_type = {}
    for r in records:
        by_type.setdefault(r["question_type"], []).append(r)
    aggregates["by_question_type"] = {
        qtype: aggregate(recs, ks) for qtype, recs in sorted(by_type.items())
    }
    return aggregates


def evaluate_db(run_search, questions, vectors, ks, hybrid=False, reranker=None):
    """Run every question against one index; return (records, aggregates,
    rerank_records, rerank_aggregates) — the rerank pair is None without a
    reranker. Base metrics come from the ranked list right before reranking,
    so a single retrieval pass yields both sets of numbers.

    hybrid searchers need both the question text and its embedding, so their
    run_search takes a (text, vector) tuple."""
    records, rerank_records = [], []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] querying ...", end="\r", flush=True)
        if hybrid:
            query = (q["question"], vectors[i - 1])
        else:
            query = q["question"] if vectors is None else vectors[i - 1]
        started = time.perf_counter()
        results = run_search(query)
        elapsed = time.perf_counter() - started
        records.append(make_record(q, results, elapsed))
        if reranker:
            started = time.perf_counter()
            reranked = reranker(q["question"], results)
            rr_elapsed = time.perf_counter() - started
            # total time to produce the reranked list = retrieval + rerank
            record = make_record(q, reranked, elapsed + rr_elapsed)
            record["rerank_seconds"] = rr_elapsed
            rerank_records.append(record)
    print()

    return (records, summarize(records, ks),
            rerank_records or None,
            summarize(rerank_records, ks) if rerank_records else None)


def print_aggregates(aggregates, ks):
    def line(label, m):
        cuts = "  ".join(f"R@{k}={m[f'recall@{k}']:.3f}" for k in ks)
        log.info(f"    {label:<12} MRR={m['mrr']:.3f}  {cuts}  NDCG@{ks[-1]}={m[f'ndcg@{ks[-1]}']:.3f}"
                 f"  t={m['avg_retrieval_time'] * 1000:.1f}ms")

    line("overall", aggregates["overall"])
    for qtype, m in aggregates["by_question_type"].items():
        line(qtype, m)


# --------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality of stored indexes against a QA file.")
    parser.add_argument("--qa", help="QA file path (data/.../<chunk dir>/qa_*.json) or list number")
    parser.add_argument("--db", help="Index path(s)/number(s), comma-separated, or 'all' for "
                                     "every index built from the QA file's chunk run")
    parser.add_argument("--hybrid", help="Hybrid retrieval: comma-separated VECTOR_PATH+BM25_PATH "
                                         "pairs, or 'auto' to pair every matching vector index "
                                         "with every matching bm25 index")
    parser.add_argument("--alpha", default=str(DEFAULT_ALPHA),
                        help="Hybrid weight of the vector score; bm25 gets 1-alpha. "
                             "Comma-separated to evaluate several weights at once "
                             f"(default {DEFAULT_ALPHA})")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                        help=f"Results to retrieve per question (default {DEFAULT_TOPK})")
    parser.add_argument("--ks", default=DEFAULT_KS,
                        help=f"Metric cutoffs, comma-separated (default {DEFAULT_KS}; "
                             "--topk is always included)")
    parser.add_argument("--rerank", choices=RERANK_PROVIDERS,
                        help="Also rerank each question's top-k results with this provider "
                             "and report with-rerank metrics alongside the without-rerank "
                             "ones (retrieval runs once per question either way)")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL,
                        help=f"Rerank model for --rerank (default {DEFAULT_RERANK_MODEL})")
    parser.add_argument("--force", action="store_true",
                        help="Evaluate even if an index was built from a different chunk run")
    args = parser.parse_args()

    setup_logging("eval_retrieval")

    if args.topk <= 0:
        sys.exit("ERROR: --topk must be > 0")
    try:
        alphas = [float(a) for a in args.alpha.split(",") if a.strip() != ""]
    except ValueError:
        sys.exit("ERROR: --alpha must be a number or comma-separated numbers")
    if not alphas or any(not 0 <= a <= 1 for a in alphas):
        sys.exit("ERROR: --alpha must be between 0 and 1")
    alphas = list(dict.fromkeys(alphas))
    ks = parse_ks(args.ks, args.topk)

    qa = choose_qa(scan_qa_files(), preselect=args.qa)
    qa_meta, questions = load_qa(qa)
    qa_chunk_run = qa_meta.get("chunk_run")
    if not qa_chunk_run and not args.force:
        sys.exit(f"ERROR: {qa['rel']} does not record its chunk_run; cannot verify indexes "
                 "(pass --force to skip the check)")

    dbs = scan_dbs()
    for db in dbs:
        db["chunk_run"] = index_chunk_run(db)

    selected = []
    if args.db or not args.hybrid:
        selected = choose_eval_dbs(dbs, qa_chunk_run, preselect=args.db, force=args.force)
    if args.hybrid:
        selected += resolve_hybrids(dbs, args.hybrid, qa_chunk_run, alphas, force=args.force)

    # Build every searcher first so questions can be embedded once per model.
    searchers = []
    for db in selected:
        if db["type"] == "hybrid":
            (method, tokenizer), run_search = search_hybrid(db["vec"], db["bm25"],
                                                            args.topk, db["alpha"])
            db["tokenizer"] = tokenizer
        else:
            method, run_search = SEARCHERS[db["type"]](db, args.topk)
        if db["type"] != "bm25" and not method:
            sys.exit(f"ERROR: could not determine embedding model for {db['rel']}")
        searchers.append((db, method, run_search))

    texts = [q["question"] for q in questions]
    vectors_by_model = {}
    for model in sorted({m for db, m, _ in searchers if db["type"] != "bm25"}):
        log.info(f"\nEmbedding {len(texts)} questions with {model} ...")
        vectors_by_model[model] = embed_query(model, texts)

    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    out_dir = qa["title_dir"] / "evaluations"
    out_dir.mkdir(exist_ok=True)

    reranker = None
    if args.rerank:
        reranker = lambda query, results: rerank_results(  # noqa: E731
            query, results, model=args.rerank_model)

    summary_rows = []
    for db, method, run_search in searchers:
        log.info(f"\nEvaluating {db['rel']} ...")
        if db["type"] == "bm25":
            vectors = None
            method_meta = {"tokenizer": method}
        elif db["type"] == "hybrid":
            vectors = vectors_by_model[method]
            method_meta = {"embedding_model": method, "tokenizer": db["tokenizer"],
                           "alpha": db["alpha"], "vector_db": db["vec"]["rel"],
                           "bm25_db": db["bm25"]["rel"]}
        else:
            vectors = vectors_by_model[method]
            method_meta = {"embedding_model": method}
        records, aggregates, rr_records, rr_aggregates = evaluate_db(
            run_search, questions, vectors, ks,
            hybrid=db["type"] == "hybrid", reranker=reranker)

        stem = db["stem"] if db["type"] == "hybrid" else db["path"].stem
        variants = [(stem, {}, records, aggregates)]
        if rr_records:
            variants.append((f"{stem}_rerank",
                             {"rerank": args.rerank, "rerank_model": args.rerank_model},
                             rr_records, rr_aggregates))
        for out_stem, rerank_meta, out_records, out_aggregates in variants:
            out = {
                "metadata": {
                    "datetime": now.isoformat(timespec="seconds"),
                    "qa_file": qa["rel"],
                    "chunk_run": qa_chunk_run,
                    "db": db["rel"],
                    "db_type": db["type"],
                    **method_meta,
                    **rerank_meta,
                    "top_k": args.topk,
                    "ks": ks,
                    "num_questions": len(questions),
                    "relevance": RELEVANCE_NOTE,
                },
                "aggregates": out_aggregates,
                "questions": out_records,
            }
            out_file = out_dir / f"eval_{stamp}_{out_stem}.json"
            out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

            if rerank_meta:
                log.info(f"  with {rerank_meta['rerank']} rerank ({rerank_meta['rerank_model']}):")
            print_aggregates(out_aggregates, ks)
            log.info(f"    -> {out_file.relative_to(ROOT)}")
            summary_rows.append({
                "db": db["rel"],
                "db_type": db["type"],
                **method_meta,
                **rerank_meta,
                "eval_file": out_file.name,
                "overall": out_aggregates["overall"],
                "by_question_type": out_aggregates["by_question_type"],
            })

    if len(summary_rows) > 1:
        summary = {
            "metadata": {
                "datetime": now.isoformat(timespec="seconds"),
                "qa_file": qa["rel"],
                "chunk_run": qa_chunk_run,
                "top_k": args.topk,
                "ks": ks,
                "num_questions": len(questions),
                "relevance": RELEVANCE_NOTE,
            },
            "indexes": summary_rows,
        }
        summary_file = out_dir / f"eval_summary_{stamp}.json"
        summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        log.info(f"\nSummary: {summary_file.relative_to(ROOT)}")

    log.info(f"\nDone: {len(summary_rows)} index(es) evaluated on {len(questions)} questions")


if __name__ == "__main__":
    main()
