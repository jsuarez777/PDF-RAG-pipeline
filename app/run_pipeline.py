#!/usr/bin/env python3
"""End-to-end RAG evaluation pipeline: extract -> chunk -> QA -> index -> eval.

Runs the existing stage scripts (chunk_text.py, generate_qa.py, index_bm25.py,
embed_chunks.py, eval_retrieval.py) over a grid of configurations:

  chunking strategies x embedding models x retrieval methods (bm25/vector/hybrid)

and collects every experiment's IR metrics into a single results file with the
best configuration by MRR. Input is either an already-extracted dataset
(--dataset data/<extractor>/<date>/<title>) or a PDF (--pdf) that is first
extracted with --method pdfplumber (text+tables) or pdf2image (page images +
EasyOCR).

Examples:
  python app/run_pipeline.py --dataset data/pdfplumber/20260715_01/uk_knowledge_and_innovation_analysis \\
      --chunk-types "fixed_size:256:50,fixed_size:512:100,sentence:5:1" \\
      --embeddings small,large --retrievals bm25,vector,hybrid --qa-num 20
  python app/run_pipeline.py --pdf docs/report.pdf --method pdfplumber --dry-run

Results go to <title>/pipeline_runs/pipeline_<datetime>.json and a summary
table is printed (Rich if available).
"""

import argparse
import json
import os
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (Path(os.environ["PDF_DATA_DIR"]) if os.environ.get("PDF_DATA_DIR")
            else ROOT / "data")  # per-user override set by the web viewer
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from chunk_text import parse_type  # noqa: E402
from logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

EXTRACT_METHODS = ("pdfplumber", "pdf2image")
RETRIEVALS = ("bm25", "vector", "hybrid")
EMBEDDING_ALIASES = {"small": "text-embedding-3-small", "large": "text-embedding-3-large"}
EMBEDDING_MODELS = tuple(EMBEDDING_ALIASES.values())
TOKENIZERS = ("simple", "word", "porter")
VECTOR_DBS = ("milvus", "chromadb")
QA_TYPES = ("direct", "inference", "paraphrased")

DEFAULT_CHUNK_TYPES = "fixed_size:256:50,fixed_size:512:100,sentence:5:1"
DEFAULT_ALPHA = 0.7


# ------------------------------------------------------------- configuration


def parse_csv(arg, valid, what, aliases=None):
    """Split a comma-separated option, map aliases, and validate every token."""
    out = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        name = (aliases or {}).get(token, token)
        if name not in valid:
            sys.exit(f"ERROR: unknown {what} '{token}'. Valid: {', '.join(valid)}")
        if name not in out:
            out.append(name)
    if not out:
        sys.exit(f"ERROR: at least one {what} is required")
    return out


def parse_vector_configs(arg):
    """Parse 'small:milvus,small:chromadb,large:milvus' into {model: [db, ...]},
    preserving order, so each embedding model gets its own set of vector DBs."""
    embed_dbs = {}
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        if token.count(":") != 1:
            sys.exit(f"ERROR: --vector-configs entry must be model:db, got '{token}'")
        raw_model, db = token.split(":")
        model = EMBEDDING_ALIASES.get(raw_model, raw_model)
        if model not in EMBEDDING_MODELS:
            sys.exit(f"ERROR: unknown embedding model '{raw_model}'. "
                     f"Valid: {', '.join(EMBEDDING_ALIASES)}")
        if db not in VECTOR_DBS:
            sys.exit(f"ERROR: unknown vector db '{db}'. Valid: {', '.join(VECTOR_DBS)}")
        dbs = embed_dbs.setdefault(model, [])
        if db not in dbs:
            dbs.append(db)
    if not embed_dbs:
        sys.exit("ERROR: --vector-configs needs at least one model:db pair")
    return embed_dbs


def parse_chunk_specs(arg):
    """Validate a comma-separated list of chunk_text.py --type specs."""
    specs = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        parse_type(token)  # exits with a message on an invalid spec
        if token not in specs:
            specs.append(token)
    if not specs:
        sys.exit("ERROR: at least one chunk type is required")
    return specs


def chunk_label(spec):
    """Filesystem/report-friendly label for a chunk spec: fixed_size:256:50 -> fixed_size_256_50."""
    return spec.replace(":", "_").replace("%", "pct")


def enumerate_experiments(chunk_specs, embed_dbs, tokenizers, retrievals, alphas):
    """One descriptor per (chunk config, retrieval variant) cell of the grid.

    embed_dbs maps each embedding model to the vector DBs chosen for it, so
    bm25 contributes one experiment per tokenizer, vector one per (model, db),
    and hybrid one per (model, db) x tokenizer x alpha combination."""
    experiments = []
    for spec in chunk_specs:
        label = chunk_label(spec)
        if "bm25" in retrievals:
            for tok in tokenizers:
                experiments.append({
                    "experiment_id": f"{tok}_{label}_bm25",
                    "chunk_type": spec, "retrieval": "bm25", "tokenizer": tok,
                })
        if "vector" in retrievals:
            for model, dbs in embed_dbs.items():
                for db in dbs:
                    experiments.append({
                        "experiment_id": f"{model}_{db}_{label}_vector",
                        "chunk_type": spec, "retrieval": "vector",
                        "embedding_model": model, "db": db,
                    })
        if "hybrid" in retrievals:
            for model, dbs in embed_dbs.items():
                for db in dbs:
                    for tok in tokenizers:
                        for alpha in alphas:
                            experiments.append({
                                "experiment_id": f"{model}+{db}+{tok}_a{alpha:g}_{label}_hybrid",
                                "chunk_type": spec, "retrieval": "hybrid",
                                "embedding_model": model, "db": db, "tokenizer": tok,
                                "alpha": alpha,
                            })
    return experiments


# -------------------------------------------------------------- orchestration


def run_step(cmd):
    """Run a stage script, streaming its output; exit the pipeline if it fails."""
    log.info("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        log.info(line.rstrip())
    code = proc.wait()
    if code != 0:
        sys.exit(f"ERROR: pipeline step failed with exit code {code}: {' '.join(cmd)}")


def new_entries(directory, before, pattern="*"):
    """Entries in `directory` matching `pattern` that were not in `before`."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob(pattern) if p not in before)


def snapshot(directory, pattern="*"):
    return set(directory.glob(pattern)) if directory.is_dir() else set()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def resolve_title_dir(arg):
    """Validate --dataset points at a data/<extractor>/<date>/<title> dir."""
    path = (ROOT / arg) if not Path(arg).is_absolute() else Path(arg)
    path = path.resolve()
    if not path.is_dir():
        sys.exit(f"ERROR: dataset dir not found: {arg}")
    try:
        parts = path.relative_to(DATA_DIR).parts
    except ValueError:
        sys.exit(f"ERROR: dataset must live under {DATA_DIR}")
    if len(parts) != 3 or parts[0] not in EXTRACT_METHODS:
        sys.exit("ERROR: dataset must be data/<pdfplumber|pdf2image>/<date>/<title>")
    return path


def extract_pdf(pdf_arg, method):
    """Extract a PDF into a fresh dataset dir; returns the title dir."""
    pdf_path = Path(pdf_arg).expanduser().resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        sys.exit(f"ERROR: not a PDF file: {pdf_arg}")
    log.info(f"\n=== Extracting {pdf_path.name} with {method} ===")
    if method == "pdfplumber":
        from pdfplumber_to_text import convert as convert_text
        return convert_text(pdf_path)
    from pdf_to_images import convert as convert_images
    out_dir = convert_images(pdf_path)
    # pdf2image pages still need OCR text before they can be chunked
    run_step([sys.executable, str(SCRIPT_DIR / "easyocr_pdfimages.py"),
              "--missing-only", str(out_dir)])
    return out_dir


# ----------------------------------------------------------------- reporting


def metrics_from_eval(eval_path):
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    return data.get("metadata", {}), data.get("aggregates", {}).get("overall", {})


def match_experiment(experiments, chunk_spec, meta):
    """Find the grid descriptor an eval file belongs to."""
    db_type = meta.get("db_type")
    retrieval = {"milvus": "vector", "chromadb": "vector"}.get(db_type, db_type)
    for exp in experiments:
        if exp["chunk_type"] != chunk_spec or exp["retrieval"] != retrieval:
            continue
        if retrieval == "bm25" and exp["tokenizer"] == meta.get("tokenizer"):
            return exp
        if (retrieval == "vector" and exp["embedding_model"] == meta.get("embedding_model")
                and exp["db"] == db_type):
            return exp
        if (retrieval == "hybrid" and exp["embedding_model"] == meta.get("embedding_model")
                and exp["db"] == db_type and exp["tokenizer"] == meta.get("tokenizer")
                and abs(exp["alpha"] - (meta.get("alpha") or 0)) < 1e-9):
            return exp
    return None


def print_summary(rows, ks_last):
    headers = ["Experiment", f"Recall@{ks_last}", "MRR", "MAP", f"NDCG@{ks_last}", "Avg Time (s)"]

    def cells(row):
        m = row["metrics"]
        return [row["experiment_id"], f"{m.get(f'recall@{ks_last}', 0):.3f}",
                f"{m.get('mrr', 0):.3f}", f"{m.get('map', 0):.3f}",
                f"{m.get(f'ndcg@{ks_last}', 0):.3f}",
                f"{m.get('avg_retrieval_time', 0):.4f}"]

    try:
        from rich.console import Console
        from rich.table import Table
        table = Table(title="RAG Evaluation Results")
        for h in headers:
            table.add_column(h, justify="right" if h != "Experiment" else "left")
        for row in rows:
            table.add_row(*cells(row))
        Console().print(table)
    except ImportError:
        widths = [max(len(h), *(len(cells(r)[i]) for r in rows)) for i, h in enumerate(headers)]
        log.info("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        for row in rows:
            log.info("  ".join(c.ljust(w) for c, w in zip(cells(row), widths)))


# --------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(
        description="Run the full RAG pipeline grid: extract, chunk, QA, index, evaluate.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--pdf", help="PDF file to extract first")
    src.add_argument("--dataset", help="Existing dataset dir (data/<extractor>/<date>/<title>)")
    parser.add_argument("--method", default="pdfplumber", choices=EXTRACT_METHODS,
                        help="PDF extraction method for --pdf (default pdfplumber)")
    parser.add_argument("--chunk-types", default=DEFAULT_CHUNK_TYPES,
                        help="Comma-separated chunk_text.py specs "
                             f"(default {DEFAULT_CHUNK_TYPES})")
    parser.add_argument("--embeddings", default="small,large",
                        help="Embedding models, comma-separated: small, large, or full names "
                             "(default small,large)")
    parser.add_argument("--vector-db", default="milvus", choices=VECTOR_DBS,
                        help="Vector store for every embedding model, unless "
                             "--vector-configs is given (default milvus)")
    parser.add_argument("--vector-configs",
                        help="Per-model vector DBs as model:db pairs, e.g. "
                             "'small:milvus,small:chromadb,large:milvus'. Overrides "
                             "--embeddings/--vector-db when given.")
    parser.add_argument("--tokenizers", default="word",
                        help=f"BM25 tokenizers, comma-separated: {', '.join(TOKENIZERS)} "
                             "(default word)")
    parser.add_argument("--retrievals", default="bm25,vector,hybrid",
                        help="Retrieval methods to evaluate, comma-separated: "
                             f"{', '.join(RETRIEVALS)} (default all)")
    parser.add_argument("--alpha", default=str(DEFAULT_ALPHA),
                        help="Hybrid vector weight; comma-separated to sweep several "
                             f"weights (default {DEFAULT_ALPHA})")
    parser.add_argument("--qa-num", type=int, default=20,
                        help="Chunks to sample per chunk config for QA generation (default 20)")
    parser.add_argument("--qa-types", default=",".join(QA_TYPES),
                        help=f"QA question types, comma-separated (default all: {','.join(QA_TYPES)})")
    parser.add_argument("--qa-model", default="gpt-4.1-mini",
                        help="Chat model for QA generation (default gpt-4.1-mini)")
    parser.add_argument("--seed", type=int, help="Random seed for QA chunk sampling")
    parser.add_argument("--topk", type=int, default=10, help="Results per query (default 10)")
    parser.add_argument("--ks", default="1,3,5,10",
                        help="Metric cutoffs, comma-separated (default 1,3,5,10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the experiment grid and count, then exit")
    args = parser.parse_args()

    setup_logging("run_pipeline")

    chunk_specs = parse_chunk_specs(args.chunk_types)
    if args.vector_configs:
        embed_dbs = parse_vector_configs(args.vector_configs)
    else:
        embeddings = parse_csv(args.embeddings, EMBEDDING_MODELS, "embedding model",
                               aliases=EMBEDDING_ALIASES)
        embed_dbs = {model: [args.vector_db] for model in embeddings}
    embeddings = list(embed_dbs)
    tokenizers = parse_csv(args.tokenizers, TOKENIZERS, "tokenizer")
    retrievals = parse_csv(args.retrievals, RETRIEVALS, "retrieval method")
    qa_types = parse_csv(args.qa_types, QA_TYPES, "question type")
    try:
        alphas = [float(a) for a in args.alpha.split(",") if a.strip() != ""]
    except ValueError:
        sys.exit("ERROR: --alpha must be a number or comma-separated numbers")
    if not alphas or any(not 0 <= a <= 1 for a in alphas):
        sys.exit("ERROR: --alpha must be between 0 and 1")
    alphas = list(dict.fromkeys(alphas))
    if args.qa_num <= 0:
        sys.exit("ERROR: --qa-num must be > 0")

    experiments = enumerate_experiments(chunk_specs, embed_dbs, tokenizers,
                                        retrievals, alphas)
    log.info(f"Pipeline grid: {len(chunk_specs)} chunk config(s) -> "
             f"{len(experiments)} experiment(s)")
    for exp in experiments:
        log.info(f"  {exp['experiment_id']}")
    if args.dry_run:
        return

    # ---------------------------------------------------------------- source
    if args.pdf:
        title_dir = extract_pdf(args.pdf, args.method)
    elif args.dataset:
        title_dir = resolve_title_dir(args.dataset)
    else:
        sys.exit("ERROR: pass --pdf or --dataset")
    log.info(f"Dataset: {rel(title_dir)}")

    needs_bm25 = "bm25" in retrievals or "hybrid" in retrievals
    needs_vector = "vector" in retrievals or "hybrid" in retrievals

    now = datetime.now()
    all_rows = []
    per_chunk = []

    for spec in chunk_specs:
        log.info(f"\n=== Chunk config: {spec} ===")

        # 1. chunk
        before = snapshot(title_dir, "*_chunk_*")
        run_step([sys.executable, str(SCRIPT_DIR / "chunk_text.py"),
                  "--type", spec, "--dataset", rel(title_dir)])
        created = [d for d in new_entries(title_dir, before, "*_chunk_*") if d.is_dir()]
        if len(created) != 1:
            sys.exit(f"ERROR: expected 1 new chunk dir, found {len(created)}")
        chunk_dir = created[0]

        # 2. QA dataset for this chunk config (fair eval: one per config)
        before = snapshot(chunk_dir, "qa_*.json")
        qa_cmd = [sys.executable, str(SCRIPT_DIR / "generate_qa.py"),
                  "--dataset", rel(chunk_dir), "--num-chunks", str(args.qa_num),
                  "--types", ",".join(qa_types), "--model", args.qa_model]
        if args.seed is not None:
            qa_cmd += ["--seed", str(args.seed)]
        run_step(qa_cmd)
        qa_files = new_entries(chunk_dir, before, "qa_*.json")
        if len(qa_files) != 1:
            sys.exit(f"ERROR: expected 1 new QA file, found {len(qa_files)}")
        qa_file = qa_files[0]

        # 3. indexes
        bm25_files, vector_files = [], []
        if needs_bm25:
            before = snapshot(title_dir / "bm25", "bm25_*.pkl")
            run_step([sys.executable, str(SCRIPT_DIR / "index_bm25.py"),
                      "--tokenizer", ",".join(tokenizers), "--dataset", rel(chunk_dir)])
            bm25_files = new_entries(title_dir / "bm25", before, "bm25_*.pkl")
        if needs_vector:
            edb = title_dir / "embedding_databases"
            before = snapshot(edb)
            # group models by db so each db is built only for the models that
            # asked for it (one embed_chunks.py call per db, over its models)
            models_per_db = {}
            for model, dbs in embed_dbs.items():
                for db in dbs:
                    models_per_db.setdefault(db, []).append(model)
            for db, models in models_per_db.items():
                run_step([sys.executable, str(SCRIPT_DIR / "embed_chunks.py"),
                          "--embedding", ",".join(models), "--db", db,
                          "--dataset", rel(chunk_dir)])
            vector_files = [p for p in new_entries(edb, before)
                            if (p.name.startswith("milvus_") and p.suffix == ".db")
                            or (p.name.startswith("chromadb_") and p.suffix == ".chroma")]

        # 4. evaluate
        eval_dbs = []
        if "bm25" in retrievals:
            eval_dbs += [rel(p) for p in bm25_files]
        if "vector" in retrievals:
            eval_dbs += [rel(p) for p in vector_files]
        eval_cmd = [sys.executable, str(SCRIPT_DIR / "eval_retrieval.py"),
                    "--qa", rel(qa_file), "--topk", str(args.topk), "--ks", args.ks]
        if eval_dbs:
            eval_cmd += ["--db", ",".join(eval_dbs)]
        if "hybrid" in retrievals:
            pairs = [f"{rel(v)}+{rel(b)}" for v in vector_files for b in bm25_files]
            eval_cmd += ["--hybrid", ",".join(pairs),
                         "--alpha", ",".join(f"{a:g}" for a in alphas)]
        eval_dir = title_dir / "evaluations"
        before = snapshot(eval_dir, "eval_*.json")
        run_step(eval_cmd)
        eval_files = [p for p in new_entries(eval_dir, before, "eval_*.json")
                      if not p.name.startswith("eval_summary_")]

        # 5. collect metrics
        for ef in eval_files:
            meta, overall = metrics_from_eval(ef)
            exp = match_experiment(experiments, spec, meta)
            row = {
                "experiment_id": exp["experiment_id"] if exp else ef.stem,
                "chunk_type": spec,
                "chunk_run": rel(chunk_dir),
                "qa_file": rel(qa_file),
                "eval_file": rel(ef),
                "retrieval": (exp or {}).get("retrieval", meta.get("db_type")),
                "embedding_model": meta.get("embedding_model"),
                "db": meta.get("db_type"),
                "tokenizer": meta.get("tokenizer"),
                "alpha": meta.get("alpha"),
                "metrics": overall,
            }
            all_rows.append(row)
        per_chunk.append({"chunk_type": spec, "chunk_run": rel(chunk_dir),
                          "qa_file": rel(qa_file),
                          "bm25_indexes": [rel(p) for p in bm25_files],
                          "vector_indexes": [rel(p) for p in vector_files]})

    # ------------------------------------------------------------- reporting
    all_rows.sort(key=lambda r: -(r["metrics"].get("mrr") or 0))
    ks_last = max(int(k) for k in args.ks.split(","))
    log.info("")
    print_summary(all_rows, ks_last)

    best = all_rows[0] if all_rows else None
    if best:
        log.info(f"\nBest configuration by MRR: {best['experiment_id']}  "
                 f"(MRR={best['metrics'].get('mrr', 0):.3f})")
    log.info(f"Total experiments: {len(all_rows)}")

    out = {
        "metadata": {
            "datetime": now.isoformat(timespec="seconds"),
            "dataset": rel(title_dir),
            "chunk_types": chunk_specs,
            "embeddings": embeddings,
            "embed_dbs": embed_dbs,
            "tokenizers": tokenizers,
            "retrievals": retrievals,
            "alpha": alphas[0] if len(alphas) == 1 else alphas,
            "alphas": alphas,
            "qa_num": args.qa_num,
            "qa_types": qa_types,
            "qa_model": args.qa_model,
            "seed": args.seed,
            "topk": args.topk,
            "ks": args.ks,
            "num_experiments": len(all_rows),
        },
        "chunk_configs": per_chunk,
        "experiments": all_rows,
        "best_by_mrr": best,
    }
    out_dir = title_dir / "pipeline_runs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"pipeline_{now.strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"\nResults: {rel(out_file)}")


if __name__ == "__main__":
    main()
