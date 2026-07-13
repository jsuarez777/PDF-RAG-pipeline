#!/usr/bin/env python3
"""Build a BM25 index from a chunked_text.json chunk run and pickle it.

Chunk runs live under data/{pdf2image,pdfplumber}/<date>/<title>/<chunk dir>/
where <chunk dir> looks like 20260712_101112_chunk_fixed_size_100_20 and
contains a chunked_text.json produced by chunk_text.py.

Tokenizers: simple (lowercase whitespace split), word (lowercase \\w+ regex),
porter (word + NLTK Porter stemming).

Outputs go to <title>/bm25/bm25_<datetime>_<tokenizer>.pkl with a
bm25_<datetime>_<tokenizer>.json metadata sidecar. The pickle is
self-contained: it holds the BM25Okapi object, the tokenizer name, and the
chunks (with text) so retrieval does not need the original chunk run.

--all-options builds an index with every tokenizer.
"""

import argparse
import json
import logging
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

EXTRACTORS = ("pdf2image", "pdfplumber")

WORD_RE = re.compile(r"\w+")


def tokenize_simple(text):
    return text.lower().split()


def tokenize_word(text):
    return WORD_RE.findall(text.lower())


def tokenize_porter(text):
    try:
        from nltk.stem import PorterStemmer
    except ImportError:
        sys.exit("ERROR: nltk not installed. Run: pip install nltk")
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in WORD_RE.findall(text.lower())]


TOKENIZERS = {
    "simple": tokenize_simple,
    "word": tokenize_word,
    "porter": tokenize_porter,
}


# ---------------------------------------------------------------- selection


def resolve_tokenizers(arg):
    """Turn a comma-separated --tokenizer value into a list of tokenizer names."""
    names = []
    for token in arg.split(","):
        token = token.strip().lower()
        if token not in TOKENIZERS:
            sys.exit(f"ERROR: unknown tokenizer '{token}'. Valid: {', '.join(TOKENIZERS)}")
        if token not in names:
            names.append(token)
    return names


def choose_from_menu(prompt, options):
    """Numbered menu; accepts a number, a name, or comma-separated mix."""
    log.info(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        log.info(f"  [{i}] {opt}")
    log.info(f"  [{len(options) + 1}] all of the above")
    choice = input("Choice (number/name, comma-separated for several): ").strip()
    if choice == str(len(options) + 1) or choice.lower() == "all":
        return list(options)
    picked = []
    for token in choice.split(","):
        token = token.strip()
        if token.isdigit() and 1 <= int(token) <= len(options):
            name = options[int(token) - 1]
        else:
            name = token
        if name not in picked:
            picked.append(name)
    return picked


# ----------------------------------------------------------------- datasets


def scan_titles():
    """Return one dict per data/<extractor>/<date>/<title> dir, with its chunk runs."""
    titles = []
    for extractor in EXTRACTORS:
        base = DATA_DIR / extractor
        if not base.is_dir():
            continue
        for date_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for title_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
                chunk_dirs = sorted(
                    p for p in title_dir.iterdir()
                    if p.is_dir() and "_chunk_" in p.name and (p / "chunked_text.json").is_file()
                )
                titles.append({
                    "title_dir": title_dir,
                    "rel": title_dir.relative_to(ROOT).as_posix(),
                    "runs": [
                        {
                            "path": cd,
                            "rel": cd.relative_to(ROOT).as_posix(),
                            "title_dir": title_dir,
                        }
                        for cd in chunk_dirs
                    ],
                })
    return titles


def choose_chunk_run(titles, preselect=None):
    """List every title with its chunk runs on separate lines; pick one run."""
    if not titles:
        sys.exit("ERROR: no datasets found under data/pdf2image or data/pdfplumber")

    runs = []
    log.info("\nAvailable datasets:")
    for t in titles:
        log.info(f"{t['rel']}")
        if not t["runs"]:
            log.info("      No chunks available")
            continue
        for run in t["runs"]:
            runs.append(run)
            log.info(f"  [{len(runs)}] {run['path'].name}")

    if not runs:
        sys.exit("ERROR: no chunk runs found; run chunk_text.py first")

    if preselect is not None:
        choice = preselect
        log.info(f"\nChunk run (from --dataset): {choice}")
    else:
        choice = input("\nChoose a chunk run (number or path): ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(runs):
        return runs[int(choice) - 1]
    norm = choice.rstrip("/")
    for run in runs:
        if norm in (run["rel"], str(run["path"])):
            return run
    sys.exit(f"ERROR: '{choice}' is not a valid chunk run choice")


def load_chunks(run):
    """Load and validate chunked_text.json; return (metadata, chunks)."""
    jpath = run["path"] / "chunked_text.json"
    try:
        data = json.loads(jpath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        sys.exit(f"ERROR: could not read {jpath.relative_to(ROOT)}: {e}")
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        sys.exit(f"ERROR: {jpath.relative_to(ROOT)} has no chunks")
    for c in chunks:
        if not isinstance(c.get("text"), str):
            sys.exit(f"ERROR: chunk {c.get('chunk_index')} in {jpath.relative_to(ROOT)} has no text")
    log.info(f"Loaded {len(chunks)} chunks from {run['rel']}")
    return data.get("metadata", {}), chunks


# ------------------------------------------------------------------ indexing


def build_index(out_dir, stamp, tokenizer_name, run, chunk_meta, chunks, now):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        sys.exit("ERROR: rank_bm25 not installed. Run: pip install rank-bm25")

    tokenize = TOKENIZERS[tokenizer_name]
    tokenized = [tokenize(c["text"]) for c in chunks]
    if not any(tokenized):
        sys.exit(f"ERROR: tokenizer '{tokenizer_name}' produced no tokens for any chunk")
    bm25 = BM25Okapi(tokenized)

    out_file = out_dir / f"bm25_{stamp}_{tokenizer_name}.pkl"
    with out_file.open("wb") as f:
        pickle.dump({
            "bm25": bm25,
            "tokenizer": tokenizer_name,
            "chunk_run": run["rel"],
            "chunks": chunks,
        }, f)

    meta = {
        "datetime": now.isoformat(timespec="seconds"),
        "chunk_run": run["rel"],
        "chunk_metadata": chunk_meta,
        "tokenizer": tokenizer_name,
        "num_chunks": len(chunks),
        "vocab_size": len(bm25.idf),
        "avg_doc_len_tokens": bm25.avgdl,
        "chunks": [
            {k: c[k] for k in ("chunk_index", "start_char", "end_char",
                               "num_chars", "start_page", "end_page") if k in c}
            for c in chunks
        ],
    }
    sidecar = out_file.with_suffix(".json")
    sidecar.write_text(json.dumps({"metadata": meta}, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    log.info(f"  bm25 ({tokenizer_name}): {out_file.relative_to(ROOT)} (+ sidecar {sidecar.name})")


# --------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index(es) from a chunk run.")
    parser.add_argument(
        "--tokenizer",
        help=f"Tokenizer(s), comma-separated: {', '.join(TOKENIZERS)}",
    )
    parser.add_argument("--dataset", help="Chunk run path (data/.../<title>/<chunk dir>) or list number")
    parser.add_argument(
        "--all-options",
        action="store_true",
        help="Build an index with every tokenizer",
    )
    args = parser.parse_args()

    setup_logging("index_bm25")

    if args.all_options:
        tokenizers = list(TOKENIZERS)
    elif args.tokenizer:
        tokenizers = resolve_tokenizers(args.tokenizer)
    else:
        tokenizers = resolve_tokenizers(",".join(choose_from_menu("Tokenizer:", list(TOKENIZERS))))
    if not tokenizers:
        sys.exit("ERROR: at least one tokenizer is required")

    titles = scan_titles()
    run = choose_chunk_run(titles, preselect=args.dataset)
    chunk_meta, chunks = load_chunks(run)

    out_dir = run["title_dir"] / "bm25"
    out_dir.mkdir(exist_ok=True)

    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")

    log.info("")
    for name in tokenizers:
        build_index(out_dir, stamp, name, run, chunk_meta, chunks, now)

    log.info(f"\nDone: {len(tokenizers)} BM25 index(es) in {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
