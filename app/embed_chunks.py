#!/usr/bin/env python3
"""Embed a chunked_text.json chunk run and store the vectors in a vector DB.

Chunk runs live under data/{pdf2image,pdfplumber}/<date>/<title>/<chunk dir>/
where <chunk dir> looks like 20260712_101112_chunk_fixed_size_100_20 and
contains a chunked_text.json produced by chunk_text.py.

Embedding models: text-embedding-3-small, text-embedding-3-large.
Vector DBs: milvus (Milvus Lite .db file), chromadb (persistent dir).

Outputs go to <title>/embedding_databases/<db>_<datetime>_<model>.<ext>
  milvus   -> milvus_<dt>_<model>.db  (+ milvus_<dt>_<model>.json metadata sidecar)
  chromadb -> chromadb_<dt>_<model>.chroma  (directory)

--all-options embeds with every model and stores each in every DB.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

from openai_client.openai_client import MyOpenAIClient  # noqa: E402

EXTRACTORS = ("pdf2image", "pdfplumber")

EMBEDDING_MODELS = ("text-embedding-3-small", "text-embedding-3-large")
EMBEDDING_ALIASES = {
    "small": "text-embedding-3-small",
    "large": "text-embedding-3-large",
}

VECTOR_DBS = ("milvus", "chromadb")

EMBED_BATCH_SIZE = 128


# ---------------------------------------------------------------- selection


def resolve_embeddings(arg):
    """Turn a comma-separated --embedding value into a list of model names."""
    models = []
    for token in arg.split(","):
        token = token.strip()
        name = EMBEDDING_ALIASES.get(token, token)
        if name not in EMBEDDING_MODELS:
            sys.exit(
                f"ERROR: unknown embedding '{token}'. "
                f"Valid: {', '.join(EMBEDDING_MODELS)} (or small/large)"
            )
        if name not in models:
            models.append(name)
    return models


def resolve_dbs(arg):
    """Turn a comma-separated --db value into a list of db names."""
    dbs = []
    for token in arg.split(","):
        token = token.strip().lower()
        if token == "chroma":
            token = "chromadb"
        if token not in VECTOR_DBS:
            sys.exit(f"ERROR: unknown vector db '{token}'. Valid: {', '.join(VECTOR_DBS)}")
        if token not in dbs:
            dbs.append(token)
    return dbs


def choose_from_menu(prompt, options):
    """Numbered menu; accepts a number, a name, or comma-separated mix."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    print(f"  [{len(options) + 1}] all of the above")
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


def scan_chunk_runs():
    """Return (runs, empty_datasets).

    runs: one dict per chunk dir found under data/<extractor>/<date>/<title>/.
    empty_datasets: rel paths of title dirs that have no chunk dirs at all.
    """
    runs = []
    empty = []
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
                if not chunk_dirs:
                    empty.append(title_dir.relative_to(ROOT).as_posix())
                    continue
                for cd in chunk_dirs:
                    runs.append({
                        "path": cd,
                        "rel": cd.relative_to(ROOT).as_posix(),
                        "title_dir": title_dir,
                    })
    return runs, empty


def choose_chunk_run(runs, empty, preselect=None):
    print("\nAvailable datasets:")
    if not runs and not empty:
        sys.exit("ERROR: no datasets found under data/pdf2image or data/pdfplumber")
    for rel in empty:
        print(f"  [-] {rel}  (No chunks)")
    for i, run in enumerate(runs, 1):
        print(f"  [{i}] {run['rel']}")

    if not runs:
        sys.exit("ERROR: no chunk runs found; run chunk_text.py first")

    if preselect is not None:
        choice = preselect
        print(f"\nChunk run (from --dataset): {choice}")
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
    print(f"Loaded {len(chunks)} chunks from {run['rel']}")
    return data.get("metadata", {}), chunks


# ---------------------------------------------------------------- embedding


def embed_texts(client, model, texts):
    """Embed texts in batches; return list of vectors (list[float])."""
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in resp.data)
        print(f"  embedded {min(start + len(batch), len(texts))}/{len(texts)} chunks", end="\r")
    print()
    if len(vectors) != len(texts):
        sys.exit(f"ERROR: got {len(vectors)} embeddings for {len(texts)} chunks")
    return vectors


# ------------------------------------------------------------------ storage


def build_metadata(run, chunk_meta, model, vectors, now):
    return {
        "datetime": now.isoformat(timespec="seconds"),
        "chunk_run": run["rel"],
        "chunk_metadata": chunk_meta,
        "embedding_model": model,
        "dimensions": len(vectors[0]),
        "num_vectors": len(vectors),
        "metric": "ip (cosine; OpenAI embeddings are unit-norm)",
    }


def store_milvus(out_dir, stamp, model, vectors, chunks, meta):
    try:
        from pymilvus import MilvusClient
    except ImportError:
        sys.exit("ERROR: pymilvus not installed. Run: pip install 'pymilvus[milvus_lite]'")

    out_file = out_dir / f"milvus_{stamp}_{model}.db"
    client = MilvusClient(str(out_file))
    client.create_collection(
        collection_name="chunks",
        dimension=len(vectors[0]),
        metric_type="IP",
        auto_id=False,
    )
    client.insert(
        collection_name="chunks",
        data=[
            {
                "id": c["chunk_index"],
                "vector": v,
                "text": c["text"],
                **{k: c[k] for k in ("start_char", "end_char", "start_page", "end_page")
                   if k in c},
            }
            for c, v in zip(chunks, vectors)
        ],
    )
    client.close()

    sidecar = out_file.with_suffix(".json")
    sidecar.write_text(json.dumps({"metadata": meta}, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"  milvus:   {out_file.relative_to(ROOT)} (+ sidecar {sidecar.name})")


def store_chromadb(out_dir, stamp, model, vectors, chunks, meta):
    try:
        import chromadb
    except ImportError:
        sys.exit("ERROR: chromadb not installed. Run: pip install chromadb")

    out_path = out_dir / f"chromadb_{stamp}_{model}.chroma"
    client = chromadb.PersistentClient(path=str(out_path))
    collection = client.create_collection(
        name="chunks",
        metadata={
            "embedding_model": model,
            "chunk_run": meta["chunk_run"],
            "datetime": meta["datetime"],
            "hnsw:space": "ip",
        },
    )
    collection.add(
        ids=[str(c["chunk_index"]) for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[
            {k: c[k] for k in ("chunk_index", "start_char", "end_char", "start_page", "end_page")
             if k in c}
            for c in chunks
        ],
    )
    print(f"  chromadb: {out_path.relative_to(ROOT)}")


STORERS = {"milvus": store_milvus, "chromadb": store_chromadb}


# --------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="Embed a chunk run and store in vector DB(s).")
    parser.add_argument(
        "--embedding",
        help="Embedding model(s), comma-separated: small, large, or full names "
             f"({', '.join(EMBEDDING_MODELS)})",
    )
    parser.add_argument(
        "--db",
        help=f"Vector db(s), comma-separated: {', '.join(VECTOR_DBS)}. "
             "If both are passed, embeddings are stored in both independently.",
    )
    parser.add_argument("--dataset", help="Chunk run path (data/.../<title>/<chunk dir>) or list number")
    parser.add_argument(
        "--all-options",
        action="store_true",
        help="Generate every embedding model and store each in every vector db",
    )
    args = parser.parse_args()

    if args.all_options:
        models = list(EMBEDDING_MODELS)
        dbs = list(VECTOR_DBS)
    else:
        if args.embedding:
            models = resolve_embeddings(args.embedding)
        else:
            models = resolve_embeddings(",".join(choose_from_menu("Embedding model:", EMBEDDING_MODELS)))
        if args.db:
            dbs = resolve_dbs(args.db)
        else:
            dbs = resolve_dbs(",".join(choose_from_menu("Vector db:", VECTOR_DBS)))
    if not models or not dbs:
        sys.exit("ERROR: at least one embedding model and one vector db are required")

    runs, empty = scan_chunk_runs()
    run = choose_chunk_run(runs, empty, preselect=args.dataset)
    chunk_meta, chunks = load_chunks(run)
    texts = [c["text"] for c in chunks]

    out_dir = run["title_dir"] / "embedding_databases"
    out_dir.mkdir(exist_ok=True)

    api = MyOpenAIClient(model=models[0])
    api.validate_api_key()
    client = api.get_client()

    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")

    for model in models:
        print(f"\nEmbedding {len(texts)} chunks with {model} ...")
        vectors = embed_texts(client, model, texts)
        meta = build_metadata(run, chunk_meta, model, vectors, now)
        for db in dbs:
            STORERS[db](out_dir, stamp, model, vectors, chunks, meta)

    print(f"\nDone: {len(models)} embedding model(s) x {len(dbs)} db(s) "
          f"-> {len(models) * len(dbs)} output(s) in {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
