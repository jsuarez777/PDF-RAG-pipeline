#!/usr/bin/env python3
"""Retrieve the top-k most relevant chunks for a query from a stored index.

Indexes are the outputs of embed_chunks.py under
data/<extractor>/<date>/<title>/embedding_databases/:
  milvus_<dt>_<model>.db        (Milvus Lite, + .json metadata sidecar)
  chromadb_<dt>_<model>.chroma  (ChromaDB persistent dir)
and of index_bm25.py under <title>/bm25/:
  bm25_<dt>_<tokenizer>.pkl     (+ .json metadata sidecar)

The query is given as --query-text "..." or --query-json <file>, where the
json file looks like {"type": "text", "query": "..."} (only type "text" for
now). For vector dbs the query is embedded with the same model the DB was
built with; for bm25 it is tokenized with the same tokenizer the index was
built with.

Results go to <title>/queries/<dt>_top<k>_<index name>.json containing the
query parameter metadata, the query, and the top-k results (rank, score,
and the full chunk with its metadata).
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

from openai_client.openai_client import MyOpenAIClient  # noqa: E402

EXTRACTORS = ("pdf2image", "pdfplumber")

QUERY_TYPES = ("text",)

CHUNK_FIELDS = ("chunk_index", "text", "start_char", "end_char", "start_page", "end_page")


# --------------------------------------------------------------- db catalog


def db_model_from_name(path):
    """Fallback: parse the embedding model out of <db>_<date>_<time>_<model>.<ext>."""
    parts = path.stem.split("_", 3)
    return parts[3] if len(parts) == 4 else None


def scan_dbs():
    """Return one dict per stored index found under embedding_databases/ or bm25/."""
    dbs = []
    for extractor in EXTRACTORS:
        base = DATA_DIR / extractor
        if not base.is_dir():
            continue
        for edb_dir in sorted(base.glob("*/*/embedding_databases")):
            for p in sorted(edb_dir.iterdir()):
                # Milvus Lite's .db is a file or a directory depending on version
                if p.name.startswith("milvus_") and p.suffix == ".db":
                    db_type = "milvus"
                elif p.name.startswith("chromadb_") and p.suffix == ".chroma" and p.is_dir():
                    db_type = "chromadb"
                else:
                    continue
                dbs.append({
                    "type": db_type,
                    "path": p,
                    "rel": p.relative_to(ROOT).as_posix(),
                    "title_dir": edb_dir.parent,
                })
        for bm_dir in sorted(base.glob("*/*/bm25")):
            for p in sorted(bm_dir.glob("bm25_*.pkl")):
                dbs.append({
                    "type": "bm25",
                    "path": p,
                    "rel": p.relative_to(ROOT).as_posix(),
                    "title_dir": bm_dir.parent,
                })
    return dbs


def choose_db(dbs, preselect=None):
    if not dbs:
        sys.exit("ERROR: no indexes found under data/*/*/*/{embedding_databases,bm25}; "
                 "run embed_chunks.py or index_bm25.py first")

    print("\nAvailable indexes:")
    for i, db in enumerate(dbs, 1):
        print(f"  [{i}] {db['rel']}")

    if preselect is not None:
        choice = preselect
        print(f"\nIndex (from --db): {choice}")
    else:
        choice = input("\nChoose an index (number or path): ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(dbs):
        return dbs[int(choice) - 1]
    norm = choice.rstrip("/")
    if norm.endswith(".json"):  # accept the bm25 sidecar as an alias for the pkl
        norm = norm[:-len(".json")] + ".pkl"
    for db in dbs:
        if norm in (db["rel"], str(db["path"])):
            return db
    # a directory (e.g. .../bm25) counts if it contains exactly one index
    matches = [db for db in dbs if db["path"].parent in (Path(norm), ROOT / norm)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        sys.exit(f"ERROR: '{choice}' contains {len(matches)} indexes; pass the file itself")
    sys.exit(f"ERROR: '{choice}' is not a valid index choice")


# -------------------------------------------------------------------- query


def load_query_json(path_str):
    """Read a query json file; return (query_text, source_rel)."""
    path = Path(path_str)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        sys.exit(f"ERROR: could not read query json {path_str}: {e}")
    qtype = data.get("type")
    if qtype not in QUERY_TYPES:
        sys.exit(f"ERROR: query json 'type' must be one of {', '.join(QUERY_TYPES)}, got '{qtype}'")
    query = data.get("query", data.get("text"))
    if not isinstance(query, str) or not query.strip():
        sys.exit(f"ERROR: query json {path_str} has no non-empty string 'query' field")
    return query, path.as_posix()


def get_query(args):
    """Return (query_text, source description) from args or interactive menu."""
    if args.query_text and args.query_json:
        sys.exit("ERROR: pass only one of --query-text / --query-json")
    if args.query_text:
        if not args.query_text.strip():
            sys.exit("ERROR: --query-text is empty")
        return args.query_text, "--query-text"
    if args.query_json:
        return load_query_json(args.query_json)

    print("\nQuery input:")
    print("  [1] text (type the query)")
    print("  [2] json file (path to {\"type\": \"text\", \"query\": ...})")
    choice = input("Choice: ").strip()
    if choice in ("1", "text"):
        query = input("Query text: ").strip()
        if not query:
            sys.exit("ERROR: query text is empty")
        return query, "interactive text"
    if choice in ("2", "json", "json file"):
        return load_query_json(input("Query json path: ").strip())
    sys.exit(f"ERROR: '{choice}' is not a valid query input choice")


def get_topk(arg):
    if arg is None:
        arg = input("\nTop-k (number of results): ").strip()
    try:
        k = int(arg)
    except ValueError:
        sys.exit(f"ERROR: top-k must be an integer, got '{arg}'")
    if k <= 0:
        sys.exit("ERROR: top-k must be > 0")
    return k


# ------------------------------------------------------------------- search


def search_milvus(db, k):
    """Return (embedding_model, results). Milvus IP 'distance' is the similarity."""
    try:
        from pymilvus import MilvusClient
    except ImportError:
        sys.exit("ERROR: pymilvus not installed. Run: pip install 'pymilvus[milvus_lite]'")

    sidecar = db["path"].with_suffix(".json")
    model = None
    if sidecar.is_file():
        try:
            model = json.loads(sidecar.read_text(encoding="utf-8"))["metadata"]["embedding_model"]
        except (json.JSONDecodeError, KeyError):
            pass
    model = model or db_model_from_name(db["path"])

    def run(vec):
        client = MilvusClient(str(db["path"]))
        client.load_collection("chunks")
        # dynamic fields must be named explicitly; "*" only returns the pk
        hits = client.search(collection_name="chunks", data=[vec], limit=k,
                             output_fields=[f for f in CHUNK_FIELDS if f != "chunk_index"])[0]
        client.close()
        results = []
        for rank, hit in enumerate(hits, 1):
            chunk = {"chunk_index": hit["id"]}
            chunk.update({f: hit["entity"][f] for f in CHUNK_FIELDS if f in hit["entity"]})
            results.append({"rank": rank, "score": hit["distance"],
                            "similarity": hit["distance"], "chunk": chunk})
        return results

    return model, run


def search_chromadb(db, k):
    """Return (embedding_model, results). Chroma ip distance = 1 - inner product."""
    try:
        import chromadb
    except ImportError:
        sys.exit("ERROR: chromadb not installed. Run: pip install chromadb")

    collection = chromadb.PersistentClient(path=str(db["path"])).get_collection("chunks")
    model = (collection.metadata or {}).get("embedding_model") or db_model_from_name(db["path"])

    def run(vec):
        res = collection.query(query_embeddings=[vec], n_results=k,
                               include=["documents", "metadatas", "distances"])
        results = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), 1
        ):
            chunk = {"chunk_index": int(cid), "text": doc}
            chunk.update({f: meta[f] for f in CHUNK_FIELDS if f in (meta or {})})
            results.append({"rank": rank, "score": dist,
                            "similarity": 1 - dist, "chunk": chunk})
        return results

    return model, run


def search_bm25(db, k):
    """Return (tokenizer_name, run). run takes the query text; score is the raw
    BM25 score (unbounded, higher is better)."""
    from app.index_bm25 import TOKENIZERS

    sidecar = db["path"].with_suffix(".json")
    if not sidecar.is_file():
        sys.exit(f"ERROR: {db['rel']} is missing its metadata sidecar {sidecar.name}")
    try:
        with db["path"].open("rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, OSError, EOFError, ImportError) as e:
        sys.exit(f"ERROR: could not load {db['rel']}: {e}")
    missing = [key for key in ("bm25", "tokenizer", "chunks") if key not in data]
    if missing:
        sys.exit(f"ERROR: {db['rel']} is missing expected data: {', '.join(missing)}")
    tokenizer = data["tokenizer"]
    if tokenizer not in TOKENIZERS:
        sys.exit(f"ERROR: {db['rel']} was built with unknown tokenizer '{tokenizer}'")

    def run(query):
        tokens = TOKENIZERS[tokenizer](query)
        if not tokens:
            sys.exit(f"ERROR: query produced no tokens with tokenizer '{tokenizer}'")
        scores = data["bm25"].get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        results = []
        for rank, i in enumerate(order, 1):
            chunk = {f: data["chunks"][i][f] for f in CHUNK_FIELDS if f in data["chunks"][i]}
            results.append({"rank": rank, "score": float(scores[i]),
                            "similarity": float(scores[i]), "chunk": chunk})
        return results

    return tokenizer, run


SEARCHERS = {"milvus": search_milvus, "chromadb": search_chromadb, "bm25": search_bm25}


EMBED_BATCH_SIZE = 1000


def embed_query(model, texts):
    """Embed a list of query texts; return one vector per text."""
    api = MyOpenAIClient(model=model)
    api.validate_api_key()
    client = api.get_client()
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        resp = client.embeddings.create(model=model, input=texts[start:start + EMBED_BATCH_SIZE])
        vectors.extend(d.embedding for d in resp.data)
    return vectors


# --------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="Retrieve top-k relevant chunks from a stored index.")
    parser.add_argument("--topk", help="Number of results to retrieve")
    parser.add_argument("--db", help="Index path (data/.../embedding_databases/<file> or "
                                     "data/.../bm25/<file>.pkl) or list number")
    parser.add_argument("--query-text", help="Query as a string")
    parser.add_argument("--query-json", help='Query json file: {"type": "text", "query": "..."}')
    args = parser.parse_args()

    dbs = scan_dbs()
    db = choose_db(dbs, preselect=args.db)
    query, query_source = get_query(args)
    k = get_topk(args.topk)

    if db["type"] == "bm25":
        tokenizer, run_search = SEARCHERS["bm25"](db, k)
        print(f"\nTokenizing query with '{tokenizer}' tokenizer ...")
        results = run_search(query)
        method_meta = {
            "tokenizer": tokenizer,
            "similarity": "bm25 score (unbounded, higher is better)",
        }
    else:
        model, run_search = SEARCHERS[db["type"]](db, k)
        if not model:
            sys.exit(f"ERROR: could not determine embedding model for {db['rel']}")
        print(f"\nEmbedding query with {model} ...")
        vector = embed_query(model, [query])[0]
        results = run_search(vector)
        method_meta = {
            "embedding_model": model,
            "similarity": "cosine (OpenAI embeddings are unit-norm); "
                          "score is the raw db value (milvus: IP, chromadb: 1 - IP)",
        }
    if len(results) < k:
        print(f"WARNING: index only returned {len(results)} results (asked for {k})")

    now = datetime.now()
    out = {
        "metadata": {
            "datetime": now.isoformat(timespec="seconds"),
            "db": db["rel"],
            "db_type": db["type"],
            **method_meta,
            "top_k": k,
            "query_type": "text",
            "query_source": query_source,
            "num_results": len(results),
        },
        "query": query,
        "results": results,
    }

    out_dir = db["title_dir"] / "queries"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{now.strftime('%Y%m%d_%H%M%S')}_top{k}_{db['path'].stem}.json"
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nTop {len(results)} results for: {query!r}")
    for r in results:
        preview = r["chunk"]["text"][:80].replace("\n", " ")
        print(f"  #{r['rank']}  sim={r['similarity']:.4f}  chunk {r['chunk']['chunk_index']}  "
              f"pages {r['chunk'].get('start_page')}-{r['chunk'].get('end_page')}  {preview}")
    print(f"\nOutput: {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
