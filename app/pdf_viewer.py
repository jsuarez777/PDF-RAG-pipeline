#!/usr/bin/env python3
"""Multi-user Flask UI for converting PDFs to page images and browsing them.

Usage:
    python app/pdf_viewer.py            # dev server, http://127.0.0.1:5001
    PDF_VIEWER_PROD=1 gunicorn -w 1 --threads 16 -b 127.0.0.1:5001 app.pdf_viewer:app

Each account gets an isolated workspace under data/users/<id>/ mirroring
the single-user layout (pdf2image/, pdfplumber/, ...); pipeline scripts are
pointed at it via the PDF_DATA_DIR environment variable. Long-running work
(conversion, OCR, chunking, indexing, evals, pipelines) executes on the
background job worker (jobs.py); endpoints enqueue and return 202 with a
job id that the frontend polls at /api/jobs/<id>.
"""

import json
import logging
import os
import queue
import re
import secrets
import sys
import threading
import time
import webbrowser
from pathlib import Path

from filelock import FileLock
from flask import (Flask, Response, abort, jsonify, render_template, request,
                   session, send_from_directory)
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auth  # noqa: E402
import db  # noqa: E402
import jobs  # noqa: E402
import pymupdf  # noqa: E402
from jobs import BroadcastHandler, JobError, run_and_stream  # noqa: E402
from logging_utils import setup_logging  # noqa: E402
from pdf_to_images import convert  # noqa: E402
from pdfplumber_to_text import convert as convert_text  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
USERS_ROOT = PROJECT_ROOT / "data" / "users"

SAFE_SEGMENT = re.compile(r"^[\w.-]+$")
PAGE_FILE = re.compile(r"^page_(\d+)\.png$")
SERVABLE_PNG = re.compile(r"^page_\d+(?:_image_\d+)?\.png$")
PLUMBER_PAGE = re.compile(r"^page_(\d+)\.json$")
QA_FILE = re.compile(r"^qa_.*\.json$")
EVAL_FILE = re.compile(r"^eval_(?!summary_).*\.json$")

DOC_TYPES = {"pdf2image": PAGE_FILE, "pdfplumber": PLUMBER_PAGE}

# ------------------------------------------------------------- guardrails
MAX_UPLOAD_MB = 50
MAX_QA_NUM = 50
MAX_TOPK = 50
MAX_CHUNK_SPECS = 8
MAX_ALPHAS = 5


def _secret_key() -> str:
    """SECRET_KEY from the environment, else a random key persisted on disk
    (so dev-server restarts don't invalidate sessions)."""
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    key_file = PROJECT_ROOT / "data" / ".secret_key"
    if key_file.is_file():
        return key_file.read_text().strip()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    key_file.write_text(key)
    key_file.chmod(0o600)
    return key


app = Flask(__name__)
app.secret_key = _secret_key()
app.config.update(
    MAX_CONTENT_LENGTH=MAX_UPLOAD_MB * 1024 * 1024,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=30 * 24 * 3600,
)
app.register_blueprint(auth.bp)
db.init_db()


@app.before_request
def before_request():
    jobs.start_worker()
    denied = auth.require_login()
    if denied is not None:
        return denied
    uid = auth.current_uid()
    if uid is not None:
        jobs.log_uid.set(uid)  # route this request's log lines to its user
    return None


# ------------------------------------------------------ per-user path helpers


def user_root_for(uid: int) -> Path:
    return USERS_ROOT / str(uid)


def user_root() -> Path:
    return user_root_for(auth.current_uid())


def user_env(uid: int) -> dict:
    """Subprocess environment pointing pipeline scripts at the user's data."""
    return {**os.environ, "PDF_DATA_DIR": str(user_root_for(uid))}


def resolve_doc_dir(uid: int, dtype: str, run: str, name: str) -> Path | None:
    """The user's data/<dtype>/<run>/<name> dir, or None if invalid/missing."""
    if dtype not in DOC_TYPES:
        return None
    if not (SAFE_SEGMENT.fullmatch(run) and SAFE_SEGMENT.fullmatch(name)):
        return None
    doc_dir = user_root_for(uid) / dtype / run / name
    return doc_dir if doc_dir.is_dir() else None


def check_segment(segment: str) -> str:
    if not SAFE_SEGMENT.fullmatch(segment):
        abort(400, "invalid path segment")
    return segment


def page_files(doc_dir: Path, pattern: re.Pattern = PAGE_FILE) -> list[str]:
    pages = []
    for f in doc_dir.iterdir():
        match = pattern.fullmatch(f.name)
        if match:
            pages.append((int(match.group(1)), f.name))
    return [name for _, name in sorted(pages)]


def job_doc_dir(uid: int, params: dict) -> Path:
    """Resolve the document a job refers to, or fail the job."""
    doc_dir = resolve_doc_dir(uid, params["dtype"], params["run"], params["name"])
    if doc_dir is None:
        raise JobError("document no longer exists")
    return doc_dir


def enqueue_job(kind: str, params: dict):
    """Queue a job for the current user and return the 202 polling response."""
    job_id = jobs.enqueue(auth.current_uid(), kind, params)
    return jsonify({"job": job_id}), 202


# ------------------------------------------------------------------- pages


@app.get("/")
def index():
    return render_template("pdf_viewer.html",
                           username=session.get("username", ""))


@app.get("/api/jobs/<int:job_id>")
def job_status(job_id: int):
    job = db.get_job(job_id, auth.current_uid())
    if job is None:
        abort(404)
    return jsonify({
        "id": job["id"],
        "kind": job["kind"],
        "status": job["status"],
        "result": json.loads(job["result"]) if job["result"] else None,
        "error": job["error"],
    })


@app.get("/api/documents")
def documents():
    docs = []
    for dtype, pattern in DOC_TYPES.items():
        root = user_root() / dtype
        if not root.is_dir():
            continue
        for run_dir in sorted(root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            for doc_dir in sorted(run_dir.iterdir()):
                if doc_dir.is_dir():
                    pages = page_files(doc_dir, pattern)
                    if pages:
                        docs.append(
                            {
                                "type": dtype,
                                "run": run_dir.name,
                                "name": doc_dir.name,
                                "pages": len(pages),
                            }
                        )
    return jsonify(docs)


@app.get("/api/documents/<run>/<name>/pages")
def pages(run: str, name: str):
    doc_dir = resolve_doc_dir(auth.current_uid(), "pdf2image", run, name)
    if doc_dir is None:
        abort(404)
    return jsonify(page_files(doc_dir))


def ocr_map(doc_dir: Path) -> dict:
    """Extracted text per page from page_<n>_easyocr.json files; null if not OCR'd."""
    out = {}
    for filename in page_files(doc_dir):
        number = int(PAGE_FILE.fullmatch(filename).group(1))
        json_path = doc_dir / f"page_{number}_easyocr.json"
        if json_path.is_file():
            try:
                out[str(number)] = json.loads(json_path.read_text()).get("extracted_text", "")
            except (json.JSONDecodeError, OSError):
                out[str(number)] = None
        else:
            out[str(number)] = None
    return out


@app.get("/api/documents/<run>/<name>/ocr")
def ocr_data(run: str, name: str):
    doc_dir = resolve_doc_dir(auth.current_uid(), "pdf2image", run, name)
    if doc_dir is None:
        abort(404)
    return jsonify(ocr_map(doc_dir))


@app.get("/api/documents/pdfplumber/<run>/<name>/content")
def plumber_content(run: str, name: str):
    """Per-page text and tables extracted by pdfplumber_to_text.py."""
    doc_dir = resolve_doc_dir(auth.current_uid(), "pdfplumber", run, name)
    if doc_dir is None:
        abort(404)
    out = []
    for filename in page_files(doc_dir, PLUMBER_PAGE):
        try:
            data = json.loads((doc_dir / filename).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "page": data.get("page"),
                "text": data.get("text", ""),
                "tables": data.get("tables") or [],
                "images": data.get("images") or [],
                "full_text": data.get("full_text", ""),
            }
        )
    return jsonify(out)


def qa_doc_dir(dtype: str, run: str, name: str) -> Path:
    doc_dir = resolve_doc_dir(auth.current_uid(), dtype, run, name)
    if doc_dir is None:
        abort(404)
    return doc_dir


def latest_chunk_run(doc_dir: Path) -> Path | None:
    """Newest <stamp>_chunk_* directory that has a chunked_text.json to draw on."""
    runs = [
        d
        for d in doc_dir.iterdir()
        if d.is_dir() and "_chunk_" in d.name and (d / "chunked_text.json").is_file()
    ]
    return max(runs, key=lambda d: d.name) if runs else None


def latest_qa_file(doc_dir: Path) -> Path | None:
    """Newest qa_<datetime>_<model>.json across the document's chunk runs."""
    qa_files = [
        f
        for chunk_dir in doc_dir.iterdir()
        if chunk_dir.is_dir() and "_chunk_" in chunk_dir.name
        for f in chunk_dir.iterdir()
        if f.is_file() and QA_FILE.fullmatch(f.name)
    ]
    return max(qa_files, key=lambda f: f.name) if qa_files else None


def qa_payload(doc_dir: Path, qa_path: Path) -> dict:
    try:
        data = json.loads(qa_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise JobError(f"could not read {qa_path.name}")
    chunks = []
    chunk_meta = {}
    chunk_file = qa_path.parent / "chunked_text.json"
    try:
        cdata = json.loads(chunk_file.read_text(encoding="utf-8"))
        chunk_meta = cdata.get("metadata", {})
        chunks = [
            {k: c.get(k) for k in ("chunk_index", "start_char", "end_char",
                                   "start_page", "end_page")}
            for c in cdata.get("chunks", [])
        ]
    except (OSError, json.JSONDecodeError):
        pass  # boxes for un-QA'd chunks just won't render
    return {
        "qa_file": qa_path.relative_to(doc_dir).as_posix(),
        "chunk_run": qa_path.parent.name,
        "items": data.get("items", []),
        "chunks": chunks,
        "chunk_metadata": chunk_meta,
    }


@app.get("/api/documents/<dtype>/<run>/<name>/qa")
def qa_data(dtype: str, run: str, name: str):
    """Latest QA dataset generated from any chunk run of this document, or null."""
    doc_dir = qa_doc_dir(dtype, run, name)
    qa_path = latest_qa_file(doc_dir)
    if qa_path is None:
        return jsonify(None)
    try:
        return jsonify(qa_payload(doc_dir, qa_path))
    except JobError as exc:
        abort(500, str(exc))


@app.get("/api/documents/<dtype>/<run>/<name>/evals")
def eval_list(dtype: str, run: str, name: str):
    """Eval files (from eval_retrieval.py), newest first, each with its full
    metadata and aggregates so the results dialog can render any of them
    without a follow-up fetch. When qa_file is given (doc-dir-relative, as
    served by qa_data), only evals pointing at that QA file are returned."""
    doc_dir = qa_doc_dir(dtype, run, name)
    qa_file = request.args.get("qa_file", "")
    eval_dir = doc_dir / "evaluations"
    out = []
    if eval_dir.is_dir():
        for f in sorted(eval_dir.iterdir(), reverse=True):
            if not (f.is_file() and EVAL_FILE.fullmatch(f.name)):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            meta = data.get("metadata", {})
            if qa_file and not str(meta.get("qa_file", "")).endswith(qa_file):
                continue
            out.append(
                {
                    "file": f.name,
                    "metadata": meta,
                    "aggregates": data.get("aggregates", {}),
                }
            )
    return jsonify(out)


@app.get("/api/documents/<dtype>/<run>/<name>/evals/<filename>")
def eval_detail(dtype: str, run: str, name: str, filename: str):
    """One eval file's per-question results, plus the text of every chunk the
    eval references (gold or retrieved) so the UI can preview them."""
    doc_dir = qa_doc_dir(dtype, run, name)
    if not EVAL_FILE.fullmatch(check_segment(filename)):
        abort(404)
    eval_path = doc_dir / "evaluations" / filename
    if not eval_path.is_file():
        abort(404)
    try:
        data = json.loads(eval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        abort(500, f"could not read {filename}")

    questions = data.get("questions", [])
    referenced = {q.get("chunk_index") for q in questions}
    for q in questions:
        referenced.update(r.get("chunk_index") for r in q.get("retrieved", []))

    chunk_texts = {}
    chunk_run = Path(str(data.get("metadata", {}).get("chunk_run", ""))).name
    if SAFE_SEGMENT.fullmatch(chunk_run):
        chunk_file = doc_dir / chunk_run / "chunked_text.json"
        try:
            cdata = json.loads(chunk_file.read_text(encoding="utf-8"))
            for c in cdata.get("chunks", []):
                if c.get("chunk_index") in referenced:
                    chunk_texts[str(c["chunk_index"])] = {
                        "text": c.get("text", ""),
                        "start_page": c.get("start_page"),
                        "end_page": c.get("end_page"),
                    }
        except (OSError, json.JSONDecodeError):
            pass  # chunk previews just won't render
    return jsonify(
        {
            "file": filename,
            "metadata": data.get("metadata", {}),
            "questions": questions,
            "chunk_texts": chunk_texts,
        }
    )


@app.post("/api/documents/<dtype>/<run>/<name>/qa/delete-pair")
def qa_delete_pair(dtype: str, run: str, name: str):
    """Delete one QA pair (and its item, if emptied) from a qa_*.json file.

    A null pair_index deletes the whole item, i.e. all of the chunk's pairs.
    Runs synchronously (it is a quick file edit); the write is guarded with
    a file lock against a concurrently running QA-generation job.
    """
    doc_dir = qa_doc_dir(dtype, run, name)
    payload = request.get_json(force=True) or {}

    parts = str(payload.get("qa_file", "")).split("/")
    if len(parts) != 2 or not QA_FILE.fullmatch(parts[1]):
        abort(400, "invalid qa_file")
    qa_path = doc_dir / check_segment(parts[0]) / parts[1]
    if not qa_path.is_file():
        abort(404)

    with FileLock(str(qa_path) + ".lock", timeout=10):
        try:
            data = json.loads(qa_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            abort(500, f"could not read {qa_path.name}")
        items = data.get("items", [])
        item_index = payload.get("item_index")
        pair_index = payload.get("pair_index")
        if not (isinstance(item_index, int) and 0 <= item_index < len(items)):
            abort(400, "invalid item_index")
        item = items[item_index]
        # guard against deleting the wrong entry when the client's view is stale
        if item.get("chunk_index") != payload.get("chunk_index"):
            abort(409, "QA file changed since it was loaded; reopen the document")

        if pair_index is None:
            log.info(f"Deleting all QA pairs of chunk {item['chunk_index']} from {qa_path.name}")
            del items[item_index]
        else:
            pairs = item.get("qa_pairs", [])
            if not (isinstance(pair_index, int) and 0 <= pair_index < len(pairs)):
                abort(400, "invalid pair_index")
            if pairs[pair_index].get("question_type") != payload.get("question_type"):
                abort(409, "QA file changed since it was loaded; reopen the document")
            log.info(f"Deleting QA pair: chunk {item['chunk_index']} "
                     f"{pairs[pair_index]['question_type']} from {qa_path.name}")
            del pairs[pair_index]
            if not pairs:
                del items[item_index]
        qa_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        return jsonify(qa_payload(doc_dir, qa_path))
    except JobError as exc:
        abort(500, str(exc))


@app.post("/api/documents/<dtype>/<run>/<name>/qa/generate")
def qa_generate(dtype: str, run: str, name: str):
    """Generate QA pairs for specific chunks via generate_qa.py and add them
    to the document's latest qa_*.json, sorted and replacing duplicates."""
    doc_dir = qa_doc_dir(dtype, run, name)
    payload = request.get_json(force=True) or {}

    chunks = payload.get("chunks")
    if not (isinstance(chunks, list) and chunks and len(chunks) <= 200
            and all(isinstance(c, int) and c >= 0 for c in chunks)):
        abort(400, "chunks must be a non-empty list of chunk indices")
    types = payload.get("types") or ["direct", "inference", "paraphrased"]
    if not (isinstance(types, list)
            and all(t in ("direct", "inference", "paraphrased") for t in types)):
        abort(400, "invalid question types")

    if latest_qa_file(doc_dir) is None:
        abort(404, "no QA dataset to append to")

    return enqueue_job("qa_generate", {
        "dtype": dtype, "run": run, "name": name,
        "chunks": sorted(set(chunks)), "types": types,
    })


@jobs.handler("qa_generate")
def job_qa_generate(uid: int, params: dict) -> dict:
    doc_dir = job_doc_dir(uid, params)
    qa_path = latest_qa_file(doc_dir)
    if qa_path is None:
        raise JobError("no QA dataset to append to")
    code = run_and_stream([
        sys.executable, str(SCRIPT_DIR / "generate_qa.py"),
        "--dataset", qa_path.parent.relative_to(PROJECT_ROOT).as_posix(),
        "--chunks", ",".join(map(str, params["chunks"])),
        "--types", ",".join(params["types"]),
        "--add", str(qa_path),
    ], env=user_env(uid))
    if code != 0:
        raise JobError(f"generate_qa.py failed with exit code {code}")
    return qa_payload(doc_dir, qa_path)


@app.post("/api/documents/<dtype>/<run>/<name>/qa/generate-all")
def qa_generate_all(dtype: str, run: str, name: str):
    """Generate a fresh QA dataset for the document's latest chunk run via
    generate_qa.py (default sampling). Used to bootstrap a document that
    has no qa_*.json yet."""
    doc_dir = qa_doc_dir(dtype, run, name)
    payload = request.get_json(force=True) or {}

    types = payload.get("types") or ["direct", "inference", "paraphrased"]
    if not (isinstance(types, list)
            and all(t in ("direct", "inference", "paraphrased") for t in types)):
        abort(400, "invalid question types")
    num = payload.get("num")
    if num is not None and not (isinstance(num, int) and 0 < num <= MAX_QA_NUM):
        abort(400, f"num must be a positive integer up to {MAX_QA_NUM}")

    if latest_chunk_run(doc_dir) is None:
        abort(404, "no chunk run to generate from; chunk the text first")

    return enqueue_job("qa_generate_all", {
        "dtype": dtype, "run": run, "name": name, "types": types, "num": num,
    })


@jobs.handler("qa_generate_all")
def job_qa_generate_all(uid: int, params: dict) -> dict:
    doc_dir = job_doc_dir(uid, params)
    chunk_run = latest_chunk_run(doc_dir)
    if chunk_run is None:
        raise JobError("no chunk run to generate from; chunk the text first")
    cmd = [
        sys.executable, str(SCRIPT_DIR / "generate_qa.py"),
        "--dataset", chunk_run.relative_to(PROJECT_ROOT).as_posix(),
        "--types", ",".join(params["types"]),
    ]
    if params.get("num") is not None:
        cmd += ["--num-chunks", str(params["num"])]
    code = run_and_stream(cmd, env=user_env(uid))
    if code != 0:
        raise JobError(f"generate_qa.py failed with exit code {code}")
    qa_path = latest_qa_file(doc_dir)
    if qa_path is None:
        raise JobError("generation produced no QA file")
    return qa_payload(doc_dir, qa_path)


@app.post("/api/documents/<dtype>/<run>/<name>/chunk")
def chunk_document(dtype: str, run: str, name: str):
    """Cut a new chunk run from the document's extracted text via chunk_text.py."""
    qa_doc_dir(dtype, run, name)  # validates existence + segments
    payload = request.get_json(force=True) or {}

    method = payload.get("type")
    if method not in ("fixed_size", "sentence", "sentence-dynamic-min",
                      "plumber-struct", "semantic"):
        abort(400, "invalid chunk type")
    if method == "fixed_size":
        size = payload.get("size")
        overlap = str(payload.get("overlap", "")).strip()
        if not (isinstance(size, int) and size > 0):
            abort(400, "chunk size must be a positive integer")
        if not re.fullmatch(r"\d+(?:\.\d+)?%|\d+", overlap):
            abort(400, "overlap must be an integer or a percent like 10%")
        type_str = f"fixed_size:{size}:{overlap}"
    elif method in ("sentence", "sentence-dynamic-min"):
        size = payload.get("size")
        overlap = payload.get("overlap", 0)
        if not (isinstance(size, int) and size > 0):
            abort(400, "sentences per chunk must be a positive integer")
        if not (isinstance(overlap, int) and 0 <= overlap < size):
            abort(400, "overlap must be a non-negative integer below the chunk size")
        type_str = f"{method}:{size}:{overlap}"
    elif method == "plumber-struct":
        size = payload.get("size")
        if size is not None and not (isinstance(size, int) and size > 0):
            abort(400, "sentences per text chunk must be a positive integer")
        type_str = f"plumber-struct:{size}" if size else "plumber-struct"
    else:
        type_str = method

    return enqueue_job("chunk", {
        "dtype": dtype, "run": run, "name": name, "type_str": type_str,
    })


@jobs.handler("chunk")
def job_chunk(uid: int, params: dict) -> dict:
    doc_dir = job_doc_dir(uid, params)
    code = run_and_stream([
        sys.executable, str(SCRIPT_DIR / "chunk_text.py"),
        "--type", params["type_str"],
        "--dataset", doc_dir.relative_to(PROJECT_ROOT).as_posix(),
    ], env=user_env(uid))
    if code != 0:
        raise JobError(f"chunk_text.py failed with exit code {code}")
    chunk_run = latest_chunk_run(doc_dir)
    return {"chunk_run": chunk_run.name if chunk_run else None}


@app.post("/api/documents/<dtype>/<run>/<name>/index")
def build_index(dtype: str, run: str, name: str):
    """Build a BM25 index or embedding store from the document's latest chunk
    run via index_bm25.py / embed_chunks.py."""
    doc_dir = qa_doc_dir(dtype, run, name)
    payload = request.get_json(force=True) or {}

    if latest_chunk_run(doc_dir) is None:
        abort(404, "no chunk run to index; chunk the text first")

    kind = payload.get("kind")
    params = {"dtype": dtype, "run": run, "name": name, "kind": kind}
    if kind == "bm25":
        tokenizers = payload.get("tokenizers")
        if not (isinstance(tokenizers, list) and tokenizers
                and all(t in ("simple", "word", "porter") for t in tokenizers)):
            abort(400, "tokenizers must be a non-empty list of: simple, word, porter")
        params["tokenizers"] = list(dict.fromkeys(tokenizers))
    elif kind == "embed":
        embeddings = payload.get("embeddings")
        dbs = payload.get("dbs")
        if not (isinstance(embeddings, list) and embeddings
                and all(e in ("small", "large", "minilm", "bge") for e in embeddings)):
            abort(400, "embeddings must be a non-empty list of: small, large, minilm, bge")
        if not (isinstance(dbs, list) and dbs
                and all(d in ("milvus", "chromadb") for d in dbs)):
            abort(400, "dbs must be a non-empty list of: milvus, chromadb")
        params["embeddings"] = list(dict.fromkeys(embeddings))
        params["dbs"] = list(dict.fromkeys(dbs))
    else:
        abort(400, "kind must be 'bm25' or 'embed'")

    return enqueue_job("index", params)


@jobs.handler("index")
def job_index(uid: int, params: dict) -> dict:
    doc_dir = job_doc_dir(uid, params)
    chunk_run = latest_chunk_run(doc_dir)
    if chunk_run is None:
        raise JobError("no chunk run to index; chunk the text first")
    dataset = chunk_run.relative_to(PROJECT_ROOT).as_posix()
    if params["kind"] == "bm25":
        cmd = [sys.executable, str(SCRIPT_DIR / "index_bm25.py"),
               "--tokenizer", ",".join(params["tokenizers"]),
               "--dataset", dataset]
    else:
        cmd = [sys.executable, str(SCRIPT_DIR / "embed_chunks.py"),
               "--embedding", ",".join(params["embeddings"]),
               "--db", ",".join(params["dbs"]),
               "--dataset", dataset]
    code = run_and_stream(cmd, env=user_env(uid))
    if code != 0:
        raise JobError(f"indexing failed with exit code {code}")
    return {"ok": True, "chunk_run": chunk_run.name}


def qa_chunk_run_rel(doc_dir: Path) -> str | None:
    """chunk_run recorded in the document's latest QA file (project-relative)."""
    qa_path = latest_qa_file(doc_dir)
    if qa_path is None:
        return None
    try:
        return json.loads(qa_path.read_text(encoding="utf-8")).get(
            "metadata", {}).get("chunk_run")
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/api/documents/<dtype>/<run>/<name>/indexes")
def list_indexes(dtype: str, run: str, name: str):
    """Stored indexes (bm25/ and embedding_databases/) for this document,
    flagged with whether each was built from the latest QA file's chunk run.
    matches is null when the chunk run couldn't be read (e.g. chromadb)."""
    doc_dir = qa_doc_dir(dtype, run, name)
    qa_run = qa_chunk_run_rel(doc_dir)

    def sidecar_chunk_run(path: Path) -> str | None:
        sidecar = path.with_suffix(".json")
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))["metadata"]["chunk_run"]
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    out = []
    bm_dir = doc_dir / "bm25"
    if bm_dir.is_dir():
        for p in sorted(bm_dir.glob("bm25_*.pkl")):
            built_from = sidecar_chunk_run(p)
            out.append({
                "rel": p.relative_to(PROJECT_ROOT).as_posix(),
                "name": p.name,
                "type": "bm25",
                "matches": None if built_from is None or qa_run is None
                           else built_from == qa_run,
            })
    edb_dir = doc_dir / "embedding_databases"
    if edb_dir.is_dir():
        for p in sorted(edb_dir.iterdir()):
            if p.name.startswith("milvus_") and p.suffix == ".db":
                db_type = "milvus"
                built_from = sidecar_chunk_run(p)
            elif p.name.startswith("chromadb_") and p.suffix == ".chroma" and p.is_dir():
                db_type = "chromadb"
                built_from = None  # stored inside the collection; not read here
            else:
                continue
            out.append({
                "rel": p.relative_to(PROJECT_ROOT).as_posix(),
                "name": p.name,
                "type": db_type,
                "matches": None if built_from is None or qa_run is None
                           else built_from == qa_run,
            })
    return jsonify({"qa_chunk_run": qa_run, "indexes": out})


@app.post("/api/documents/<dtype>/<run>/<name>/eval")
def run_eval(dtype: str, run: str, name: str):
    """Evaluate indexes against the document's latest QA file via
    eval_retrieval.py."""
    doc_dir = qa_doc_dir(dtype, run, name)
    payload = request.get_json(force=True) or {}

    if latest_qa_file(doc_dir) is None:
        abort(404, "no QA dataset to evaluate; generate QA pairs first")

    db_sel = payload.get("db", "all")
    if db_sel == "all":
        db_arg = "all"
    elif isinstance(db_sel, list) and db_sel and all(isinstance(d, str) for d in db_sel):
        for d in db_sel:
            resolved = (PROJECT_ROOT / d).resolve()
            if not resolved.is_relative_to(doc_dir.resolve()) or not resolved.exists():
                abort(400, f"invalid index path: {d}")
        db_arg = ",".join(db_sel)
    else:
        abort(400, "db must be 'all' or a non-empty list of index paths")

    topk = payload.get("topk", 10)
    if not (isinstance(topk, int) and 0 < topk <= MAX_TOPK):
        abort(400, f"topk must be a positive integer up to {MAX_TOPK}")
    ks = str(payload.get("ks", "1,3,5,10")).replace(" ", "")
    if not re.fullmatch(r"\d+(,\d+)*", ks):
        abort(400, "ks must be comma-separated integers")
    if any(int(k) > topk for k in ks.split(",")):
        abort(400, f"each cutoff (ks) must be ≤ top-k ({topk})")

    return enqueue_job("eval", {
        "dtype": dtype, "run": run, "name": name,
        "db_arg": db_arg, "topk": topk, "ks": ks,
        "force": bool(payload.get("force")),
    })


@jobs.handler("eval")
def job_eval(uid: int, params: dict) -> list:
    doc_dir = job_doc_dir(uid, params)
    qa_path = latest_qa_file(doc_dir)
    if qa_path is None:
        raise JobError("no QA dataset to evaluate; generate QA pairs first")
    cmd = [sys.executable, str(SCRIPT_DIR / "eval_retrieval.py"),
           "--qa", qa_path.relative_to(PROJECT_ROOT).as_posix(),
           "--db", params["db_arg"], "--topk", str(params["topk"]),
           "--ks", params["ks"]]
    if params.get("force"):
        cmd.append("--force")

    started = time.time()
    code = run_and_stream(cmd, env=user_env(uid))
    if code != 0:
        raise JobError(f"eval_retrieval.py failed with exit code {code}")

    results = []
    eval_dir = doc_dir / "evaluations"
    if eval_dir.is_dir():
        for f in sorted(eval_dir.iterdir()):
            if not (f.is_file() and EVAL_FILE.fullmatch(f.name)
                    and f.stat().st_mtime >= started - 1):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            results.append({
                "file": f.name,
                "metadata": data.get("metadata", {}),
                "aggregates": data.get("aggregates", {}),
            })
    return results


CHUNK_SPEC = re.compile(
    r"fixed_size:\d+:(?:\d+(?:\.\d+)?%|\d+)"
    r"|(?:sentence|sentence-dynamic-min):\d+(?::\d+)?"
    r"|plumber-struct(?::\d+)?"
    r"|semantic"
)


def chunk_spec_errors(spec: str) -> list[str]:
    """Numeric-rule violations for one chunk spec, mirroring the Run Pipeline
    popup: fixed_size is character-based (size >= 20), sentence methods count
    sentences (size >= 1), and overlap is always >= 0 and below the size."""
    errs = []
    parts = spec.split(":")
    method = parts[0]
    if method == "fixed_size":
        size = int(parts[1])
        if size < 20:
            errs.append(f"{spec}: chunk size must be at least 20 characters")
        ov = parts[2]
        if ov.endswith("%"):
            if not 0 <= float(ov[:-1]) <= 90:
                errs.append(f"{spec}: overlap percent must be between 0% and 90%")
        elif int(ov) > 0.9 * size:
            errs.append(f"{spec}: overlap ({ov}) must be between 0 and 90% of the "
                        f"chunk size ({int(0.9 * size)})")
    elif method in ("sentence", "sentence-dynamic-min"):
        size = int(parts[1])
        if size < 1:
            errs.append(f"{spec}: sentences per chunk must be at least 1")
        if len(parts) == 3 and int(parts[2]) >= size:
            errs.append(f"{spec}: overlap ({parts[2]}) must be less than sentences per chunk ({size})")
    elif method == "plumber-struct" and len(parts) == 2 and int(parts[1]) < 1:
        errs.append(f"{spec}: sentences per text chunk must be at least 1")
    return errs


@app.post("/api/documents/<dtype>/<run>/<name>/pipeline")
def run_full_pipeline(dtype: str, run: str, name: str):
    """Run the full grid pipeline (run_pipeline.py) on this document."""
    qa_doc_dir(dtype, run, name)  # validates existence + segments
    payload = request.get_json(force=True) or {}

    chunk_types = payload.get("chunk_types")
    if not (isinstance(chunk_types, list) and chunk_types
            and len(chunk_types) <= MAX_CHUNK_SPECS
            and all(isinstance(c, str) and CHUNK_SPEC.fullmatch(c) for c in chunk_types)):
        abort(400, "chunk_types must be a list of up to "
                   f"{MAX_CHUNK_SPECS} chunk specs like fixed_size:256:50")
    spec_errors = [e for c in chunk_types for e in chunk_spec_errors(c)]
    if spec_errors:
        abort(400, "; ".join(spec_errors))

    def csv_of(key, valid, default):
        vals = payload.get(key) or default
        if not (isinstance(vals, list) and vals and all(v in valid for v in vals)):
            abort(400, f"{key} must be a non-empty list from: {', '.join(valid)}")
        return ",".join(dict.fromkeys(vals))

    tokenizers = csv_of("tokenizers", ("simple", "word", "porter"), ["word"])
    retrievals = csv_of("retrievals", ("bm25", "vector", "hybrid"),
                        ["bm25", "vector", "hybrid"])
    qa_types = csv_of("qa_types", ("direct", "inference", "paraphrased"),
                      ["direct", "inference", "paraphrased"])

    # vector_configs: model:db pairs (e.g. "small:milvus"), one per checked
    # embedding model x its checked DBs. Only required when a vector-using
    # retrieval method (vector/hybrid) is selected.
    vc = payload.get("vector_configs") or []
    needs_vector = "vector" in retrievals.split(",") or "hybrid" in retrievals.split(",")
    if not (isinstance(vc, list) and all(isinstance(v, str) for v in vc)):
        abort(400, "vector_configs must be a list of model:db strings")
    if not all(re.fullmatch(r"(?:small|large|minilm|bge):(?:milvus|chromadb)", v) for v in vc):
        abort(400, "each vector_config must look like small:milvus")
    vector_configs = ",".join(dict.fromkeys(vc))
    if needs_vector and not vector_configs:
        abort(400, "pick at least one embedding model and vector DB")
    # alphas: one or more hybrid vector weights to sweep (each 0..1)
    alphas = payload.get("alphas")
    if alphas is None:
        alphas = [payload.get("alpha", 0.7)]
    if not (isinstance(alphas, list) and alphas and len(alphas) <= MAX_ALPHAS
            and all(isinstance(a, (int, float)) and 0 <= a <= 1 for a in alphas)):
        abort(400, f"alphas must be a list of up to {MAX_ALPHAS} numbers between 0 and 1")
    alphas = list(dict.fromkeys(alphas))
    qa_num = payload.get("qa_num", 20)
    if not (isinstance(qa_num, int) and 0 < qa_num <= MAX_QA_NUM):
        abort(400, f"qa_num must be a positive integer up to {MAX_QA_NUM}")
    topk = payload.get("topk", 10)
    if not (isinstance(topk, int) and 0 < topk <= MAX_TOPK):
        abort(400, f"topk must be a positive integer up to {MAX_TOPK}")
    ks = str(payload.get("ks", "1,3,5,10")).replace(" ", "")
    if not re.fullmatch(r"\d+(,\d+)*", ks):
        abort(400, "ks must be comma-separated integers")
    if any(int(k) > topk for k in ks.split(",")):
        abort(400, f"each cutoff (ks) must be ≤ top-k ({topk})")
    seed = payload.get("seed")
    if seed is not None and not isinstance(seed, int):
        abort(400, "seed must be an integer")

    return enqueue_job("pipeline", {
        "dtype": dtype, "run": run, "name": name,
        "chunk_types": ",".join(dict.fromkeys(chunk_types)),
        "tokenizers": tokenizers, "retrievals": retrievals,
        "alphas": ",".join(f"{a:g}" for a in alphas),
        "qa_num": qa_num, "qa_types": qa_types,
        "topk": topk, "ks": ks,
        "vector_configs": vector_configs, "seed": seed,
    })


@jobs.handler("pipeline")
def job_pipeline(uid: int, params: dict) -> dict:
    doc_dir = job_doc_dir(uid, params)
    cmd = [sys.executable, str(SCRIPT_DIR / "run_pipeline.py"),
           "--dataset", doc_dir.relative_to(PROJECT_ROOT).as_posix(),
           "--chunk-types", params["chunk_types"],
           "--tokenizers", params["tokenizers"],
           "--retrievals", params["retrievals"],
           "--alpha", params["alphas"],
           "--qa-num", str(params["qa_num"]), "--qa-types", params["qa_types"],
           "--topk", str(params["topk"]), "--ks", params["ks"]]
    if params.get("vector_configs"):
        cmd += ["--vector-configs", params["vector_configs"]]
    if params.get("seed") is not None:
        cmd += ["--seed", str(params["seed"])]

    started = time.time()
    code = run_and_stream(cmd, env=user_env(uid))
    if code != 0:
        raise JobError(f"run_pipeline.py failed with exit code {code}")

    results_dir = doc_dir / "pipeline_runs"
    candidates = [f for f in results_dir.glob("pipeline_*.json")
                  if f.stat().st_mtime >= started - 1] if results_dir.is_dir() else []
    if not candidates:
        raise JobError("pipeline ran but produced no results file")
    newest = max(candidates, key=lambda f: f.name)
    try:
        return json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise JobError(f"could not read {newest.name}")


@app.get("/api/documents/<dtype>/<run>/<name>/pipeline-runs")
def pipeline_runs(dtype: str, run: str, name: str):
    """Past pipeline results for this document, newest first."""
    doc_dir = qa_doc_dir(dtype, run, name)
    results_dir = doc_dir / "pipeline_runs"
    out = []
    if results_dir.is_dir():
        for f in sorted(results_dir.glob("pipeline_*.json"), reverse=True):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return jsonify(out)


@app.post("/api/documents/<run>/<name>/ocr/<int:page>")
def run_page_ocr(run: str, name: str, page: int):
    """Run (or re-run) EasyOCR on a single page via easyocr_pdfimages.py."""
    doc_dir = resolve_doc_dir(auth.current_uid(), "pdf2image", run, name)
    if doc_dir is None or not (doc_dir / f"page_{page}.png").is_file():
        abort(404)
    return enqueue_job("ocr_page", {
        "dtype": "pdf2image", "run": run, "name": name, "page": page,
    })


@jobs.handler("ocr_page")
def job_ocr_page(uid: int, params: dict) -> dict:
    doc_dir = job_doc_dir(uid, params)
    page = params["page"]
    image_path = doc_dir / f"page_{page}.png"
    if not image_path.is_file():
        raise JobError("page image no longer exists")
    code = run_and_stream(
        [sys.executable, str(SCRIPT_DIR / "easyocr_pdfimages.py"), str(image_path)],
        env=user_env(uid))
    if code != 0:
        raise JobError(f"OCR failed with exit code {code}")
    json_path = doc_dir / f"page_{page}_easyocr.json"
    try:
        text = json.loads(json_path.read_text()).get("extracted_text", "")
    except (OSError, json.JSONDecodeError):
        raise JobError("OCR ran but no result file was produced")
    return {"page": page, "extracted_text": text}


@app.post("/api/documents/<run>/<name>/ocr")
def run_all_ocr(run: str, name: str):
    """OCR every page still missing easyocr data, in one script run."""
    doc_dir = resolve_doc_dir(auth.current_uid(), "pdf2image", run, name)
    if doc_dir is None:
        abort(404)
    return enqueue_job("ocr_all", {"dtype": "pdf2image", "run": run, "name": name})


@jobs.handler("ocr_all")
def job_ocr_all(uid: int, params: dict) -> dict:
    doc_dir = job_doc_dir(uid, params)
    code = run_and_stream(
        [sys.executable, str(SCRIPT_DIR / "easyocr_pdfimages.py"),
         "--missing-only", str(doc_dir)],
        env=user_env(uid))
    if code != 0:
        raise JobError(f"OCR failed with exit code {code}")
    return ocr_map(doc_dir)


@app.get("/images/<dtype>/<run>/<name>/<filename>")
def image(dtype: str, run: str, name: str, filename: str):
    if not SERVABLE_PNG.fullmatch(filename):
        abort(404)
    doc_dir = resolve_doc_dir(auth.current_uid(), dtype, run, name)
    if doc_dir is None:
        abort(404)
    return send_from_directory(doc_dir, filename)


@app.get("/logs/stream")
def log_stream():
    broadcaster = jobs.broadcaster_for(auth.current_uid())

    def generate():
        q, backlog = broadcaster.listen()
        try:
            for line in backlog:
                yield f"data: {line}\n\n"
            while True:
                try:
                    yield f"data: {q.get(timeout=15)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            broadcaster.drop(q)

    return Response(generate(), mimetype="text/event-stream")


@app.post("/upload")
def upload():
    file = request.files.get("pdf")
    if file is None or not file.filename:
        abort(400, "no file provided")
    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        abort(400, "not a PDF file")

    method = request.form.get("method", "pdf2image")
    if method not in DOC_TYPES:
        abort(400, f"unknown method: {method}")

    # one subdir per upload so the original filename (= document title) survives
    uploads = user_root() / "uploads" / time.strftime("%Y%m%d_%H%M%S")
    uploads.mkdir(parents=True, exist_ok=True)
    pdf_path = uploads / filename
    file.save(pdf_path)

    try:
        with pymupdf.open(pdf_path) as doc:
            page_count = len(doc)
    except Exception:
        pdf_path.unlink(missing_ok=True)
        abort(400, "could not open the file as a PDF")

    log.info(f"Upload received: {filename} ({page_count} pages, method: {method})")
    return enqueue_job("convert", {
        "pdf": pdf_path.relative_to(PROJECT_ROOT).as_posix(), "method": method,
    })


@jobs.handler("convert")
def job_convert(uid: int, params: dict) -> dict:
    pdf_path = (PROJECT_ROOT / params["pdf"]).resolve()
    if not pdf_path.is_relative_to(user_root_for(uid).resolve()) or not pdf_path.is_file():
        raise JobError("uploaded file no longer exists")
    method = params["method"]
    converter = convert if method == "pdf2image" else convert_text
    try:
        out_dir = converter(pdf_path, output_root=user_root_for(uid) / method)
    finally:
        pdf_path.unlink(missing_ok=True)  # the page images/JSON are the artifact
        try:
            pdf_path.parent.rmdir()  # its per-upload directory
        except OSError:
            pass
    return {
        "type": method,
        "run": out_dir.parent.name,
        "name": out_dir.name,
        "pages": len(page_files(out_dir, DOC_TYPES[method])),
    }


if os.environ.get("PDF_VIEWER_PROD") == "1":
    # Under gunicorn the __main__ block never runs; configure logging here.
    setup_logging("pdf_viewer", extra_handlers=[BroadcastHandler()], console=True)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


if __name__ == "__main__":
    # Debug mode runs this file twice (reloader parent + serving child);
    # only the parent opens the browser, and only the serving child logs,
    # so a single log file is created per (re)start.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        # console=False keeps the terminal quiet: activity streams to the
        # browser log pane and the log file instead.
        log_file = setup_logging(
            "pdf_viewer", extra_handlers=[BroadcastHandler()], console=False
        )
        if os.getenv("LOG_HTTP") != "1":
            logging.getLogger("werkzeug").setLevel(logging.WARNING)
        log.info(f"Logging to {log_file}")
    else:
        threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5001")).start()
    app.run(debug=True, port=5001, threaded=True)
