#!/usr/bin/env python3
"""Flask UI for converting PDFs to page images and browsing the results.

Usage:
    python app/pdf_viewer.py            # serves http://127.0.0.1:5001

Documents live under data/pdf2image/<YYYYMMDD_NN>/<name>/page_<n>.png,
as produced by pdf_to_images.py; uploads are converted with the same code.
"""

import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logging_utils import setup_logging  # noqa: E402
from pdf_to_images import OUTPUT_ROOT, convert  # noqa: E402
from pdfplumber_to_text import OUTPUT_ROOT as PLUMBER_ROOT  # noqa: E402
from pdfplumber_to_text import convert as convert_text  # noqa: E402

log = logging.getLogger(__name__)

app = Flask(__name__)

SAFE_SEGMENT = re.compile(r"^[\w.-]+$")
PAGE_FILE = re.compile(r"^page_(\d+)\.png$")
SERVABLE_PNG = re.compile(r"^page_\d+(?:_image_\d+)?\.png$")
PLUMBER_PAGE = re.compile(r"^page_(\d+)\.json$")
QA_FILE = re.compile(r"^qa_.*\.json$")
EVAL_FILE = re.compile(r"^eval_(?!summary_).*\.json$")


class LogBroadcaster:
    """Fan-out of log lines to every connected SSE client, with replay history."""

    def __init__(self, history: int = 200):
        self._clients: list[queue.Queue] = []
        self._history: deque[str] = deque(maxlen=history)
        self._lock = threading.Lock()

    def publish(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        stamped = f"[{time.strftime('%H:%M:%S')}] {line}"
        with self._lock:
            self._history.append(stamped)
            for q in self._clients:
                q.put(stamped)

    def listen(self) -> tuple[queue.Queue, list[str]]:
        q: queue.Queue = queue.Queue()
        with self._lock:
            backlog = list(self._history)
            self._clients.append(q)
        return q, backlog

    def drop(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)


logs = LogBroadcaster()


class BroadcastHandler(logging.Handler):
    """Logging handler that mirrors every record to the SSE log pane."""

    def emit(self, record: logging.LogRecord) -> None:
        for line in self.format(record).splitlines():
            logs.publish(line)


def run_and_stream(cmd: list[str]) -> int:
    """Run a command, streaming its output through the viewer's logging."""
    log.info("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log.info(line.rstrip())
    code = proc.wait()
    log.info(f"[exit code {code}]")
    return code


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


@app.get("/")
def index():
    return render_template("pdf_viewer.html")


DOC_TYPES = {
    "pdf2image": (OUTPUT_ROOT, PAGE_FILE),
    "pdfplumber": (PLUMBER_ROOT, PLUMBER_PAGE),
}


@app.get("/api/documents")
def documents():
    docs = []
    for dtype, (root, pattern) in DOC_TYPES.items():
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
    doc_dir = OUTPUT_ROOT / check_segment(run) / check_segment(name)
    if not doc_dir.is_dir():
        abort(404)
    return jsonify(page_files(doc_dir))


@app.get("/api/documents/<run>/<name>/ocr")
def ocr_data(run: str, name: str):
    """Extracted text per page from page_<n>_easyocr.json files; null if not OCR'd."""
    doc_dir = OUTPUT_ROOT / check_segment(run) / check_segment(name)
    if not doc_dir.is_dir():
        abort(404)
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
    return jsonify(out)


@app.get("/api/documents/pdfplumber/<run>/<name>/content")
def plumber_content(run: str, name: str):
    """Per-page text and tables extracted by pdfplumber_to_text.py."""
    doc_dir = PLUMBER_ROOT / check_segment(run) / check_segment(name)
    if not doc_dir.is_dir():
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
    if dtype not in DOC_TYPES:
        abort(404)
    doc_dir = DOC_TYPES[dtype][0] / check_segment(run) / check_segment(name)
    if not doc_dir.is_dir():
        abort(404)
    return doc_dir


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
        abort(500, f"could not read {qa_path.name}")
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
    """Latest QA dataset generated from any chunk run of this document, or null.

    generate_qa.py writes qa_<datetime>_<model>.json into chunk dirs
    (<title dir>/<stamp>_chunk_*/), so the newest file across all chunk
    runs is picked by its timestamped name.
    """
    doc_dir = qa_doc_dir(dtype, run, name)
    qa_path = latest_qa_file(doc_dir)
    if qa_path is None:
        return jsonify(None)
    return jsonify(qa_payload(doc_dir, qa_path))


@app.get("/api/documents/<dtype>/<run>/<name>/evals")
def eval_list(dtype: str, run: str, name: str):
    """Eval files (from eval_retrieval.py) whose metadata points at the given
    QA file, newest first. qa_file is doc-dir-relative, as served by qa_data."""
    doc_dir = qa_doc_dir(dtype, run, name)
    qa_file = request.args.get("qa_file", "")
    eval_dir = doc_dir / "evaluations"
    out = []
    if eval_dir.is_dir():
        for f in sorted(eval_dir.iterdir(), reverse=True):
            if not (f.is_file() and EVAL_FILE.fullmatch(f.name)):
                continue
            try:
                meta = json.loads(f.read_text(encoding="utf-8")).get("metadata", {})
            except (OSError, json.JSONDecodeError):
                continue
            if qa_file and not str(meta.get("qa_file", "")).endswith(qa_file):
                continue
            out.append(
                {
                    "file": f.name,
                    "db_type": meta.get("db_type"),
                    "embedding_model": meta.get("embedding_model"),
                    "top_k": meta.get("top_k"),
                    "datetime": meta.get("datetime"),
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
    """
    doc_dir = qa_doc_dir(dtype, run, name)
    payload = request.get_json(force=True) or {}

    parts = str(payload.get("qa_file", "")).split("/")
    if len(parts) != 2 or not QA_FILE.fullmatch(parts[1]):
        abort(400, "invalid qa_file")
    qa_path = doc_dir / check_segment(parts[0]) / parts[1]
    if not qa_path.is_file():
        abort(404)

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
    return jsonify(qa_payload(doc_dir, qa_path))


@app.post("/api/documents/<dtype>/<run>/<name>/qa/generate")
def qa_generate(dtype: str, run: str, name: str):
    """Generate QA pairs for specific chunks via generate_qa.py and add them
    to the document's latest qa_*.json, sorted and replacing duplicates."""
    doc_dir = qa_doc_dir(dtype, run, name)
    payload = request.get_json(force=True) or {}

    chunks = payload.get("chunks")
    if not (isinstance(chunks, list) and chunks
            and all(isinstance(c, int) and c >= 0 for c in chunks)):
        abort(400, "chunks must be a non-empty list of chunk indices")
    types = payload.get("types") or ["direct", "inference", "paraphrased"]
    if not (isinstance(types, list)
            and all(t in ("direct", "inference", "paraphrased") for t in types)):
        abort(400, "invalid question types")

    qa_path = latest_qa_file(doc_dir)
    if qa_path is None:
        abort(404, "no QA dataset to append to")

    project_root = Path(__file__).resolve().parent.parent
    script = Path(__file__).resolve().parent / "generate_qa.py"
    code = run_and_stream([
        sys.executable, str(script),
        "--dataset", qa_path.parent.relative_to(project_root).as_posix(),
        "--chunks", ",".join(map(str, sorted(set(chunks)))),
        "--types", ",".join(types),
        "--add", str(qa_path),
    ])
    if code != 0:
        abort(500, f"generate_qa.py failed with exit code {code}")
    return jsonify(qa_payload(doc_dir, qa_path))


@app.post("/api/documents/<run>/<name>/ocr/<int:page>")
def run_page_ocr(run: str, name: str, page: int):
    """Run (or re-run) EasyOCR on a single page via easyocr_pdfimages.py."""
    doc_dir = OUTPUT_ROOT / check_segment(run) / check_segment(name)
    image_path = doc_dir / f"page_{page}.png"
    if not image_path.is_file():
        abort(404)

    script = Path(__file__).resolve().parent / "easyocr_pdfimages.py"
    code = run_and_stream([sys.executable, str(script), str(image_path)])
    if code != 0:
        abort(500, f"OCR failed with exit code {code}")

    json_path = doc_dir / f"page_{page}_easyocr.json"
    try:
        text = json.loads(json_path.read_text()).get("extracted_text", "")
    except (OSError, json.JSONDecodeError):
        abort(500, "OCR ran but no result file was produced")
    return jsonify({"page": page, "extracted_text": text})


@app.post("/api/documents/<run>/<name>/ocr")
def run_all_ocr(run: str, name: str):
    """OCR every page still missing easyocr data, in one script run."""
    doc_dir = OUTPUT_ROOT / check_segment(run) / check_segment(name)
    if not doc_dir.is_dir():
        abort(404)

    script = Path(__file__).resolve().parent / "easyocr_pdfimages.py"
    code = run_and_stream([sys.executable, str(script), "--missing-only", str(doc_dir)])
    if code != 0:
        abort(500, f"OCR failed with exit code {code}")
    return ocr_data(run, name)


@app.get("/images/<dtype>/<run>/<name>/<filename>")
def image(dtype: str, run: str, name: str, filename: str):
    if dtype not in DOC_TYPES or not SERVABLE_PNG.fullmatch(filename):
        abort(404)
    doc_dir = DOC_TYPES[dtype][0] / check_segment(run) / check_segment(name)
    return send_from_directory(doc_dir, filename)


@app.get("/logs/stream")
def log_stream():
    def generate():
        q, backlog = logs.listen()
        try:
            for line in backlog:
                yield f"data: {line}\n\n"
            while True:
                try:
                    yield f"data: {q.get(timeout=15)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            logs.drop(q)

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
    converter = convert if method == "pdf2image" else convert_text

    log.info(f"Upload received: {filename} (method: {method})")
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / filename
        file.save(pdf_path)
        try:
            out_dir = converter(pdf_path)
        except Exception as exc:
            log.error(f"ERROR: {exc}")
            raise

    pattern = DOC_TYPES[method][1]
    return jsonify(
        {
            "type": method,
            "run": out_dir.parent.name,
            "name": out_dir.name,
            "pages": len(page_files(out_dir, pattern)),
        }
    )


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
