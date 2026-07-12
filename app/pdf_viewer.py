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
