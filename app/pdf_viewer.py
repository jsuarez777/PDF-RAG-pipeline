#!/usr/bin/env python3
"""Flask UI for converting PDFs to page images and browsing the results.

Usage:
    python app/pdf_viewer.py            # serves http://127.0.0.1:5001

Documents live under data/pdf2image/<YYYYMMDD_NN>/<name>/page_<n>.png,
as produced by pdf_to_images.py; uploads are converted with the same code.
"""

import os
import re
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_to_images import OUTPUT_ROOT, convert  # noqa: E402

app = Flask(__name__)

SAFE_SEGMENT = re.compile(r"^[\w.-]+$")
PAGE_FILE = re.compile(r"^page_(\d+)\.png$")


def check_segment(segment: str) -> str:
    if not SAFE_SEGMENT.fullmatch(segment):
        abort(400, "invalid path segment")
    return segment


def page_files(doc_dir: Path) -> list[str]:
    pages = []
    for f in doc_dir.iterdir():
        match = PAGE_FILE.fullmatch(f.name)
        if match:
            pages.append((int(match.group(1)), f.name))
    return [name for _, name in sorted(pages)]


@app.get("/")
def index():
    return render_template("pdf_viewer.html")


@app.get("/api/documents")
def documents():
    docs = []
    if OUTPUT_ROOT.is_dir():
        for run_dir in sorted(OUTPUT_ROOT.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            for doc_dir in sorted(run_dir.iterdir()):
                if doc_dir.is_dir():
                    pages = page_files(doc_dir)
                    if pages:
                        docs.append(
                            {"run": run_dir.name, "name": doc_dir.name, "pages": len(pages)}
                        )
    return jsonify(docs)


@app.get("/api/documents/<run>/<name>/pages")
def pages(run: str, name: str):
    doc_dir = OUTPUT_ROOT / check_segment(run) / check_segment(name)
    if not doc_dir.is_dir():
        abort(404)
    return jsonify(page_files(doc_dir))


@app.get("/images/<run>/<name>/<filename>")
def image(run: str, name: str, filename: str):
    if not PAGE_FILE.fullmatch(filename):
        abort(404)
    doc_dir = OUTPUT_ROOT / check_segment(run) / check_segment(name)
    return send_from_directory(doc_dir, filename)


@app.post("/upload")
def upload():
    file = request.files.get("pdf")
    if file is None or not file.filename:
        abort(400, "no file provided")
    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        abort(400, "not a PDF file")

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / filename
        file.save(pdf_path)
        out_dir = convert(pdf_path)

    return jsonify(
        {"run": out_dir.parent.name, "name": out_dir.name, "pages": len(page_files(out_dir))}
    )


if __name__ == "__main__":
    # Debug mode runs this file twice (reloader parent + serving child);
    # only the parent opens the browser, and only once.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5001")).start()
    app.run(debug=True, port=5001)
