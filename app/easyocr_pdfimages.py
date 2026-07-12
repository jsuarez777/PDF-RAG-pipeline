#!/usr/bin/env python3
"""OCR the page images produced by pdf_to_images.py using EasyOCR.

Usage:
    python app/easyocr_pdfimages.py [--missing-only] [path/to/image/folder | path/to/page_<n>.png]

A folder argument processes every page in it (only pages without an
existing page_<n>_easyocr.json when --missing-only is given); a single page_<n>.png
argument processes only that page. If no argument is given, the script
lists the documents available under
data/pdf2image/<YYYYMMDD_NN>/<name>/ and prompts for a selection. Each
page is OCR'd in English with paragraph grouping. Progress goes to
stdout and the run's log file; the full extracted text is written to
the log file only. Results are saved alongside each image as
page_<n>_easyocr.json with metadata and the extracted text.
"""

import json
import logging
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logging_utils import setup_logging  # noqa: E402
from pdf_to_images import OUTPUT_ROOT  # noqa: E402

log = logging.getLogger(__name__)

# EasyOCR hardcodes pin_memory=True in its torch DataLoader; unsupported on
# Apple's MPS backend, so torch warns once per page. Harmless — silence it.
warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")

PAGE_FILE = re.compile(r"^page_(\d+)\.png$")
SEPARATOR_WIDTH = 60


def page_images(folder: Path) -> list[tuple[int, Path]]:
    """Return (page number, path) pairs for page_<n>.png files, in order."""
    pages = []
    for f in folder.iterdir():
        match = PAGE_FILE.fullmatch(f.name)
        if match:
            pages.append((int(match.group(1)), f))
    return sorted(pages)


def list_documents() -> list[Path]:
    """Document folders under data/pdf2image/, newest run first."""
    docs = []
    if OUTPUT_ROOT.is_dir():
        for run_dir in sorted(OUTPUT_ROOT.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            for doc_dir in sorted(run_dir.iterdir()):
                if doc_dir.is_dir() and page_images(doc_dir):
                    docs.append(doc_dir)
    return docs


def choose_document() -> Path:
    docs = list_documents()
    if not docs:
        log.error(f"No scanned documents found under {OUTPUT_ROOT}")
        raise SystemExit(1)

    log.info("Available documents:")
    for i, doc in enumerate(docs, start=1):
        label = f"{doc.parent.name}/{doc.name}"
        log.info(f"  {i}. {label} ({len(page_images(doc))} page(s))")

    while True:
        raw = input(f"Select a document [1-{len(docs)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(docs):
            return docs[int(raw) - 1]
        log.info("Invalid selection, try again.")


def page_marker(number: int) -> str:
    return f" Page {number} ".center(SEPARATOR_WIDTH, "=")


def make_reader():
    log.info("Loading EasyOCR (English) ...")
    import easyocr  # deferred: slow import, pulls in torch

    try:
        version = importlib_metadata.version("easyocr")
    except importlib_metadata.PackageNotFoundError:
        version = "unknown"
    return easyocr.Reader(["en"], verbose=False), version


def ocr_page(reader, version: str, number: int, image_path: Path) -> None:
    log.info(f"OCR page {number}: {image_path.name} ...")
    started = time.monotonic()
    ran_at = datetime.now(timezone.utc).isoformat()
    results = reader.readtext(str(image_path), paragraph=True)
    duration = time.monotonic() - started

    # With paragraph=True items are [bbox, text]; without it, a
    # confidence float is appended — keep it when present.
    regions = []
    for item in results:
        region = {
            "bbox": [[int(x), int(y)] for x, y in item[0]],
            "text": item[1],
        }
        if len(item) > 2:
            region["confidence"] = float(item[2])
        regions.append(region)

    folder = image_path.parent
    record = {
        "document": folder.name,
        "run": folder.parent.name,
        "page": number,
        "image_file": image_path.name,
        "ocr": {
            "engine": "easyocr",
            "version": version,
            "languages": ["en"],
            "paragraph": True,
            "ran_at": ran_at,
            "duration_seconds": round(duration, 3),
            "regions": len(regions),
        },
        "regions": regions,
        "extracted_text": "\n\n".join(r["text"] for r in regions),
    }

    json_path = image_path.with_name(f"page_{number}_easyocr.json")
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    log.info(f"  wrote {json_path.name} ({len(regions)} region(s), {duration:.1f}s)")

    # Full page text only at DEBUG, so it lands in the log file but not
    # on the console or in the viewer's log pane.
    log.debug(page_marker(number))
    log.debug(record["extracted_text"])
    log.debug("")


def main() -> int:
    log_file = setup_logging("easyocr_pdfimages", debug_file=True)
    log.info(f"Logging to {log_file}")

    args = sys.argv[1:]
    missing_only = "--missing-only" in args
    if missing_only:
        args.remove("--missing-only")

    if args:
        target = Path(args[0]).expanduser().resolve()
        if target.is_file():
            match = PAGE_FILE.fullmatch(target.name)
            if not match:
                log.error(f"Error: not a page_<n>.png file: {target}")
                return 1
            pages = [(int(match.group(1)), target)]
        elif target.is_dir():
            pages = page_images(target)
            if not pages:
                log.error(f"Error: no page_<n>.png files in {target}")
                return 1
        else:
            log.error(f"Error: no such file or folder: {target}")
            return 1
    else:
        pages = page_images(choose_document())

    if missing_only:
        skipped = [n for n, p in pages if p.with_name(f"page_{n}_easyocr.json").is_file()]
        pages = [(n, p) for n, p in pages if n not in skipped]
        if skipped:
            log.info(f"Skipping {len(skipped)} page(s) with existing OCR data")
        if not pages:
            log.info("Nothing to do: every page already has OCR data")
            return 0

    reader, version = make_reader()
    for number, image_path in pages:
        ocr_page(reader, version, number, image_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
