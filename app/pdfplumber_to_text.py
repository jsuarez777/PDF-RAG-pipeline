#!/usr/bin/env python3
"""Extract per-page text and tables from a PDF using pdfplumber.

Usage:
    python app/pdfplumber_to_text.py [path/to/document.pdf]

If no path is given, the script prompts for one. Output goes to
data/pdfplumber/<YYYYMMDD_NN>/<normalized-pdf-name>/page_<n>.json,
mirroring the folder conventions of pdf_to_images.py. Each JSON holds
`text` (prose with table content filtered out), `tables` (structured
rows for each detected table), `full_text` (the unfiltered
extraction, kept as a fallback since table detection is heuristic),
and `images` (metadata for embedded images, each crop-rendered to
page_<n>_image_<m>.png alongside the JSON).
"""

import json
import logging
import sys
from pathlib import Path

import pdfplumber
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logging_utils import setup_logging  # noqa: E402
from pdf_to_images import PROJECT_ROOT, next_run_dir, normalize_name  # noqa: E402

log = logging.getLogger(__name__)

OUTPUT_ROOT = PROJECT_ROOT / "data" / "pdfplumber"


def extract_page(page) -> dict:
    """Split a page into prose-only text, structured tables, and full text.

    Characters whose center falls inside a detected table's bbox are
    filtered out of `text`, so table content lives only in `tables`.
    `full_text` keeps the unfiltered extraction as a safety net, since
    table detection is heuristic.
    """
    full_text = page.extract_text() or ""
    tables = page.find_tables()
    if not tables:
        return {"text": full_text, "tables": [], "full_text": full_text}

    bboxes = [t.bbox for t in tables]

    def outside_tables(obj) -> bool:
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        return not any(
            x0 <= cx <= x1 and top <= cy <= bottom for (x0, top, x1, bottom) in bboxes
        )

    return {
        "text": page.filter(outside_tables).extract_text() or "",
        "tables": [t.extract() for t in tables],
        "full_text": full_text,
    }


def extract_images(page, out_dir: Path, page_number: int) -> list[dict]:
    """Crop-and-render each embedded image on the page to a PNG file.

    Bounding boxes are clamped to the page, since embedded images can
    bleed past the visible page edges and page.crop rejects that.
    """
    records = []
    for number, image in enumerate(page.images, start=1):
        x0 = max(image["x0"], page.bbox[0])
        top = max(image["top"], page.bbox[1])
        x1 = min(image["x1"], page.bbox[2])
        bottom = min(image["bottom"], page.bbox[3])
        if x0 >= x1 or top >= bottom:
            continue

        filename = f"page_{page_number}_image_{number}.png"
        target = out_dir / filename
        page.crop((x0, top, x1, bottom)).to_image(resolution=150).save(target)
        log.info(f"  wrote {target.relative_to(PROJECT_ROOT)}")
        records.append(
            {
                "file": filename,
                "image_number": number,
                "bbox": {
                    "x0": image["x0"],
                    "top": image["top"],
                    "x1": image["x1"],
                    "bottom": image["bottom"],
                },
                "width": image["width"],
                "height": image["height"],
                "source_size": list(image["srcsize"]),
                "name": image.get("name"),
            }
        )
    return records


def convert(pdf_path: Path) -> Path:
    out_dir = next_run_dir(OUTPUT_ROOT) / normalize_name(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Extracting text from {pdf_path} with pdfplumber ...")
    with pdfplumber.open(pdf_path) as pdf:
        count = len(pdf.pages)
        for number, page in enumerate(pdf.pages, start=1):
            record = {
                "document": out_dir.name,
                "run": out_dir.parent.name,
                "page": number,
                **extract_page(page),
                "images": extract_images(page, out_dir, number),
            }
            target = out_dir / f"page_{number}.json"
            target.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            log.info(f"  wrote {target.relative_to(PROJECT_ROOT)}")

    # Page preview images for the viewer (pymupdf needs no poppler).
    with pymupdf.open(pdf_path) as doc:
        for number, page in enumerate(doc, start=1):
            target = out_dir / f"page_{number}.png"
            page.get_pixmap(dpi=150).save(target)
            log.info(f"  wrote {target.relative_to(PROJECT_ROOT)}")

    log.info(f"Done: {count} page(s) -> {out_dir.relative_to(PROJECT_ROOT)}")
    return out_dir


def main() -> int:
    log_file = setup_logging("pdfplumber_to_text")
    log.info(f"Logging to {log_file}")

    if len(sys.argv) > 1:
        raw = sys.argv[1]
    else:
        raw = input("Path to PDF file: ").strip().strip("'\"")

    pdf_path = Path(raw).expanduser().resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        log.error(f"Error: not a PDF file: {pdf_path}")
        return 1

    convert(pdf_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
