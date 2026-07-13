"""Extract text, tables, and images from a PDF using pdfplumber.

Usage:
    python app/pdfplumber_extract.py [path/to/document.pdf]

If no path is given, the script prompts for one. Output goes to
data/pdfplumber/<YYYYMMDD_NN>/page_<n>.json, where NN is an iteration
counter that increments per run on the same day. Each page JSON holds
metadata (document name, page number, extract datetime, pdfplumber
version), the page text, the tables (ordered top to bottom), and a list
of image entries pointing at page_<n>_image_<m>.png files saved
alongside the JSON.
"""

import json
import logging
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "data" / "pdfplumber"
IMAGE_RESOLUTION = 200


def next_run_dir(root: Path) -> Path:
    """Return data/pdfplumber/<YYYYMMDD_NN> with NN incremented per day."""
    today = date.today().strftime("%Y%m%d")
    iteration = 1
    for existing in root.glob(f"{today}_*"):
        match = re.fullmatch(rf"{today}_(\d+)", existing.name)
        if match:
            iteration = max(iteration, int(match.group(1)) + 1)
    return root / f"{today}_{iteration:02d}"


def clamp_bbox(bbox, page) -> tuple[float, float, float, float]:
    """Clamp a bounding box to the page bounds so cropping never fails."""
    x0, top, x1, bottom = bbox
    return (
        max(x0, 0),
        max(top, 0),
        min(x1, page.width),
        min(bottom, page.height),
    )


def extract_tables(page) -> list[dict]:
    """Extract tables ordered top to bottom, with bbox and cell data.

    pdfplumber exposes no caption or title metadata for tables, so the
    metadata is limited to position and shape.
    """
    tables = sorted(page.find_tables(), key=lambda t: (t.bbox[1], t.bbox[0]))
    results = []
    for number, table in enumerate(tables, start=1):
        rows = table.extract()
        results.append(
            {
                "table_number": number,
                "bbox": list(table.bbox),
                "row_count": len(rows),
                "column_count": max((len(row) for row in rows), default=0),
                "rows": rows,
            }
        )
    return results


def extract_images(page, out_dir: Path, page_number: int) -> list[dict]:
    """Save each embedded image as a PNG and return metadata entries."""
    results = []
    for number, image in enumerate(page.images, start=1):
        filename = f"page_{page_number}_image_{number}.png"
        bbox = clamp_bbox((image["x0"], image["top"], image["x1"], image["bottom"]), page)
        entry = {
            "image_number": number,
            "name": image.get("name"),
            "bbox": list(bbox),
            "width": image["width"],
            "height": image["height"],
            "file": filename,
        }
        try:
            page.crop(bbox).to_image(resolution=IMAGE_RESOLUTION).save(out_dir / filename)
        except Exception as exc:  # degenerate bboxes (zero area) can't render
            entry["file"] = None
            entry["error"] = str(exc)
        results.append(entry)
    return results


def extract(pdf_path: Path) -> Path:
    out_dir = next_run_dir(OUTPUT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Extracting {pdf_path} with pdfplumber {pdfplumber.__version__} ...")
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            number = page.page_number
            images = extract_images(page, out_dir, number)
            page_data = {
                "metadata": {
                    "document": pdf_path.name,
                    "page": number,
                    "extracted_at": datetime.now().isoformat(timespec="seconds"),
                    "pdfplumber_version": pdfplumber.__version__,
                },
                "text": page.extract_text() or "",
                "tables": extract_tables(page),
                "images": images,
            }
            target = out_dir / f"page_{number}.json"
            target.write_text(json.dumps(page_data, indent=2, ensure_ascii=False))
            log.info(
                f"  wrote {target.relative_to(PROJECT_ROOT)}"
                f" ({len(page_data['tables'])} table(s), {len(images)} image(s))"
            )
            page.close()
        log.info(f"Done: {len(pdf.pages)} page(s) -> {out_dir.relative_to(PROJECT_ROOT)}")
    return out_dir


def main() -> int:
    log_file = setup_logging("pdfplumber_extract")
    log.info(f"Logging to {log_file}")

    if len(sys.argv) > 1:
        raw = sys.argv[1]
    else:
        raw = input("Path to PDF file: ").strip().strip("'\"")

    pdf_path = Path(raw).expanduser().resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        log.error(f"Error: not a PDF file: {pdf_path}")
        return 1

    extract(pdf_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
