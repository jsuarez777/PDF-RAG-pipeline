"""Convert every page of a PDF into PNG images.

Usage:
    python app/pdf_to_images.py [path/to/document.pdf]

If no path is given, the script prompts for one. Output goes to
data/pdf2image/<YYYYMMDD_NN>/<normalized-pdf-name>/page_<n>.png,
where NN is an iteration counter that increments per run on the same day.
"""

import logging
import re
import sys
from datetime import date
from pathlib import Path

from logging_utils import setup_logging
from pdf2image import convert_from_path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "data" / "pdf2image"
DPI = 200


def normalize_name(pdf_path: Path) -> str:
    """Normalize a PDF filename using git branch naming rules."""
    name = pdf_path.stem
    # Whitespace becomes underscore.
    name = re.sub(r"\s+", "_", name)
    # Drop characters git refs forbid: ~ ^ : ? * [ ] \ and control chars.
    name = re.sub(r"[~^:?*\[\]\\\x00-\x1f\x7f]", "", name)
    # Sequences like ".." and "@{" are invalid in refs.
    name = re.sub(r"\.{2,}", ".", name)
    name = name.replace("@{", "_")
    # No leading/trailing dots, no trailing ".lock", no leading dashes.
    name = name.strip(".").lstrip("-")
    if name.endswith(".lock"):
        name = name[: -len(".lock")]
    return name or "document"


def next_run_dir(root: Path) -> Path:
    """Return data/pdf2image/<YYYYMMDD_NN> with NN incremented per day."""
    today = date.today().strftime("%Y%m%d")
    iteration = 1
    for existing in root.glob(f"{today}_*"):
        match = re.fullmatch(rf"{today}_(\d+)", existing.name)
        if match:
            iteration = max(iteration, int(match.group(1)) + 1)
    return root / f"{today}_{iteration:02d}"


def convert(pdf_path: Path, output_root: Path | None = None) -> Path:
    out_dir = next_run_dir(output_root or OUTPUT_ROOT) / normalize_name(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Converting {pdf_path} at {DPI} dpi ...")
    pages = convert_from_path(pdf_path, dpi=DPI)
    for number, page in enumerate(pages, start=1):
        target = out_dir / f"page_{number}.png"
        page.save(target, "PNG")
        log.info(f"  wrote {target.relative_to(PROJECT_ROOT)}")
    log.info(f"Done: {len(pages)} page(s) -> {out_dir.relative_to(PROJECT_ROOT)}")
    return out_dir


def main() -> int:
    log_file = setup_logging("pdf_to_images")
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
