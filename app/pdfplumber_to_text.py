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
import re
import sys
from collections import defaultdict
from pathlib import Path

import pdfplumber
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logging_utils import setup_logging  # noqa: E402
from pdf_to_images import PROJECT_ROOT, next_run_dir, normalize_name  # noqa: E402

log = logging.getLogger(__name__)

OUTPUT_ROOT = PROJECT_ROOT / "data" / "pdfplumber"


def table_is_sparse(rows: list[list]) -> bool:
    """True for grids that are mostly empty cells.

    Chart frames (axis gridlines) get picked up by find_tables as huge
    near-empty grids; real tables in these documents are >60% filled.
    """
    cells = [c for row in rows for c in row]
    filled = sum(1 for c in cells if c and c.strip())
    return not cells or filled / len(cells) < 0.25


def rule_rows(page) -> list[dict]:
    """Find horizontal rules drawn as rows of thin rects.

    Open-style tables draw their ruling as one thin rect per column,
    abutting with ~1pt gaps at column boundaries. Returns each rule-row
    with its merged segments; rows whose segments have large gaps (e.g.
    two unrelated underlines at the same height) are excluded.
    """
    thin = [
        r
        for r in page.rects
        if (r["bottom"] - r["top"]) <= 3 and (r["x1"] - r["x0"]) >= 15
    ]
    clusters = defaultdict(list)
    for r in sorted(thin, key=lambda r: r["top"]):
        key = next((k for k in clusters if abs(k - r["top"]) <= 2), None)
        clusters[key if key is not None else r["top"]].append(r)

    rows = []
    for y in sorted(clusters):
        segs = []
        for r in sorted(clusters[y], key=lambda r: r["x0"]):
            if segs and r["x0"] - segs[-1][1] <= 0.5:  # double-drawn rule
                segs[-1][1] = max(segs[-1][1], r["x1"])
            else:
                segs.append([r["x0"], r["x1"]])
        gaps = [b[0] - a[1] for a, b in zip(segs, segs[1:])]
        if (
            len(segs) >= 2
            and all(g <= 6 for g in gaps)
            and segs[-1][1] - segs[0][0] >= 150
        ):
            rows.append({"y": y, "segs": segs, "x0": segs[0][0], "x1": segs[-1][1]})
    return rows


def group_rule_tables(rows: list[dict]) -> list[list[dict]]:
    """Group rule-rows into tables.

    Tables open with a pair of close-together header rules, so a row
    that follows a large gap AND is itself followed closely by another
    rule starts a new table; a lone far rule is a bottom border.
    """
    groups = []
    for i, row in enumerate(rows):
        g = groups[-1] if groups else None
        starts_new = (
            g is None
            or abs(row["x0"] - g[0]["x0"]) > 10
            or abs(row["x1"] - g[0]["x1"]) > 10
            or (
                row["y"] - g[-1]["y"] > 60
                and i + 1 < len(rows)
                and rows[i + 1]["y"] - row["y"] <= 35
            )
        )
        if starts_new:
            groups.append([row])
        else:
            g.append(row)
    return [g for g in groups if len(g) >= 3]


def extract_rule_table(page, group: list[dict]) -> tuple[tuple, list[list] | None]:
    """Extract one rule-row group as a table via explicit line strategies.

    Columns come from the segment boundaries of the ruling; row lines
    come from clustering word baselines inside the region, since the
    body rows have no ruling of their own.
    """
    x0 = min(r["x0"] for r in group)
    x1 = max(r["x1"] for r in group)
    top, bottom = group[0]["y"] - 2, group[-1]["y"] + 4
    crop = page.crop((x0, top, x1, bottom))

    row_tops = []
    for w in sorted(crop.extract_words(), key=lambda w: w["top"]):
        if not row_tops or w["top"] - row_tops[-1] > 4:
            row_tops.append(w["top"])

    breaks = sorted(
        (a[1] + b[0]) / 2
        for row in group
        for a, b in zip(row["segs"], row["segs"][1:])
    )
    merged = []
    for b in breaks:
        if merged and b - merged[-1][-1] <= 3:
            merged[-1].append(b)
        else:
            merged.append([b])

    table = crop.extract_table(
        {
            "vertical_strategy": "explicit",
            "explicit_vertical_lines": [x0] + [sum(m) / len(m) for m in merged] + [x1],
            "horizontal_strategy": "explicit",
            "explicit_horizontal_lines": [top] + [t - 1 for t in row_tops[1:]] + [bottom],
        }
    )
    return (x0, top, x1, bottom), table


def extract_page(page) -> dict:
    """Split a page into prose-only text, structured tables, and full text.

    Tables come from two detectors: pdfplumber's default lines strategy
    (bordered tables), with sparse chart-frame false positives dropped,
    and a rule-rect detector for open-style tables whose ruling is
    horizontal-only with whitespace-aligned columns. Characters whose
    center falls inside a detected table's bbox are filtered out of
    `text`, so table content lives only in `tables`. `full_text` keeps
    the unfiltered extraction as a safety net, since table detection is
    heuristic.
    """
    full_text = page.extract_text() or ""

    found = [
        (t.bbox, rows)
        for t in page.find_tables()
        if not table_is_sparse(rows := t.extract())
    ]
    for group in group_rule_tables(rule_rows(page)):
        bbox, rows = extract_rule_table(page, group)
        overlaps_found = any(
            bbox[0] < fx1 and fx0 < bbox[2] and bbox[1] < fb and ft < bbox[3]
            for (fx0, ft, fx1, fb), _ in found
        )
        if rows and not overlaps_found and not table_is_sparse(rows):
            found.append((bbox, rows))
    found.sort(key=lambda item: item[0][1])

    if not found:
        return {"text": full_text, "tables": [], "full_text": full_text}

    bboxes = [bbox for bbox, _ in found]

    def outside_tables(obj) -> bool:
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        return not any(
            x0 <= cx <= x1 and top <= cy <= bottom for (x0, top, x1, bottom) in bboxes
        )

    return {
        "text": page.filter(outside_tables).extract_text() or "",
        "tables": [rows for _, rows in found],
        "full_text": full_text,
    }


def cluster_image_boxes(boxes: list[tuple], pad: float = 1.5) -> list[tuple]:
    """Union-find merge of bboxes that overlap or nearly touch once padded.

    Some PDF exporters (e.g. gradient shading, tiled map screenshots) split
    one visual into hundreds of adjacent image XObjects, individually too
    thin to rasterize (they round to 0px at typical resolutions). Grouping
    by adjacency and cropping the union bbox recovers the original figure
    as a single image instead of many broken slivers.
    """
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    padded = [(x0 - pad, top - pad, x1 + pad, bottom + pad) for x0, top, x1, bottom in boxes]
    for i in range(n):
        ax0, at, ax1, ab = padded[i]
        for j in range(i + 1, n):
            bx0, bt, bx1, bb = padded[j]
            if ax0 < bx1 and bx0 < ax1 and at < bb and bt < ab:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idxs in groups.values():
        x0 = min(boxes[i][0] for i in idxs)
        top = min(boxes[i][1] for i in idxs)
        x1 = max(boxes[i][2] for i in idxs)
        bottom = max(boxes[i][3] for i in idxs)
        merged.append((x0, top, x1, bottom))
    return merged


CAPTION_RE = re.compile(r"^Figure\s+\d+(\.\d+)*\b", re.IGNORECASE)


def page_text_lines(page, y_tol: float = 2, x_gap: float = 20) -> list[dict]:
    """Reading-order text lines, split so side-by-side columns don't merge.

    Words are bucketed into rows by top-proximity (same trick as
    rule_rows), then each row is split into contiguous horizontal segments
    wherever two words are more than x_gap apart — otherwise two charts
    sharing a row of axis labels would concatenate into one wide "line".
    """
    words = page.extract_words()
    rows = defaultdict(list)
    for w in sorted(words, key=lambda w: w["top"]):
        key = next((k for k in rows if abs(k - w["top"]) <= y_tol), None)
        rows[key if key is not None else w["top"]].append(w)

    lines = []
    for row_words in rows.values():
        seg = None
        for w in sorted(row_words, key=lambda w: w["x0"]):
            if seg and w["x0"] - seg["x1"] <= x_gap:
                seg["text"] += " " + w["text"]
                seg["x1"] = max(seg["x1"], w["x1"])
                seg["bottom"] = max(seg["bottom"], w["bottom"])
            else:
                if seg:
                    lines.append(seg)
                seg = {"top": w["top"], "bottom": w["bottom"], "x0": w["x0"], "x1": w["x1"], "text": w["text"]}
        if seg:
            lines.append(seg)
    return sorted(lines, key=lambda l: l["top"])


def find_captions(lines: list[dict], min_gap: float = 6) -> list[dict]:
    """Lines starting with 'Figure N.M' that follow a real paragraph break.

    Without the gap check, a body-text line that happens to word-wrap at
    "figure 5.15) is always..." mid-sentence would also match the regex.
    Real captions always sit in the whitespace right below their figure.
    """
    captions = []
    prev_bottom = None
    for l in lines:
        gap = l["top"] - prev_bottom if prev_bottom is not None else None
        if CAPTION_RE.match(l["text"].strip()) and (gap is None or gap > min_gap):
            captions.append(l)
        prev_bottom = l["bottom"]
    return captions


def figure_regions(page, lines: list[dict], captions: list[dict]) -> list[dict]:
    """For each caption, find the figure's extent above it.

    The bottom edge is the caption's top. The top edge is the bottom of the
    nearest preceding "wide" line (close to full column width) above the
    caption — that's the last line of the body-text paragraph before the
    figure, since chart titles/tick-labels/legend text are all much
    narrower than a wrapped prose line. Once that vertical band is known,
    every image, non-trivial rect, and text line inside it (axes frame,
    data-point markers, tick labels, legend) is unioned into one bbox, so
    a figure fragmented across hundreds of image XObjects (or accompanied
    by vector-drawn axes) still renders as a single clean crop.
    """
    wide_min = 180
    regions = []
    for cap in captions:
        prev_bottom = max(
            (c["bottom"] for c in captions if c["top"] < cap["top"]),
            default=page.bbox[1],
        )
        band_top = prev_bottom
        for l in sorted(
            (l for l in lines if prev_bottom <= l["top"] < cap["top"]),
            key=lambda l: -l["top"],
        ):
            if (l["x1"] - l["x0"]) >= wide_min:
                band_top = l["bottom"]
                break
        band_bottom = cap["top"]

        boxes = []
        image_indexes = []
        for idx, image in enumerate(page.images):
            x0 = max(image["x0"], page.bbox[0])
            top = max(image["top"], page.bbox[1])
            x1 = min(image["x1"], page.bbox[2])
            bottom = min(image["bottom"], page.bbox[3])
            if x0 < x1 and band_top - 1 <= top and bottom <= band_bottom + 1:
                boxes.append((x0, top, x1, bottom))
                image_indexes.append(idx)
        for r in page.rects:
            if (r["x1"] - r["x0"]) * (r["bottom"] - r["top"]) < 4:
                continue
            if band_top - 1 <= r["top"] and r["bottom"] <= band_bottom + 1:
                boxes.append((r["x0"], r["top"], r["x1"], r["bottom"]))
        for l in lines:
            if band_top - 1 <= l["top"] and l["bottom"] <= band_bottom + 1:
                boxes.append((l["x0"], l["top"], l["x1"], l["bottom"]))

        if not boxes:
            continue
        x0 = min(b[0] for b in boxes)
        top = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        bottom = max(b[3] for b in boxes)
        regions.append(
            {"caption": cap["text"], "bbox": (x0, top, x1, bottom), "image_indexes": image_indexes}
        )
    return regions


def extract_images(page, out_dir: Path, page_number: int) -> list[dict]:
    """Crop-and-render each figure on the page to a PNG file.

    Figures with a "Figure N.M" caption are extracted whole (see
    figure_regions): axes, tick labels, legend, and every data-point image
    fragment inside the caption's figure band, unioned into one crop. Any
    remaining images not covered by a caption (or on pages with no
    captions at all) fall back to adjacency clustering (see
    cluster_image_boxes), since some PDF exporters fragment a single
    visual into hundreds of touching image XObjects too thin to rasterize
    on their own.
    """
    lines = page_text_lines(page)
    captions = find_captions(lines)
    regions = figure_regions(page, lines, captions)
    consumed = {idx for region in regions for idx in region["image_indexes"]}

    boxes = []
    for idx, image in enumerate(page.images):
        if idx in consumed:
            continue
        x0 = max(image["x0"], page.bbox[0])
        top = max(image["top"], page.bbox[1])
        x1 = min(image["x1"], page.bbox[2])
        bottom = min(image["bottom"], page.bbox[3])
        if x0 >= x1 or top >= bottom:
            continue
        boxes.append((x0, top, x1, bottom))

    crops = [(region["bbox"], region["caption"]) for region in regions]
    crops += [(box, None) for box in cluster_image_boxes(boxes)]

    records = []
    for number, ((x0, top, x1, bottom), caption) in enumerate(crops, start=1):
        filename = f"page_{page_number}_image_{number}.png"
        target = out_dir / filename
        try:
            page.crop((x0, top, x1, bottom)).to_image(resolution=150).save(target)
        except (ValueError, AttributeError) as exc:
            log.warning(
                f"  skipped page_{page_number}_image_{number}: render failed ({exc})"
            )
            continue
        log.info(f"  wrote {target.relative_to(PROJECT_ROOT)}")
        record = {
            "file": filename,
            "image_number": number,
            "bbox": {"x0": x0, "top": top, "x1": x1, "bottom": bottom},
            "width": x1 - x0,
            "height": bottom - top,
            "source_size": [round((x1 - x0) * 150 / 72), round((bottom - top) * 150 / 72)],
        }
        if caption:
            record["caption"] = caption
        records.append(record)
    return records


def convert(pdf_path: Path, output_root: Path | None = None) -> Path:
    out_dir = next_run_dir(output_root or OUTPUT_ROOT) / normalize_name(pdf_path)
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
