#!/usr/bin/env python3
"""Chunk extracted page text from a verified dataset into a chunked_text.json.

Datasets live under data/{pdf2image,pdfplumber}/<date>/<title>/.
  - pdf2image:  page_N_easyocr.json, text field "extracted_text"
  - pdfplumber: page_N.json,         text field "full_text"

Chunk types: fixed_size, sentence, semantic (only fixed_size implemented).
fixed_size is given as  fixed_size:<size>:<overlap>  where overlap is a
character count, or a percent of size if it ends with "%".
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

EXTRACTORS = {
    "pdf2image": {"json_suffix": "_easyocr.json", "text_field": "extracted_text"},
    "pdfplumber": {"json_suffix": ".json", "text_field": "full_text"},
}

CHUNK_TYPES = ("fixed_size", "sentence", "semantic")

PAGE_RE = re.compile(r"^page_(\d+)")


def parse_type(type_str):
    """Return (method, size, overlap). size/overlap are None unless fixed_size."""
    parts = type_str.split(":")
    method = parts[0]
    if method not in CHUNK_TYPES:
        sys.exit(f"ERROR: unknown chunk type '{method}'. Valid types: {', '.join(CHUNK_TYPES)}")

    if method != "fixed_size":
        if len(parts) > 1:
            sys.exit(f"ERROR: chunk type '{method}' does not take extra parameters")
        return method, None, None

    if len(parts) != 3:
        sys.exit("ERROR: fixed_size must be of form fixed_size:<chunk size>:<overlap>, e.g. fixed_size:100:20 or fixed_size:200:10%")

    try:
        size = int(parts[1])
    except ValueError:
        sys.exit(f"ERROR: invalid chunk size '{parts[1]}' (must be an integer)")
    if size <= 0:
        sys.exit("ERROR: chunk size must be > 0")

    overlap = parse_overlap(parts[2], size)
    return method, size, overlap


def parse_overlap(overlap_str, size):
    overlap_str = overlap_str.strip()
    if overlap_str.endswith("%"):
        try:
            pct = float(overlap_str[:-1])
        except ValueError:
            sys.exit(f"ERROR: invalid overlap percent '{overlap_str}'")
        if pct < 0:
            sys.exit("ERROR: overlap percent must be >= 0")
        overlap = int(size * pct / 100)
    else:
        try:
            overlap = int(overlap_str)
        except ValueError:
            sys.exit(f"ERROR: invalid overlap '{overlap_str}' (must be an integer or percent like 10%)")
        if overlap < 0:
            sys.exit("ERROR: overlap must be >= 0")

    if overlap >= size:
        sys.exit(f"ERROR: overlap ({overlap}) must be less than chunk size ({size})")
    if overlap > size / 2:
        print(f"WARNING: overlap ({overlap}) is more than 50% of chunk size ({size})")
    return overlap


def page_num(path):
    m = PAGE_RE.match(path.name)
    return int(m.group(1)) if m else None


def scan_datasets():
    """Return list of dataset dicts for every data/<extractor>/<date>/<title> folder."""
    datasets = []
    for extractor, cfg in EXTRACTORS.items():
        base = DATA_DIR / extractor
        if not base.is_dir():
            continue
        for date_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for title_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
                pages = set()
                json_pages = set()
                for f in title_dir.iterdir():
                    n = page_num(f)
                    if n is None:
                        continue
                    pages.add(n)
                    if f.name == f"page_{n}{cfg['json_suffix']}":
                        json_pages.add(n)
                missing = sorted(pages - json_pages)
                datasets.append({
                    "extractor": extractor,
                    "path": title_dir,
                    "rel": title_dir.relative_to(ROOT).as_posix(),
                    "pages": sorted(pages),
                    "missing_json": missing,
                })
    return datasets


def choose_dataset(datasets, preselect=None):
    if not datasets:
        sys.exit("ERROR: no datasets found under data/pdf2image or data/pdfplumber")

    print("\nAvailable datasets:")
    for i, ds in enumerate(datasets, 1):
        if ds["missing_json"]:
            status = f"ERROR: {len(ds['missing_json'])} pages missing json"
        else:
            status = f"{len(ds['pages'])} pages"
        print(f"  [{i}] {ds['rel']}  ({status})")

    if preselect is not None:
        choice = preselect
        print(f"\nDataset (from --dataset): {choice}")
    else:
        choice = input("\nChoose a dataset (number or path): ").strip()

    selected = None
    if choice.isdigit() and 1 <= int(choice) <= len(datasets):
        selected = datasets[int(choice) - 1]
    else:
        norm = choice.rstrip("/")
        for ds in datasets:
            if norm in (ds["rel"], str(ds["path"])):
                selected = ds
                break

    if selected is None:
        sys.exit(f"ERROR: '{choice}' is not a valid dataset choice")
    if selected["missing_json"]:
        print(
            f"ERROR: dataset {selected['rel']} cannot be chosen: "
            f"{len(selected['missing_json'])} pages are missing their json file "
            f"(pages: {', '.join(map(str, selected['missing_json'][:10]))}"
            f"{', ...' if len(selected['missing_json']) > 10 else ''})",
            file=sys.stderr,
        )
        sys.exit(-1)
    if not selected["pages"]:
        sys.exit(f"ERROR: dataset {selected['rel']} contains no pages")
    return selected


def load_and_validate_pages(ds):
    """Validate every page json has the extractor's text field; return [(page, text)]."""
    cfg = EXTRACTORS[ds["extractor"]]
    field = cfg["text_field"]
    pages = []
    for n in ds["pages"]:
        jpath = ds["path"] / f"page_{n}{cfg['json_suffix']}"
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            sys.exit(f"ERROR: could not read {jpath.relative_to(ROOT)}: {e}")
        if field not in data or not isinstance(data[field], str):
            sys.exit(
                f"ERROR: {jpath.relative_to(ROOT)} is missing required string field "
                f"'{field}' for extractor '{ds['extractor']}'"
            )
        pages.append((n, data[field]))
    print(f"Validated {len(pages)} pages: all have string '{field}'")
    return pages


def fixed_size_chunks(pages, size, overlap):
    """Concatenate page texts (newline-joined) and cut fixed-size overlapping chunks."""
    # Build full text with per-page character spans so chunks can report
    # which pages they start/end on.
    full_parts = []
    spans = []  # (page, start, end) in full-text coordinates
    pos = 0
    for i, (n, text) in enumerate(pages):
        if i > 0:
            full_parts.append("\n")
            pos += 1
        spans.append((n, pos, pos + len(text)))
        full_parts.append(text)
        pos += len(text)
    full_text = "".join(full_parts)

    def page_at(char_pos):
        for n, start, end in spans:
            if start <= char_pos < end:
                return n
        # position falls on a page-separator newline or past the end
        for n, start, end in spans:
            if char_pos < start:
                return n
        return spans[-1][0]

    chunks = []
    step = size - overlap
    start = 0
    while start < len(full_text):
        text = full_text[start:start + size]
        end = start + len(text)
        chunks.append({
            "chunk_index": len(chunks),
            "text": text,
            "start_char": start,
            "end_char": end,
            "num_chars": len(text),
            "start_page": page_at(start),
            "end_page": page_at(end - 1),
        })
        if end >= len(full_text):
            break
        start += step
    return chunks, len(full_text)


def main():
    parser = argparse.ArgumentParser(description="Chunk extracted dataset text.")
    parser.add_argument(
        "--type",
        help="Chunk type: fixed_size:<size>:<overlap>, sentence, or semantic. "
             "Overlap is characters, or percent of size if it ends with %%.",
    )
    parser.add_argument("--dataset", help="Dataset path (data/<extractor>/<date>/<title>) or list number")
    args = parser.parse_args()

    type_str = args.type
    if not type_str:
        type_str = input(f"Chunk type ({', '.join(CHUNK_TYPES)}): ").strip()
        if type_str == "fixed_size":
            size_in = input("Chunk size (characters): ").strip()
            overlap_in = input("Overlap (characters, or percent like 10%): ").strip()
            type_str = f"fixed_size:{size_in}:{overlap_in}"

    method, size, overlap = parse_type(type_str)
    if method != "fixed_size":
        sys.exit(f"ERROR: chunk type '{method}' is not implemented yet (only fixed_size)")

    datasets = scan_datasets()
    ds = choose_dataset(datasets, preselect=args.dataset)
    pages = load_and_validate_pages(ds)

    chunks, total_chars = fixed_size_chunks(pages, size, overlap)

    now = datetime.now()
    out_dir = ds["path"] / f"{now.strftime('%Y%m%d_%H%M%S')}_chunk_{method}_{size}_{overlap}"
    out_dir.mkdir(parents=True, exist_ok=False)

    result = {
        "metadata": {
            "datetime": now.isoformat(timespec="seconds"),
            "dataset": ds["rel"],
            "extractor": ds["extractor"],
            "chunk_method": method,
            "chunk_size": size,
            "overlap": overlap,
            "num_pages": len(pages),
            "total_chars": total_chars,
            "num_chunks": len(chunks),
        },
        "chunks": chunks,
    }

    out_file = out_dir / "chunked_text.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks ({total_chars} chars from {len(pages)} pages)")
    print(f"Output: {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
