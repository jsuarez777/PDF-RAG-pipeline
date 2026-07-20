#!/usr/bin/env python3
"""Chunk extracted page text from a verified dataset into a chunked_text.json.

Datasets live under data/{pdf2image,pdfplumber}/<date>/<title>/.
  - pdf2image:  page_N_easyocr.json, text field "extracted_text"
  - pdfplumber: page_N.json,         text field "full_text"

Chunk types: fixed_size, sentence, sentence-dynamic-min, plumber-struct,
semantic.
  fixed_size:<size>:<overlap>       size/overlap in tokens (tiktoken cl100k_base,
                                    the encoding of the OpenAI embedding models
                                    used downstream); overlap may also be a
                                    percent of size ("10%"). Token windows are
                                    mapped back to char offsets so chunk
                                    boundaries stay addressable in the text.
  sentence:<n>[:<overlap>]          exactly n sentences/chunk (optional sentence
                                    overlap); oversized "sentences" (flattened
                                    tables) hard-split at a token cap from n.
  sentence-dynamic-min:<n>[:<ov>]   like sentence, but keeps packing past n
                                    until a min-tokens floor is met, so short
                                    fragments (shredded tables) aren't left tiny.
  plumber-struct[:<n>]              pdfplumber only: prose (table-filtered
                                    'text') as sentence chunks, tables as
                                    header-labeled row chunks, images as
                                    descriptor chunks; each chunk records its
                                    source kind.
  semantic:<max>[:<percentile>]     embed each sentence and cut a chunk wherever
                                    the topic shifts (cosine distance between
                                    consecutive sentences above the <percentile>
                                    of all such distances, default 90) or adding
                                    the next sentence would exceed <max> tokens.
                                    Groups sentences by meaning while keeping
                                    every chunk under the token cap. Requires an
                                    OpenAI API key (embeds with
                                    text-embedding-3-small).
"""

import argparse
import json
import os
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (Path(os.environ["PDF_DATA_DIR"]) if os.environ.get("PDF_DATA_DIR")
            else ROOT / "data")  # per-user override set by the web viewer
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))  # so semantic chunking can import the openai_client package

from logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

EXTRACTORS = {
    "pdf2image": {"json_suffix": "_easyocr.json", "text_field": "extracted_text"},
    "pdfplumber": {"json_suffix": ".json", "text_field": "full_text"},
}

CHUNK_TYPES = ("fixed_size", "sentence", "sentence-dynamic-min", "plumber-struct", "semantic")

# Sentence methods share parsing/grouping; only the min-chars floor differs.
SENTENCE_METHODS = ("sentence", "sentence-dynamic-min")

# plumber-struct: sentences packed per text chunk when no :n is given.
PLUMBER_STRUCT_DEFAULT_SENTENCES = 5

# semantic: embedding model used to score sentence-to-sentence topic shifts, and
# the default breakpoint percentile (top 10% largest distances become seams).
SEMANTIC_EMBED_MODEL = "text-embedding-3-small"
SEMANTIC_DEFAULT_PERCENTILE = 90

PAGE_RE = re.compile(r"^page_(\d+)")

# Sentence-chunk sizing, calibrated against the pdfplumber corpus
# (mean 34.5 words/sentence, ~1.3 cl100k tokens/word for English prose).
# A "sentence" that exceeds the derived cap is almost always a flattened
# table/chart with no real boundary, so it is hard-split; see chunk-strategy
# eval notes.
SENT_AVG_WORDS = 35
SENT_MIN_WORDS = 15
SENT_TOKENS_PER_WORD = 1.3
SENT_CAP_BUFFER = 0.20

# All token counting uses the encoding of the OpenAI text-embedding-3-* models
# that embed_chunks.py feeds these chunks to.
TOKEN_ENCODING = "cl100k_base"

_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        import tiktoken  # local import: pulled in only when chunking runs
        _encoder = tiktoken.get_encoding(TOKEN_ENCODING)
    return _encoder


def count_tokens(text):
    return len(get_encoder().encode(text, disallowed_special=()))


def token_starts(text):
    """Encode text; return (tokens, starts) where starts[i] is the char offset
    at which token i begins and starts[len(tokens)] == len(text).

    Token byte sequences concatenate exactly to the utf-8 encoding of the text,
    so cumulative token-byte lengths give byte offsets, which are then mapped
    to char offsets. A token that begins mid-character (a multi-byte char split
    across tokens) maps to that character's offset, so slicing the text at
    starts[] never cuts a character in half.
    """
    enc = get_encoder()
    tokens = enc.encode(text, disallowed_special=())
    byte_ends = []  # byte offset just past char i
    b = 0
    for ch in text:
        b += len(ch.encode("utf-8"))
        byte_ends.append(b)
    starts = []
    ci = 0
    pos = 0
    for tok in tokens:
        while ci < len(byte_ends) and byte_ends[ci] <= pos:
            ci += 1
        starts.append(ci)
        pos += len(enc.decode_single_token_bytes(tok))
    starts.append(len(text))
    return tokens, starts


def sentence_cap_tokens(n_per_chunk):
    """Max tokens a sentence chunk may reach before it is force-split."""
    return int(n_per_chunk * SENT_AVG_WORDS * SENT_TOKENS_PER_WORD * (1 + SENT_CAP_BUFFER))


def sentence_min_tokens(n_per_chunk):
    """Floor (tokens) a sentence-dynamic-min chunk must reach; it keeps absorbing
    sentences past n_per_chunk until met, so short fragments get packed together."""
    return int(n_per_chunk * SENT_MIN_WORDS * SENT_TOKENS_PER_WORD)


def parse_type(type_str):
    """Return (method, size, overlap).

    fixed_size:            size=tokens/chunk, overlap=tokens.
    sentence:              size=sentences/chunk, overlap=sentences (default 0).
    sentence-dynamic-min:  same params; packs past size until a min-chars floor.
    semantic:              size=max tokens/chunk, overlap=breakpoint percentile.
    """
    parts = type_str.split(":")
    method = parts[0]
    if method not in CHUNK_TYPES:
        sys.exit(f"ERROR: unknown chunk type '{method}'. Valid types: {', '.join(CHUNK_TYPES)}")

    if method in SENTENCE_METHODS:
        if len(parts) not in (2, 3):
            sys.exit(f"ERROR: {method} must be of form {method}:<sentences per chunk>[:<overlap sentences>], e.g. {method}:3 or {method}:3:1")
        try:
            size = int(parts[1])
        except ValueError:
            sys.exit(f"ERROR: invalid sentences-per-chunk '{parts[1]}' (must be an integer)")
        if size <= 0:
            sys.exit("ERROR: sentences per chunk must be > 0")
        overlap = 0
        if len(parts) == 3:
            try:
                overlap = int(parts[2])
            except ValueError:
                sys.exit(f"ERROR: invalid overlap '{parts[2]}' (must be an integer number of sentences)")
            if overlap < 0:
                sys.exit("ERROR: overlap must be >= 0")
            if overlap >= size:
                sys.exit(f"ERROR: overlap ({overlap}) must be less than sentences per chunk ({size})")
        return method, size, overlap

    if method == "plumber-struct":
        if len(parts) > 2:
            sys.exit("ERROR: plumber-struct must be of form plumber-struct[:<sentences per text chunk>]")
        size = PLUMBER_STRUCT_DEFAULT_SENTENCES
        if len(parts) == 2:
            try:
                size = int(parts[1])
            except ValueError:
                sys.exit(f"ERROR: invalid sentences-per-chunk '{parts[1]}' (must be an integer)")
            if size <= 0:
                sys.exit("ERROR: sentences per chunk must be > 0")
        return method, size, 0

    if method == "semantic":
        if len(parts) not in (2, 3):
            sys.exit("ERROR: semantic must be of form semantic:<max tokens>[:<breakpoint percentile>], e.g. semantic:512 or semantic:512:90")
        try:
            size = int(parts[1])
        except ValueError:
            sys.exit(f"ERROR: invalid max tokens '{parts[1]}' (must be an integer)")
        if size <= 0:
            sys.exit("ERROR: max tokens must be > 0")
        percentile = SEMANTIC_DEFAULT_PERCENTILE
        if len(parts) == 3:
            try:
                percentile = int(parts[2])
            except ValueError:
                sys.exit(f"ERROR: invalid breakpoint percentile '{parts[2]}' (must be an integer)")
            if not 0 < percentile < 100:
                sys.exit("ERROR: breakpoint percentile must be between 1 and 99")
        return method, size, percentile

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
            sys.exit(f"ERROR: invalid overlap '{overlap_str}' (must be an integer number of tokens or percent like 10%)")
        if overlap < 0:
            sys.exit("ERROR: overlap must be >= 0")

    if overlap >= size:
        sys.exit(f"ERROR: overlap ({overlap}) must be less than chunk size ({size})")
    if overlap > size / 2:
        log.warning(f"WARNING: overlap ({overlap}) is more than 50% of chunk size ({size})")
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

    # newest valid dataset (by data/<extractor>/<date>/ dir name) is the default choice
    eligible = [i for i, ds in enumerate(datasets) if not ds["missing_json"] and ds["pages"]]
    default = max(eligible, key=lambda i: datasets[i]["path"].parent.name) + 1 if eligible else None

    log.info("\nAvailable datasets:")
    for i, ds in enumerate(datasets, 1):
        if ds["missing_json"]:
            status = f"ERROR: {len(ds['missing_json'])} pages missing json"
        else:
            status = f"{len(ds['pages'])} pages"
        mark = "  (latest)" if i == default else ""
        log.info(f"  [{i}] {ds['rel']}  ({status}){mark}")

    if preselect is not None:
        choice = preselect
        log.info(f"\nDataset (from --dataset): {choice}")
    else:
        hint = f", Enter for [{default}]" if default else ""
        choice = input(f"\nChoose a dataset (number or path{hint}): ").strip()
        if not choice and default:
            choice = str(default)

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
    log.info(f"Validated {len(pages)} pages: all have string '{field}'")
    return pages


def build_full_text(pages):
    """Join page texts with newlines; return (full_text, page_at) where page_at
    maps a char position back to the page number it falls on."""
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

    return full_text, page_at


def fixed_size_chunks(pages, size, overlap):
    """Concatenate page texts (newline-joined) and cut fixed-size overlapping
    token windows, mapped back to char offsets so every chunk records exactly
    where its boundaries fall in the text."""
    full_text, page_at = build_full_text(pages)
    tokens, starts = token_starts(full_text)

    chunks = []
    step = size - overlap
    t = 0
    while t < len(tokens):
        t_end = min(t + size, len(tokens))
        start = starts[t]
        end = starts[t_end]
        text = full_text[start:end]
        chunks.append({
            "chunk_index": len(chunks),
            "text": text,
            "start_char": start,
            "end_char": end,
            "num_chars": len(text),
            "num_tokens": t_end - t,
            "start_page": page_at(start),
            "end_page": page_at(max(end - 1, start)),
        })
        if t_end >= len(tokens):
            break
        t += step
    return chunks, len(full_text)


def segment_sentences(full_text, cap):
    """Split full_text into (sentence_text, start_char, num_tokens) triples
    using pysbd in char_span mode, so offsets come straight from the segmenter.
    (pysbd can alter sentence text even with clean=False — e.g. swapping a
    comma and a space — so recovering offsets by substring search drifts and
    can lock onto a duplicate far ahead, producing giant mis-sliced chunks.)

    Any single sentence longer than `cap` tokens (almost always a flattened
    table/chart with no real boundary) is hard-split into cap-sized token
    windows, mapped back to char offsets, so no downstream chunk can be
    oversized.
    """
    import pysbd  # local import: only needed for the sentence method

    seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
    out = []
    prev_end = 0
    for span in seg.segment(full_text):
        # pysbd spans occasionally overlap by a char or two; clamp so
        # offsets stay monotonic and sentences never share text.
        start = max(span.start, prev_end)
        end = max(span.end, start)
        prev_end = end
        sent = full_text[start:end]
        if not sent.strip():
            continue
        tokens, starts = token_starts(sent)
        if len(tokens) <= cap:
            out.append((sent, start, len(tokens)))
        else:
            for t in range(0, len(tokens), cap):
                t_end = min(t + cap, len(tokens))
                piece = sent[starts[t]:starts[t_end]]
                if piece:
                    out.append((piece, start + starts[t], t_end - t))
    return out


def sentence_chunks(pages, n_per_chunk, overlap, min_tokens=None):
    """Group sentences into chunks of n_per_chunk (with sentence overlap).

    A chunk closes early if adding the next sentence would exceed the token cap
    derived from n_per_chunk, so table blobs can't inflate a chunk. If
    min_tokens is set (sentence-dynamic-min), a chunk instead keeps absorbing
    sentences *past* n_per_chunk until it reaches the floor, so short pysbd
    fragments (shredded tables) get packed together rather than left as tiny
    chunks.
    """
    full_text, page_at = build_full_text(pages)
    cap = sentence_cap_tokens(n_per_chunk)
    sentences = segment_sentences(full_text, cap)

    chunks = []
    i = 0
    while i < len(sentences):
        group = []
        toks = 0
        j = i
        while j < len(sentences):
            sent, sstart, ntok = sentences[j]
            if group:
                if toks + ntok > cap:
                    break  # cap always wins: this sentence starts the next chunk
                # Stop once we have enough sentences AND (if a floor is set) enough tokens.
                reached_floor = min_tokens is None or toks >= min_tokens
                if len(group) >= n_per_chunk and reached_floor:
                    break
            group.append((sent, sstart))
            toks += ntok
            j += 1

        start_char = group[0][1]
        last_sent, last_start = group[-1]
        end_char = last_start + len(last_sent)
        text = full_text[start_char:end_char]
        chunks.append({
            "chunk_index": len(chunks),
            "text": text,
            "start_char": start_char,
            "end_char": end_char,
            "num_chars": len(text),
            "num_tokens": count_tokens(text),
            "num_sentences": len(group),
            "start_page": page_at(start_char),
            "end_page": page_at(end_char - 1),
        })
        if j >= len(sentences):
            break
        # Advance by however many sentences this chunk emitted, minus overlap;
        # always make progress (>=1) even if the cap forced a 1-sentence chunk.
        i += max(1, len(group) - overlap)
    return chunks, len(full_text)


def load_plumber_pages(ds):
    """Load full pdfplumber page records (text/tables/images) for plumber-struct.

    Unlike load_and_validate_pages, which only reads the flattened full_text,
    this keeps the structured fields so tables and images can be chunked on
    their own terms."""
    if ds["extractor"] != "pdfplumber":
        sys.exit("ERROR: plumber-struct chunking needs a pdfplumber dataset "
                 "(it uses the structured text/tables/images fields)")
    pages = []
    for n in ds["pages"]:
        jpath = ds["path"] / f"page_{n}.json"
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            sys.exit(f"ERROR: could not read {jpath.relative_to(ROOT)}: {e}")
        if not isinstance(data.get("text"), str):
            sys.exit(f"ERROR: {jpath.relative_to(ROOT)} is missing required string field 'text'")
        pages.append({
            "page": n,
            "text": data["text"],
            "tables": data.get("tables") or [],
            "images": data.get("images") or [],
        })
    log.info(f"Validated {len(pages)} pages: all have string 'text'")
    return pages


def table_row_lines(table):
    """Flatten one pdfplumber table into 'Header: value; ...' lines, one per
    data row. A single-row table has no data rows, so its cells become the line."""
    rows = [[("" if cell is None else str(cell).strip()) for cell in row] for row in table]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []
    if len(rows) == 1:
        return ["; ".join(cell for cell in rows[0] if cell)]
    header = rows[0]
    lines = []
    for row in rows[1:]:
        fields = []
        for i, cell in enumerate(row):
            if not cell:
                continue
            label = header[i] if i < len(header) and header[i] else f"col{i + 1}"
            fields.append(f"{label}: {cell}")
        if fields:
            lines.append("; ".join(fields))
    return lines


def pack_lines(lines, cap):
    """Group lines into newline-joined blocks of at most cap tokens each
    (joining newlines are not counted against the cap)."""
    blocks = []
    group = []
    toks = 0
    for line in lines:
        ntok = count_tokens(line)
        if group and toks + ntok > cap:
            blocks.append("\n".join(group))
            group, toks = [], 0
        group.append(line)
        toks += ntok
    if group:
        blocks.append("\n".join(group))
    return blocks


def plumber_struct_chunks(pages, n_per_chunk):
    """Chunk pdfplumber pages by content kind instead of the flattened full_text.

    Per page, in order: prose (the table-filtered 'text' field) is packed into
    sentence chunks with the dynamic-min floor; each detected table becomes
    row chunks with header-labeled fields (packed up to the token cap); each
    embedded image becomes a small descriptor chunk. Every chunk records its
    source kind so downstream analysis can compare them."""
    cap = sentence_cap_tokens(n_per_chunk)
    floor = sentence_min_tokens(n_per_chunk)
    chunks = []
    total_chars = 0

    def add(text, page, source, extra=None):
        chunks.append({
            "chunk_index": len(chunks),
            "text": text,
            "num_chars": len(text),
            "num_tokens": count_tokens(text),
            "start_page": page,
            "end_page": page,
            "source": source,
            **(extra or {}),
        })

    for p in pages:
        total_chars += len(p["text"])
        prose = p["text"].strip()
        if prose:
            groups, _ = sentence_chunks([(p["page"], prose)], n_per_chunk, 0, min_tokens=floor)
            for g in groups:
                add(g["text"], p["page"], "text",
                    {"num_sentences": g.get("num_sentences")})
        for t_num, table in enumerate(p["tables"], 1):
            lines = table_row_lines(table)
            total_chars += sum(len(line) for line in lines)
            parts = pack_lines(lines, cap)
            for part_num, block in enumerate(parts, 1):
                part = f" (part {part_num}/{len(parts)})" if len(parts) > 1 else ""
                add(f"Table {t_num} on page {p['page']}{part}:\n{block}",
                    p["page"], "table", {"table_number": t_num})
        for img in p["images"]:
            name = img.get("name") or f"image {img.get('image_number', '?')}"
            size = ""
            if img.get("width") and img.get("height"):
                size = f", {round(img['width'])}x{round(img['height'])} px"
            file_note = f" (file {img['file']})" if img.get("file") else ""
            add(f"Image on page {p['page']}: {name}{size}{file_note}",
                p["page"], "image", {"image_file": img.get("file")})
    return chunks, total_chars


def percentile(values, pct):
    """Linear-interpolated percentile of a list (pct in 1..99). Empty -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def embed_sentences(texts):
    """Embed sentence strings with the semantic-chunk embedding model.

    Returns one vector per input text, in order. OpenAI embeddings are
    unit-norm, so a dot product of two vectors is their cosine similarity."""
    from openai_client.openai_client import MyOpenAIClient  # local: only for semantic

    api = MyOpenAIClient(model=SEMANTIC_EMBED_MODEL)
    api.validate_api_key()
    client = api.get_client()

    vectors = []
    batch = 128  # matches embed_chunks.py's embedding batch size
    for start in range(0, len(texts), batch):
        resp = client.embeddings.create(model=SEMANTIC_EMBED_MODEL, input=texts[start:start + batch])
        vectors.extend(item.embedding for item in resp.data)
        log.info(f"  embedded {min(start + batch, len(texts))}/{len(texts)} sentences")
    if len(vectors) != len(texts):
        sys.exit(f"ERROR: got {len(vectors)} embeddings for {len(texts)} sentences")
    return vectors


def cosine_distance(a, b):
    """1 - cosine similarity of two (unit-norm) embedding vectors."""
    return 1.0 - sum(x * y for x, y in zip(a, b))


def semantic_chunks(pages, max_tokens, breakpoint_percentile):
    """Group sentences into chunks by meaning, bounded by max_tokens.

    Each sentence is embedded and the cosine distance between consecutive
    sentence embeddings measures how far the topic shifts across that boundary.
    A chunk is closed when either that distance is a "seam" (>= the given
    percentile of all consecutive distances, so the largest topic shifts end a
    chunk) or adding the next sentence would exceed max_tokens (the hard cap
    always wins). Any single sentence longer than max_tokens was already
    hard-split by segment_sentences, so every chunk stays under the cap.

    Char offsets are carried through so each chunk records where it falls in the
    concatenated page text, exactly like the other chunkers.
    """
    full_text, page_at = build_full_text(pages)
    sentences = segment_sentences(full_text, max_tokens)
    if not sentences:
        return [], len(full_text)

    vectors = embed_sentences([s for s, _, _ in sentences])
    distances = [cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    threshold = percentile(distances, breakpoint_percentile) if distances else None

    chunks = []
    group = []  # (sentence_text, start_char)
    toks = 0

    def flush():
        nonlocal group, toks
        if not group:
            return
        start_char = group[0][1]
        last_sent, last_start = group[-1]
        end_char = last_start + len(last_sent)
        text = full_text[start_char:end_char]
        chunks.append({
            "chunk_index": len(chunks),
            "text": text,
            "start_char": start_char,
            "end_char": end_char,
            "num_chars": len(text),
            "num_tokens": count_tokens(text),
            "num_sentences": len(group),
            "start_page": page_at(start_char),
            "end_page": page_at(end_char - 1),
        })
        group, toks = [], 0

    for i, (sent, sstart, ntok) in enumerate(sentences):
        # Cap wins: close the current chunk before a sentence that would overflow it.
        if group and toks + ntok > max_tokens:
            flush()
        group.append((sent, sstart))
        toks += ntok
        # Close the chunk on a semantic seam right after this sentence.
        if threshold is not None and i < len(distances) and distances[i] >= threshold:
            flush()
    flush()  # trailing sentences
    return chunks, len(full_text)


def main():
    parser = argparse.ArgumentParser(description="Chunk extracted dataset text.")
    parser.add_argument(
        "--type",
        help="Chunk type: fixed_size:<size>:<overlap>, sentence, or semantic. "
             "Size and overlap are tokens (cl100k_base); overlap may be a "
             "percent of size if it ends with %%.",
    )
    parser.add_argument("--dataset", help="Dataset path (data/<extractor>/<date>/<title>) or list number")
    args = parser.parse_args()

    setup_logging("chunk_text")

    type_str = args.type
    if not type_str:
        log.info("\nChunk type:")
        for i, name in enumerate(CHUNK_TYPES, 1):
            log.info(f"  [{i}] {name}")
        choice = input("Choice (number or name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(CHUNK_TYPES):
            type_str = CHUNK_TYPES[int(choice) - 1]
        elif choice in CHUNK_TYPES:
            type_str = choice
        else:
            sys.exit(f"ERROR: '{choice}' is not a valid chunk type choice")
        if type_str == "fixed_size":
            size_in = input("Chunk size (tokens): ").strip()
            overlap_in = input("Overlap (tokens, or percent like 10%): ").strip()
            type_str = f"fixed_size:{size_in}:{overlap_in}"
        elif type_str in SENTENCE_METHODS:
            n_in = input("Sentences per chunk: ").strip()
            ov_in = input("Overlap (sentences, blank for 0): ").strip()
            type_str = f"{type_str}:{n_in}:{ov_in}" if ov_in else f"{type_str}:{n_in}"
        elif type_str == "plumber-struct":
            n_in = input(f"Sentences per text chunk (blank for {PLUMBER_STRUCT_DEFAULT_SENTENCES}): ").strip()
            if n_in:
                type_str = f"plumber-struct:{n_in}"
        elif type_str == "semantic":
            mt_in = input("Max tokens per chunk: ").strip()
            pctl_in = input(f"Breakpoint percentile (blank for {SEMANTIC_DEFAULT_PERCENTILE}): ").strip()
            type_str = f"semantic:{mt_in}:{pctl_in}" if pctl_in else f"semantic:{mt_in}"

    method, size, overlap = parse_type(type_str)

    datasets = scan_datasets()
    ds = choose_dataset(datasets, preselect=args.dataset)

    if method == "plumber-struct":
        plumber_pages = load_plumber_pages(ds)
        chunks, total_chars = plumber_struct_chunks(plumber_pages, size)
        pages = plumber_pages
    elif method in SENTENCE_METHODS:
        pages = load_and_validate_pages(ds)
        min_tokens = sentence_min_tokens(size) if method == "sentence-dynamic-min" else None
        chunks, total_chars = sentence_chunks(pages, size, overlap, min_tokens=min_tokens)
    elif method == "semantic":
        pages = load_and_validate_pages(ds)
        chunks, total_chars = semantic_chunks(pages, size, overlap)
    else:
        pages = load_and_validate_pages(ds)
        chunks, total_chars = fixed_size_chunks(pages, size, overlap)

    now = datetime.now()
    out_dir = ds["path"] / f"{now.strftime('%Y%m%d_%H%M%S')}_chunk_{method}_{size}_{overlap}"
    out_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "datetime": now.isoformat(timespec="seconds"),
        "dataset": ds["rel"],
        "extractor": ds["extractor"],
        "chunk_method": method,
        "num_pages": len(pages),
        "total_chars": total_chars,
        "num_chunks": len(chunks),
    }
    metadata["token_encoding"] = TOKEN_ENCODING
    if method == "plumber-struct":
        metadata["sentences_per_text_chunk"] = size
        metadata["token_cap"] = sentence_cap_tokens(size)
        metadata["token_floor"] = sentence_min_tokens(size)
        by_source = {}
        for c in chunks:
            by_source[c["source"]] = by_source.get(c["source"], 0) + 1
        metadata["chunks_by_source"] = by_source
    elif method in SENTENCE_METHODS:
        metadata["sentences_per_chunk"] = size
        metadata["overlap_sentences"] = overlap
        metadata["token_cap"] = sentence_cap_tokens(size)
        if method == "sentence-dynamic-min":
            metadata["token_floor"] = sentence_min_tokens(size)
    elif method == "semantic":
        metadata["max_tokens"] = size
        metadata["breakpoint_percentile"] = overlap
        metadata["embedding_model"] = SEMANTIC_EMBED_MODEL
    else:
        metadata["chunk_size_tokens"] = size
        metadata["overlap_tokens"] = overlap

    result = {"metadata": metadata, "chunks": chunks}

    out_file = out_dir / "chunked_text.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Wrote {len(chunks)} chunks ({total_chars} chars from {len(pages)} pages)")
    log.info(f"Output: {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
