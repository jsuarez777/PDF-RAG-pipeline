"""CLI boundary tests for app/chunk_text.py.

Static fixture datasets live in tests/datasets/chunk_text/ (mirroring the
data/{pdf2image,pdfplumber}/<date>/<title> layout). File-based tests copy them
to tmp_path so output folders never pollute the repo.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

from app import chunk_text

FIXTURES = Path(__file__).parent / "datasets" / "chunk_text"


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Copy fixture datasets to tmp_path and point the script at them."""
    import logging_utils
    shutil.copytree(FIXTURES, tmp_path / "data")
    monkeypatch.setattr(chunk_text, "ROOT", tmp_path)
    monkeypatch.setattr(chunk_text, "DATA_DIR", tmp_path / "data")
    # keep setup_logging() calls in main() from writing into the repo's logs/
    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path / "logs")
    return tmp_path


# ---------------------------------------------------------------- parse_type

def test_parse_char_overlap():
    assert chunk_text.parse_type("fixed_size:100:20") == ("fixed_size", 100, 20)


def test_parse_percent_overlap():
    assert chunk_text.parse_type("fixed_size:200:10%") == ("fixed_size", 200, 20)


def test_parse_zero_overlap_ok():
    assert chunk_text.parse_type("fixed_size:100:0") == ("fixed_size", 100, 0)


def test_parse_overlap_just_below_size_ok(caplog):
    assert chunk_text.parse_type("fixed_size:100:99") == ("fixed_size", 100, 99)
    assert "WARNING" in caplog.text


@pytest.mark.parametrize(
    "spec",
    [
        "fixed_size:100:120",     # overlap > size
        "fixed_size:100:100",     # overlap == size
        "fixed_size:100:120%",    # percent overlap > size
        "fixed_size:100:100%",    # percent overlap == size
        "fixed_size:100:-5",      # negative overlap
        "fixed_size:100:-10%",    # negative percent
        "fixed_size:0:0",         # zero size
        "fixed_size:-100:20",     # negative size
        "fixed_size:abc:20",      # non-integer size
        "fixed_size:100:xyz",     # non-integer overlap
        "fixed_size:100:x%",      # non-numeric percent
        "fixed_size:100",         # missing overlap
        "fixed_size",             # missing size and overlap
        "fixed_size:100:20:5",    # too many parts
        "bogus_type",             # unknown type
        "sentence",               # sentence types need a sentences-per-chunk count
        "sentence:0",             # zero sentences per chunk
        "sentence:5:5",           # sentence overlap must be below the chunk size
        "sentence:5:1:2",         # too many parts
        "sentence-dynamic-min",   # same rules as sentence
        "semantic:5",             # semantic takes no params
    ],
)
def test_parse_type_rejects_bad_specs(spec):
    with pytest.raises(SystemExit):
        chunk_text.parse_type(spec)


def test_parse_warns_over_half_overlap(caplog):
    chunk_text.parse_type("fixed_size:100:60")
    assert "WARNING" in caplog.text


def test_parse_no_warning_at_exactly_half(caplog):
    chunk_text.parse_type("fixed_size:100:50")
    assert "WARNING" not in caplog.text


def test_parse_semantic_takes_no_params():
    assert chunk_text.parse_type("semantic") == ("semantic", None, None)


def test_parse_sentence_specs():
    assert chunk_text.parse_type("sentence:5") == ("sentence", 5, 0)
    assert chunk_text.parse_type("sentence:5:1") == ("sentence", 5, 1)
    assert chunk_text.parse_type("sentence-dynamic-min:3:2") == ("sentence-dynamic-min", 3, 2)


# ----------------------------------------------------------- token_starts

def test_token_starts_tiles_the_text():
    text = "hello world, this is a test"
    tokens, starts = chunk_text.token_starts(text)
    assert starts[0] == 0
    assert starts[-1] == len(text)
    assert len(starts) == len(tokens) + 1
    assert all(a <= b for a, b in zip(starts, starts[1:]))
    # slicing at consecutive starts reassembles the text exactly
    assert "".join(text[starts[i]:starts[i + 1]] for i in range(len(tokens))) == text


def test_token_starts_multibyte_text():
    text = "naïve café — π ≈ 3.14159, 数据处理 test"
    tokens, starts = chunk_text.token_starts(text)
    assert starts[0] == 0 and starts[-1] == len(text)
    assert all(a <= b for a, b in zip(starts, starts[1:]))
    assert "".join(text[starts[i]:starts[i + 1]] for i in range(len(tokens))) == text


# --------------------------------------------------------- fixed_size_chunks

def test_chunk_stride_and_last_chunk():
    text = "The quick brown fox jumps over the lazy dog. " * 10
    tokens, starts = chunk_text.token_starts(text)
    assert len(tokens) > 50  # sanity: enough tokens for several windows
    size, overlap = 40, 10
    chunks, total = chunk_text.fixed_size_chunks([(1, text)], size=size, overlap=overlap)
    assert total == len(text)
    step = size - overlap
    for k, c in enumerate(chunks):
        t0, t1 = k * step, min(k * step + size, len(tokens))
        assert c["start_char"] == starts[t0]
        assert c["end_char"] == starts[t1]
        assert c["num_tokens"] == t1 - t0
        assert c["text"] == text[c["start_char"]:c["end_char"]]
        assert c["num_chars"] == c["end_char"] - c["start_char"]
    # stops at the chunk that reaches the end of the token stream
    assert (len(chunks) - 1) * step + size >= len(tokens)
    assert chunks[-1]["end_char"] == len(text)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_short_final_chunk():
    text = "one small page of text with a handful of tokens in it"
    tokens, _ = chunk_text.token_starts(text)
    chunks, _ = chunk_text.fixed_size_chunks([(1, text)], size=len(tokens) - 2, overlap=0)
    assert [c["num_tokens"] for c in chunks] == [len(tokens) - 2, 2]
    assert chunks[-1]["end_char"] == len(text)


def test_chunk_spans_page_boundary():
    pages = [(1, "a" * 50), (2, "b" * 50)]
    chunks, total = chunk_text.fixed_size_chunks(pages, size=60, overlap=0)
    assert total == 101  # 50 + newline separator + 50
    assert chunks[0]["start_page"] == 1
    assert chunks[0]["end_page"] == 2
    assert chunks[-1]["end_page"] == 2


def test_chunk_no_text_lost():
    pages = [(1, "one two three four five"), (2, "six seven eight nine ten")]
    chunks, _ = chunk_text.fixed_size_chunks(pages, size=12, overlap=4)
    combined = " ".join(c["text"] for c in chunks)
    for word in "one two three four five six seven eight nine ten".split():
        assert word in combined


def test_chunk_empty_middle_page():
    pages = [(1, "aaaa"), (2, ""), (3, "bbbb")]
    chunks, total = chunk_text.fixed_size_chunks(pages, size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0]["start_page"] == 1
    assert chunks[0]["end_page"] == 3
    assert "aaaa" in chunks[0]["text"] and "bbbb" in chunks[0]["text"]


# ------------------------------------------------ dataset scan/choose/validate

def test_scan_finds_all_datasets(data_root):
    datasets = chunk_text.scan_datasets()
    rels = {ds["rel"] for ds in datasets}
    assert rels == {
        "data/pdf2image/20260101_01/good",
        "data/pdf2image/20260101_02/incomplete",
        "data/pdfplumber/20260101_01/good",
        "data/pdfplumber/20260101_02/badfield",
    }


def test_scan_flags_missing_json(data_root):
    datasets = {ds["rel"]: ds for ds in chunk_text.scan_datasets()}
    assert datasets["data/pdf2image/20260101_02/incomplete"]["missing_json"] == [2]
    assert datasets["data/pdf2image/20260101_01/good"]["missing_json"] == []


def test_choose_valid_dataset_by_number(data_root):
    datasets = chunk_text.scan_datasets()
    idx = next(i for i, ds in enumerate(datasets, 1) if ds["rel"].endswith("pdf2image/20260101_01/good"))
    ds = chunk_text.choose_dataset(datasets, preselect=str(idx))
    assert ds["rel"] == "data/pdf2image/20260101_01/good"


def test_choose_valid_dataset_by_path(data_root):
    datasets = chunk_text.scan_datasets()
    ds = chunk_text.choose_dataset(datasets, preselect="data/pdfplumber/20260101_01/good")
    assert ds["extractor"] == "pdfplumber"


def test_choose_incomplete_dataset_exits_minus_one(data_root, capsys):
    datasets = chunk_text.scan_datasets()
    with pytest.raises(SystemExit) as exc:
        chunk_text.choose_dataset(datasets, preselect="data/pdf2image/20260101_02/incomplete")
    assert exc.value.code == -1
    assert "missing their json file" in capsys.readouterr().err


def test_choose_invalid_input_exits(data_root):
    datasets = chunk_text.scan_datasets()
    for bad in ("0", str(len(datasets) + 1), "not/a/dataset", ""):
        with pytest.raises(SystemExit):
            chunk_text.choose_dataset(datasets, preselect=bad)


def test_validate_accepts_empty_text_field(data_root):
    datasets = chunk_text.scan_datasets()
    ds = chunk_text.choose_dataset(datasets, preselect="data/pdf2image/20260101_01/good")
    pages = chunk_text.load_and_validate_pages(ds)
    assert [n for n, _ in pages] == [1, 2, 3]
    assert pages[1][1] == ""  # blank page allowed


def test_validate_rejects_wrong_field_type(data_root):
    datasets = chunk_text.scan_datasets()
    ds = chunk_text.choose_dataset(datasets, preselect="data/pdfplumber/20260101_02/badfield")
    with pytest.raises(SystemExit) as exc:
        chunk_text.load_and_validate_pages(ds)
    assert "full_text" in str(exc.value)


def test_validate_rejects_missing_field(data_root):
    bad = data_root / "data/pdf2image/20260101_01/good/page_2_easyocr.json"
    bad.write_text(json.dumps({"page": 2}), encoding="utf-8")
    datasets = chunk_text.scan_datasets()
    ds = chunk_text.choose_dataset(datasets, preselect="data/pdf2image/20260101_01/good")
    with pytest.raises(SystemExit) as exc:
        chunk_text.load_and_validate_pages(ds)
    assert "extracted_text" in str(exc.value)


# ------------------------------------------------------------------ main / e2e

def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["chunk_text.py", *argv])
    chunk_text.main()


def test_main_end_to_end(data_root, monkeypatch):
    run_main(monkeypatch, "--type", "fixed_size:10:10%", "--dataset", "data/pdfplumber/20260101_01/good")
    out_dirs = list((data_root / "data/pdfplumber/20260101_01/good").glob("*_chunk_fixed_size_10_1"))
    assert len(out_dirs) == 1
    result = json.loads((out_dirs[0] / "chunked_text.json").read_text(encoding="utf-8"))
    meta = result["metadata"]
    assert meta["dataset"] == "data/pdfplumber/20260101_01/good"
    assert meta["chunk_method"] == "fixed_size"
    assert meta["chunk_size_tokens"] == 10
    assert meta["overlap_tokens"] == 1
    assert meta["token_encoding"] == "cl100k_base"
    assert meta["num_pages"] == 2
    assert meta["num_chunks"] == len(result["chunks"])
    chunks = result["chunks"]
    assert len(chunks) > 1
    assert all(c["num_tokens"] == 10 for c in chunks[:-1])
    assert 0 < chunks[-1]["num_tokens"] <= 10
    starts = [c["start_char"] for c in chunks]
    assert starts[0] == 0
    assert all(a < b for a, b in zip(starts, starts[1:]))
    assert chunks[-1]["end_char"] == meta["total_chars"]


def test_main_rejects_unimplemented_type(data_root, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, "--type", "semantic")
    assert "not implemented" in str(exc.value)


def test_main_bad_overlap_exits_before_dataset_prompt(data_root, monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--type", "fixed_size:100:120")


# ------------------------------------------------------ plumber-struct chunks

def test_parse_plumber_struct_specs():
    assert chunk_text.parse_type("plumber-struct") == (
        "plumber-struct", chunk_text.PLUMBER_STRUCT_DEFAULT_SENTENCES, 0)
    assert chunk_text.parse_type("plumber-struct:3") == ("plumber-struct", 3, 0)
    with pytest.raises(SystemExit):
        chunk_text.parse_type("plumber-struct:3:1")
    with pytest.raises(SystemExit):
        chunk_text.parse_type("plumber-struct:0")


def test_table_row_lines_labels_cells_with_headers():
    table = [["Year", "Rate", None],
             ["2012", "45%", ""],
             [None, "38%", "note"]]
    lines = chunk_text.table_row_lines(table)
    assert lines == ["Year: 2012; Rate: 45%", "Rate: 38%; col3: note"]


def test_table_row_lines_single_row_and_empty():
    assert chunk_text.table_row_lines([["a", None, "b"]]) == ["a; b"]
    assert chunk_text.table_row_lines([]) == []
    assert chunk_text.table_row_lines([[None, ""], ["", None]]) == []


def test_pack_lines_respects_cap():
    lines = ["the quick brown fox", "jumps over the lazy dog", "and runs far away"]
    ntok = [chunk_text.count_tokens(line) for line in lines]
    # a cap (in tokens) that fits the first two lines exactly but not the third
    blocks = chunk_text.pack_lines(lines, cap=ntok[0] + ntok[1])
    assert blocks == [lines[0] + "\n" + lines[1], lines[2]]
    # a single oversized line still becomes its own block
    assert chunk_text.pack_lines([lines[0]], cap=1) == [lines[0]]


def test_plumber_struct_chunks_sources():
    pages = [
        {"page": 1,
         "text": "One sentence here. Another sentence follows. A third one ends it.",
         "tables": [[["H1", "H2"], ["a", "b"], ["c", "d"]]],
         "images": [{"file": "page_1_image_1.png", "image_number": 1,
                     "name": "Im0", "width": 100.0, "height": 50.0}]},
        {"page": 2, "text": "", "tables": [], "images": []},
    ]
    chunks, total = chunk_text.plumber_struct_chunks(pages, n_per_chunk=5)
    by_source = {}
    for c in chunks:
        by_source.setdefault(c["source"], []).append(c)
    assert set(by_source) == {"text", "table", "image"}
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    (table_chunk,) = by_source["table"]
    assert "H1: a; H2: b" in table_chunk["text"]
    assert table_chunk["text"].startswith("Table 1 on page 1")
    (image_chunk,) = by_source["image"]
    assert "Im0" in image_chunk["text"] and "page_1_image_1.png" in image_chunk["text"]
    assert all(c["start_page"] == 1 for c in chunks)  # page 2 had no content
