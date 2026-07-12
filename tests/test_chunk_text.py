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
    shutil.copytree(FIXTURES, tmp_path / "data")
    monkeypatch.setattr(chunk_text, "ROOT", tmp_path)
    monkeypatch.setattr(chunk_text, "DATA_DIR", tmp_path / "data")
    return tmp_path


# ---------------------------------------------------------------- parse_type

def test_parse_char_overlap():
    assert chunk_text.parse_type("fixed_size:100:20") == ("fixed_size", 100, 20)


def test_parse_percent_overlap():
    assert chunk_text.parse_type("fixed_size:200:10%") == ("fixed_size", 200, 20)


def test_parse_zero_overlap_ok():
    assert chunk_text.parse_type("fixed_size:100:0") == ("fixed_size", 100, 0)


def test_parse_overlap_just_below_size_ok(capsys):
    assert chunk_text.parse_type("fixed_size:100:99") == ("fixed_size", 100, 99)
    assert "WARNING" in capsys.readouterr().out


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
        "sentence:5",             # non-fixed_size type takes no params
    ],
)
def test_parse_type_rejects_bad_specs(spec):
    with pytest.raises(SystemExit):
        chunk_text.parse_type(spec)


def test_parse_warns_over_half_overlap(capsys):
    chunk_text.parse_type("fixed_size:100:60")
    assert "WARNING" in capsys.readouterr().out


def test_parse_no_warning_at_exactly_half(capsys):
    chunk_text.parse_type("fixed_size:100:50")
    assert "WARNING" not in capsys.readouterr().out


def test_parse_other_types_take_no_params():
    assert chunk_text.parse_type("sentence") == ("sentence", None, None)
    assert chunk_text.parse_type("semantic") == ("semantic", None, None)


# --------------------------------------------------------- fixed_size_chunks

def test_chunk_stride_and_last_chunk():
    pages = [(1, "a" * 100)]
    chunks, total = chunk_text.fixed_size_chunks(pages, size=40, overlap=10)
    assert total == 100
    # Stops at the chunk that reaches the end of text: a chunk starting at 90
    # would add no new characters (it lies entirely inside chunk 60..100).
    assert [c["start_char"] for c in chunks] == [0, 30, 60]
    assert [c["num_chars"] for c in chunks] == [40, 40, 40]
    assert chunks[-1]["end_char"] == 100
    assert all(c["end_char"] - c["start_char"] == c["num_chars"] for c in chunks)
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2]


def test_chunk_short_final_chunk():
    pages = [(1, "a" * 105)]
    chunks, _ = chunk_text.fixed_size_chunks(pages, size=40, overlap=10)
    assert [c["start_char"] for c in chunks] == [0, 30, 60, 90]
    assert chunks[-1]["num_chars"] == 15


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
    run_main(monkeypatch, "--type", "fixed_size:50:10%", "--dataset", "data/pdfplumber/20260101_01/good")
    out_dirs = list((data_root / "data/pdfplumber/20260101_01/good").glob("*_chunk_fixed_size_50_5"))
    assert len(out_dirs) == 1
    result = json.loads((out_dirs[0] / "chunked_text.json").read_text(encoding="utf-8"))
    meta = result["metadata"]
    assert meta["dataset"] == "data/pdfplumber/20260101_01/good"
    assert meta["chunk_method"] == "fixed_size"
    assert meta["chunk_size"] == 50
    assert meta["overlap"] == 5
    assert meta["num_pages"] == 2
    assert meta["num_chunks"] == len(result["chunks"])
    starts = [c["start_char"] for c in result["chunks"]]
    assert starts[:3] == [0, 45, 90]
    assert result["chunks"][-1]["end_char"] == meta["total_chars"]


def test_main_rejects_unimplemented_type(data_root, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, "--type", "sentence")
    assert "not implemented" in str(exc.value)


def test_main_bad_overlap_exits_before_dataset_prompt(data_root, monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--type", "fixed_size:100:120")
