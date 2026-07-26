"""Tests for the summary/pipeline backfill logic in app/rerank_eval.py."""

import json

from app.rerank_eval import update_eval_summaries


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def make_files(tmp_path):
    """A base eval, its cohere twin, and a summary listing the base row."""
    base = tmp_path / "eval_20260101_000000_bm25_x_word.json"
    twin = tmp_path / "eval_20260101_000000_bm25_x_word_rerank_cohere.json"
    summary = tmp_path / "eval_summary_20260101_000000.json"
    write_json(base, {"metadata": {}, "aggregates": {}, "questions": []})
    write_json(twin, {
        "metadata": {"rerank": "cohere"},
        "aggregates": {"overall": {"mrr": 0.9}, "by_question_type": {"direct": {"mrr": 0.95}}},
        "questions": [],
    })
    write_json(summary, {
        "metadata": {"qa_file": "qa.json"},
        "indexes": [
            {"db": "bm25/x.pkl", "db_type": "bm25", "tokenizer": "word",
             "eval_file": base.name, "overall": {"mrr": 0.7}, "by_question_type": {}},
            {"db": "milvus.db", "db_type": "milvus",
             "eval_file": "eval_20260101_000000_milvus_y.json",
             "overall": {"mrr": 0.8}, "by_question_type": {}},
        ],
    })
    return base, twin, summary


def test_twin_row_inserted_after_base(tmp_path):
    base, twin, summary = make_files(tmp_path)
    update_eval_summaries(base, twin, "cohere", "rerank-v3.5")

    rows = json.loads(summary.read_text())["indexes"]
    assert [r["eval_file"] for r in rows] == [base.name, twin.name,
                                              "eval_20260101_000000_milvus_y.json"]
    twin_row = rows[1]
    # carries the base row's index identity plus the twin's rerank fields/metrics
    assert twin_row["db_type"] == "bm25" and twin_row["tokenizer"] == "word"
    assert twin_row["rerank"] == "cohere" and twin_row["rerank_model"] == "rerank-v3.5"
    assert twin_row["overall"] == {"mrr": 0.9}
    assert twin_row["by_question_type"] == {"direct": {"mrr": 0.95}}


def test_rerun_does_not_duplicate(tmp_path):
    base, twin, summary = make_files(tmp_path)
    update_eval_summaries(base, twin, "cohere", "rerank-v3.5")
    update_eval_summaries(base, twin, "cohere", "rerank-v3.5")
    rows = json.loads(summary.read_text())["indexes"]
    assert len(rows) == 3


def test_summary_without_base_row_untouched(tmp_path):
    base, twin, summary = make_files(tmp_path)
    other = tmp_path / "eval_summary_20260101_111111.json"
    write_json(other, {"metadata": {}, "indexes": [
        {"db": "z", "db_type": "bm25", "eval_file": "eval_other.json",
         "overall": {}, "by_question_type": {}}]})
    update_eval_summaries(base, twin, "cohere", "rerank-v3.5")
    assert len(json.loads(other.read_text())["indexes"]) == 1
