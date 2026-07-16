"""Tests for the hybrid retrieval score fusion in app/retriever_topk.py."""

import pytest

from app.retriever_topk import combine_hybrid, minmax_normalize


def result(chunk_index, similarity, text=None):
    chunk = {"chunk_index": chunk_index}
    if text is not None:
        chunk["text"] = text
    return {"rank": 0, "score": similarity, "similarity": similarity, "chunk": chunk}


# ----------------------------------------------------------- minmax_normalize

def test_minmax_maps_to_unit_range():
    norm = minmax_normalize([result(0, 2.0), result(1, 18.0), result(2, 10.0)])
    assert norm[0] == 0.0
    assert norm[1] == 1.0
    assert norm[2] == pytest.approx(0.5)


def test_minmax_all_equal_scores_normalize_to_one():
    norm = minmax_normalize([result(0, 7.0), result(1, 7.0)])
    assert norm == {0: 1.0, 1: 1.0}


def test_minmax_empty():
    assert minmax_normalize([]) == {}


def test_minmax_handles_negative_similarities():
    norm = minmax_normalize([result(0, -0.5), result(1, 0.5)])
    assert norm == {0: 0.0, 1: 1.0}


# -------------------------------------------------------------- combine_hybrid

def test_combine_weights_by_alpha():
    vec = [result(0, 0.9), result(1, 0.1)]
    bm = [result(1, 20.0), result(0, 2.0)]
    # alpha=0.7: chunk 0 = 0.7*1 + 0.3*0 = 0.7; chunk 1 = 0.7*0 + 0.3*1 = 0.3
    results = combine_hybrid(vec, bm, alpha=0.7, k=10)
    assert [(r["chunk"]["chunk_index"], r["score"]) for r in results] == [
        (0, pytest.approx(0.7)), (1, pytest.approx(0.3))]
    assert [r["rank"] for r in results] == [1, 2]


def test_combine_alpha_one_is_pure_vector():
    vec = [result(0, 0.9), result(1, 0.5), result(2, 0.1)]
    bm = [result(2, 30.0), result(1, 3.0)]
    results = combine_hybrid(vec, bm, alpha=1.0, k=10)
    assert [r["chunk"]["chunk_index"] for r in results] == [0, 1, 2]


def test_combine_alpha_zero_is_pure_bm25():
    vec = [result(0, 0.9), result(1, 0.5)]
    bm = [result(1, 30.0), result(0, 3.0)]
    results = combine_hybrid(vec, bm, alpha=0.0, k=10)
    assert [r["chunk"]["chunk_index"] for r in results] == [1, 0]


def test_combine_chunk_missing_from_one_side_scores_zero_there():
    vec = [result(0, 0.8), result(1, 0.2)]
    bm = [result(2, 5.0)]  # all-equal -> normalizes to 1.0
    results = combine_hybrid(vec, bm, alpha=0.5, k=10)
    scores = {r["chunk"]["chunk_index"]: r["score"] for r in results}
    assert scores[0] == pytest.approx(0.5)   # vector-only hit
    assert scores[2] == pytest.approx(0.5)   # bm25-only hit
    assert scores[1] == pytest.approx(0.0)


def test_combine_truncates_to_k_and_ranks_sequentially():
    vec = [result(i, 1.0 - i / 10) for i in range(8)]
    results = combine_hybrid(vec, [], alpha=0.5, k=3)
    assert len(results) == 3
    assert [r["rank"] for r in results] == [1, 2, 3]
    assert [r["chunk"]["chunk_index"] for r in results] == [0, 1, 2]


def test_combine_both_empty():
    assert combine_hybrid([], [], alpha=0.7, k=5) == []


def test_combine_prefers_chunk_copy_with_text():
    vec = [result(0, 0.9)]  # milvus/chroma results carry text
    bm = [result(0, 5.0, text="the chunk text")]
    results = combine_hybrid(vec, bm, alpha=0.7, k=5)
    assert results[0]["chunk"]["text"] == "the chunk text"


def test_combine_scores_stay_in_unit_range():
    vec = [result(i, s) for i, s in enumerate([0.99, 0.5, -0.2])]
    bm = [result(i, s) for i, s in enumerate([55.0, 3.0, 0.4])]
    for r in combine_hybrid(vec, bm, alpha=0.7, k=10):
        assert 0.0 <= r["score"] <= 1.0
