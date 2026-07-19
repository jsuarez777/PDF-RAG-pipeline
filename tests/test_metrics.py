"""Tests for the IR metrics in app/eval_retrieval.py, checked against
hand-computed values.

`gold_rank` finds the 1-based rank of the gold chunk in a ranked result list.
`aggregate` turns a list of per-question gold ranks (None = miss) into
Recall@K, Precision@K, MRR, MAP, and NDCG@K over the whole question set.
"""

import math

from app.eval_retrieval import aggregate, gold_rank


def result(rank, chunk_index):
    return {"rank": rank, "chunk": {"chunk_index": chunk_index}}


def record(gold_rank_value, retrieval_seconds=0.0):
    return {"gold_rank": gold_rank_value, "retrieval_seconds": retrieval_seconds}


# --------------------------------------------------------------- gold_rank

def test_gold_rank_returns_one_based_rank_of_match():
    results = [result(1, 5), result(2, 2), result(3, 9)]
    assert gold_rank(results, gold=2) == 2


def test_gold_rank_returns_none_when_not_retrieved():
    results = [result(1, 5), result(2, 2), result(3, 9)]
    assert gold_rank(results, gold=99) is None


# ----------------------------------------------------------------- aggregate
#
# 5 queries, gold ranks [1, 2, 3, None, 5] (4 hits, 1 miss), evaluated at
# k = 1, 3, 5.

RANKS = [1, 2, 3, None, 5]


def make_records():
    return [record(r) for r in RANKS]


def test_aggregate_mrr_and_map():
    metrics = aggregate(make_records(), ks=[1, 3, 5])
    expected = (1 / 1 + 1 / 2 + 1 / 3 + 1 / 5) / 5
    assert metrics["mrr"] == expected
    # Single gold chunk per query -> MAP collapses to MRR (AP of one relevant
    # doc at rank r is exactly 1/r).
    assert metrics["map"] == expected


def test_aggregate_recall_at_k():
    metrics = aggregate(make_records(), ks=[1, 3, 5])
    assert metrics["recall@1"] == 1 / 5   # only rank-1 hit qualifies
    assert metrics["recall@3"] == 3 / 5   # ranks 1, 2, 3
    assert metrics["recall@5"] == 4 / 5   # ranks 1, 2, 3, 5 (miss excluded)


def test_aggregate_precision_at_k():
    metrics = aggregate(make_records(), ks=[1, 3, 5])
    assert metrics["precision@1"] == 1 / (5 * 1)
    assert metrics["precision@3"] == 3 / (5 * 3)
    assert metrics["precision@5"] == 4 / (5 * 5)


def test_aggregate_ndcg_at_k():
    metrics = aggregate(make_records(), ks=[1, 3, 5])
    ndcg_1 = (1 / math.log2(2)) / 5
    ndcg_3 = (1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)) / 5
    ndcg_5 = (1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
              + 1 / math.log2(6)) / 5
    assert metrics["ndcg@1"] == ndcg_1
    assert metrics["ndcg@3"] == ndcg_3
    assert metrics["ndcg@5"] == ndcg_5


def test_aggregate_avg_retrieval_time():
    records = [record(1, retrieval_seconds=0.1), record(2, retrieval_seconds=0.3)]
    metrics = aggregate(records, ks=[1])
    assert metrics["avg_retrieval_time"] == (0.1 + 0.3) / 2


def test_aggregate_perfect_retrieval_hits_metric_ceilings():
    records = [record(1), record(1), record(1)]
    metrics = aggregate(records, ks=[1])
    assert metrics["mrr"] == 1.0
    assert metrics["map"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert metrics["precision@1"] == 1.0
    assert metrics["ndcg@1"] == 1.0


def test_aggregate_all_misses_yields_zero_metrics_without_dividing_by_zero():
    records = [record(None), record(None)]
    metrics = aggregate(records, ks=[1, 5])
    assert metrics["mrr"] == 0.0
    assert metrics["map"] == 0.0
    assert metrics["recall@1"] == 0.0
    assert metrics["precision@5"] == 0.0
    assert metrics["ndcg@5"] == 0.0
