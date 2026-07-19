"""Tests for the grid configuration logic in app/run_pipeline.py."""

import pytest

from app import run_pipeline
from app.run_pipeline import (
    chunk_label,
    enumerate_experiments,
    parse_chunk_specs,
    parse_csv,
)

# ------------------------------------------------------------------ parse_csv

def test_parse_csv_maps_aliases_and_dedupes():
    models = parse_csv("small,large,small", run_pipeline.EMBEDDING_MODELS,
                       "embedding model", aliases=run_pipeline.EMBEDDING_ALIASES)
    assert models == ["text-embedding-3-small", "text-embedding-3-large"]


def test_parse_csv_rejects_unknown():
    with pytest.raises(SystemExit):
        parse_csv("bm25,bogus", run_pipeline.RETRIEVALS, "retrieval method")


def test_parse_csv_rejects_empty():
    with pytest.raises(SystemExit):
        parse_csv(" , ", run_pipeline.TOKENIZERS, "tokenizer")


# ----------------------------------------------------------- parse_chunk_specs

def test_parse_chunk_specs_valid():
    specs = parse_chunk_specs("fixed_size:256:50, sentence:5:1,sentence-dynamic-min:3")
    assert specs == ["fixed_size:256:50", "sentence:5:1", "sentence-dynamic-min:3"]


def test_parse_chunk_specs_dedupes():
    assert parse_chunk_specs("sentence:5,sentence:5") == ["sentence:5"]


def test_parse_chunk_specs_rejects_bad_spec():
    with pytest.raises(SystemExit):
        parse_chunk_specs("fixed_size:100")  # missing overlap


def test_chunk_label():
    assert chunk_label("fixed_size:256:50") == "fixed_size_256_50"
    assert chunk_label("fixed_size:200:10%") == "fixed_size_200_10pct"


# ------------------------------------------------------- enumerate_experiments

def test_grid_counts_per_retrieval_method():
    exps = enumerate_experiments(
        chunk_specs=["fixed_size:256:50", "sentence:5:1"],
        embed_dbs={"text-embedding-3-small": ["milvus"], "text-embedding-3-large": ["milvus"]},
        tokenizers=["word"],
        retrievals=["bm25", "vector", "hybrid"],
        alphas=[0.7],
    )
    # per chunk config: 1 bm25 (tokenizers) + 2 vector (embeddings) + 2 hybrid (emb x tok)
    assert len(exps) == 2 * (1 + 2 + 2)
    by_retrieval = {}
    for e in exps:
        by_retrieval.setdefault(e["retrieval"], []).append(e)
    assert len(by_retrieval["bm25"]) == 2
    assert len(by_retrieval["vector"]) == 4
    assert len(by_retrieval["hybrid"]) == 4


def test_grid_experiment_ids_are_unique():
    exps = enumerate_experiments(
        ["fixed_size:256:50", "fixed_size:512:100"],
        {"text-embedding-3-small": ["milvus"]}, ["simple", "word"],
        ["bm25", "vector", "hybrid"], [0.5])
    ids = [e["experiment_id"] for e in exps]
    assert len(ids) == len(set(ids))


def test_grid_single_method_only():
    exps = enumerate_experiments(["sentence:5"], {"text-embedding-3-small": ["milvus"]},
                                 ["word"], ["vector"], [0.7])
    assert len(exps) == 1
    exp = exps[0]
    assert exp["retrieval"] == "vector"
    assert exp["embedding_model"] == "text-embedding-3-small"
    assert exp["db"] == "milvus"
    assert exp["chunk_type"] == "sentence:5"


def test_grid_hybrid_records_alpha_and_both_sides():
    (exp,) = enumerate_experiments(["sentence:5"], {"text-embedding-3-small": ["milvus"]},
                                   ["porter"], ["hybrid"], [0.3])
    assert exp["alpha"] == 0.3
    assert exp["embedding_model"] == "text-embedding-3-small"
    assert exp["db"] == "milvus"
    assert exp["tokenizer"] == "porter"


def test_match_experiment_maps_eval_metadata_back_to_grid():
    exps = enumerate_experiments(["sentence:5"], {"text-embedding-3-small": ["milvus"]},
                                 ["word"], ["bm25", "vector", "hybrid"], [0.7])
    cases = [
        ({"db_type": "bm25", "tokenizer": "word"}, "bm25"),
        ({"db_type": "milvus", "embedding_model": "text-embedding-3-small"}, "vector"),
        # Hybrid eval metadata's db_type is the literal "hybrid"; the underlying
        # vector db name is recovered from the "vector_db" rel path instead.
        ({"db_type": "hybrid", "embedding_model": "text-embedding-3-small",
          "tokenizer": "word", "alpha": 0.7,
          "vector_db": "data/pdfplumber/20260716_01/doc/embedding_databases/"
                       "milvus_20260716_text-embedding-3-small.db"}, "hybrid"),
    ]
    for meta, expected in cases:
        exp = run_pipeline.match_experiment(exps, "sentence:5", meta)
        assert exp is not None and exp["retrieval"] == expected


def test_match_experiment_returns_none_for_unknown():
    exps = enumerate_experiments(["sentence:5"], {"text-embedding-3-small": ["milvus"]},
                                 ["word"], ["vector"], [0.7])
    assert run_pipeline.match_experiment(
        exps, "sentence:5", {"db_type": "bm25", "tokenizer": "word"}) is None
