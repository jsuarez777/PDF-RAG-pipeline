#!/usr/bin/env python3
"""Cohere reranking of retrieved chunks, shared by retriever_topk.py,
eval_retrieval.py and (through it) run_pipeline.py.

A reranker is a cross-encoder that scores each (query, document) pair jointly,
which is more accurate than the bi-encoder similarity used for first-stage
retrieval but too slow to run over a whole corpus. So it is applied as a
second stage: retrieve top-k candidates with bm25/vector/hybrid, then let the
reranker reorder those k chunks.

Reads COHERE_API_KEY from the environment (falling back to ~/.profile, like
the OpenAI key). The result schema is unchanged: rank/score/similarity/chunk,
with score and similarity replaced by Cohere's relevance score (0..1) and the
pre-rerank position kept in pre_rerank_rank.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai_client.openai_client import load_env_var_from_profile  # noqa: E402

RERANK_PROVIDERS = ("cohere",)
DEFAULT_RERANK_MODEL = "rerank-v3.5"

_client = None


def cohere_client():
    global _client
    if _client is None:
        try:
            import cohere
        except ImportError:
            sys.exit("ERROR: cohere not installed. Run: pip install cohere")
        key = os.getenv("COHERE_API_KEY") or load_env_var_from_profile("COHERE_API_KEY")
        if not key:
            sys.exit("ERROR: COHERE_API_KEY environment variable is not set. "
                     "Please set it before running: export COHERE_API_KEY='...'")
        os.environ.setdefault("COHERE_API_KEY", key)  # visible to child code too
        _client = cohere.ClientV2(api_key=key)
    return _client


def rerank_results(query, results, model=DEFAULT_RERANK_MODEL):
    """Rerank a top-k result list against the query with Cohere.

    Returns a new list (same length, same schema) in reranked order: score and
    similarity become Cohere's relevance score, and each entry records where
    first-stage retrieval had ranked it under pre_rerank_rank."""
    if not results:
        return []
    docs = [r["chunk"].get("text") or "" for r in results]
    resp = cohere_client().rerank(model=model, query=query,
                                  documents=docs, top_n=len(docs))
    out = []
    for rank, hit in enumerate(resp.results, 1):
        src = results[hit.index]
        out.append({"rank": rank, "score": hit.relevance_score,
                    "similarity": hit.relevance_score,
                    "pre_rerank_rank": src["rank"], "chunk": src["chunk"]})
    return out
