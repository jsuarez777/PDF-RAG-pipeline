#!/usr/bin/env python3
"""Reranking of retrieved chunks, shared by retriever_topk.py,
eval_retrieval.py and (through it) run_pipeline.py.

A reranker is a cross-encoder that scores each (query, document) pair jointly,
which is more accurate than the bi-encoder similarity used for first-stage
retrieval but too slow to run over a whole corpus. So it is applied as a
second stage: retrieve top-k candidates with bm25/vector/hybrid, then let the
reranker reorder those k chunks.

Providers:
  cohere  Cohere rerank API (rerank-v3.5). Reads COHERE_API_KEY or CO_API_KEY
          from the environment, falling back to ~/.profile like the OpenAI key.
          Trial keys are limited to 10 calls/min and 1000/month.
  local   sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2),
          runs on this machine with no key or rate limit; scores are
          sigmoid-squashed so both providers report relevance in [0, 1].

The result schema is unchanged: rank/score/similarity/chunk, with score and
similarity replaced by the reranker's relevance score (0..1) and the
pre-rerank position kept in pre_rerank_rank.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai_client.openai_client import load_env_var_from_profile  # noqa: E402

RERANK_PROVIDERS = ("cohere", "local")
DEFAULT_RERANK_MODELS = {
    "cohere": "rerank-v3.5",
    "local": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

_client = None
_cross_encoders = {}


def cohere_client():
    global _client
    if _client is None:
        try:
            import cohere
        except ImportError:
            sys.exit("ERROR: cohere not installed. Run: pip install cohere")
        key = next((k for name in ("COHERE_API_KEY", "CO_API_KEY")
                    for k in (os.getenv(name), load_env_var_from_profile(name))
                    if k), None)
        if not key:
            sys.exit("ERROR: Cohere API key not found. Set COHERE_API_KEY (or "
                     "CO_API_KEY) in the environment or in ~/.profile.")
        os.environ.setdefault("COHERE_API_KEY", key)  # visible to child code too
        _client = cohere.ClientV2(api_key=key)
    return _client


def cross_encoder(model):
    if model not in _cross_encoders:
        # Milvus Lite imports faiss lazily; on macOS importing faiss after
        # torch crashes on duplicate OpenMP runtimes, so bring faiss in first
        # (same guard as embedding_backends._load).
        try:
            import faiss  # noqa: F401
        except ImportError:
            pass
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            sys.exit("ERROR: sentence-transformers not installed. "
                     "Run: pip install sentence-transformers")
        _cross_encoders[model] = CrossEncoder(model)
    return _cross_encoders[model]


def _rank_cohere(query, docs, model):
    """Return [(candidate index, relevance score)] best-first via the API.

    Trial keys allow 10 calls/min (rolling window), so on a 429 wait out the
    window and retry rather than dying mid-eval. Capped so a monthly-quota
    429 (also 10xx-style, but waiting won't help) eventually surfaces."""
    client = cohere_client()
    for attempt in range(6):
        try:
            resp = client.rerank(model=model, query=query,
                                 documents=docs, top_n=len(docs))
            return [(hit.index, float(hit.relevance_score)) for hit in resp.results]
        except Exception as e:
            if getattr(e, "status_code", None) != 429 or attempt == 5:
                raise
            print("Cohere rate limit hit (trial keys: 10 calls/min) — "
                  "waiting 65s before retrying ...", flush=True)
            time.sleep(65)


def _rank_local(query, docs, model):
    """Return [(candidate index, relevance score)] best-first via a local
    CrossEncoder. Sigmoid squashes the raw logits into [0, 1] so scores are
    comparable in shape to Cohere's."""
    ce = cross_encoder(model)
    import inspect
    import torch  # already loaded by sentence_transformers
    # sentence-transformers >= 5 renamed activation_fct to activation_fn
    arg = ("activation_fn" if "activation_fn" in inspect.signature(ce.predict).parameters
           else "activation_fct")
    scores = ce.predict([(query, d) for d in docs], **{arg: torch.nn.Sigmoid()})
    # stable sort: first-stage order breaks ties
    return sorted(enumerate(float(s) for s in scores), key=lambda t: -t[1])


def rerank_results(query, results, provider="cohere", model=None):
    """Rerank a top-k result list against the query.

    Returns a new list (same length, same schema) in reranked order: score and
    similarity become the reranker's relevance score, and each entry records
    where first-stage retrieval had ranked it under pre_rerank_rank."""
    if not results:
        return []
    model = model or DEFAULT_RERANK_MODELS[provider]
    docs = [r["chunk"].get("text") or "" for r in results]
    rank_fn = _rank_cohere if provider == "cohere" else _rank_local
    out = []
    for rank, (index, score) in enumerate(rank_fn(query, docs, model), 1):
        src = results[index]
        out.append({"rank": rank, "score": score, "similarity": score,
                    "pre_rerank_rank": src["rank"], "chunk": src["chunk"]})
    return out
