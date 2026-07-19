"""Local (sentence-transformers) embedding models shared by embed and query time.

Each entry maps the short model name used in filenames and CLI args to its
Hugging Face repo and the literal prefixes the model was trained with.
Asymmetric models (bge) need the query prefix at search time but embed
passages bare; skipping it silently costs recall, so both embed_chunks.py
and retriever_topk.embed_query go through embed_local().
"""

import sys

LOCAL_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "bge-base-en-v1.5": {
        "repo": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
}

_loaded = {}


def is_local_model(model):
    return model in LOCAL_EMBEDDING_MODELS


def _load(model):
    # Milvus Lite imports faiss lazily; on macOS importing faiss after torch
    # crashes on duplicate OpenMP runtimes, so bring faiss in before torch.
    try:
        import faiss  # noqa: F401
    except ImportError:
        pass
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("ERROR: sentence-transformers not installed. "
                 "Run: pip install sentence-transformers")
    if model not in _loaded:
        _loaded[model] = SentenceTransformer(LOCAL_EMBEDDING_MODELS[model]["repo"])
    return _loaded[model]


def embed_local(model, texts, kind):
    """Embed texts with a local model; kind is 'query' or 'passage'.

    Vectors are L2-normalized so the stores' IP metric equals cosine,
    matching the OpenAI models' unit-norm output.
    """
    prefix = LOCAL_EMBEDDING_MODELS[model][f"{kind}_prefix"]
    st = _load(model)
    vectors = st.encode([prefix + t for t in texts], batch_size=64,
                        normalize_embeddings=True,
                        show_progress_bar=len(texts) > 64)
    return [v.tolist() for v in vectors]
