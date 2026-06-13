"""
retrieve.py — Query → top-K chunks from ChromaDB.

Pure retrieval; no LLM. Takes a string query, embeds it with the same
Sentence Transformers model used at ingest time, runs cosine similarity
search against the ChromaDB collection, and returns ranked chunks with
their metadata.

Optional cross-encoder reranking: when reranker=True, the top-K results
from vector search are re-scored by a CrossEncoder and reordered. Catches
cases where dense retrieval misses lexical signals.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

# Match the ingest-time settings
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from .ingest import COLLECTION_NAME, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Reranker — Apache 2.0, ~85 MB. Loaded lazily on first use.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level singletons so repeated retrievals don't reload the models
_collection_cache: Dict[str, Any] = {}
_reranker_cache: Dict[str, Any] = {}


def _get_collection(chroma_db_path: Path):
    """Open the ChromaDB collection (cached per-process)."""
    key = str(chroma_db_path.resolve())
    if key not in _collection_cache:
        client = chromadb.PersistentClient(
            path=key,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _collection_cache[key] = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection_cache[key]


def _get_reranker():
    """Lazy-load the CrossEncoder. Downloads ~85 MB on first call."""
    if "model" not in _reranker_cache:
        from sentence_transformers import CrossEncoder  # local import — heavy
        _reranker_cache["model"] = CrossEncoder(RERANKER_MODEL)
    return _reranker_cache["model"]


def retrieve(
    query: str,
    chroma_db_path: Path,
    k: int = 10,
    rerank: bool = False,
    rerank_top_n: int = 5,
) -> Dict[str, Any]:
    """
    Return top-K chunks for a query, with metadata and (cosine) distance scores.

    Args:
        query:        natural-language question
        chroma_db_path: where the persistent ChromaDB lives
        k:            how many candidates to fetch from the vector store
        rerank:       if True, run a CrossEncoder over the top-K and reorder
        rerank_top_n: how many results to keep after rerank

    Returns:
        {
          "query":        original query string,
          "chunks":       [{text, pdf_name, page, chunk_index, score, rerank_score?}, ...],
          "retrieval_ms": int,
          "rerank_ms":    int | None,
        }
    """
    t0 = time.monotonic()
    collection = _get_collection(chroma_db_path)
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    retrieval_ms = int((time.monotonic() - t0) * 1000)

    # Flatten ChromaDB's nested-list format (each list is per query; we send one)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    chunks: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "text":        doc,
            "pdf_name":    meta.get("pdf_name", ""),
            "page":        meta.get("page", -1),
            "chunk_index": meta.get("chunk_index", -1),
            "score":       round(1.0 - dist, 4),  # cosine sim = 1 - cosine dist
        })

    rerank_ms: Optional[int] = None
    if rerank and chunks:
        t1 = time.monotonic()
        reranker = _get_reranker()
        pairs = [(query, c["text"]) for c in chunks]
        rerank_scores = reranker.predict(pairs).tolist()
        for ck, rs in zip(chunks, rerank_scores):
            ck["rerank_score"] = round(float(rs), 4)
        chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
        chunks = chunks[:rerank_top_n]
        rerank_ms = int((time.monotonic() - t1) * 1000)

    return {
        "query":        query,
        "chunks":       chunks,
        "retrieval_ms": retrieval_ms,
        "rerank_ms":    rerank_ms,
    }
