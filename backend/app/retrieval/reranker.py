"""
Cross-Encoder Reranking  (optional — set RERANKER_MODEL env var)
===================================================================
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from app.core import state
from app.core.config import RERANKER_MODEL
from app.retrieval.embeddings import encode_text

logger = logging.getLogger(__name__)


def init_cross_encoder() -> None:
    if not RERANKER_MODEL:
        logger.info("Cross-encoder disabled (set RERANKER_MODEL to enable)")
        return
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder: %s", RERANKER_MODEL)
        state.cross_encoder = CrossEncoder(RERANKER_MODEL, max_length=512)
        logger.info("Cross-encoder ready")
    except Exception as e:
        logger.warning("Could not load cross-encoder %s: %s", RERANKER_MODEL, e)


def rerank_docs(query: str, docs: List[dict]) -> List[dict]:
    """
    Two-stage reranking:
      • If cross_encoder is available → use it (genuine cross-attention scoring).
      • Otherwise → refined cosine similarity with exact-match bonus.
    """
    if state.cross_encoder is not None:
        pairs = [(query, d["content"]) for d in docs]
        ce_scores = state.cross_encoder.predict(pairs, show_progress_bar=False)
        # Sigmoid-normalise to [0, 1]
        ce_scores = 1.0 / (1.0 + np.exp(-ce_scores))
        reranked = []
        for doc, ce in zip(docs, ce_scores):
            reranked.append({
                **doc,
                "pre_rerank_score": doc["final_score"],
                "ce_score": round(float(ce), 4),
            })
    else:
        # Fallback: recompute cosine with query re-embedding
        q_emb = encode_text(query)
        reranked = []
        for doc in docs:
            idx = next((i for i, d in enumerate(state.KNOWLEDGE_BASE) if d["id"] == doc["id"]), None)
            if idx is not None and state.doc_embeddings is not None:
                cosine = float(state.doc_embeddings[idx] @ q_emb)
                ce = (cosine + 1.0) / 2.0
            else:
                ce = doc["final_score"]
            # Boost for title match
            if any(t in doc["title"] for t in query.split() if len(t) > 1):
                ce = min(0.99, ce + 0.04)
            reranked.append({
                **doc,
                "pre_rerank_score": doc["final_score"],
                "ce_score": round(ce, 4),
            })

    reranked.sort(key=lambda x: x["ce_score"], reverse=True)
    return reranked
