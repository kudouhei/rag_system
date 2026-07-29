"""
Sparse Retrieval — BM25 + jieba tokenisation
==============================================
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from app.core import state

logger = logging.getLogger(__name__)


def init_bm25(docs: List[dict]) -> None:
    try:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        state.tokenize_fn = lambda t: list(jieba.cut(t))
        logger.info("BM25 using jieba tokeniser")
    except ImportError:
        state.tokenize_fn = lambda t: t.lower().split()
        logger.warning("jieba not found — using whitespace tokeniser")

    from rank_bm25 import BM25Okapi
    tokenized = [state.tokenize_fn(d["title"] + " " + d["content"]) for d in docs]
    state.bm25_index = BM25Okapi(tokenized)
    logger.info("BM25 index ready (%d docs)", len(docs))


def compute_bm25_scores(query: str) -> np.ndarray:
    tokens = state.tokenize_fn(query)
    scores = state.bm25_index.get_scores(tokens).astype(np.float32)
    mx = scores.max()
    return scores / mx if mx > 0 else scores
