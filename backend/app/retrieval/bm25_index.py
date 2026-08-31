"""
Sparse Retrieval — BM25
=========================
Tokenisation auto-detects the corpus language: regulatory/fund documents are
tokenised with a lightweight regex word tokenizer (handles hyphenated legal
terms like "anti-money-laundering" and codes like "REG-FM-102" sensibly);
jieba is used automatically as a fallback for CJK content when installed.
"""
from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

from app.core import state

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def _english_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _looks_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text[:200])


def init_bm25(docs: List[dict]) -> None:
    sample = " ".join(d.get("title", "") + d.get("content", "") for d in docs[:20])
    if _looks_cjk(sample):
        try:
            import jieba
            jieba.setLogLevel(logging.WARNING)
            state.tokenize_fn = lambda t: list(jieba.cut(t))
            logger.info("BM25 using jieba tokeniser (CJK corpus detected)")
        except ImportError:
            state.tokenize_fn = _english_tokenize
            logger.warning("jieba not found — falling back to regex tokeniser")
    else:
        state.tokenize_fn = _english_tokenize
        logger.info("BM25 using regex word tokeniser (English corpus detected)")

    from rank_bm25 import BM25Okapi
    tokenized = [state.tokenize_fn(d["title"] + " " + d["content"]) for d in docs]
    state.bm25_index = BM25Okapi(tokenized)
    logger.info("BM25 index ready (%d docs)", len(docs))


def compute_bm25_scores(query: str) -> np.ndarray:
    tokens = state.tokenize_fn(query)
    scores = state.bm25_index.get_scores(tokens).astype(np.float32)
    mx = scores.max()
    return scores / mx if mx > 0 else scores
