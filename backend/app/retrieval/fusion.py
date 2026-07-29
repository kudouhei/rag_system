"""
Score Fusion
============
Combines dense (embedding), sparse (BM25), and graph retrieval lanes into a
single ranking score per retrieval strategy.
"""
from __future__ import annotations

import numpy as np


def fuse_scores(
    emb_arr:    np.ndarray,
    bm25_arr:   np.ndarray,
    graph_arr:  np.ndarray,
    strategy:   str,
    use_graph:  bool,
) -> np.ndarray:
    """Central score fusion. When GraphRAG is active, shifts weights to accommodate graph lane."""
    if strategy == "vector":
        return emb_arr
    if strategy == "bm25":
        return bm25_arr
    # hybrid / adaptive
    if use_graph and graph_arr.any():
        return (0.50 * emb_arr + 0.30 * bm25_arr + 0.20 * graph_arr).astype(np.float32)
    return (0.60 * emb_arr + 0.40 * bm25_arr).astype(np.float32)
