"""
Shared Retrieval Scoring
==========================
Single source of truth for computing dense/sparse/graph score arrays and
turning them into per-document scored dicts.

Previously this ~30-line combination (dense + sparse + graph scoring, fused
into a per-document `final_score`/`strategy_used` dict) was independently
duplicated in `run_rag_pipeline`, `query_rag`, and `run_compliance_check`.
All three now call the two functions below.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.core import state
from app.retrieval.bm25_index import compute_bm25_scores
from app.retrieval.embeddings import compute_embedding_scores
from app.retrieval.fusion import fuse_scores
from app.retrieval.graph_rag import compute_graph_scores


async def compute_score_arrays(query: str, enable_graph: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run dense/sparse/graph scoring (each CPU-bound) off the event loop."""
    loop = asyncio.get_event_loop()
    emb_scores = await loop.run_in_executor(None, compute_embedding_scores, query)
    bm25_arr = await loop.run_in_executor(None, compute_bm25_scores, query)
    graph_arr = (
        await loop.run_in_executor(None, compute_graph_scores, query)
        if enable_graph
        else np.zeros(len(state.KNOWLEDGE_BASE), dtype=np.float32)
    )
    return emb_scores, bm25_arr, graph_arr


def build_scored_docs(
    emb_scores: np.ndarray,
    bm25_arr: np.ndarray,
    graph_arr: np.ndarray,
    strategy: str,
    enable_graph: bool,
    doc_mask: Optional[Sequence[bool]] = None,
) -> List[dict]:
    """
    Combine per-document score arrays (indexed against `state.KNOWLEDGE_BASE`)
    into a list of scored document copies with `embedding_score`, `bm25_score`,
    `graph_score`, `final_score`, and `strategy_used` set.

    `doc_mask` (same length/order as `state.KNOWLEDGE_BASE`), if given, skips
    excluded documents — used by Compliance Check's metadata filters.
    """
    final_arr = fuse_scores(emb_scores, bm25_arr, graph_arr, strategy, enable_graph)
    docs_scored: List[dict] = []
    for idx, doc in enumerate(state.KNOWLEDGE_BASE):
        if doc_mask is not None and not doc_mask[idx]:
            continue
        d = doc.copy()
        es = float(emb_scores[idx])
        bs = float(bm25_arr[idx])
        gs = float(graph_arr[idx])
        if strategy == "vector":
            d.update(embedding_score=es, bm25_score=0.0, graph_score=0.0, final_score=es)
        elif strategy == "bm25":
            d.update(embedding_score=0.0, bm25_score=bs, graph_score=0.0, final_score=bs)
        else:
            d.update(embedding_score=es, bm25_score=bs, graph_score=gs, final_score=float(final_arr[idx]))
        d["strategy_used"] = strategy
        docs_scored.append(d)
    return docs_scored
