"""
RAGAS-style Evaluation  (Es et al., 2023)
============================================
"""
from __future__ import annotations

from typing import List

import numpy as np

from app.core import state
from app.retrieval.embeddings import encode_text


def compute_ragas_metrics(query: str, docs: List[dict], answer: str) -> dict:
    """
    Simplified RAGAS metrics (no external NLI model required):

    • context_relevance   — avg cosine(query, chunk)  ∈ [0, 1]
    • context_precision   — fraction of chunks with similarity > 0.45
    • answer_relevance    — cosine(query_emb, answer_emb)  ∈ [0, 1]
    • answer_faithfulness — token-overlap proxy between answer and context
    """
    if not docs or not query:
        return {}

    q_emb = encode_text(query)

    # Context Relevance & Precision
    sims = []
    for doc in docs:
        idx = next((i for i, d in enumerate(state.KNOWLEDGE_BASE) if d["id"] == doc["id"]), None)
        if idx is not None and state.doc_embeddings is not None:
            sim = (float(state.doc_embeddings[idx] @ q_emb) + 1.0) / 2.0
            sims.append(sim)

    context_relevance = round(float(np.mean(sims)), 3) if sims else 0.0
    context_precision = round(sum(1 for s in sims if s > 0.45) / max(len(sims), 1), 3)

    # Answer Relevance — cosine(answer_emb, query_emb)
    if answer.strip():
        a_emb = encode_text(answer)
        ar = (float(a_emb @ q_emb) + 1.0) / 2.0
        answer_relevance = round(ar, 3)
    else:
        answer_relevance = 0.0

    # Answer Faithfulness (token-overlap proxy)
    context_tokens = set(
        " ".join(d["content"] for d in docs[:3]).lower().split()
    )
    answer_tokens = set(answer.lower().split())
    if answer_tokens:
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        answer_faithfulness = round(min(1.0, overlap * 1.8), 3)
    else:
        answer_faithfulness = 0.0

    return {
        "context_relevance":    context_relevance,
        "context_precision":    context_precision,
        "answer_relevance":     answer_relevance,
        "answer_faithfulness":  answer_faithfulness,
    }
