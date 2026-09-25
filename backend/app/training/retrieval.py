"""Option-aware retrieval for GDPR training explanations."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.core import state
from app.retrieval.reranker import rerank_docs
from app.retrieval.scoring import (
    build_scored_docs,
    compute_score_arrays,
)
from app.training.access import build_training_doc_mask
from app.training.query_planner import build_retrieval_queries
from app.training.schemas import TrainingExplanationRequest


@dataclass(slots=True)
class TrainingRetrievalResult:
    eligible_doc_count: int
    results_by_query: dict[str, list[dict]]
    merged_docs: list[dict]


async def retrieve_training_candidates(
    request: TrainingExplanationRequest,
) -> TrainingRetrievalResult:
    query_plan = build_retrieval_queries(request)
    doc_mask = build_training_doc_mask(
        state.KNOWLEDGE_BASE,
        request,
    )
    eligible_doc_count = sum(doc_mask)

    if eligible_doc_count == 0:
        return TrainingRetrievalResult(
            eligible_doc_count=0,
            results_by_query={
                query.query_id: []
                for query in query_plan
            },
            merged_docs=[],
        )

    results_by_query: dict[str, list[dict]] = {}
    merged_by_doc_id: dict[str, dict] = {}

    for query in query_plan:
        emb_scores, bm25_scores, graph_scores = (
            await compute_score_arrays(
                query.text,
                enable_graph=False,
            )
        )

        scored_docs = build_scored_docs(
            emb_scores,
            bm25_scores,
            graph_scores,
            strategy="hybrid",
            enable_graph=False,
            doc_mask=doc_mask,
        )

        ranked_docs = sorted(
            scored_docs,
            key=lambda doc: doc["final_score"],
            reverse=True,
        )[:request.top_k]

        if state.cross_encoder is not None and ranked_docs:
            loop = asyncio.get_running_loop()
            ranked_docs = await loop.run_in_executor(
                None,
                rerank_docs,
                query.text,
                ranked_docs,
            )

        results_by_query[query.query_id] = ranked_docs

        for rank, doc in enumerate(ranked_docs, start=1):
            doc_id = doc["id"]
            score = float(
                doc.get("ce_score", doc["final_score"])
            )

            if doc_id not in merged_by_doc_id:
                merged_doc = doc.copy()
                merged_doc["matched_query_ids"] = []
                merged_doc["matched_option_ids"] = []
                merged_doc["rrf_score"] = 0.0
                merged_doc["training_score"] = score
                merged_by_doc_id[doc_id] = merged_doc

            merged_doc = merged_by_doc_id[doc_id]
            merged_doc["matched_query_ids"].append(query.query_id)

            if query.option_id is not None:
                merged_doc["matched_option_ids"].append(
                    query.option_id
                )

            # Reciprocal Rank Fusion across query result lists.
            merged_doc["rrf_score"] += 1.0 / (60.0 + rank)
            merged_doc["training_score"] = max(
                merged_doc["training_score"],
                score,
            )

    merged_docs = sorted(
        merged_by_doc_id.values(),
        key=lambda doc: (
            doc["rrf_score"],
            doc["training_score"],
        ),
        reverse=True,
    )

    return TrainingRetrievalResult(
        eligible_doc_count=eligible_doc_count,
        results_by_query=results_by_query,
        merged_docs=merged_docs,
    )