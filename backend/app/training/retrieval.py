"""Option-aware retrieval for GDPR training explanations."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from app.core import state
from app.core.provenance import compute_corpus_version
from app.retrieval.reranker import rerank_docs
from app.retrieval.scoring import (
    build_scored_docs,
    compute_score_arrays,
)
from app.training.access import build_training_doc_mask
from app.training.query_planner import build_retrieval_queries
from app.training.schemas import TrainingExplanationRequest


if TYPE_CHECKING:
    from app.training.trace import TrainingTraceCollector


@dataclass(slots=True)
class TrainingRetrievalResult:
    eligible_doc_count: int
    results_by_query: dict[str, list[dict]]
    merged_docs: list[dict]
    eligible_corpus_version: str


def _serialize_trace_candidate(
    doc: dict,
) -> dict[str, object]:
    """Return a bounded, JSON-safe candidate representation."""

    final_score = doc.get(
        "ce_score",
        doc.get(
            "final_score",
            0.0,
        ),
    )

    return {
        "chunk_id": (
            doc.get("stable_id")
            or doc.get("id")
        ),
        "source": doc.get("source"),
        "title": doc.get("title"),
        "embedding_score": round(
            float(
                doc.get(
                    "embedding_score",
                    0.0,
                )
            ),
            4,
        ),
        "bm25_score": round(
            float(
                doc.get(
                    "bm25_score",
                    0.0,
                )
            ),
            4,
        ),
        "final_score": round(
            float(final_score),
            4,
        ),
    }


async def retrieve_training_candidates(
    request: TrainingExplanationRequest,
    trace: TrainingTraceCollector | None = None,
) -> TrainingRetrievalResult:
    knowledge_base = state.KNOWLEDGE_BASE

    # ------------------------------------------------------------------
    # Stage 2: Access Control
    #
    # Determine which chunks this tenant and course may access before
    # executing retrieval.
    # ------------------------------------------------------------------

    access_started_at = perf_counter()

    doc_mask = build_training_doc_mask(
        knowledge_base,
        request,
    )

    eligible_docs = [
        doc
        for doc, allowed in zip(
            knowledge_base,
            doc_mask,
        )
        if allowed
    ]

    eligible_doc_count = len(eligible_docs)

    eligible_corpus_version = compute_corpus_version(
        eligible_docs
    )

    if trace is not None:
        trace.record(
            stage="access_control",
            status="complete",
            started_at=access_started_at,
            summary=(
                f"{eligible_doc_count} of "
                f"{len(knowledge_base)} chunks are eligible."
            ),
            details={
                "total_chunk_count": len(knowledge_base),
                "eligible_chunk_count": eligible_doc_count,
                "excluded_chunk_count": (
                    len(knowledge_base)
                    - eligible_doc_count
                ),
                "tenant_id": request.tenant_id,
                "course_id": request.course_id,
                "knowledge_domain": "gdpr",
                "eligible_corpus_version": (
                    eligible_corpus_version
                ),
            },
        )

    # ------------------------------------------------------------------
    # Stage 3: Query Planning
    #
    # Build one question-level query and one neutral assessment query for
    # each answer option.
    # ------------------------------------------------------------------

    query_started_at = perf_counter()

    query_plan = build_retrieval_queries(
        request
    )

    if trace is not None:
        trace.record(
            stage="query_planning",
            status="complete",
            started_at=query_started_at,
            summary=(
                f"Built {len(query_plan)} retrieval queries."
            ),
            details={
                "queries": [
                    {
                        "query_id": query.query_id,
                        "option_id": query.option_id,
                        "text": query.text,
                    }
                    for query in query_plan
                ],
            },
        )

    # ------------------------------------------------------------------
    # Stage 4: Retrieval
    #
    # Execute hybrid retrieval against the global arrays, while enforcing
    # the access mask when candidates are constructed.
    # ------------------------------------------------------------------

    retrieval_started_at = perf_counter()

    if eligible_doc_count == 0:
        empty_results = {
            query.query_id: []
            for query in query_plan
        }

        if trace is not None:
            trace.record(
                stage="retrieval",
                status="complete",
                started_at=retrieval_started_at,
                summary=(
                    "No eligible chunks were available "
                    "for retrieval."
                ),
                details={
                    "top_k": request.top_k,
                    "unique_candidate_count": 0,
                    "results_by_query": empty_results,
                },
            )

        return TrainingRetrievalResult(
            eligible_doc_count=0,
            eligible_corpus_version=(
                eligible_corpus_version
            ),
            results_by_query=empty_results,
            merged_docs=[],
        )

    results_by_query: dict[str, list[dict]] = {}
    merged_by_doc_id: dict[str, dict] = {}

    for query in query_plan:
        (
            embedding_scores,
            bm25_scores,
            graph_scores,
        ) = await compute_score_arrays(
            query.text,
            enable_graph=False,
        )

        scored_docs = build_scored_docs(
            embedding_scores,
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

        if (
            state.cross_encoder is not None
            and ranked_docs
        ):
            loop = asyncio.get_running_loop()

            ranked_docs = await loop.run_in_executor(
                None,
                rerank_docs,
                query.text,
                ranked_docs,
            )

        results_by_query[
            query.query_id
        ] = ranked_docs

        for rank, doc in enumerate(
            ranked_docs,
            start=1,
        ):
            doc_id = doc["id"]

            score = float(
                doc.get(
                    "ce_score",
                    doc["final_score"],
                )
            )

            if doc_id not in merged_by_doc_id:
                merged_doc = doc.copy()

                merged_doc[
                    "matched_query_ids"
                ] = []

                merged_doc[
                    "matched_option_ids"
                ] = []

                merged_doc["rrf_score"] = 0.0
                merged_doc["training_score"] = score

                merged_by_doc_id[
                    doc_id
                ] = merged_doc

            merged_doc = merged_by_doc_id[
                doc_id
            ]

            merged_doc[
                "matched_query_ids"
            ].append(query.query_id)

            if query.option_id is not None:
                merged_doc[
                    "matched_option_ids"
                ].append(query.option_id)

            # Reciprocal Rank Fusion combines ranks from the question-level
            # query and the option-level queries.
            merged_doc["rrf_score"] += (
                1.0 / (60.0 + rank)
            )

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

    if trace is not None:
        trace.record(
            stage="retrieval",
            status="complete",
            started_at=retrieval_started_at,
            summary=(
                f"Retrieved {len(merged_docs)} "
                "unique candidates across "
                f"{len(results_by_query)} queries."
            ),
            details={
                "top_k": request.top_k,
                "unique_candidate_count": (
                    len(merged_docs)
                ),
                "cross_encoder_enabled": (
                    state.cross_encoder is not None
                ),
                "results_by_query": {
                    query_id: [
                        _serialize_trace_candidate(
                            doc
                        )
                        for doc in documents[:5]
                    ]
                    for query_id, documents
                    in results_by_query.items()
                },
            },
        )

    return TrainingRetrievalResult(
        eligible_doc_count=eligible_doc_count,
        eligible_corpus_version=(
            eligible_corpus_version
        ),
        results_by_query=results_by_query,
        merged_docs=merged_docs,
    )