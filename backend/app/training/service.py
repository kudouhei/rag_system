"""Application service for GDPR training explanations."""
from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from app.training.domain import determine_learner_result
from app.training.evidence import (
    MaterializedTrainingEvidence,
    materialize_training_evidence,
    select_training_evidence,
)
from app.training.retrieval import (
    retrieve_training_candidates,
)
from app.training.schemas import (
    TrainingExplanationRequest,
    TrainingExplanationResponse,
    TrainingOptionExplanation,
)


if TYPE_CHECKING:
    from app.training.trace import TrainingTraceCollector


def _compute_evidence_relevance_score(
    materialized: MaterializedTrainingEvidence,
) -> float:
    evidence_by_id = {
        item.evidence_id: item
        for item in materialized.evidence
    }

    per_option_scores: list[float] = []

    for evidence_ids in (
        materialized.evidence_ids_by_option.values()
    ):
        scores = [
            evidence_by_id[
                evidence_id
            ].relevance_score
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
        ]

        per_option_scores.append(
            max(
                scores,
                default=0.0,
            )
        )

    if not per_option_scores:
        return 0.0

    return round(
        sum(per_option_scores)
        / len(per_option_scores),
        4,
    )


def _build_fallback_explanation(
    option_id: str,
    is_correct: bool,
    evidence_ids: list[str],
) -> str:
    verdict = (
        "correct"
        if is_correct
        else "incorrect"
    )

    if evidence_ids:
        references = ", ".join(
            evidence_ids
        )

        return (
            f"The course answer key marks option {option_id} as "
            f"{verdict}. Relevant regulatory evidence candidates: "
            f"{references}. Automated explanation generation has not "
            "yet been applied, so the cited excerpts require review."
        )

    return (
        f"The course answer key marks option {option_id} as "
        f"{verdict}, but no relevant regulatory evidence candidate "
        "was retrieved. Manual review is required."
    )


async def build_training_explanation(
    request: TrainingExplanationRequest,
    trace: TrainingTraceCollector | None = None,
) -> TrainingExplanationResponse:
    # ------------------------------------------------------------------
    # Stage 1: Request
    #
    # FastAPI/Pydantic has already validated the request contract before
    # the service is called.
    # ------------------------------------------------------------------

    request_started_at = perf_counter()

    if trace is not None:
        trace.record(
            stage="request",
            status="complete",
            started_at=request_started_at,
            summary=(
                f"Accepted training question "
                f"{request.question_id!r}."
            ),
            details={
                "tenant_id": request.tenant_id,
                "course_id": request.course_id,
                "question_id": request.question_id,
                "option_count": len(request.options),
                "option_ids": [
                    option.option_id
                    for option in request.options
                ],
                "correct_option_ids": (
                    request.correct_option_ids
                ),
                "selected_option_ids": (
                    request.selected_option_ids
                ),
                "jurisdiction": request.jurisdiction,
                "language": request.language,
                "top_k": request.top_k,
            },
        )

    # ------------------------------------------------------------------
    # Stages 2–4:
    # Access Control → Query Planning → Retrieval
    #
    # These stages are executed and traced inside retrieval.py because
    # that module owns their real intermediate data.
    # ------------------------------------------------------------------

    retrieval = await retrieve_training_candidates(
        request,
        trace=trace,
    )

    # ------------------------------------------------------------------
    # Stage 5: Evidence Selection
    #
    # Convert the retrieval candidate pool into bounded, option-linked
    # evidence suitable for the explanation response.
    # ------------------------------------------------------------------

    evidence_started_at = perf_counter()

    selection = select_training_evidence(
        request=request,
        retrieval=retrieval,
    )

    materialized = materialize_training_evidence(
        selection
    )

    learner_result = determine_learner_result(
        correct_option_ids=(
            request.correct_option_ids
        ),
        selected_option_ids=(
            request.selected_option_ids
        ),
    )

    correct_ids = set(
        request.correct_option_ids
    )

    selected_ids = set(
        request.selected_option_ids
    )

    option_explanations: list[
        TrainingOptionExplanation
    ] = []

    for option in request.options:
        evidence_ids = (
            materialized.evidence_ids_by_option.get(
                option.option_id,
                [],
            )
        )

        is_correct = (
            option.option_id
            in correct_ids
        )

        option_explanations.append(
            TrainingOptionExplanation(
                option_id=option.option_id,
                is_correct=is_correct,
                selected_by_learner=(
                    option.option_id
                    in selected_ids
                ),
                explanation=(
                    _build_fallback_explanation(
                        option_id=option.option_id,
                        is_correct=is_correct,
                        evidence_ids=evidence_ids,
                    )
                ),
                evidence_ids=evidence_ids,
            )
        )

    used_evidence_ids = {
        evidence_id
        for explanation in option_explanations
        for evidence_id in explanation.evidence_ids
    }

    response_evidence = [
        evidence
        for evidence in materialized.evidence
        if evidence.evidence_id
        in used_evidence_ids
    ]

    if trace is not None:
        trace.record(
            stage="evidence_selection",
            status="complete",
            started_at=evidence_started_at,
            summary=(
                f"Published {len(response_evidence)} "
                "option-linked evidence items."
            ),
            details={
                "retrieved_candidate_count": (
                    len(retrieval.merged_docs)
                ),
                "selected_context_count": (
                    len(selection.docs)
                ),
                "published_evidence_count": (
                    len(response_evidence)
                ),
                "dropped_unreferenced_count": (
                    len(materialized.evidence)
                    - len(response_evidence)
                ),
                "evidence_ids_by_option": (
                    materialized.evidence_ids_by_option
                ),
                "options_without_candidates": (
                    selection.option_ids_without_candidates
                ),
                "published_evidence": [
                    {
                        "evidence_id": (
                            evidence.evidence_id
                        ),
                        "chunk_id": (
                            evidence.chunk_id
                        ),
                        "source": evidence.source,
                        "section": evidence.section,
                        "relevance_score": (
                            evidence.relevance_score
                        ),
                    }
                    for evidence in response_evidence
                ],
            },
        )

    # ------------------------------------------------------------------
    # Stage 6: Generation
    #
    # The deterministic fallback above remains active until the controlled
    # LLM generator is connected and validated.
    # ------------------------------------------------------------------

    generation_started_at = perf_counter()

    if trace is not None:
        trace.record(
            stage="generation",
            status="skipped",
            started_at=generation_started_at,
            summary=(
                "Automated explanation generation "
                "is not enabled."
            ),
            details={
                "generator_model": None,
                "fallback_used": True,
            },
        )

    # ------------------------------------------------------------------
    # Stage 7: Grounding
    #
    # A retrieval relevance score is available, but claim-level grounding
    # has not yet been implemented.
    # ------------------------------------------------------------------

    grounding_started_at = perf_counter()

    if trace is not None:
        trace.record(
            stage="grounding",
            status="not_evaluated",
            started_at=grounding_started_at,
            summary=(
                "Claim-level grounding validation "
                "has not yet been applied."
            ),
            details={
                "grounding_score": None,
            },
        )

    # ------------------------------------------------------------------
    # Stage 8: Final Response
    # ------------------------------------------------------------------

    final_started_at = perf_counter()

    has_missing_option_evidence = any(
        not explanation.evidence_ids
        for explanation in option_explanations
    )

    if (
        not response_evidence
        or has_missing_option_evidence
    ):
        response_status = (
            "insufficient_evidence"
        )

        summary = (
            "The system could not retrieve sufficient regulatory "
            "evidence for every answer option. "
            "Manual review is required."
        )
    else:
        response_status = "review_required"

        summary = (
            "Relevant regulatory evidence was retrieved for each option. "
            "Automated explanation generation is not yet enabled, so the "
            "evidence must be reviewed before publication."
        )

    response = TrainingExplanationResponse(
        trace_id=str(uuid4()),
        question_id=request.question_id,
        status=response_status,
        learner_result=learner_result,
        summary=summary,
        option_explanations=(
            option_explanations
        ),
        evidence=response_evidence,
        evidence_relevance_score=(
            _compute_evidence_relevance_score(
                materialized
            )
        ),
        grounding_score=None,
        corpus_version=(
            retrieval.eligible_corpus_version
        ),
        generator_model=None,
        generated_at=datetime.now(
            timezone.utc
        ),
    )

    if trace is not None:
        trace.record(
            stage="final_response",
            status="complete",
            started_at=final_started_at,
            summary=(
                f"Built a {response.status!r} "
                "training explanation response."
            ),
            details={
                "trace_id": response.trace_id,
                "status": response.status,
                "learner_result": (
                    response.learner_result
                ),
                "evidence_relevance_score": (
                    response.evidence_relevance_score
                ),
                "grounding_score": (
                    response.grounding_score
                ),
                "evidence_count": (
                    len(response.evidence)
                ),
                "corpus_version": (
                    response.corpus_version
                ),
                "generator_model": (
                    response.generator_model
                ),
            },
        )

    return response