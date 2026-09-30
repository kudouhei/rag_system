"""Application service for GDPR training explanations."""
from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from app.training.domain import (
    determine_learner_result,
)
from app.training.evidence import (
    MaterializedTrainingEvidence,
    materialize_training_evidence,
    select_training_evidence,
)
from app.training.generation import (
    generate_training_content,
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
    from app.training.trace import (
        TrainingTraceCollector,
    )


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
                "option_count": len(
                    request.options
                ),
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
                "jurisdiction": (
                    request.jurisdiction
                ),
                "language": request.language,
                "top_k": request.top_k,
            },
        )

    # ------------------------------------------------------------------
    # Stages 2–4:
    # Access Control → Query Planning → Retrieval
    # ------------------------------------------------------------------

    retrieval = await retrieve_training_candidates(
        request,
        trace=trace,
    )

    # ------------------------------------------------------------------
    # Stage 5: Evidence Selection
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

    if trace is not None:
        trace.record(
            stage="evidence_selection",
            status="complete",
            started_at=evidence_started_at,
            summary=(
                f"Linked evidence to "
                f"{len(request.options)} answer options."
            ),
            details={
                "retrieved_candidate_count": (
                    len(retrieval.merged_docs)
                ),
                "selected_context_count": (
                    len(selection.docs)
                ),
                "materialized_evidence_count": (
                    len(materialized.evidence)
                ),
                "evidence_ids_by_option": (
                    materialized.evidence_ids_by_option
                ),
                "options_without_candidates": (
                    selection.option_ids_without_candidates
                ),
            },
        )

    # ------------------------------------------------------------------
    # Stage 6: Controlled Generation
    #
    # The generator may write narrative text and select allowed evidence
    # IDs. It may not decide correctness or learner result.
    # ------------------------------------------------------------------

    generation_started_at = perf_counter()

    generation_attempt = (
        await generate_training_content(
            request=request,
            materialized=materialized,
            learner_result=learner_result,
        )
    )

    generated_content = (
        generation_attempt.content
    )

    generated_by_option_id = {
        option.option_id: option
        for option in (
            generated_content.option_explanations
            if generated_content is not None
            else []
        )
    }

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
        is_correct = (
            option.option_id
            in correct_ids
        )

        generated_option = (
            generated_by_option_id.get(
                option.option_id
            )
        )

        if generated_option is not None:
            explanation_text = (
                generated_option.explanation
            )

            evidence_ids = (
                generated_option.evidence_ids
            )
        else:
            evidence_ids = (
                materialized.evidence_ids_by_option.get(
                    option.option_id,
                    [],
                )
            )

            explanation_text = (
                _build_fallback_explanation(
                    option_id=option.option_id,
                    is_correct=is_correct,
                    evidence_ids=evidence_ids,
                )
            )

        option_explanations.append(
            TrainingOptionExplanation(
                option_id=option.option_id,
                is_correct=is_correct,
                selected_by_learner=(
                    option.option_id
                    in selected_ids
                ),
                explanation=explanation_text,
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

    if generation_attempt.status == "generated":
        generation_trace_status = "complete"
    elif generation_attempt.status in {
        "disabled",
        "insufficient_evidence",
    }:
        generation_trace_status = "skipped"
    else:
        generation_trace_status = "failed"

    if trace is not None:
        trace.record(
            stage="generation",
            status=generation_trace_status,
            started_at=generation_started_at,
            summary=generation_attempt.reason,
            details={
                "generation_status": (
                    generation_attempt.status
                ),
                "generator_model": (
                    generation_attempt.model
                ),
                "fallback_used": (
                    generated_content is None
                ),
                "published_evidence_count": (
                    len(response_evidence)
                ),
                "published_evidence_ids": [
                    evidence.evidence_id
                    for evidence in response_evidence
                ],
            },
        )

    # ------------------------------------------------------------------
    # Stage 7: Grounding
    #
    # Generated content is still review_required until claim-level
    # grounding validation is implemented.
    # ------------------------------------------------------------------

    grounding_started_at = perf_counter()

    grounding_score = None

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
                "grounding_score": grounding_score,
                "generated_content_present": (
                    generated_content is not None
                ),
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

        response_summary = (
            "The system could not retrieve sufficient regulatory "
            "evidence for every answer option. "
            "Manual review is required."
        )

    elif generated_content is not None:
        # Generation succeeded, but grounding has not yet been evaluated.
        response_status = "review_required"
        response_summary = (
            generated_content.summary
        )

    else:
        response_status = "review_required"

        response_summary = (
            "Relevant regulatory evidence was retrieved for each option. "
            "Automated explanation generation was unavailable or rejected, "
            "so the deterministic fallback requires review."
        )

    response_generator_model = (
        generation_attempt.model
        if generation_attempt.status
        == "generated"
        else None
    )

    response = TrainingExplanationResponse(
        trace_id=str(uuid4()),
        question_id=request.question_id,
        status=response_status,
        learner_result=learner_result,
        summary=response_summary,
        option_explanations=(
            option_explanations
        ),
        evidence=response_evidence,
        evidence_relevance_score=(
            _compute_evidence_relevance_score(
                materialized
            )
        ),
        grounding_score=grounding_score,
        corpus_version=(
            retrieval.eligible_corpus_version
        ),
        generator_model=(
            response_generator_model
        ),
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
                "generation_status": (
                    generation_attempt.status
                ),
            },
        )

    return response