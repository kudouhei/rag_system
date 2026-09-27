"""Application service for GDPR training explanations."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.training.domain import determine_learner_result
from app.training.evidence import (
    MaterializedTrainingEvidence,
    materialize_training_evidence,
    select_training_evidence,
)
from app.training.retrieval import retrieve_training_candidates
from app.training.schemas import (
    TrainingExplanationRequest,
    TrainingExplanationResponse,
    TrainingOptionExplanation,
)

def _compute_evidence_relevance_score(
    materialized: MaterializedTrainingEvidence,
) -> float:
    evidence_by_id = { item.evidence_id: item for item in materialized.evidence }

    per_option_scores: list[float] = []

    for evidence_ids in materialized.evidence_ids_by_option.values():
        scores = [
            evidence_by_id[evidence_id].relevance_score
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
        ]
        per_option_scores.append(max(scores, default=0.0))

    if not per_option_scores: return 0.0

    return round(sum(per_option_scores) / len(per_option_scores), 4)

def _build_fallback_explanation(
    option_id: str,
    is_correct: bool,
    evidence_ids: list[str],
) -> str:
    if evidence_ids:
        references = ", ".join(evidence_ids)
        verdict = "correct" if is_correct else "incorrect"

        return (
            f"The course answer key marks option {option_id} as "
            f"{verdict}. Relevant regulatory evidence candidates: "
            f"{references}. Automated explanation generation has not "
            "yet been applied, so the cited excerpts require review."
        )

    verdict = "correct" if is_correct else "incorrect"

    return (
        f"The course answer key marks option {option_id} as "
        f"{verdict}, but no relevant regulatory evidence candidate "
        "was retrieved. Manual review is required."
    )

async def build_training_explanation(
    request: TrainingExplanationRequest,
) -> TrainingExplanationResponse:
    retrieval = await retrieve_training_candidates(request)

    selection = select_training_evidence(
        request=request,
        retrieval=retrieval,
    )

    materialized = materialize_training_evidence(selection)

    learner_result = determine_learner_result(
        correct_option_ids=request.correct_option_ids,
        selected_option_ids=request.selected_option_ids,
    )

    correct_ids = set(request.correct_option_ids)
    selected_ids = set(request.selected_option_ids)

    option_explanations: list[TrainingOptionExplanation] = []

    used_evidence_ids = {
        evidence_id
        for explanation in option_explanations
        for evidence_id in explanation.evidence_ids
    }

    response_evidence = [
        evidence
        for evidence in materialized.evidence
        if evidence.evidence_id in used_evidence_ids
    ]

    for option in request.options:
        evidence_ids = materialized.evidence_ids_by_option.get(
            option.option_id,
            [],
        )
        is_correct = option.option_id in correct_ids

        option_explanations.append(
            TrainingOptionExplanation(
                option_id=option.option_id,
                is_correct=is_correct,
                selected_by_learner=(
                    option.option_id in selected_ids
                ),
                explanation=_build_fallback_explanation(
                    option_id=option.option_id,
                    is_correct=is_correct,
                    evidence_ids=evidence_ids,
                ),
                evidence_ids=evidence_ids,
            )
        )

    has_missing_option_evidence = any(
        not explanation.evidence_ids
        for explanation in option_explanations
    )

    if (
        not response_evidence
        or has_missing_option_evidence
    ):
        status = "insufficient_evidence"
        summary = (
            "The system could not retrieve sufficient regulatory "
            "evidence for every answer option. Manual review is required."
        )
    else:
        status = "review_required"
        summary = (
            "Relevant regulatory evidence was retrieved for each option. "
            "Automated explanation generation is not yet enabled, so the "
            "evidence must be reviewed before publication."
        )

    return TrainingExplanationResponse(
        trace_id=str(uuid4()),
        question_id=request.question_id,
        status=status,
        learner_result=learner_result,
        summary=summary,
        option_explanations=option_explanations,
        evidence=response_evidence,
        evidence_relevance_score=_compute_evidence_relevance_score(
            materialized
        ),
        grounding_score=None,
        corpus_version=retrieval.eligible_corpus_version,
        generator_model=None,
        generated_at=datetime.now(timezone.utc),
    )
