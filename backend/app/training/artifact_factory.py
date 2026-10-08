"""Convert runtime training responses into review artifacts."""
from __future__ import annotations

from uuid import UUID, uuid4

from app.training.artifacts import (
    ExplanationArtifactEvidence,
    ExplanationArtifactOption,
    ExplanationArtifactProvenance,
    TrainingExplanationArtifact,
)
from app.training.generation import (
    GENERATION_PROMPT_VERSION,
)
from app.training.grounding import (
    GROUNDING_PROMPT_VERSION,
)
from app.training.schemas import (
    TrainingExplanationRequest,
    TrainingExplanationResponse,
)


def build_review_artifact(
    *,
    request: TrainingExplanationRequest,
    response: TrainingExplanationResponse,
    grounding_model: str | None,
    created_by: str = "system",
    artifact_id: UUID | None = None,
) -> TrainingExplanationArtifact:
    """Build a learner-independent review artifact."""

    evidence_by_id = {
        item.evidence_id: item
        for item in response.evidence
    }

    artifact_options = []

    for option in response.option_explanations:
        evidence_chunk_ids = [
            evidence_by_id[evidence_id].chunk_id
            for evidence_id in option.evidence_ids
        ]

        artifact_options.append(
            ExplanationArtifactOption(
                option_id=option.option_id,
                is_correct=option.is_correct,
                explanation=option.explanation,
                evidence_chunk_ids=(
                    evidence_chunk_ids
                ),
            )
        )

    artifact_evidence = [
        ExplanationArtifactEvidence(
            chunk_id=item.chunk_id,
            source=item.source,
            title=item.title,
            section=item.section,
            excerpt=item.excerpt,
            regulation_number=(
                item.regulation_number
            ),
            issuing_authority=(
                item.issuing_authority
            ),
            effective_date=item.effective_date,
        )
        for item in response.evidence
    ]

    grounding_was_evaluated = (
        response.grounding_score is not None
    )

    provenance = ExplanationArtifactProvenance(
        corpus_version=response.corpus_version,
        content_origin=(
            "llm_generated"
            if response.generator_model is not None
            else "deterministic_fallback"
        ),
        generation_prompt_version=(
            GENERATION_PROMPT_VERSION
        ),
        generator_model=response.generator_model,
        grounding_prompt_version=(
            GROUNDING_PROMPT_VERSION
            if grounding_was_evaluated
            else None
        ),
        grounding_model=(
            grounding_model
            if grounding_was_evaluated
            else None
        ),
        grounding_score=response.grounding_score,
        source_trace_id=response.trace_id,
    )

    return TrainingExplanationArtifact(
        artifact_id=artifact_id or uuid4(),
        tenant_id=request.tenant_id,
        course_id=request.course_id,
        question_id=request.question_id,
        question_fingerprint=(
            response.question_fingerprint
        ),
        explanation_version=(
            response.explanation_version
        ),
        jurisdiction=request.jurisdiction,
        language=request.language,
        summary=response.summary,
        option_explanations=artifact_options,
        evidence=artifact_evidence,
        provenance=provenance,
        created_by=created_by,
        created_at=response.generated_at,
    )