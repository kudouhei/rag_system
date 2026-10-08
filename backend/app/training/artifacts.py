"""Persistent domain models for reviewed explanation content."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.training.lifecycle import (
    ExplanationStatus,
)


class ExplanationArtifactOption(BaseModel):
    """Learner-independent explanation for one answer option."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    option_id: str = Field(
        min_length=1,
        max_length=100,
    )
    is_correct: bool
    explanation: str = Field(
        min_length=1,
        max_length=8000,
    )
    evidence_chunk_ids: list[str] = Field(
        min_length=1,
        max_length=8,
    )


class ExplanationArtifactEvidence(BaseModel):
    """Immutable snapshot of evidence used by the explanation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: str = Field(
        min_length=1,
        max_length=200,
    )
    source: str = Field(
        min_length=1,
        max_length=500,
    )
    title: str | None = Field(
        default=None,
        max_length=500,
    )
    section: str | None = Field(
        default=None,
        max_length=500,
    )
    excerpt: str = Field(
        min_length=1,
        max_length=4000,
    )

    regulation_number: str | None = Field(
        default=None,
        max_length=100,
    )
    issuing_authority: str | None = Field(
        default=None,
        max_length=200,
    )
    effective_date: str | None = Field(
        default=None,
        max_length=50,
    )


class ExplanationArtifactProvenance(BaseModel):
    """Technical lineage of an explanation candidate."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    corpus_version: str = Field(
        min_length=1,
        max_length=100,
    )
    content_origin: Literal[
        "llm_generated",
        "deterministic_fallback",
    ]

    generation_prompt_version: str = Field(
        min_length=1,
        max_length=100,
    )
    generator_model: str | None = Field(
        default=None,
        max_length=200,
    )

    grounding_prompt_version: str | None = Field(
        default=None,
        max_length=100,
    )
    grounding_model: str | None = Field(
        default=None,
        max_length=200,
    )
    grounding_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    source_trace_id: str = Field(
        min_length=1,
        max_length=100,
    )


class TrainingExplanationArtifact(BaseModel):
    """One immutable explanation version awaiting governance."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    artifact_id: UUID

    tenant_id: str = Field(
        min_length=1,
        max_length=100,
    )
    course_id: str = Field(
        min_length=1,
        max_length=100,
    )
    question_id: str = Field(
        min_length=1,
        max_length=100,
    )

    question_fingerprint: str = Field(
        min_length=1,
        max_length=100,
    )
    explanation_version: str = Field(
        min_length=1,
        max_length=100,
    )

    status: ExplanationStatus = (
        ExplanationStatus.REVIEW_REQUIRED
    )

    jurisdiction: str = Field(
        min_length=2,
        max_length=20,
    )
    language: str = Field(
        min_length=2,
        max_length=20,
    )

    summary: str = Field(
        min_length=1,
        max_length=8000,
    )
    option_explanations: list[
        ExplanationArtifactOption
    ] = Field(
        min_length=2,
        max_length=10,
    )
    evidence: list[
        ExplanationArtifactEvidence
    ] = Field(
        min_length=1,
    )

    provenance: ExplanationArtifactProvenance

    created_by: str = Field(
        min_length=1,
        max_length=200,
    )
    created_at: datetime

    @model_validator(mode="after")
    def validate_content_references(
        self,
    ) -> Self:
        option_ids = [
            option.option_id
            for option in self.option_explanations
        ]

        if len(option_ids) != len(set(option_ids)):
            raise ValueError(
                "artifact contains duplicate option_id values"
            )

        chunk_ids = [
            item.chunk_id
            for item in self.evidence
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "artifact contains duplicate chunk_id values"
            )

        available_chunk_ids = set(chunk_ids)

        for option in self.option_explanations:
            referenced_chunk_ids = (
                option.evidence_chunk_ids
            )

            if len(referenced_chunk_ids) != len(
                set(referenced_chunk_ids)
            ):
                raise ValueError(
                    "artifact option "
                    f"{option.option_id!r} contains duplicate "
                    "evidence_chunk_ids"
                )

            unknown_chunk_ids = (
                set(referenced_chunk_ids)
                - available_chunk_ids
            )

            if unknown_chunk_ids:
                raise ValueError(
                    "artifact option "
                    f"{option.option_id!r} references unknown "
                    "evidence chunks: "
                    f"{sorted(unknown_chunk_ids)}"
                )

        return self