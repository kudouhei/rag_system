"""API contracts for the GDPR training explanation feature."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class TrainingQuestionOption(BaseModel):
    option_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=4000)


class TrainingExplanationRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=100)
    course_id: str = Field(min_length=1, max_length=100)
    question_id: str = Field(min_length=1, max_length=100)

    question: str = Field(min_length=1, max_length=8000)
    options: list[TrainingQuestionOption] = Field(
        min_length=2,
        max_length=10,
    )
    correct_option_ids: list[str] = Field(min_length=1)

    selected_option_ids: list[str] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    reference_analysis: str | None = Field(default=None, max_length=10000)

    jurisdiction: str = Field(default="LU", min_length=2, max_length=20)
    language: str = Field(default="en", min_length=2, max_length=20)
    top_k: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def validate_option_references(self) -> Self:
        option_ids = [option.option_id for option in self.options]

        if len(option_ids) != len(set(option_ids)):
            raise ValueError("options contain duplicate option_id values")

        if len(self.correct_option_ids) != len(set(self.correct_option_ids)):
            raise ValueError("correct_option_ids contain duplicate values")

        if len(self.selected_option_ids) != len(set(self.selected_option_ids)):
            raise ValueError("selected_option_ids contain duplicate values")

        available_ids = set(option_ids)

        unknown_correct_ids = set(self.correct_option_ids) - available_ids
        if unknown_correct_ids:
            raise ValueError(
                "correct_option_ids contain unknown values: "
                f"{sorted(unknown_correct_ids)}"
            )

        unknown_selected_ids = set(self.selected_option_ids) - available_ids
        if unknown_selected_ids:
            raise ValueError(
                "selected_option_ids contain unknown values: "
                f"{sorted(unknown_selected_ids)}"
            )

        return self

class TrainingEvidence(BaseModel):
    evidence_id: str = Field(min_length=1, max_length=100)
    chunk_id: str = Field(min_length=1, max_length=200)
    source: str = Field(min_length=1, max_length=500)
    title: str | None = Field(default=None, max_length=500)
    section: str | None = Field(default=None, max_length=500)
    excerpt: str = Field(min_length=1, max_length=4000)

    relevance_score: float = Field(ge=0.0, le=1.0)

    regulation_number: str | None = Field(default=None, max_length=100)
    issuing_authority: str | None = Field(default=None, max_length=200)
    effective_date: str | None = Field(default=None, max_length=50)


class TrainingOptionExplanation(BaseModel):
    option_id: str = Field(min_length=1, max_length=100)
    is_correct: bool
    selected_by_learner: bool = False
    explanation: str = Field(min_length=1, max_length=8000)
    evidence_ids: list[str] = Field(default_factory=list)


class TrainingExplanationResponse(BaseModel):
    trace_id: str = Field(min_length=1, max_length=100)
    question_id: str = Field(min_length=1, max_length=100)

    status: Literal[
        "grounded",
        "insufficient_evidence",
        "review_required",
    ]
    learner_result: Literal[
        "correct",
        "incorrect",
        "partially_correct",
        "not_answered",
    ]

    summary: str = Field(min_length=1, max_length=8000)
    option_explanations: list[TrainingOptionExplanation] = Field(
        min_length=2,
        max_length=10,
    )
    evidence: list[TrainingEvidence] = Field(default_factory=list)

    grounding_score: float = Field(ge=0.0, le=1.0)
    corpus_version: str = Field(min_length=1, max_length=100)
    generator_model: str | None = Field(default=None, max_length=200)
    generated_at: datetime

    @model_validator(mode="after")
    def validate_evidence_references(self) -> Self:
        evidence_ids = [item.evidence_id for item in self.evidence]

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence contains duplicate evidence_id values")

        option_ids = [
            explanation.option_id
            for explanation in self.option_explanations
        ]

        if len(option_ids) != len(set(option_ids)):
            raise ValueError(
                "option_explanations contain duplicate option_id values"
            )

        available_evidence_ids = set(evidence_ids)

        for explanation in self.option_explanations:
            referenced_ids = explanation.evidence_ids

            if len(referenced_ids) != len(set(referenced_ids)):
                raise ValueError(
                    "option explanation "
                    f"{explanation.option_id!r} contains duplicate evidence_ids"
                )

            unknown_ids = set(referenced_ids) - available_evidence_ids
            if unknown_ids:
                raise ValueError(
                    "option explanation "
                    f"{explanation.option_id!r} references unknown evidence: "
                    f"{sorted(unknown_ids)}"
                )

        if self.status == "grounded":
            if not self.evidence:
                raise ValueError(
                    "grounded response must contain evidence"
                )

            options_without_evidence = [
                explanation.option_id
                for explanation in self.option_explanations
                if not explanation.evidence_ids
            ]
            if options_without_evidence:
                raise ValueError(
                    "grounded response contains options without evidence: "
                    f"{sorted(options_without_evidence)}"
                )

        return self