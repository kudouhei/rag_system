"""Stable version identifiers for training content."""
from __future__ import annotations

import hashlib
import json

from collections.abc import Sequence

from app.training.schemas import (
    TrainingEvidence,
    TrainingExplanationRequest,
    TrainingOptionExplanation,
)


def compute_question_fingerprint(
    request: TrainingExplanationRequest,
) -> str:
    """Hash the question content that affects its explanation."""

    payload = {
        "question_id": request.question_id,
        "question": request.question,
        "options": [
            {
                "option_id": option.option_id,
                "text": option.text,
            }
            for option in request.options
        ],
        "correct_option_ids": sorted(
            request.correct_option_ids
        ),
        "jurisdiction": request.jurisdiction,
        "language": request.language,
    }

    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        canonical_payload.encode("utf-8")
    ).hexdigest()

    return f"sha256:{digest}"

def compute_explanation_version(
    *,
    question_fingerprint: str,
    summary: str,
    option_explanations: Sequence[TrainingOptionExplanation],
    evidence: Sequence[TrainingEvidence],
) -> str:
    """Hash learner-independent explanation content."""

    evidence_ids = [
        item.evidence_id
        for item in evidence
    ]

    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError(
            "evidence contains duplicate evidence_id values"
        )

    evidence_by_id = {item.evidence_id: item for item in evidence}

    referenced_evidence_ids = {
        evidence_id
        for option in option_explanations
        for evidence_id in option.evidence_ids
    }

    unknown_evidence_ids = (
        referenced_evidence_ids
        - set(evidence_by_id)
    )

    if unknown_evidence_ids:
        raise ValueError(
            "option explanations reference unknown evidence: "
            f"{sorted(unknown_evidence_ids)}"
        )

    canonical_options = []

    for option in sorted(
        option_explanations,
        key=lambda item: item.option_id,
    ):
        referenced_chunk_ids = sorted(
            evidence_by_id[evidence_id].chunk_id
            for evidence_id in option.evidence_ids
        )

        canonical_options.append(
            {
                "option_id": option.option_id,
                "is_correct": option.is_correct,
                "explanation": option.explanation,
                "referenced_chunk_ids": (
                    referenced_chunk_ids
                ),
            }
        )

    canonical_evidence = []
    for evidence_id in referenced_evidence_ids:
        item = evidence_by_id[evidence_id]

        canonical_evidence.append(
            {
                "chunk_id": item.chunk_id,
                "source": item.source,
                "title": item.title,
                "section": item.section,
                "excerpt": item.excerpt,
                "regulation_number": (
                    item.regulation_number
                ),
                "issuing_authority": (
                    item.issuing_authority
                ),
                "effective_date": (
                    item.effective_date
                ),
            }
        )

    canonical_evidence.sort(
        key=lambda item: item["chunk_id"]
    )

    payload = {
        "question_fingerprint": question_fingerprint,
        "summary": summary,
        "option_explanations": canonical_options,
        "evidence": canonical_evidence,
    }

    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        canonical_payload.encode("utf-8")
    ).hexdigest()

    return f"sha256:{digest}"