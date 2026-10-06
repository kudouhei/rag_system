"""Stable version identifiers for training content."""
from __future__ import annotations

import hashlib
import json

from app.training.schemas import (
    TrainingExplanationRequest,
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