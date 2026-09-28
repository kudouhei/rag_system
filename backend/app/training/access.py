"""Document access rules for the training explanation feature."""
from __future__ import annotations

from collections.abc import Sequence

from app.training.schemas import TrainingExplanationRequest


def _normalise_ids(value: object) -> set[str]:
    if value is None:
        return set()

    if isinstance(value, str):
        cleaned = value.strip()
        return {cleaned} if cleaned else set()

    if isinstance(value, Sequence):
        return {
            str(item).strip()
            for item in value
            if str(item).strip()
        }

    return set()


def build_training_doc_mask(
    docs: Sequence[dict],
    request: TrainingExplanationRequest,
) -> list[bool]:
    mask: list[bool] = []

    for doc in docs:
        # Compatibility default for the existing curated repository corpus.
        # New uploads will later be required to declare access_scope explicitly.
        access_scope = str(
            doc.get("access_scope", "shared")
        ).strip().lower()

        if access_scope == "shared":
            tenant_allowed = True
        elif access_scope == "tenant":
            tenant_allowed = (
                str(doc.get("tenant_id", "")).strip()
                == request.tenant_id
            )
        else:
            # Unknown access scopes fail closed.
            tenant_allowed = False

        allowed_course_ids = _normalise_ids(doc.get("course_ids"))
        course_allowed = (
            not allowed_course_ids
            or request.course_id in allowed_course_ids
        )

        mask.append(tenant_allowed and course_allowed)

    return mask