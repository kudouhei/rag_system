"""Core tests for explanation lifecycle governance."""
from __future__ import annotations

import unittest

from app.training.lifecycle import (
    ExplanationStatus,
    InvalidExplanationStatusTransition,
    allowed_next_statuses,
    validate_status_transition,
)


class TrainingLifecycleTests(unittest.TestCase):
    def test_reviewed_content_can_be_approved(
        self,
    ) -> None:
        validate_status_transition(
            current=ExplanationStatus.REVIEW_REQUIRED,
            target=ExplanationStatus.APPROVED,
        )

    def test_content_cannot_bypass_approval(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            InvalidExplanationStatusTransition,
            "review_required.*published",
        ):
            validate_status_transition(
                current=(
                    ExplanationStatus.REVIEW_REQUIRED
                ),
                target=ExplanationStatus.PUBLISHED,
            )

    def test_published_content_can_only_be_retired(
        self,
    ) -> None:
        self.assertEqual(
            allowed_next_statuses(
                ExplanationStatus.PUBLISHED
            ),
            frozenset(
                {
                    ExplanationStatus.RETIRED,
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()