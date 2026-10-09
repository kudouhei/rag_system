"""Core tests for explanation review events."""
from __future__ import annotations

from datetime import datetime, timezone
import unittest
from uuid import uuid4

from pydantic import ValidationError

from app.training.lifecycle import ExplanationStatus
from app.training.review_events import (
    ExplanationReviewEvent,
    ReviewAction,
)


class TrainingReviewEventTests(unittest.TestCase):
    @staticmethod
    def valid_approval_payload() -> dict:
        return {
            "event_id": uuid4(),
            "artifact_id": uuid4(),
            "tenant_id": "bank-a",
            "action": ReviewAction.APPROVE,
            "from_status": (
                ExplanationStatus.REVIEW_REQUIRED
            ),
            "to_status": ExplanationStatus.APPROVED,
            "actor_id": "reviewer-001",
            "actor_role": "reviewer",
            "comment": None,
            "created_at": datetime.now(timezone.utc),
        }

    def test_valid_approval_event_is_accepted(self) -> None:
        event = ExplanationReviewEvent.model_validate(
            self.valid_approval_payload()
        )

        self.assertEqual(
            event.action,
            ReviewAction.APPROVE,
        )
        self.assertEqual(
            event.to_status,
            ExplanationStatus.APPROVED,
        )

    def test_approve_cannot_record_publication(self) -> None:
        cases = [
            (
                ExplanationStatus.REVIEW_REQUIRED,
                "Cannot transition",
            ),
            (
                ExplanationStatus.APPROVED,
                "action 'approve' requires",
            ),
        ]

        for from_status, expected_error in cases:
            with self.subTest(from_status=from_status):
                payload = self.valid_approval_payload()
                payload["from_status"] = from_status
                payload["to_status"] = (
                    ExplanationStatus.PUBLISHED
                )

                with self.assertRaisesRegex(
                    ValidationError,
                    expected_error,
                ):
                    ExplanationReviewEvent.model_validate(
                        payload
                    )

    def test_rejection_requires_nonblank_comment(self) -> None:
        for comment in (None, "", "   "):
            with self.subTest(comment=comment):
                payload = self.valid_approval_payload()
                payload["action"] = ReviewAction.REJECT
                payload["to_status"] = (
                    ExplanationStatus.REJECTED
                )
                payload["comment"] = comment

                with self.assertRaisesRegex(
                    ValidationError,
                    "requires a comment",
                ):
                    ExplanationReviewEvent.model_validate(
                        payload
                    )


if __name__ == "__main__":
    unittest.main()