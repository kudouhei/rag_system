"""Core tests for stable training-question versions."""
from __future__ import annotations

import unittest

from app.training.schemas import TrainingExplanationRequest
from app.training.versioning import compute_question_fingerprint


class TrainingVersioningTests(unittest.TestCase):
    @staticmethod
    def build_request(
        *,
        selected_option_ids: list[str] | None = None,
        correct_option_ids: list[str] | None = None,
    ) -> TrainingExplanationRequest:
        return TrainingExplanationRequest(
            tenant_id="bank-a",
            course_id="gdpr-foundations",
            question_id="gdpr-art5-data-minimisation-001",
            question=(
                "Which option best describes the GDPR "
                "data minimisation principle?"
            ),
            options=[
                {
                    "option_id": "A",
                    "text": (
                        "Collect only personal data that is "
                        "necessary for the stated purpose."
                    ),
                },
                {
                    "option_id": "B",
                    "text": (
                        "Collect as much personal data as may "
                        "become useful later."
                    ),
                },
            ],
            correct_option_ids=correct_option_ids or ["A"],
            selected_option_ids=selected_option_ids or [],
            jurisdiction="LU",
            language="en",
        )

    def test_learner_selection_does_not_change_fingerprint(
        self,
    ) -> None:
        selected_a = self.build_request(
            selected_option_ids=["A"],
        )
        selected_b = self.build_request(
            selected_option_ids=["B"],
        )

        self.assertEqual(
            compute_question_fingerprint(selected_a),
            compute_question_fingerprint(selected_b),
        )

    def test_answer_key_change_creates_new_fingerprint(
        self,
    ) -> None:
        original = self.build_request(
            correct_option_ids=["A"],
        )
        changed_answer = self.build_request(
            correct_option_ids=["B"],
        )

        self.assertNotEqual(
            compute_question_fingerprint(original),
            compute_question_fingerprint(changed_answer),
        )


if __name__ == "__main__":
    unittest.main()