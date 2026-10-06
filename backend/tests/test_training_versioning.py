"""Core tests for stable training-content versions."""
from __future__ import annotations

import unittest

from app.training.schemas import (
    TrainingEvidence,
    TrainingExplanationRequest,
    TrainingOptionExplanation,
)
from app.training.versioning import (
    compute_explanation_version,
    compute_question_fingerprint,
)


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
            question_id=(
                "gdpr-art5-data-minimisation-001"
            ),
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
            correct_option_ids=(
                correct_option_ids or ["A"]
            ),
            selected_option_ids=(
                selected_option_ids or []
            ),
            jurisdiction="LU",
            language="en",
        )

    @staticmethod
    def build_explanation_content(
        *,
        evidence_id: str = "E1",
        selected_option_id: str = "B",
        explanation_a: str = (
            "Data minimisation requires limiting personal "
            "data to what is necessary."
        ),
    ) -> tuple[
        list[TrainingOptionExplanation],
        list[TrainingEvidence],
    ]:
        option_explanations = [
            TrainingOptionExplanation(
                option_id="A",
                is_correct=True,
                selected_by_learner=(
                    selected_option_id == "A"
                ),
                explanation=explanation_a,
                evidence_ids=[evidence_id],
            ),
            TrainingOptionExplanation(
                option_id="B",
                is_correct=False,
                selected_by_learner=(
                    selected_option_id == "B"
                ),
                explanation=(
                    "Collecting data merely because it may "
                    "be useful later conflicts with data "
                    "minimisation."
                ),
                evidence_ids=[evidence_id],
            ),
        ]

        evidence = [
            TrainingEvidence(
                evidence_id=evidence_id,
                chunk_id="chunk:article-5-stable",
                source=(
                    "gdpr/eu-gdpr-article-05-en.md"
                ),
                title=(
                    "Article 5 - Principles relating to "
                    "processing of personal data"
                ),
                section="Article 5",
                excerpt=(
                    "Personal data shall be adequate, "
                    "relevant and limited to what is "
                    "necessary."
                ),
                relevance_score=0.9352,
                regulation_number=(
                    "Regulation (EU) 2016/679"
                ),
                issuing_authority=(
                    "European Parliament and Council "
                    "of the European Union"
                ),
                effective_date="2018-05-25",
            )
        ]

        return option_explanations, evidence

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

    def test_learner_selection_does_not_change_explanation_version(
        self,
    ) -> None:
        selected_a = self.build_explanation_content(
            selected_option_id="A",
        )
        selected_b = self.build_explanation_content(
            selected_option_id="B",
        )

        version_a = compute_explanation_version(
            question_fingerprint="sha256:question-v1",
            option_explanations=selected_a[0],
            evidence=selected_a[1],
            summary="GDPR Article 5 applies.",
        )
        version_b = compute_explanation_version(
            question_fingerprint="sha256:question-v1",
            option_explanations=selected_b[0],
            evidence=selected_b[1],
            summary="GDPR Article 5 applies.",
        )

        self.assertEqual(version_a, version_b)

    def test_temporary_evidence_id_does_not_change_version(
        self,
    ) -> None:
        evidence_e1 = self.build_explanation_content(
            evidence_id="E1",
        )
        evidence_e8 = self.build_explanation_content(
            evidence_id="E8",
        )

        version_e1 = compute_explanation_version(
            question_fingerprint="sha256:question-v1",
            option_explanations=evidence_e1[0],
            evidence=evidence_e1[1],
            summary="GDPR Article 5 applies.",
        )
        version_e8 = compute_explanation_version(
            question_fingerprint="sha256:question-v1",
            option_explanations=evidence_e8[0],
            evidence=evidence_e8[1],
            summary="GDPR Article 5 applies.",
        )

        self.assertEqual(version_e1, version_e8)

    def test_explanation_change_creates_new_version(
        self,
    ) -> None:
        original = self.build_explanation_content()
        changed = self.build_explanation_content(
            explanation_a=(
                "Article 5 requires personal data to be "
                "adequate, relevant and limited."
            ),
        )

        original_version = compute_explanation_version(
            question_fingerprint="sha256:question-v1",
            option_explanations=original[0],
            evidence=original[1],
            summary="GDPR Article 5 applies.",
        )
        changed_version = compute_explanation_version(
            question_fingerprint="sha256:question-v1",
            option_explanations=changed[0],
            evidence=changed[1],
            summary="GDPR Article 5 applies.",
        )

        self.assertNotEqual(
            original_version,
            changed_version,
        )


if __name__ == "__main__":
    unittest.main()