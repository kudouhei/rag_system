"""Core contract tests for controlled training generation."""
from __future__ import annotations

import json
import unittest

from app.training.evidence import (
    MaterializedTrainingEvidence,
)
from app.training.generation import (
    InvalidGeneratedTrainingContent,
    parse_generated_training_content,
)
from app.training.schemas import (
    TrainingEvidence,
    TrainingExplanationRequest,
    TrainingQuestionOption,
)


class TrainingGenerationContractTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.request = TrainingExplanationRequest(
            tenant_id="bank-a",
            course_id="gdpr-foundations",
            question_id=(
                "gdpr-art5-data-minimisation-001"
            ),
            question=(
                "Which GDPR principle requires personal data "
                "to be limited to what is necessary?"
            ),
            options=[
                TrainingQuestionOption(
                    option_id="A",
                    text="Data minimisation",
                ),
                TrainingQuestionOption(
                    option_id="B",
                    text=(
                        "Collect as much personal data "
                        "as might become useful later"
                    ),
                ),
                TrainingQuestionOption(
                    option_id="C",
                    text="Storage limitation",
                ),
            ],
            correct_option_ids=["A"],
            selected_option_ids=["B"],
            jurisdiction="LU",
            language="en",
            top_k=3,
        )

        self.materialized = (
            MaterializedTrainingEvidence(
                evidence=[
                    TrainingEvidence(
                        evidence_id="E1",
                        chunk_id="chunk:article-5",
                        source=(
                            "gdpr/"
                            "eu-gdpr-article-05-en.md"
                        ),
                        title="GDPR Article 5",
                        section="Article 5",
                        excerpt=(
                            "Personal data shall be adequate, "
                            "relevant and limited to what is "
                            "necessary."
                        ),
                        relevance_score=0.93,
                    )
                ],
                evidence_ids_by_option={
                    "A": ["E1"],
                    "B": ["E1"],
                    "C": ["E1"],
                },
            )
        )

    def test_accepts_valid_generated_content(
        self,
    ) -> None:
        raw_content = json.dumps(
            {
                "summary": (
                    "The learner selected an incorrect "
                    "answer."
                ),
                "option_explanations": [
                    {
                        "option_id": "A",
                        "explanation": (
                            "This is the data minimisation "
                            "principle."
                        ),
                        "evidence_ids": ["E1"],
                    },
                    {
                        "option_id": "B",
                        "explanation": (
                            "Collecting data merely because "
                            "it may be useful is inconsistent "
                            "with data minimisation."
                        ),
                        "evidence_ids": ["E1"],
                    },
                    {
                        "option_id": "C",
                        "explanation": (
                            "Storage limitation concerns how "
                            "long personal data is retained."
                        ),
                        "evidence_ids": ["E1"],
                    },
                ],
            }
        )

        generated = (
            parse_generated_training_content(
                raw_content,
                self.request,
                self.materialized,
            )
        )

        self.assertEqual(
            generated.summary,
            (
                "The learner selected an incorrect "
                "answer."
            ),
        )

        self.assertEqual(
            [
                option.option_id
                for option
                in generated.option_explanations
            ],
            ["A", "B", "C"],
        )

    def test_rejects_disallowed_evidence_id(
        self,
    ) -> None:
        raw_content = json.dumps(
            {
                "summary": "Invalid evidence reference.",
                "option_explanations": [
                    {
                        "option_id": "A",
                        "explanation": "Explanation A.",
                        "evidence_ids": ["E1"],
                    },
                    {
                        "option_id": "B",
                        "explanation": "Explanation B.",
                        "evidence_ids": ["E99"],
                    },
                    {
                        "option_id": "C",
                        "explanation": "Explanation C.",
                        "evidence_ids": ["E1"],
                    },
                ],
            }
        )

        with self.assertRaisesRegex(
            InvalidGeneratedTrainingContent,
            "disallowed evidence",
        ):
            parse_generated_training_content(
                raw_content,
                self.request,
                self.materialized,
            )

    def test_rejects_missing_option(
        self,
    ) -> None:
        raw_content = json.dumps(
            {
                "summary": "One option is missing.",
                "option_explanations": [
                    {
                        "option_id": "A",
                        "explanation": "Explanation A.",
                        "evidence_ids": ["E1"],
                    },
                    {
                        "option_id": "B",
                        "explanation": "Explanation B.",
                        "evidence_ids": ["E1"],
                    },
                ],
            }
        )

        with self.assertRaisesRegex(
            InvalidGeneratedTrainingContent,
            "option IDs do not match",
        ):
            parse_generated_training_content(
                raw_content,
                self.request,
                self.materialized,
            )


if __name__ == "__main__":
    unittest.main()