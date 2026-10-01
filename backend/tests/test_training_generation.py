"""Core contract tests for controlled training generation."""
from __future__ import annotations

import json
import unittest

from unittest.mock import (
    AsyncMock,
    patch,
)

from app.training.evidence import (
    MaterializedTrainingEvidence,
)
from app.training.generation import (
    GeneratedTrainingContent,
    InvalidGeneratedTrainingContent,
    generate_training_content,
    parse_generated_training_content,
)
from app.training.schemas import (
    TrainingEvidence,
    TrainingExplanationRequest,
    TrainingQuestionOption,
)


class TrainingGenerationContractTests(
    unittest.IsolatedAsyncioTestCase
):
    def setUp(self) -> None:
        self.supporting_quote = (
            "Personal data shall be adequate, "
            "relevant and limited to what is necessary."
        )
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
                        excerpt=self.supporting_quote,
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

    def _generated_option(self, option_id: str, explanation: str, evidence_id: str = "E1", supporting_quote: str | None = None) -> dict:
        """Build one model-output fixture using the current contract."""

        return {
            "option_id": option_id,
            "explanation": explanation,
            "evidence_ids": [evidence_id],
            "claims": [
                {
                    "claim": (
                        "Personal data must be limited "
                        "to what is necessary."
                    ),
                    "evidence_id": evidence_id,
                    "supporting_quote": (
                        supporting_quote
                        if supporting_quote is not None
                        else self.supporting_quote
                    ),
                }
            ],
        }

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
                    self._generated_option(
                        "A",
                        (
                            "This is the data minimisation "
                            "principle."
                        ),
                    ),
                    self._generated_option(
                        "B",
                        (
                            "Collecting data merely because "
                            "it may be useful is inconsistent "
                            "with data minimisation."
                        ),
                    ),
                    self._generated_option(
                        "C",
                        (
                            "Storage limitation concerns how "
                            "long personal data is retained."
                        ),
                    ),
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
                    self._generated_option(
                        "A",
                        "Explanation A.",
                    ),
                    self._generated_option(
                        "B",
                        "Explanation B.",
                        evidence_id="E99",
                    ),
                    self._generated_option(
                        "C",
                        "Explanation C.",
                    ),
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
                    self._generated_option(
                        "A",
                        "Explanation A.",
                    ),
                    self._generated_option(
                        "B",
                        "Explanation B.",
                    ),
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

    def test_rejects_supporting_quote_not_in_evidence(
        self,
    ) -> None:
        raw_content = json.dumps(
            {
                "summary": "Contains a fabricated quotation.",
                "option_explanations": [
                    self._generated_option(
                        "A",
                        "Explanation A.",
                        supporting_quote=(
                            "This quotation does not exist "
                            "in the supplied evidence."
                        ),
                    ),
                    self._generated_option(
                        "B",
                        "Explanation B.",
                    ),
                    self._generated_option(
                        "C",
                        "Explanation C.",
                    ),
                ],
            }
        )

        with self.assertRaisesRegex(
            InvalidGeneratedTrainingContent,
            "supporting quote",
        ):
            parse_generated_training_content(
                raw_content,
                self.request,
                self.materialized,
            )

    @patch(
        "app.training.generation.get_active_llm_model",
        return_value="mock-training-model",
    )
    @patch(
        "app.training.generation.llm_structured_call",
        new_callable=AsyncMock,
    )
    async def test_generates_when_llm_output_is_valid(
        self,
        mock_llm_structured_call,
        mock_get_active_llm_model,
    ) -> None:
        mock_llm_structured_call.return_value = json.dumps(
            {
                "summary": (
                    "The learner selected an incorrect answer."
                ),
                "option_explanations": [
                    self._generated_option(
                        "A",
                        (
                            "Option A expresses the data "
                            "minimisation principle."
                        ),
                    ),
                    self._generated_option(
                        "B",
                        (
                            "Option B permits unnecessary "
                            "collection and is incorrect."
                        ),
                    ),
                    self._generated_option(
                        "C",
                        (
                            "Option C concerns retention time, "
                            "not collection necessity."
                        ),
                    ),
                ],
            }
        )

        attempt = await generate_training_content(
            request=self.request,
            materialized=self.materialized,
            learner_result="incorrect",
        )

        self.assertEqual(
            attempt.status,
            "generated",
        )

        self.assertEqual(
            attempt.model,
            "mock-training-model",
        )

        self.assertIsNotNone(
            attempt.content
        )

        self.assertEqual(
            attempt.content.summary,
            (
                "The learner selected an incorrect answer."
            ),
        )

        mock_get_active_llm_model.assert_called_once_with()

        mock_llm_structured_call.assert_awaited_once()

        structured_call_kwargs = (
            mock_llm_structured_call.await_args.kwargs
        )

        self.assertIs(
            structured_call_kwargs["response_model"],
            GeneratedTrainingContent,
        )

if __name__ == "__main__":
    unittest.main()