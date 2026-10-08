"""Core tests for persistent explanation artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import unittest

from pydantic import ValidationError

from app.training.artifacts import (
    TrainingExplanationArtifact,
)
from app.training.lifecycle import (
    ExplanationStatus,
)


class TrainingArtifactTests(unittest.TestCase):
    @staticmethod
    def valid_artifact_payload() -> dict:
        return {
            "artifact_id": (
                "de305d54-75b4-431b-adb2-eb6b9e546014"
            ),
            "tenant_id": "bank-a",
            "course_id": "gdpr-foundations",
            "question_id": (
                "gdpr-art5-data-minimisation-001"
            ),
            "question_fingerprint": (
                "sha256:question-v1"
            ),
            "explanation_version": (
                "sha256:explanation-v1"
            ),
            "jurisdiction": "LU",
            "language": "en",
            "summary": (
                "Article 5 establishes the data "
                "minimisation principle."
            ),
            "option_explanations": [
                {
                    "option_id": "A",
                    "is_correct": True,
                    "explanation": (
                        "Option A correctly describes "
                        "data minimisation."
                    ),
                    "evidence_chunk_ids": [
                        "chunk:article-5-stable"
                    ],
                },
                {
                    "option_id": "B",
                    "is_correct": False,
                    "explanation": (
                        "Option B conflicts with the "
                        "data minimisation principle."
                    ),
                    "evidence_chunk_ids": [
                        "chunk:article-5-stable"
                    ],
                },
            ],
            "evidence": [
                {
                    "chunk_id": "chunk:article-5-stable",
                    "source": (
                        "gdpr/"
                        "eu-gdpr-article-05-en.md"
                    ),
                    "title": "GDPR Article 5",
                    "section": "Article 5",
                    "excerpt": (
                        "Personal data shall be adequate, "
                        "relevant and limited to what is "
                        "necessary."
                    ),
                    "regulation_number": (
                        "Regulation (EU) 2016/679"
                    ),
                    "issuing_authority": (
                        "European Parliament and Council "
                        "of the European Union"
                    ),
                    "effective_date": "2018-05-25",
                }
            ],
            "provenance": {
                "corpus_version": "sha256:corpus-v1",
                "content_origin": "llm_generated",
                "generation_prompt_version": (
                    "training-explanation-generation-v1"
                ),
                "generator_model": "gpt-5-mini",
                "grounding_prompt_version": (
                    "grounding-entailment-v1"
                ),
                "grounding_model": "gpt-5-mini",
                "grounding_score": 1.0,
                "source_trace_id": "trace-001",
            },
            "created_by": "system",
            "created_at": datetime.now(timezone.utc),
        }

    def test_valid_artifact_starts_in_review(
        self,
    ) -> None:
        artifact = TrainingExplanationArtifact.model_validate(
            self.valid_artifact_payload()
        )

        self.assertEqual(
            artifact.status,
            ExplanationStatus.REVIEW_REQUIRED,
        )

    def test_learner_specific_fields_are_rejected(
        self,
    ) -> None:
        learner_fields = (
            ("learner_result", "incorrect"),
            ("selected_option_ids", ["B"]),
        )

        for field_name, field_value in learner_fields:
            with self.subTest(field_name=field_name):
                payload = self.valid_artifact_payload()
                payload[field_name] = field_value

                with self.assertRaisesRegex(
                    ValidationError,
                    field_name,
                ):
                    TrainingExplanationArtifact.model_validate(
                        payload
                    )

    def test_unknown_evidence_chunk_is_rejected(
        self,
    ) -> None:
        payload = self.valid_artifact_payload()

        payload["option_explanations"][0][
            "evidence_chunk_ids"
        ] = ["chunk:missing"]

        with self.assertRaisesRegex(
            ValidationError,
            "references unknown evidence chunks",
        ):
            TrainingExplanationArtifact.model_validate(
                payload
            )


if __name__ == "__main__":
    unittest.main()