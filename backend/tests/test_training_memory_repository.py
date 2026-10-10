"""Core tests for in-memory artifact storage."""
from __future__ import annotations

from datetime import datetime, timezone
import unittest
from uuid import uuid4

from app.training.artifacts import (
    TrainingExplanationArtifact,
)
from app.training.memory_repository import (
    InMemoryExplanationArtifactRepository,
)
from app.training.repository import (
    ExplanationArtifactAlreadyExists,
)


class TrainingMemoryRepositoryTests(
    unittest.IsolatedAsyncioTestCase
):
    def setUp(self) -> None:
        self.repository = (
            InMemoryExplanationArtifactRepository()
        )

        self.artifact = (
            TrainingExplanationArtifact.model_validate(
                {
                    "artifact_id": uuid4(),
                    "tenant_id": "bank-a",
                    "course_id": "gdpr-foundations",
                    "question_id": "q-001",
                    "question_fingerprint": (
                        "sha256:question-v1"
                    ),
                    "explanation_version": (
                        "sha256:explanation-v1"
                    ),
                    "jurisdiction": "LU",
                    "language": "en",
                    "summary": (
                        "Article 5 requires data minimisation."
                    ),
                    "option_explanations": [
                        {
                            "option_id": "A",
                            "is_correct": True,
                            "explanation": (
                                "Collect only necessary data."
                            ),
                            "evidence_chunk_ids": [
                                "chunk:article-5"
                            ],
                        },
                        {
                            "option_id": "B",
                            "is_correct": False,
                            "explanation": (
                                "Collecting all potentially useful "
                                "data conflicts with minimisation."
                            ),
                            "evidence_chunk_ids": [
                                "chunk:article-5"
                            ],
                        },
                    ],
                    "evidence": [
                        {
                            "chunk_id": "chunk:article-5",
                            "source": (
                                "gdpr/eu-gdpr-article-05-en.md"
                            ),
                            "excerpt": (
                                "Personal data shall be adequate, "
                                "relevant and limited to what "
                                "is necessary."
                            ),
                        }
                    ],
                    "provenance": {
                        "corpus_version": "sha256:corpus-v1",
                        "content_origin": "llm_generated",
                        "generation_prompt_version": (
                            "training-explanation-generation-v1"
                        ),
                        "generator_model": "gpt-5-mini",
                        "source_trace_id": "trace-001",
                    },
                    "created_by": "system",
                    "created_at": datetime.now(timezone.utc),
                }
            )
        )

    async def test_other_tenant_cannot_read_artifact(
        self,
    ) -> None:
        await self.repository.add(self.artifact)

        own_artifact = await self.repository.get_by_id(
            tenant_id="bank-a",
            artifact_id=self.artifact.artifact_id,
        )
        self.assertIsNotNone(own_artifact)

        other_artifact = await self.repository.get_by_id(
            tenant_id="bank-b",
            artifact_id=self.artifact.artifact_id,
        )
        self.assertIsNone(other_artifact)

        other_version = (
            await self.repository.find_by_version(
                tenant_id="bank-b",
                course_id=self.artifact.course_id,
                question_id=self.artifact.question_id,
                explanation_version=(
                    self.artifact.explanation_version
                ),
            )
        )
        self.assertIsNone(other_version)

    async def test_same_content_with_new_id_is_rejected(
        self,
    ) -> None:
        await self.repository.add(self.artifact)

        duplicate = self.artifact.model_copy(
            update={"artifact_id": uuid4()},
            deep=True,
        )

        with self.assertRaises(
            ExplanationArtifactAlreadyExists
        ):
            await self.repository.add(duplicate)

        saved = await self.repository.find_by_version(
            tenant_id=self.artifact.tenant_id,
            course_id=self.artifact.course_id,
            question_id=self.artifact.question_id,
            explanation_version=(
                self.artifact.explanation_version
            ),
        )

        self.assertIsNotNone(saved)
        self.assertEqual(
            saved.artifact_id,
            self.artifact.artifact_id,
        )

    async def test_storage_isolated_from_caller_mutations(
        self,
    ) -> None:
        await self.repository.add(self.artifact)

        # Mutating the original must not change stored content.
        self.artifact.option_explanations.clear()

        first_read = await self.repository.get_by_id(
            tenant_id=self.artifact.tenant_id,
            artifact_id=self.artifact.artifact_id,
        )

        self.assertIsNotNone(first_read)
        self.assertEqual(
            len(first_read.option_explanations),
            2,
        )

        # Mutating a returned snapshot must also be harmless.
        first_read.evidence.clear()

        second_read = (
            await self.repository.find_by_version(
                tenant_id=self.artifact.tenant_id,
                course_id=self.artifact.course_id,
                question_id=self.artifact.question_id,
                explanation_version=(
                    self.artifact.explanation_version
                ),
            )
        )

        self.assertIsNotNone(second_read)
        self.assertEqual(
            len(second_read.option_explanations),
            2,
        )
        self.assertEqual(
            len(second_read.evidence),
            1,
        )


if __name__ == "__main__":
    unittest.main()