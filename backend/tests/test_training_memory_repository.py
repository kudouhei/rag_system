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
    ExplanationArtifactStatusConflict,
    ExplanationPublicationConflict,
)

from app.training.lifecycle import (
    ExplanationStatus,
)
from app.training.review_events import (
    ExplanationReviewEvent,
    ReviewAction,
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

    async def apply_action(
        self,
        artifact: TrainingExplanationArtifact,
        *,
        action: ReviewAction,
        target: ExplanationStatus,
        comment: str | None = None,
    ) -> TrainingExplanationArtifact:
        event = ExplanationReviewEvent(
            event_id=uuid4(),
            artifact_id=artifact.artifact_id,
            tenant_id=artifact.tenant_id,
            action=action,
            from_status=artifact.status,
            to_status=target,
            actor_id="admin-001",
            actor_role="administrator",
            comment=comment,
            created_at=datetime.now(timezone.utc),
        )

        return await self.repository.apply_review_event(
            tenant_id=artifact.tenant_id,
            expected_revision=artifact.revision,
            event=event,
        )

    async def test_approval_saves_state_and_event(self) -> None:
        await self.repository.add(self.artifact)

        updated = await self.apply_action(
            self.artifact,
            action=ReviewAction.APPROVE,
            target=ExplanationStatus.APPROVED,
        )

        saved = await self.repository.get_by_id(
            tenant_id=self.artifact.tenant_id,
            artifact_id=self.artifact.artifact_id,
        )
        events = await self.repository.list_review_events(
            tenant_id=self.artifact.tenant_id,
            artifact_id=self.artifact.artifact_id,
        )

        self.assertEqual(saved, updated)
        self.assertEqual(
            updated.status,
            ExplanationStatus.APPROVED,
        )
        self.assertEqual(updated.revision, 1)
        self.assertEqual(
            updated.explanation_version,
            self.artifact.explanation_version,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0].action,
            ReviewAction.APPROVE,
        )
        self.assertEqual(events[0].actor_id, "admin-001")
        self.assertEqual(
            events[0].artifact_id,
            self.artifact.artifact_id,
        )

    async def test_stale_revision_leaves_state_and_history_unchanged(
        self,
    ) -> None:
        await self.repository.add(self.artifact)

        approved = await self.apply_action(
            self.artifact,
            action=ReviewAction.APPROVE,
            target=ExplanationStatus.APPROVED,
        )
        returned = await self.apply_action(
            approved,
            action=ReviewAction.RETURN_TO_REVIEW,
            target=ExplanationStatus.REVIEW_REQUIRED,
            comment="Another review is needed.",
        )

        history_before = (
            await self.repository.list_review_events(
                tenant_id=self.artifact.tenant_id,
                artifact_id=self.artifact.artifact_id,
            )
        )

        # The original snapshot still has revision 0.
        with self.assertRaises(
            ExplanationArtifactStatusConflict
        ):
            await self.apply_action(
                self.artifact,
                action=ReviewAction.APPROVE,
                target=ExplanationStatus.APPROVED,
            )

        saved = await self.repository.get_by_id(
            tenant_id=self.artifact.tenant_id,
            artifact_id=self.artifact.artifact_id,
        )
        history_after = (
            await self.repository.list_review_events(
                tenant_id=self.artifact.tenant_id,
                artifact_id=self.artifact.artifact_id,
            )
        )

        self.assertEqual(saved, returned)
        self.assertEqual(saved.revision, 2)
        self.assertEqual(
            saved.status,
            ExplanationStatus.REVIEW_REQUIRED,
        )
        self.assertEqual(history_after, history_before)

    async def test_second_publication_is_rejected_without_changes(
        self,
    ) -> None:
        await self.repository.add(self.artifact)

        approved = await self.apply_action(
            self.artifact,
            action=ReviewAction.APPROVE,
            target=ExplanationStatus.APPROVED,
        )
        published = await self.apply_action(
            approved,
            action=ReviewAction.PUBLISH,
            target=ExplanationStatus.PUBLISHED,
        )

        second_payload = self.artifact.model_dump()
        second_payload.update(
            {
                "artifact_id": uuid4(),
                "explanation_version": "sha256:explanation-v2",
                "summary": "An alternative explanation.",
            }
        )
        second = TrainingExplanationArtifact.model_validate(
            second_payload
        )
        await self.repository.add(second)

        second_approved = await self.apply_action(
            second,
            action=ReviewAction.APPROVE,
            target=ExplanationStatus.APPROVED,
        )
        history_before = (
            await self.repository.list_review_events(
                tenant_id=second.tenant_id,
                artifact_id=second.artifact_id,
            )
        )

        with self.assertRaises(
            ExplanationPublicationConflict
        ):
            await self.apply_action(
                second_approved,
                action=ReviewAction.PUBLISH,
                target=ExplanationStatus.PUBLISHED,
            )

        second_saved = await self.repository.get_by_id(
            tenant_id=second.tenant_id,
            artifact_id=second.artifact_id,
        )
        first_saved = await self.repository.get_by_id(
            tenant_id=published.tenant_id,
            artifact_id=published.artifact_id,
        )
        history_after = (
            await self.repository.list_review_events(
                tenant_id=second.tenant_id,
                artifact_id=second.artifact_id,
            )
        )

        self.assertEqual(second_saved, second_approved)
        self.assertEqual(first_saved, published)
        self.assertEqual(history_after, history_before)


if __name__ == "__main__":
    unittest.main()