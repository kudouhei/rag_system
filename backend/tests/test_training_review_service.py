"""Core tests for review-service authorization."""
from __future__ import annotations

from datetime import datetime, timezone
import unittest
from uuid import uuid4

from app.training.artifacts import (
    TrainingExplanationArtifact,
)
from app.training.lifecycle import (
    ExplanationStatus,
)
from app.training.memory_repository import (
    InMemoryExplanationArtifactRepository,
)
from app.training.review_events import (
    ReviewAction,
)
from app.training.review_service import (
    ExplanationReviewPermissionDenied,
    ExplanationReviewService,
)


class TrainingReviewServiceTests(
    unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        self.repository = (
            InMemoryExplanationArtifactRepository()
        )
        self.service = ExplanationReviewService(
            self.repository
        )

        self.artifact = TrainingExplanationArtifact(
            artifact_id=uuid4(),
            tenant_id="bank-a",
            course_id="gdpr-foundations",
            question_id="q-001",
            question_fingerprint="sha256:question-v1",
            explanation_version="sha256:explanation-v1",
            jurisdiction="LU",
            language="en",
            summary="Article 5 requires data minimisation.",
            option_explanations=[
                {
                    "option_id": "A",
                    "is_correct": True,
                    "explanation": "Collect only necessary data.",
                    "evidence_chunk_ids": ["chunk:article-5"],
                },
                {
                    "option_id": "B",
                    "is_correct": False,
                    "explanation": (
                        "Collecting all potentially useful data "
                        "conflicts with data minimisation."
                    ),
                    "evidence_chunk_ids": ["chunk:article-5"],
                },
            ],
            evidence=[
                {
                    "chunk_id": "chunk:article-5",
                    "source": "gdpr/eu-gdpr-article-05-en.md",
                    "excerpt": (
                        "Personal data shall be adequate, "
                        "relevant and limited to what is necessary."
                    ),
                }
            ],
            provenance={
                "corpus_version": "sha256:corpus-v1",
                "content_origin": "llm_generated",
                "generation_prompt_version": (
                    "training-explanation-generation-v1"
                ),
                "generator_model": "gpt-5-mini",
                "source_trace_id": "trace-001",
            },
            created_by="system",
            created_at=datetime.now(timezone.utc),
        )

        await self.repository.add(self.artifact)

    async def approve_as_reviewer(
        self,
    ) -> TrainingExplanationArtifact:
        return await self.service.execute_action(
            tenant_id=self.artifact.tenant_id,
            artifact_id=self.artifact.artifact_id,
            expected_revision=0,
            action=ReviewAction.APPROVE,
            actor_id="reviewer-001",
            actor_role="reviewer",
        )

    async def test_reviewer_can_approve_with_audit_record(
        self,
    ) -> None:
        updated = await self.approve_as_reviewer()

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

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actor_id, "reviewer-001")
        self.assertEqual(events[0].actor_role, "reviewer")
        self.assertEqual(
            events[0].action,
            ReviewAction.APPROVE,
        )
        self.assertEqual(
            events[0].from_status,
            ExplanationStatus.REVIEW_REQUIRED,
        )
        self.assertEqual(
            events[0].to_status,
            ExplanationStatus.APPROVED,
        )

    async def test_reviewer_cannot_publish_approved_content(
        self,
    ) -> None:
        approved = await self.approve_as_reviewer()

        history_before = (
            await self.repository.list_review_events(
                tenant_id=self.artifact.tenant_id,
                artifact_id=self.artifact.artifact_id,
            )
        )

        with self.assertRaises(
            ExplanationReviewPermissionDenied
        ):
            await self.service.execute_action(
                tenant_id=approved.tenant_id,
                artifact_id=approved.artifact_id,
                expected_revision=approved.revision,
                action=ReviewAction.PUBLISH,
                actor_id="reviewer-001",
                actor_role="reviewer",
            )

        saved = await self.repository.get_by_id(
            tenant_id=approved.tenant_id,
            artifact_id=approved.artifact_id,
        )
        history_after = (
            await self.repository.list_review_events(
                tenant_id=approved.tenant_id,
                artifact_id=approved.artifact_id,
            )
        )

        self.assertEqual(saved, approved)
        self.assertEqual(history_after, history_before)


if __name__ == "__main__":
    unittest.main()