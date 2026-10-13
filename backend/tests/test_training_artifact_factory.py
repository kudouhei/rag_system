"""Core tests for runtime-to-artifact conversion."""
from __future__ import annotations

from datetime import datetime, timezone
import unittest
from uuid import UUID

from unittest.mock import AsyncMock, patch

from app.training.authoring_service import (
    ExplanationAuthoringService,
    InsufficientExplanationEvidence,
)
from app.training.memory_repository import (
    InMemoryExplanationArtifactRepository,
)
from app.training.service import (
    TrainingExplanationBuildResult,
)

from app.training.artifact_factory import (
    build_review_artifact,
)
from app.training.generation import (
    GENERATION_PROMPT_VERSION,
)
from app.training.grounding import (
    GROUNDING_PROMPT_VERSION,
)
from app.training.lifecycle import (
    ExplanationStatus,
)
from app.training.schemas import (
    TrainingExplanationRequest,
    TrainingExplanationResponse,
)


class TrainingArtifactFactoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.request = TrainingExplanationRequest(
            tenant_id="bank-a",
            course_id="gdpr-foundations",
            question_id=(
                "gdpr-art5-data-minimisation-001"
            ),
            question=(
                "Which option describes data minimisation?"
            ),
            options=[
                {
                    "option_id": "A",
                    "text": "Collect only necessary data.",
                },
                {
                    "option_id": "B",
                    "text": "Collect all potentially useful data.",
                },
            ],
            correct_option_ids=["A"],
            selected_option_ids=["B"],
            jurisdiction="LU",
            language="en",
        )

        self.response = TrainingExplanationResponse(
            trace_id="trace-001",
            question_id=self.request.question_id,
            question_fingerprint="sha256:question-v1",
            explanation_version=(
                "sha256:explanation-v1"
            ),
            status="review_required",
            learner_result="incorrect",
            summary=(
                "Article 5 establishes data minimisation."
            ),
            option_explanations=[
                {
                    "option_id": "A",
                    "is_correct": True,
                    "selected_by_learner": False,
                    "explanation": (
                        "Option A correctly limits collection "
                        "to necessary personal data."
                    ),
                    "evidence_ids": ["E1"],
                },
                {
                    "option_id": "B",
                    "is_correct": False,
                    "selected_by_learner": True,
                    "explanation": (
                        "Option B conflicts with data "
                        "minimisation."
                    ),
                    "evidence_ids": ["E1"],
                },
            ],
            evidence=[
                {
                    "evidence_id": "E1",
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
                    "relevance_score": 0.9352,
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
            evidence_relevance_score=0.9352,
            grounding_score=1.0,
            corpus_version="sha256:corpus-v1",
            generator_model="gpt-5-mini",
            generated_at=datetime.now(timezone.utc),
        )

    def build_artifact(self):
        return build_review_artifact(
            request=self.request,
            response=self.response,
            grounding_model="gpt-5-mini",
            created_by="system",
            artifact_id=UUID(
                "de305d54-75b4-431b-adb2-eb6b9e546014"
            ),
        )

    def test_factory_removes_learner_specific_data(
        self,
    ) -> None:
        artifact = self.build_artifact()
        payload = artifact.model_dump(
            mode="json"
        )

        self.assertNotIn(
            "learner_result",
            payload,
        )
        self.assertNotIn(
            "selected_option_ids",
            payload,
        )

        for option in payload["option_explanations"]:
            self.assertNotIn(
                "selected_by_learner",
                option,
            )

        self.assertEqual(
            artifact.status,
            ExplanationStatus.REVIEW_REQUIRED,
        )

    def test_factory_maps_evidence_and_provenance(
        self,
    ) -> None:
        artifact = self.build_artifact()

        self.assertEqual(
            artifact.option_explanations[
                0
            ].evidence_chunk_ids,
            ["chunk:article-5-stable"],
        )
        self.assertEqual(
            artifact.evidence[0].chunk_id,
            "chunk:article-5-stable",
        )

        self.assertEqual(
            artifact.provenance.content_origin,
            "llm_generated",
        )
        self.assertEqual(
            artifact.provenance.generation_prompt_version,
            GENERATION_PROMPT_VERSION,
        )
        self.assertEqual(
            artifact.provenance.grounding_prompt_version,
            GROUNDING_PROMPT_VERSION,
        )
        self.assertEqual(
            artifact.provenance.source_trace_id,
            "trace-001",
        )

    async def test_authoring_saves_candidate_to_review_queue(
        self,
    ) -> None:
        repository = InMemoryExplanationArtifactRepository()
        service = ExplanationAuthoringService(repository)

        build_result = TrainingExplanationBuildResult(
            response=self.response,
            grounding_model="judge-test-deployment",
        )

        with patch(
            "app.training.authoring_service."
            "build_training_explanation_result",
            new_callable=AsyncMock,
            return_value=build_result,
        ) as pipeline:
            artifact = await service.create_candidate(
                request=self.request,
                created_by="author-001",
            )

        pipeline.assert_awaited_once()

        submitted_request = pipeline.await_args.args[0]
        self.assertEqual(
            submitted_request.selected_option_ids,
            [],
        )
        self.assertEqual(
            self.request.selected_option_ids,
            ["B"],
        )

        queue = await repository.list_by_status(
            tenant_id=self.request.tenant_id,
            course_id=self.request.course_id,
            status=ExplanationStatus.REVIEW_REQUIRED,
        )

        self.assertEqual(queue, [artifact])
        self.assertEqual(artifact.created_by, "author-001")
        self.assertEqual(
            artifact.provenance.grounding_model,
            "judge-test-deployment",
        )
        self.assertEqual(
            artifact.provenance.source_trace_id,
            self.response.trace_id,
        )
        self.assertEqual(
            artifact.option_explanations[0].evidence_chunk_ids,
            ["chunk:article-5-stable"],
        )

    async def test_insufficient_evidence_is_not_saved(
        self,
    ) -> None:
        repository = InMemoryExplanationArtifactRepository()
        service = ExplanationAuthoringService(repository)

        payload = self.response.model_dump()
        payload["status"] = "insufficient_evidence"
        payload["summary"] = "No sufficient evidence was found."
        payload["evidence"] = []
        payload["evidence_relevance_score"] = 0.0
        payload["grounding_score"] = None
        payload["generator_model"] = None

        for option in payload["option_explanations"]:
            option["evidence_ids"] = []

        insufficient_response = (
            TrainingExplanationResponse.model_validate(
                payload
            )
        )

        build_result = TrainingExplanationBuildResult(
            response=insufficient_response,
            grounding_model=None,
        )

        with patch(
            "app.training.authoring_service."
            "build_training_explanation_result",
            new_callable=AsyncMock,
            return_value=build_result,
        ):
            with self.assertRaises(
                InsufficientExplanationEvidence
            ):
                await service.create_candidate(
                    request=self.request,
                    created_by="author-001",
                )

        queue = await repository.list_by_status(
            tenant_id=self.request.tenant_id,
            status=ExplanationStatus.REVIEW_REQUIRED,
        )

        self.assertEqual(queue, [])

if __name__ == "__main__":
    unittest.main()