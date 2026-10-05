"""Core tests for claim-level grounding scoring."""
from __future__ import annotations
import json
import unittest

from unittest.mock import (
    AsyncMock,
    patch,
)

from app.training.generation import (
    GeneratedTrainingContent,
)
from app.training.grounding import (
    GroundingJudgeOutput,
    InvalidGroundingOutput,
    calculate_grounding_score,
    evaluate_training_grounding,
)


class TrainingGroundingScoreTests(
    unittest.IsolatedAsyncioTestCase
):
    def setUp(self) -> None:
        self.generated = (
            GeneratedTrainingContent.model_validate(
                {
                    "summary": (
                        "The learner selected an "
                        "incorrect answer."
                    ),
                    "option_explanations": [
                        {
                            "option_id": "A",
                            "explanation": (
                                "Option A is correct."
                            ),
                            "evidence_ids": ["E1"],
                            "claims": [
                                {
                                    "claim": (
                                        "The data must be "
                                        "adequate and relevant."
                                    ),
                                    "evidence_id": "E1",
                                    "supporting_quote": (
                                        "adequate, relevant and "
                                        "limited to what is necessary"
                                    ),
                                },
                                {
                                    "claim": (
                                        "The data must be limited "
                                        "to what is necessary."
                                    ),
                                    "evidence_id": "E1",
                                    "supporting_quote": (
                                        "limited to what is necessary"
                                    ),
                                },
                            ],
                        },
                        {
                            "option_id": "B",
                            "explanation": (
                                "Option B is incorrect."
                            ),
                            "evidence_ids": ["E1"],
                            "claims": [
                                {
                                    "claim": (
                                        "Collecting unnecessary "
                                        "data conflicts with the rule."
                                    ),
                                    "evidence_id": "E1",
                                    "supporting_quote": (
                                        "limited to what is necessary"
                                    ),
                                }
                            ],
                        },
                    ],
                }
            )
        )

    def test_calculates_supported_claim_ratio(
        self,
    ) -> None:
        judged = GroundingJudgeOutput.model_validate(
            {
                "verdicts": [
                    {
                        "option_id": "A",
                        "claim_index": 0,
                        "supported": True,
                        "reason": (
                            "The quote explicitly requires "
                            "adequacy and relevance."
                        ),
                    },
                    {
                        "option_id": "A",
                        "claim_index": 1,
                        "supported": True,
                        "reason": (
                            "The quote explicitly states "
                            "the necessity limit."
                        ),
                    },
                    {
                        "option_id": "B",
                        "claim_index": 0,
                        "supported": False,
                        "reason": (
                            "The supplied quote does not "
                            "explicitly discuss collection."
                        ),
                    },
                ]
            }
        )

        score = calculate_grounding_score(
            generated=self.generated,
            judged=judged,
        )

        self.assertEqual(
            score,
            0.6667,
        )

    def test_rejects_missing_claim_verdict(
        self,
    ) -> None:
        judged = GroundingJudgeOutput.model_validate(
            {
                "verdicts": [
                    {
                        "option_id": "A",
                        "claim_index": 0,
                        "supported": True,
                        "reason": "The quote supports the claim.",
                    },
                    {
                        "option_id": "B",
                        "claim_index": 0,
                        "supported": True,
                        "reason": "The quote supports the claim.",
                    },
                ]
            }
        )

        with self.assertRaisesRegex(
            InvalidGroundingOutput,
            "do not match generated claims",
        ):
            calculate_grounding_score(
                generated=self.generated,
                judged=judged,
            )

    @patch(
        "app.training.grounding.get_active_llm_model",
        return_value="mock-grounding-model",
    )
    @patch(
        "app.training.grounding.llm_structured_call",
        new_callable=AsyncMock,
    )
    async def test_evaluates_every_generated_claim(
        self,
        mock_llm_structured_call,
        mock_get_active_llm_model,
    ) -> None:
        mock_llm_structured_call.return_value = (
            json.dumps(
                {
                    "verdicts": [
                        {
                            "option_id": "A",
                            "claim_index": 0,
                            "supported": True,
                            "reason": (
                                "The quote supports adequacy "
                                "and relevance."
                            ),
                        },
                        {
                            "option_id": "A",
                            "claim_index": 1,
                            "supported": True,
                            "reason": (
                                "The quote states the "
                                "necessity limitation."
                            ),
                        },
                        {
                            "option_id": "B",
                            "claim_index": 0,
                            "supported": False,
                            "reason": (
                                "The quote does not directly "
                                "support the collection claim."
                            ),
                        },
                    ]
                }
            )
        )

        attempt = await evaluate_training_grounding(
            self.generated
        )

        self.assertEqual(
            attempt.status,
            "evaluated",
        )

        self.assertEqual(
            attempt.score,
            0.6667,
        )

        self.assertEqual(
            attempt.model,
            "mock-grounding-model",
        )

        mock_get_active_llm_model.assert_called_once_with()

        mock_llm_structured_call.assert_awaited_once()

        call_kwargs = (
            mock_llm_structured_call.await_args.kwargs
        )

        self.assertIs(
            call_kwargs["response_model"],
            GroundingJudgeOutput,
        )

        self.assertEqual(
            call_kwargs["max_tokens"],
            2500,
        )

        self.assertEqual(
            call_kwargs["reasoning_effort"],
            "low",
        )

        user_payload = json.loads(
            call_kwargs["messages"][1]["content"]
        )

        claim_keys = [
            (
                item["option_id"],
                item["claim_index"],
            )
            for item in user_payload["claims"]
        ]

        self.assertEqual(
            claim_keys,
            [
                ("A", 0),
                ("A", 1),
                ("B", 0),
            ],
        )

if __name__ == "__main__":
    unittest.main()