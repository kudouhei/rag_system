"""Core tests for claim-level grounding scoring."""
from __future__ import annotations

import unittest

from app.training.generation import (
    GeneratedTrainingContent,
)
from app.training.grounding import (
    GroundingJudgeOutput,
    InvalidGroundingOutput,
    calculate_grounding_score,
)


class TrainingGroundingScoreTests(
    unittest.TestCase
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


if __name__ == "__main__":
    unittest.main()