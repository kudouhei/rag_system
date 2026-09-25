from datetime import datetime, timezone
import unittest

from pydantic import ValidationError

from app.training.schemas import (
    TrainingExplanationRequest,
    TrainingExplanationResponse,
)


class TrainingSchemaTests(unittest.TestCase):
    def valid_request(self) -> dict:
        return {
            "tenant_id": "bank-a",
            "course_id": "gdpr-foundations",
            "question_id": "q-001",
            "question": "Which principle requires limiting collected data?",
            "options": [
                {
                    "option_id": "A",
                    "text": "Data minimisation",
                },
                {
                    "option_id": "B",
                    "text": "Unlimited retention",
                },
            ],
            "correct_option_ids": ["A"],
            "selected_option_ids": ["B"],
        }

    def valid_response(self) -> dict:
        return {
            "trace_id": "trace-001",
            "question_id": "q-001",
            "status": "grounded",
            "learner_result": "incorrect",
            "summary": "Data minimisation is the correct principle.",
            "option_explanations": [
                {
                    "option_id": "A",
                    "is_correct": True,
                    "selected_by_learner": False,
                    "explanation": (
                        "The principle requires personal data to be "
                        "adequate, relevant and limited."
                    ),
                    "evidence_ids": ["E1"],
                },
                {
                    "option_id": "B",
                    "is_correct": False,
                    "selected_by_learner": True,
                    "explanation": (
                        "Unlimited retention conflicts with storage "
                        "limitation requirements."
                    ),
                    "evidence_ids": ["E1"],
                },
            ],
            "evidence": [
                {
                    "evidence_id": "E1",
                    "chunk_id": "gdpr-article-5-chunk-1",
                    "source": "GDPR Article 5",
                    "section": "Article 5(1)(c)",
                    "excerpt": (
                        "Personal data shall be adequate, relevant and "
                        "limited to what is necessary."
                    ),
                    "relevance_score": 0.94,
                }
            ],
            "grounding_score": 0.90,
            "corpus_version": "test-v1",
            "generator_model": None,
            "generated_at": datetime.now(timezone.utc),
        }

    def test_valid_request_is_accepted(self):
        request = TrainingExplanationRequest(**self.valid_request())

        self.assertEqual(request.correct_option_ids, ["A"])
        self.assertEqual(request.selected_option_ids, ["B"])

    def test_unknown_correct_option_is_rejected(self):
        payload = self.valid_request()
        payload["correct_option_ids"] = ["C"]

        with self.assertRaisesRegex(
            ValidationError,
            "correct_option_ids contain unknown values",
        ):
            TrainingExplanationRequest(**payload)

    def test_unknown_evidence_reference_is_rejected(self):
        payload = self.valid_response()
        payload["option_explanations"][0]["evidence_ids"] = ["E99"]

        with self.assertRaisesRegex(
            ValidationError,
            "references unknown evidence",
        ):
            TrainingExplanationResponse(**payload)


if __name__ == "__main__":
    unittest.main()