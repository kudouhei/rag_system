"""Run the labelled training-grounding evaluation set."""
from __future__ import annotations

import asyncio
import json

from pathlib import Path

from app.llm.client import init_llm_client
from app.training.generation import (
    GeneratedTrainingContent,
)
from app.training.grounding import (
    GROUNDING_PROMPT_VERSION,
    evaluate_training_grounding,
)


EVAL_FILE = Path(__file__).with_name(
    "training_grounding_cases.json"
)


def load_cases() -> list[dict]:
    """Load and minimally validate human-labelled cases."""

    payload = json.loads(
        EVAL_FILE.read_text(
            encoding="utf-8"
        )
    )

    cases = payload.get("cases")

    if not isinstance(cases, list):
        raise ValueError(
            "Eval dataset must contain a cases list"
        )

    if not 2 <= len(cases) <= 10:
        raise ValueError(
            "The initial eval runner supports "
            "between 2 and 10 cases"
        )

    case_ids = [
        case["case_id"]
        for case in cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "Eval dataset contains duplicate case IDs"
        )

    for case in cases:
        if not isinstance(
            case.get("expected_supported"),
            bool,
        ):
            raise ValueError(
                "expected_supported must be boolean "
                f"for case {case['case_id']!r}"
            )

    return cases


def build_eval_generation(
    cases: list[dict],
) -> GeneratedTrainingContent:
    """Convert eval cases into the production grounding input."""

    return GeneratedTrainingContent.model_validate(
        {
            "summary": (
                "Offline grounding evaluation batch."
            ),
            "option_explanations": [
                {
                    "option_id": case["case_id"],
                    "explanation": case["claim"],
                    "evidence_ids": ["EVAL"],
                    "claims": [
                        {
                            "claim": case["claim"],
                            "evidence_id": "EVAL",
                            "supporting_quote": (
                                case["supporting_quote"]
                            ),
                        }
                    ],
                }
                for case in cases
            ],
        }
    )


async def run_eval() -> int:
    """Execute the eval and print a compact confusion matrix."""

    cases = load_cases()

    generated = build_eval_generation(
        cases
    )

    init_llm_client()

    attempt = await evaluate_training_grounding(
        generated
    )

    if (
        attempt.status != "evaluated"
        or attempt.output is None
    ):
        print(
            "Evaluation failed:",
            attempt.status,
            attempt.reason,
        )
        return 2

    predictions = {
        verdict.option_id: verdict
        for verdict in attempt.output.verdicts
    }

    expected_case_ids = {
        case["case_id"]
        for case in cases
    }

    if set(predictions) != expected_case_ids:
        print(
            "Evaluation failed: verdict IDs "
            "do not match case IDs"
        )
        return 2

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    print(
        "model:",
        attempt.model,
    )
    print(
        "prompt_version:",
        GROUNDING_PROMPT_VERSION,
    )
    print()

    for case in cases:
        case_id = case["case_id"]
        expected = case["expected_supported"]
        verdict = predictions[case_id]
        predicted = verdict.supported

        if expected and predicted:
            true_positive += 1
        elif not expected and not predicted:
            true_negative += 1
        elif not expected and predicted:
            false_positive += 1
        else:
            false_negative += 1

        outcome = (
            "PASS"
            if expected == predicted
            else "FAIL"
        )

        print(
            f"{outcome} {case_id}"
        )
        print(
            "  expected:",
            expected,
        )
        print(
            "  predicted:",
            predicted,
        )
        print(
            "  reason:",
            verdict.reason,
        )

    total = len(cases)

    accuracy = (
        true_positive + true_negative
    ) / total

    false_positive_rate = (
        false_positive
        / (false_positive + true_negative)
        if false_positive + true_negative
        else 0.0
    )

    false_negative_rate = (
        false_negative
        / (false_negative + true_positive)
        if false_negative + true_positive
        else 0.0
    )

    print()
    print("confusion_matrix:")
    print(
        "  true_positive:",
        true_positive,
    )
    print(
        "  true_negative:",
        true_negative,
    )
    print(
        "  false_positive:",
        false_positive,
    )
    print(
        "  false_negative:",
        false_negative,
    )
    print(
        "accuracy:",
        round(accuracy, 4),
    )
    print(
        "false_positive_rate:",
        round(false_positive_rate, 4),
    )
    print(
        "false_negative_rate:",
        round(false_negative_rate, 4),
    )

    return (
        0
        if (
            false_positive == 0
            and false_negative == 0
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            run_eval()
        )
    )