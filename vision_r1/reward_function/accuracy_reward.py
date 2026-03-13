"""
Vision-R1 accuracy-only reward function (baseline).

Scores the response based solely on whether the \\boxed{} answer matches
the ground truth.
"""

from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

REWARD_NAME = "vision_r1_accuracy_reward"
REWARD_TYPE = "sequential"


def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        answer = extract_boxed_content(response)
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    except Exception:
        return 0.0


def compute_score(reward_input: dict[str, Any]) -> dict[str, float]:
    response = reward_input["response"]
    ground_truth = reward_input["ground_truth"]

    acc = accuracy_reward(response, ground_truth)

    return {
        "overall": acc,
        "accuracy": acc,
    }
