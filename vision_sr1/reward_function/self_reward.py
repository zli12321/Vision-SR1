"""
Vision-SR1 Self-Reward function.

Scores both the main response (with image) and the description-only
second-hop response (without image). The overall reward combines:
  - format_score: whether the response follows <description>...<think>...\\boxed{} format
  - accuracy_score: whether the final \\boxed{} answer is correct
  - description_accuracy: whether the text-only second-hop answer is correct

This rewards the model for producing descriptions that are detailed enough
to answer the question without seeing the image.
"""

import re
from typing import Any, Optional

from mathruler.grader import extract_boxed_content, grade_answer

REWARD_NAME = "vision_sr1_self_reward"
REWARD_TYPE = "sequential"


def format_reward(response: str) -> float:
    """Check if response follows <description>...<think>...\\boxed{} format."""
    pattern = re.compile(
        r"^\s*<description>.*?</description>\s*"
        r"<think>.*?</think>\s*"
        r"\\boxed\{.*?\}\s*$",
        re.DOTALL,
    )
    return 1.0 if pattern.fullmatch(response) else 0.0


def description_format_reward(response: str) -> float:
    """Check if second-hop response follows <think>...\\boxed{} format."""
    pattern = re.compile(
        r"<think>.*?</think>\s*"
        r"\\boxed\{.*?\}\s*$",
        re.DOTALL,
    )
    return 1.0 if pattern.fullmatch(response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Check if the \\boxed{} answer matches the ground truth."""
    try:
        answer = extract_boxed_content(response)
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    except Exception:
        return 0.0


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.1) -> dict[str, float]:
    response = reward_input["response"]
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
    ground_truth = reward_input["ground_truth"]
    description_answer = reward_input.get("description_answer", "")

    fmt_score = format_reward(response)
    acc_score = accuracy_reward(response, ground_truth)

    desc_acc = 0.0
    desc_fmt = 0.0
    if description_answer:
        try:
            desc_acc = accuracy_reward(description_answer, ground_truth)
        except Exception:
            pass
        try:
            desc_fmt = description_format_reward(description_answer)
        except Exception:
            pass

    overall = (1 - format_weight) * acc_score + format_weight * fmt_score + desc_acc

    return {
        "overall": overall,
        "format": fmt_score,
        "accuracy": acc_score,
        "description_format": desc_fmt,
        "description_accuracy": desc_acc,
    }
