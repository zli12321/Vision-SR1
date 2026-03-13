"""
Evaluation reward function for EasyR1.

Scores responses on accuracy (boxed answer vs ground truth) and format
compliance. Works for both CoT (<think>...\boxed{}) and See-Think
(<description>...<think>...\boxed{}) formats.
"""

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

REWARD_NAME = "eval_accuracy"
REWARD_TYPE = "sequential"


def cot_format_reward(response: str) -> float:
    pattern = re.compile(
        r"<think>.*?</think>\s*"
        r"\\boxed\{.*?\}\s*$",
        re.DOTALL,
    )
    return 1.0 if pattern.fullmatch(response.strip()) else 0.0


def see_think_format_reward(response: str) -> float:
    pattern = re.compile(
        r"^\s*<description>.*?</description>\s*"
        r"<think>.*?</think>\s*"
        r"\\boxed\{.*?\}\s*$",
        re.DOTALL,
    )
    return 1.0 if pattern.fullmatch(response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        answer = extract_boxed_content(response)
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    except Exception:
        return 0.0


def compute_score(reward_input: dict[str, Any]) -> dict[str, float]:
    response = reward_input["response"]
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
    ground_truth = reward_input["ground_truth"]

    acc = accuracy_reward(response, ground_truth)
    cot_fmt = cot_format_reward(response)
    st_fmt = see_think_format_reward(response)
    fmt = max(cot_fmt, st_fmt)

    return {
        "overall": acc,
        "accuracy": acc,
        "format": fmt,
    }
