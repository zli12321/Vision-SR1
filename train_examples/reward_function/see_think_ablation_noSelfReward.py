# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict, List, Optional
from mathruler.grader import extract_boxed_content, grade_answer


'''
This reward function does not use self reward as visual signals. It only uses the final answer's correctness as the final reward.
'''

def format_reward(predict: str) -> float:
    pattern = re.compile(
        r"^\s*<description>.*?</description>\s*"    # the image description block
        r"<think>.*?</think>\s*"                   # the reasoning block
        r"\\boxed\{.*?\}\s*$",                     # the final answer
        re.DOTALL
    )
    return 1.0 if pattern.fullmatch(predict) else 0.0


def description_format_reward(predict: str) -> float:
    pattern = re.compile(
        r"<think>.*?</think>\s*"                   # the reasoning block
        r"\\boxed\{.*?\}\s*$",                     # the final answer
        re.DOTALL
    )
    return 1.0 if pattern.fullmatch(predict) else 0.0

def extract_description(predict: str) -> Optional[str]:
    """
    Extracts the content of the <answer>…</answer> block from `predict`.
    Returns the inner text (with leading/trailing whitespace stripped),
    or None if no <answer> tag is found.
    """
    match = re.search(r"<description>([\s\S]*?)</description>", predict, re.DOTALL)
    if not match:
        return predict
    return match.group(1).strip()


def extract_answer(predict: str) -> Optional[str]:
    """
    Extracts the content of the <answer>…</answer> block from `predict`.
    Returns the inner text (with leading/trailing whitespace stripped),
    or None if no <answer> tag is found.
    """
    match = re.search(r"<answer>([\s\S]*?)</answer>", predict, re.DOTALL)
    if not match:
        return predict
    return match.group(1).strip()



def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    # answer = extract_answer(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predicts: List[str], ground_truths: List[str], questions: List[str], description_answers: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    
    for predict, ground_truth, question, desc_ans in zip(
        predicts, ground_truths, questions, description_answers
    ):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        
        
        scores.append(
            {
                # "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score
            }
        )
    
    print('-'*10)
    print("Regular generated answers: ", predicts[0])
    print('*'*10)
    print('Ground Truth answer: ', ground_truths[0])
    print('Reward: ', scores[0]["accuracy"])
    print('-'*10)

    return scores
