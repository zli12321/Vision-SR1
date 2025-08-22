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
from typing import List, Dict, Any, Optional
from mathruler.grader import extract_boxed_content, grade_answer
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep


'''
This reward function uses an external LLM to reward visual grounding
'''

VLLM_IP = "YOUR VLLM API ADDRESS"
VLLM_API_KEY = "YOUR VLLM API KEY"
VLLM_MODEL = "YOUR DEPLOYED VLLM MODEL"

client = OpenAI(
    base_url=VLLM_IP,  # your vLLM server
    api_key=VLLM_API_KEY,    # if you set --api-key when launching
)

ABS_Verify_Prompt = '''Text description: {Description}\nQuestion: {Question}\nYou are provided a text description of a problem and a question. Determine the answer to the question based on the text description. First provide an internal step-by-step reasoning within <think> </think> tags, then provide a single word or phrase answer in \\boxed{}.'''

def chat_batch(
    client,
    all_message_batches: List[List[Dict[str, str]]],
    *,
    model: str = VLLM_MODEL,
    max_workers: int = 8,
    retries: int = 2,
    backoff: float = 0.5,
    timeout: Optional[float] = None,
) -> List[str]:
    """
    Send many chat requests in parallel and return replies as a list of strings,
    preserving the order of `all_message_batches`.
    """

    def _chat_once_with_retry(messages: List[Dict[str, str]]) -> str:
        last_err: Optional[BaseException] = None
        for attempt in range(retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=timeout,
                )
                # Different SDKs expose content slightly differently; handle common cases.
                choice = resp.choices[0]
                if hasattr(choice, "message") and getattr(choice.message, "content", None) is not None:
                    return choice.message.content
                if hasattr(choice, "text") and choice.text is not None:
                    return choice.text
                # Fallback to stringifying the choice if structure is unexpected.
                return str(choice)
            except Exception as e:
                last_err = e
                if attempt < retries:
                    sleep(backoff * (2 ** attempt))
        return f"Error: {last_err!r}"

    results: List[Optional[str]] = [None] * len(all_message_batches)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_chat_once_with_retry, batch): i
            for i, batch in enumerate(all_message_batches)
        }
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()

    # mypy-friendly cast: no Nones remain at this point
    return [r if r is not None else "Error: Unknown failure" for r in results]

def extract_MCQ(text: str) -> Optional[str]:
    """
    Return everything after the first occurrence of the literal token 'Answer:'.

    Examples
    --------
    extract_answer("Question: 2+2=? Answer: 4")
    '4'
    extract_answer("…\nAnswer:   The capital is Paris.\nReasoning: …")
    'The capital is Paris.'
    extract_answer("No answer tag here")
    None
    """
    m = re.search(r"Answer:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        # grab capture group 1, then split on the *first* newline
        # so we ignore anything after the answer line
        return m.group(1).splitlines()[0].strip()
    return text


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
    extracted_descriptions = [extract_description(ele) for ele in predicts]
    batch_messages = []
    
    for index in range(len(extracted_descriptions)):
        curr_msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": ABS_Verify_Prompt.replace('{Description}', extracted_descriptions[index]).replace('{Question}', questions[index])}
        ]
        batch_messages.append(curr_msg)
        
    # batched_description_outputs = chat_batch(client, batch_messages)
    batched_description_outputs = extracted_descriptions
    
    for predict, ground_truth, question, desc_ans in zip(
        predicts, ground_truths, questions, batched_description_outputs
    ):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        
        
        try:
            description_format_score = format_reward(desc_ans)
        except:
            description_format_score = 0
        
        try:  
            description_reward = accuracy_reward(desc_ans, ground_truth)
        except:
            description_reward = 0

        
        scores.append(
            {
                # "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                # "overall": (1 - format_weight) * accuracy_score + format_weight * format_score + (1 - format_weight) * description_reward + format_weight * description_format_score,
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score + description_reward,
                "format": format_score,
                "accuracy": accuracy_score,
                "description_format": description_format_score,
                "description_accuracy":  description_reward
            }
        )
    
    print('-'*10)
    print("Regular generated answers: ", predicts[0])
    print('*'*10)
    print('Description generated answers example: ', description_answers[0])
    print('Ground Truth answer: ', ground_truths[0])
    print('Description Reward: ', scores[0]["description_accuracy"])
    print('Regular answer reward: ', scores[0]["accuracy"])
    print('-'*10)

    return scores
