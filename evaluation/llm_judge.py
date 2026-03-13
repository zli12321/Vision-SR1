#!/usr/bin/env python3
"""
LLM Judge for evaluation responses.

1. Extracts the final answer from \\boxed{} in each model response (regex).
2. Sends the extracted answer + ground truth to an LLM judge (Qwen2.5-14B-Instruct
   via vLLM) to determine correctness.

This keeps prompts very short (just two short strings to compare), avoiding
context-window issues with long chain-of-thought reasoning.

Usage:
    python evaluation/llm_judge.py --response_dir ./evaluation/responses/vision_r1_full/Qwen2.5-VL-7B-Instruct

    python evaluation/llm_judge.py --response_file ./evaluation/responses/vision_r1_full/Qwen2.5-VL-7B-Instruct/mmstar.jsonl

    python evaluation/llm_judge.py --response_dir ... --judge_model Qwen/Qwen2.5-14B-Instruct --tp_size 4
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from mathruler.grader import extract_boxed_content


JUDGE_PROMPT_TEMPLATE = """\
You are an expert answer evaluator. Determine whether the model's extracted answer is correct by comparing it to the ground truth.

## Model's Extracted Answer:
{extracted_answer}

## Ground Truth Answer:
{ground_truth}

## Instructions:
Compare the two answers. Be flexible with formatting differences (e.g. "1/2" vs "0.5", "A" vs "(A)", equivalent expressions) but strict on mathematical/factual correctness.
Return your judgment as a JSON object with EXACTLY this format (no other text):

{{"verdict": "correct" or "incorrect", "reasoning": "<brief explanation>"}}
"""


def extract_boxed(response: str) -> str:
    """Extract content from \\boxed{} in the response."""
    try:
        answer = extract_boxed_content(response)
        return answer if answer else ""
    except Exception:
        return ""


def build_judge_messages(extracted_answer: str, ground_truth: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a precise answer evaluation assistant. Always respond with valid JSON only."},
        {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(
            extracted_answer=extracted_answer if extracted_answer else "(no answer extracted)",
            ground_truth=ground_truth,
        )},
    ]


def parse_judge_output(text: str) -> dict[str, Any]:
    """Extract the JSON verdict from the judge's response."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {"verdict": "incorrect", "reasoning": f"Failed to parse judge output: {text[:200]}"}


def collect_response_files(response_dir: str | None, response_file: str | None) -> list[str]:
    files = []
    if response_file:
        files.append(response_file)
    if response_dir:
        pattern = os.path.join(response_dir, "**", "*.jsonl")
        files.extend(sorted(glob.glob(pattern, recursive=True)))
        if not files:
            pattern = os.path.join(response_dir, "*.jsonl")
            files.extend(sorted(glob.glob(pattern)))
    return files


def response_path_to_judgment_path(response_path: str) -> str:
    """Mirror the response path structure under evaluation/judgments/."""
    p = Path(response_path)
    parts = list(p.parts)
    try:
        idx = parts.index("responses")
        parts[idx] = "judgments"
    except ValueError:
        parts.insert(-1, "judgments")
    return str(Path(*parts))


def load_responses(filepath: str) -> list[dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="LLM Judge: extract \\boxed{} then judge with LLM")
    parser.add_argument("--response_dir", type=str, default=None,
                        help="Directory containing response JSONL files")
    parser.add_argument("--response_file", type=str, default=None,
                        help="Single response JSONL file to judge")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                        help="Judge model name or path")
    parser.add_argument("--tp_size", type=int, default=None,
                        help="Tensor parallel size (default: auto-detect GPU count)")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max tokens for judge response")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for judge")
    args = parser.parse_args()

    if not args.response_dir and not args.response_file:
        parser.error("Provide --response_dir or --response_file")

    response_files = collect_response_files(args.response_dir, args.response_file)
    if not response_files:
        print("No response JSONL files found.")
        sys.exit(1)

    print(f"Found {len(response_files)} response file(s) to judge.")

    # --- Step 1: Extract \\boxed{} from all responses ---
    all_file_records: list[tuple[str, list[dict]]] = []
    total_samples = 0
    for resp_file in response_files:
        records = load_responses(resp_file)
        for rec in records:
            rec["_extracted_answer"] = extract_boxed(rec["response"])
        all_file_records.append((resp_file, records))
        total_samples += len(records)

    print(f"Extracted \\boxed{{}} answers from {total_samples} total samples.")

    # --- Step 2: Load LLM judge ---
    tp_size = args.tp_size or torch.cuda.device_count()
    if tp_size < 1:
        tp_size = 1
    print(f"Loading judge model: {args.judge_model} (tp={tp_size})")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # --- Step 3: Judge each file ---
    for resp_file, records in all_file_records:
        print(f"\n{'='*60}")
        print(f"Judging: {resp_file}")
        print(f"{'='*60}")

        if not records:
            print("  (empty file, skipping)")
            continue

        prompts = []
        for rec in records:
            messages = build_judge_messages(rec["_extracted_answer"], rec["ground_truth"])
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)

        print(f"  Running LLM judge on {len(prompts)} samples...")
        outputs = llm.generate(prompts, sampling_params)

        judgments = []
        correct_count = 0
        for rec, output in zip(records, outputs):
            judge_text = output.outputs[0].text
            parsed = parse_judge_output(judge_text)

            judgment = {
                "prompt": rec.get("prompt", ""),
                "response": rec["response"],
                "ground_truth": rec["ground_truth"],
                "extracted_answer": rec["_extracted_answer"],
                "judge_verdict": parsed.get("verdict", "incorrect"),
                "judge_reasoning": parsed.get("reasoning", ""),
                "original_score": rec.get("score", None),
            }
            judgments.append(judgment)
            if judgment["judge_verdict"] == "correct":
                correct_count += 1

        judgment_path = response_path_to_judgment_path(resp_file)
        os.makedirs(os.path.dirname(judgment_path), exist_ok=True)
        with open(judgment_path, "w", encoding="utf-8") as f:
            for j in judgments:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")

        acc = correct_count / len(judgments) * 100 if judgments else 0.0
        dataset_name = Path(resp_file).stem
        print(f"  {dataset_name}: {correct_count}/{len(judgments)} correct ({acc:.1f}%)")
        print(f"  Saved to: {judgment_path}")

    print("\nDone. All judgments saved.")


if __name__ == "__main__":
    main()
