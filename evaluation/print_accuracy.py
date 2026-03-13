#!/usr/bin/env python3
"""
Print accuracy from judgment JSONL files.

Reads judgment files produced by llm_judge.py (boxed extraction + grading)
and prints a formatted table with per-dataset accuracy.

Usage:
    python evaluation/print_accuracy.py --judgment_dir ./evaluation/judgments/vision_r1_full/Qwen2.5-VL-7B-Instruct
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def load_judgments(filepath: str) -> list[dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_accuracy(judgments: list[dict]) -> dict:
    n = len(judgments)
    if n == 0:
        return {"samples": 0, "llm_judge_acc": 0.0, "rule_based_acc": 0.0}

    llm_correct = sum(1 for j in judgments if j.get("judge_verdict") == "correct")
    rule_scores = [j.get("original_score") for j in judgments if j.get("original_score") is not None]
    rule_correct = sum(1 for s in rule_scores if s >= 0.5)
    rule_total = len(rule_scores) if rule_scores else n

    return {
        "samples": n,
        "llm_judge_acc": llm_correct / n * 100,
        "rule_based_acc": rule_correct / rule_total * 100 if rule_total > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Print LLM Judge accuracy table")
    parser.add_argument("--judgment_dir", type=str, required=True,
                        help="Directory containing judgment JSONL files")
    parser.add_argument("--save_summary", action="store_true", default=True,
                        help="Save summary JSON alongside judgments")
    args = parser.parse_args()

    pattern = os.path.join(args.judgment_dir, "**", "*.jsonl")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        pattern = os.path.join(args.judgment_dir, "*.jsonl")
        files = sorted(glob.glob(pattern))

    if not files:
        print(f"No judgment files found in {args.judgment_dir}")
        sys.exit(1)

    # Compute per-dataset stats
    results = []
    for f in files:
        dataset_name = Path(f).stem
        judgments = load_judgments(f)
        stats = compute_accuracy(judgments)
        stats["dataset"] = dataset_name
        results.append(stats)

    # Print table
    col_ds = max(len("Dataset"), max(len(r["dataset"]) for r in results)) + 2
    col_n = max(len("Samples"), 7)
    col_llm = max(len("LLM-Judge Acc"), 13)
    col_rule = max(len("Rule-Based Acc"), 14)

    header = f"  {'Dataset':<{col_ds}} {'Samples':>{col_n}} {'LLM-Judge Acc':>{col_llm}} {'Rule-Based Acc':>{col_rule}}"
    separator = "  " + "─" * (col_ds + col_n + col_llm + col_rule + 6)

    print()
    print(header)
    print(separator)

    total_samples = 0
    total_llm_correct = 0
    total_rule_correct = 0
    total_rule_samples = 0

    for r in results:
        ds = r["dataset"]
        n = r["samples"]
        llm_acc = r["llm_judge_acc"]
        rule_acc = r["rule_based_acc"]

        print(f"  {ds:<{col_ds}} {n:>{col_n}} {llm_acc:>{col_llm - 1}.1f}% {rule_acc:>{col_rule - 1}.1f}%")

        total_samples += n
        total_llm_correct += round(llm_acc * n / 100)
        total_rule_correct += round(rule_acc * n / 100)
        total_rule_samples += n

    print(separator)

    avg_llm = total_llm_correct / total_samples * 100 if total_samples > 0 else 0.0
    avg_rule = total_rule_correct / total_rule_samples * 100 if total_rule_samples > 0 else 0.0
    print(f"  {'Average':<{col_ds}} {total_samples:>{col_n}} {avg_llm:>{col_llm - 1}.1f}% {avg_rule:>{col_rule - 1}.1f}%")
    print()

    # Save summary
    if args.save_summary:
        summary = {
            "datasets": results,
            "total_samples": total_samples,
            "average_llm_judge_acc": avg_llm,
            "average_rule_based_acc": avg_rule,
        }
        summary_path = os.path.join(args.judgment_dir, "accuracy_summary.json")
        os.makedirs(args.judgment_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Summary saved to: {summary_path}")
        print()


if __name__ == "__main__":
    main()
