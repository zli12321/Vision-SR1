# from utils.math_utils import *
from utils.gpt_eval import *
from utils.gemini_eval import *
from utils.math_utils import *
from mathruler.grader import extract_boxed_content
import json
from typing import List, Dict, Union
from pathlib import Path
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.ERROR)
import json
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import argparse
from datasets import load_dataset

logging.getLogger().setLevel(logging.ERROR)

ONLY_FILE = "MLLM_test"

'''
Check for utils/gemini_eval.py for the evaluation function details and evaluation prompts.
Redefine the generate() function in utils/gemini_eval.py to place your LLM API key to generate responses.
'''


def read_jsonl(path: Path) -> List[Dict]:
    """Read a .jsonl file into a list of dicts."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}")
    return records


# ---------- argument parsing ----------
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score model outputs against a dataset with accuracy reward. If accuracy reward is 0, use LLM to judge correctness."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("./Raw-Outputs/7B-Vision-SR1"),
        help="Directory that contains the model-generated JSONL files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./Raw-Outputs/LLM-Eval-out/7B-Vision-SR1"),
        help="Where to write the scored JSONL files.",
    )
    return parser.parse_args()
# --------------------------------------


def process_file(
    jsonl_path: Path,
    output_dir: Path,
    problems: List[str],
    answers: List[str],
    position: int = 0,
) -> None:
    """Score one JSONL file and write the result next to it in output_dir."""
    records = read_jsonl(jsonl_path)
    out_path = output_dir / jsonl_path.name

    with out_path.open("w", encoding="utf-8") as fout, tqdm(
        total=len(records),
        desc=jsonl_path.name,
        position=position,
        leave=True,
    ) as pbar:
        for idx, rec in enumerate(records):
            question = problems[idx]
            gold_answer = answers[idx]
            model_ans = rec["response"]

            extracted = extract_boxed_content(model_ans)
            if extracted.lower() == "none":
                extracted = model_ans

            if accuracy_reward(model_ans, gold_answer) == 1:
                accuracy_output = "correct"
                accuracy_judgment = "correct"
            else:
                accuracy_output = generate(question, gold_answer, extracted)
                accuracy_judgment = extract_judgment(accuracy_output).lower()
                # Optional debug prints:
                # print("Question:", question)
                # print("Gold:", gold_answer)
                # print("Model:", extracted)
                # print("Accuracy output:", accuracy_output)

            # attach new fields
            rec["gold_answer"] = gold_answer
            rec["accuracy_output"] = accuracy_output
            rec["accuracy_judgment"] = accuracy_judgment

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pbar.update(1)

    print(f"[{jsonl_path.name}] Done, wrote {len(records)} records")


def main() -> None:
    args = get_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # dataset loading
    try:
        ds = load_dataset(f"zli12321/{ONLY_FILE}")
    except FileNotFoundError:
        ds = load_dataset(f"HuggingFaceH4/{ONLY_FILE}")

    answers = ds["test"]["answer"]
    problems = [p.replace("<image>", "") for p in ds["test"]["problem"]]

    target_file = input_dir / f"{ONLY_FILE}.jsonl"
    if not target_file.exists():
        raise FileNotFoundError(target_file)

    process_file(
        jsonl_path=target_file,
        output_dir=output_dir,
        problems=problems,
        answers=answers,
        position=0,
    )


if __name__ == "__main__":
    main()