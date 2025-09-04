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
from datasets import load_dataset

def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}")
    return records



# ONLY_FILE = "hallusionbench"
ONLY_FILE = "MLLM_test"

INPUT_DIR  = Path('./Raw-Outputs/7B-Vision-SR1')
OUTPUT_DIR = Path('./Raw-Outputs/7b_Vision-SR1-v2')

try:
    ds = load_dataset(f'zli12321/{ONLY_FILE}')
except:
    ds = load_dataset(f'HuggingFaceH4/{ONLY_FILE}')

# dataset_type = ds['test']['file_name']
answers = ds['test']['answer']
problems = [ele.replace('<image>', '' ) for ele in ds['test']['problem']]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_file(jsonl_path: Path, position: int):
    records = read_jsonl(jsonl_path)
    out_path = OUTPUT_DIR / jsonl_path.name

    # one tqdm bar per file, positioned by `position`
    with out_path.open('w', encoding='utf-8') as fout, \
         tqdm(total=len(records),
              desc=f"{jsonl_path.name}",
              position=position,
              leave=True) as pbar:

        for index, rec in enumerate(records):
            # question    = rec['problem']
            # gold_answer = rec['gold_answer']
            question = problems[index]
            gold_answer = answers[index]
            model_ans   = rec['response']
            extracted_box_content = extract_boxed_content(model_ans)
            if extracted_box_content.lower() == 'none':
                extracted_box_content = model_ans
            
            
            if accuracy_reward(model_ans, gold_answer) == 1:
                accuracy_output   = "correct"
                accuracy_judgment = "correct"
            else:
                accuracy_output = generate(question, gold_answer, extracted_box_content)
                accuracy_judgment = extract_judgment(accuracy_output).lower()
                print('Question: ', question)
                print(gold_answer)
                print(extracted_box_content)
                print('Accuracy: output: ', accuracy_output)

            # attach new fields
            rec['gold_answer'] = gold_answer
            rec['accuracy_output']   = accuracy_output
            rec['accuracy_judgment'] = accuracy_judgment

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            pbar.update(1)

    print(f"[{jsonl_path.name}] Done, wrote {len(records)} records")


def main():
    # --- 1️⃣  EDIT THIS: point to the one file you want ---
    ONLY_THIS = INPUT_DIR / f"{ONLY_FILE}.jsonl"      # ⬅️  change the name
    # ------------------------------------------------------

    if not ONLY_THIS.exists():
        raise FileNotFoundError(ONLY_THIS)

    # position = 0 → puts the tqdm bar on the first row
    process_file(ONLY_THIS, position=0)


if __name__ == "__main__":
    main()