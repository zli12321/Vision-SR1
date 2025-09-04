import json
from typing import Iterator, List, Dict
from mathruler.grader import extract_boxed_content
from datasets import load_dataset
from utils.math_utils import *
from collections import defaultdict
from typing import List, Dict, Sequence, Tuple, Iterable
from tqdm import tqdm
from pathlib import Path
import argparse, os


answer_tag_re   = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
final_answer_re = re.compile(r"Final\s+Answer\s*:?\s*(.*)", re.I | re.S)

def iter_jsonl(path: str) -> Iterator[Dict]:
    """Yield one JSON object per line from a .jsonl file."""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: str) -> List[Dict]:
    """Read an entire .jsonl file into a list of dicts."""
    return list(iter_jsonl(path))


def get_final_answer(text: str) -> Optional[str]:
    """
    1. Find the text between <answer> … </answer>.
    2. Inside that text, return whatever follows 'Final Answer:'.
       If the tag is present but the phrase is missing, return the whole tag-content.
       Return None if the <answer> tag is absent altogether.
    """
    tag_match = answer_tag_re.search(text)
    if not tag_match:                 # no <answer> … </answer>
        return None

    inner = tag_match.group(1).strip()

    fa_match = final_answer_re.search(inner)
    return fa_match.group(1).strip() if fa_match else inner


def compute_accuracy_by_dataset(
    dataset_types: List[str],
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Returns a dict mapping each dataset type to its accuracy.
    Also prints out numerator/denominator and percentage.
    """
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    # accumulate counts
    for ds_type, pred, true in zip(dataset_types, predictions, references):
        stats[ds_type]["total"] += 1
        # if pred == true:
        if accuracy_reward(pred, true) == 1:
            stats[ds_type]["correct"] += 1

    # compute and print accuracies
    accuracies = {}
    for ds_type, v in stats.items():
        acc = v["correct"] / v["total"]
        accuracies[ds_type] = acc
        print(f"{ds_type}: {v['correct']}/{v['total']} → {acc:.2%}")

    return accuracies


# --------------------------------------------------------------------
# 1) llm-Eval judgments  --------------------------------------------
# --------------------------------------------------------------------

def compute_llmEval_stats(
    dataset_types: List[str],
    judgments: List[str],
    *,
    show_progress: bool = True,
) -> Dict[str, Tuple[int, float]]:
    """
    Return {dataset_type: (count, accuracy)} with an optional progress bar.
    """
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    iterable = zip(dataset_types, judgments)
    if show_progress:
        iterable = tqdm(iterable, total=len(dataset_types), desc="llm-Eval")

    for ds_type, j in iterable:
        stats[ds_type]["total"] += 1
        if "incorrect" not in j.lower():
            stats[ds_type]["correct"] += 1

    return {
        ds_type: (v["total"], v["correct"] / v["total"])
        for ds_type, v in stats.items()
    }


# --------------------------------------------------------------------
# 2) MCQ accuracy  ---------------------------------------------------
# --------------------------------------------------------------------
def compute_MCQ_stats(
    file_name: str,
    *,
    result_dirs: Sequence[str] = (
        "./7b_sft_description_r1_Train1",
        # "./gpt_eval_out/7b_sft_description_r1_Train1",
    ),
    hf_owner: str = "zli12321",
    show_progress: bool = True,
) -> Tuple[int, float]:
    """
    Return (count, accuracy) for one MCQ dataset, with progress bar.
    """
    # locate results file
    result_path = next(
        (p for d in result_dirs if (p := Path(d) / f"{file_name}.jsonl").is_file()),
        None,
    )
    if result_path is None:
        raise FileNotFoundError(
            f"No JSONL results for '{file_name}' in {list(result_dirs)}"
        )

    data  = load_jsonl(result_path)
    gold  = load_dataset(f"{hf_owner}/{file_name}", split="test")["answer"]

    iterator = enumerate(data)
    if show_progress:
        iterator = tqdm(iterator, total=len(data), desc=file_name)

    correct = 0
    for i, sample in iterator:
        pred_txt = (
            get_final_answer(sample["response"])
            if "Vision-R1(By Huang)" in str(result_path)
            else sample["response"]
        )
        correct += (
            grade_answer(pred_txt, gold[i])
            if "Vision-R1(By Huang)" in str(result_path)
            else accuracy_reward(pred_txt, gold[i])
        )

    total = len(data)
    return total, correct / total


def compute_llm_rule_stats(
    *,
    file_stem: str               = "MLLM_test",   # HF dataset name     → zli12321/<file_stem>
    directory_folder: str        = "./gpt_eval_out/7b_sft_description_r1_Train1/",
    hf_owner: str                = "zli12321",
    show_progress: bool          = True,
) -> Dict[str, Tuple[int, float]]:
    """
    Return {dataset_type: (count, accuracy)} where correctness is decided by
    `accuracy_reward(pred, gold)`.

    The function:
    1) loads `<directory_folder>/<file_stem>.jsonl`  ➜ model responses
    2) loads HuggingFace dataset `hf_owner/file_stem` (split='test')
       – expects columns 'file_name' (dataset type) and 'answer'.

    Parameters
    ----------
    file_stem : str, optional
        Base name of both the JSONL file and the HF dataset repo.
    directory_folder : str, optional
        Where the JSONL file lives.
    hf_owner : str, optional
        HF Hub user/org that owns the ground-truth dataset.
    show_progress : bool, optional
        Wrap the main loop in a tqdm progress bar if True.

    Returns
    -------
    dict
        Mapping {dataset_type: (n_samples, accuracy)}.
    """
    # ------------------------------------------------------------------
    # 1) read data
    # ------------------------------------------------------------------
    jsonl_path = Path(directory_folder) / f"{file_stem}.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Cannot find responses file: {jsonl_path}")

    records   = load_jsonl(jsonl_path)
    ds        = load_dataset(f"{hf_owner}/{file_stem}", split="test")
    ds_types  = ds["file_name"]          # per-row dataset identifier
    gold_ans  = ds["answer"]

    # sanity check
    assert len(records) == len(ds_types) == len(gold_ans), \
        "JSONL and HF dataset lengths do not match"

    # ------------------------------------------------------------------
    # 2) accumulate per-dataset stats
    # ------------------------------------------------------------------
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    iterator = zip(ds_types, records, gold_ans)
    if show_progress:
        iterator = tqdm(iterator, total=len(ds_types), desc="Computing rule-based Results")

    for ds_type, rec, gold in iterator:
        pred = rec["response"]
        stats[ds_type]["total"]   += 1
        stats[ds_type]["correct"] += accuracy_reward(pred, gold)

    # ------------------------------------------------------------------
    # 3) final dict {dataset: (count, accuracy)}
    # ------------------------------------------------------------------
    return {
        ds_type: (v["total"], v["correct"] / v["total"])
        for ds_type, v in stats.items()
    }

# --------------------------------------------------------------------
# 3) Weighted-subset helper (unchanged) ------------------------------
# --------------------------------------------------------------------
def add_weighted_subset(
    results: Dict[str, Tuple[int, float]],
    *,
    patterns: Iterable[str] = ("mmmu-pro", "mmmu_pro"),
    label: str = "weighted_mmmu_pro",
    print_table: bool = True,
) -> Dict[str, Tuple[int, float]]:
    # ------------------------------------------------------------------
    # 1) weighted aggregate
    # ------------------------------------------------------------------
    matches = [k for k in results if any(p in k for p in patterns)]
    total   = sum(results[k][0] for k in matches)
    weighted_acc = (
        sum(results[k][0] * results[k][1] for k in matches) / total if total else 0.0
    )
    merged = {**results, label: (total, weighted_acc)}

    # ------------------------------------------------------------------
    # 2) pretty-print with key rename
    # ------------------------------------------------------------------
    if print_table:
        rename = {"mmmu-pro": "mmmu_pro_4_options"}   # <-- your custom mapping
        print("\nDataset".ljust(22), "Accuracy")
        print("-" * 34)
        for k, (_, acc) in merged.items():
            shown_key = rename.get(k, k)             # fall back to original
            print(f"{shown_key.ljust(22)} {acc:.3%}")
    return merged


def pretty_print_stats(
    stats: Dict[str, Tuple[int, float]],
    *,
    rename: Optional[Dict[str, str]] = None,
    order: Optional[List[str]] = None,
    title: str = ""
) -> float:
    """
    Pretty-prints the stats table, then prints:
        Average = (a1 + a2 + ... + ak) / k
    where the ai are the accuracies of all datasets EXCEPT the individual
    mmmu-pro splits (we keep only 'weighted_mmmu_pro'). Returns that average.
    """
    rename = rename or {}
    keys   = list(stats.keys())

    # honour custom ordering
    if order is None:
        keys = sorted(keys)
    else:
        rest = sorted(k for k in keys if k not in order)
        keys = order + rest

    # ---------- main table ----------
    if title:
        print(title)
    print("Dataset".ljust(22), "Accuracy")
    print("-" * 34)
    for k in keys:
        print(f"{rename.get(k, k).ljust(22)} {stats[k][1]:.3%}")

    # ---------- average (skip individual mmmu-pro sets) ----------
    mmmu_pat  = re.compile(r"mmmu[-_]pro", re.I)
    avg_keys  = [
        k for k in keys
        if not (mmmu_pat.search(k) and k != "weighted_mmmu_pro")
    ]
    values    = [stats[k][1] for k in avg_keys]
    mean_acc  = sum(values) / len(values)

    # ---------- formula ----------
    nums      = " + ".join(f"{v:.3%}" for v in values)
    print("-" * 34)
    print(f"{'Average'.ljust(22)} {mean_acc:.3%}")
    print(f"        = ({nums}) / {len(values)}\n")

    return mean_acc


def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute accuracy tables for LLM-Eval and MCQ runs"
    )
    parser.add_argument(
        "--llm_eval_dir",
        default="./gpt_eval_out/7b_sft_description_r1_Train1",
        help="Folder that holds MLLM_test.jsonl and other llm-eval files",
    )
    parser.add_argument(
        "--mcq_dir",
        default="./7b_sft_description_r1_Train1",
        help="Folder that holds the per-dataset MCQ result *.jsonl files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cli_args()

    # make sure we have trailing slashes for later path joins
    directory_folder     = os.path.join(os.path.expanduser(args.llm_eval_dir), "")
    mcq_directory_folder = os.path.join(os.path.expanduser(args.mcq_dir),  "")

    ds_full       = load_dataset("zli12321/MLLM_test")
    dataset_type  = ds_full["test"]["file_name"]
    llm_judgments = [
        r["accuracy_judgment"]
        for r in load_jsonl(directory_folder + "MLLM_test.jsonl")
    ]

    # ──────────────────────────────────
    # 1) collect stats  (progress bars ON)
    # ──────────────────────────────────
    merged_stats: Dict[str, Tuple[int, float]] = {}
    merged_stats.update(
        compute_llmEval_stats(dataset_type, llm_judgments, show_progress=True)
    )

    mcq_files = [
        "mmmu_pro_10options",
        "mmmu-pro-vision",
        "MMMU",
        "visnumbench",
        "hallusionbench",
    ]
    for fname in tqdm(mcq_files, desc="MCQ sets"):
        merged_stats[fname] = compute_MCQ_stats(
            fname,
            result_dirs=[mcq_directory_folder],   # ← uses CLI arg
            show_progress=True,
        )

    augmented   = add_weighted_subset(merged_stats, print_table=False)
    rule_stats  = compute_llm_rule_stats(
        file_stem="MLLM_test",
        directory_folder=directory_folder,       # ← uses CLI arg
        hf_owner="zli12321",
        show_progress=True,
    )
    mcq_only        = {k: v for k, v in merged_stats.items() if k in mcq_files}
    rule_mcq_stats  = {**mcq_only, **rule_stats}
    final_stats     = add_weighted_subset(rule_mcq_stats, print_table=False)

    # ──────────────────────────────────
    # 2) pretty-print summaries (unchanged)
    # ──────────────────────────────────
    rename_map  = {"mmmu-pro": "mmmu_pro_4_options"}
    ordered_mcq = [
        "mmmu_pro_10options",
        "mmmu-pro-vision",
        "MMMU",
        "visnumbench",
        "hallusionbench",
    ]

    pretty_print_stats(
        augmented,
        rename=rename_map,
        order=ordered_mcq + [
            "mmmu-pro", "clevr_count_70k", "mm-vet", "mathverse",
            "mathvista", "mathvision", "realWorldQA", "weighted_mmmu_pro",
        ],
        title="\n--- LLM-Eval + MCQ ---",
    )

    pretty_print_stats(
        final_stats,
        rename=rename_map,
        order=ordered_mcq + [
            "mmmu-pro", "clevr_count_70k", "mm-vet", "mathverse",
            "mathvista", "mathvision", "realWorldQA", "weighted_mmmu_pro",
        ],
        title="--- Rule-Based Evaluation ---",
    )

