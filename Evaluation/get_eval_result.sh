#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Flags your eval.py expects.
# Change these only if you renamed the CLI arguments in eval.py.
# ---------------------------------------------------------------
CMD_EVAL="--llm_eval_dir"
CMD_MCQ="--mcq_dir"

# ---------------------------------------------------------------
# One line per run:  Label | LLM-Eval directory | MCQ directory
# (No quotes inside the three columns; use a literal | as delimiter.)
# ---------------------------------------------------------------
CONFIGS=(
  # ---------- 3-B models ----------
  "3B-baseline                | ./Raw-Outputs/LLM-Eval-out/3B-baseline            | ./Raw-Outputs/3B-baseline"
  "3B-SFT                     | ./Raw-Outputs/LLM-Eval-out/3B-SFT                 | ./Raw-Outputs/3B-SFT"
  "3B-Vision-R1               | ./Raw-Outputs/LLM-Eval-out/3B-Vision-R1           | ./Raw-Outputs/3B-Vision-R1"
  "3B-Vision-SR1              | ./Raw-Outputs/LLM-Eval-out/3B-Vision-SR1          | ./Raw-Outputs/3B-Vision-SR1"
  "3B-Vision-R1-ablation      | ./Raw-Outputs/LLM-Eval-out/3B-Vision-R1-ablation  | ./Raw-Outputs/3B-Vision-R1-ablation"

  # ---------- 7-B models ----------
  "7B-baseline                | ./Raw-Outputs/LLM-Eval-out/7B-baseline            | ./Raw-Outputs/7B-baseline"
  "7B-SFT                     | ./Raw-Outputs/LLM-Eval-out/7B-SFT                 | ./Raw-Outputs/7B-SFT"
  "7B-Vision-R1               | ./Raw-Outputs/LLM-Eval-out/7B-Vision-R1           | ./Raw-Outputs/7B-Vision-R1"
  "7B-Vision-SR1              | ./Raw-Outputs/LLM-Eval-out/7B-Vision-SR1          | ./Raw-Outputs/7B-Vision-SR1"
  "7B-Vision-SR1-ablation     | ./Raw-Outputs/LLM-Eval-out/7B-Vision-SR1-ablation | ./Raw-Outputs/7B-Vision-SR1-ablation"

  # ---------- other baselines ----------
  "7B-Vision-R1 (By Huang)    | ./Raw-Outputs/LLM-Eval-out/7B-Vision-R1(By Huang)  | ./Raw-Outputs/7B-Vision-R1(By Huang)"
  "7B-Perception-R1           | ./Raw-Outputs/LLM-Eval-out/7B-Perception-R1       | ./Raw-Outputs/7B-Perception-R1"
  "3B-Visionary-R1            | ./Raw-Outputs/LLM-Eval-out/3B-Visionary-R1        | ./Raw-Outputs/3B-Visionary-R1"
)

python_bin="python"   # point to your Python interpreter if needed

# ---------------------------------------------------------------
# Helper: trim leading/trailing whitespace (portable)
# ---------------------------------------------------------------
trim() { echo "$1" | xargs; }

# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r RAW_LABEL RAW_EVAL RAW_MCQ <<< "$entry"

  LABEL=$(trim "$RAW_LABEL")
  EVAL_DIR=$(trim "$RAW_EVAL")
  MCQ_DIR=$(trim "$RAW_MCQ")

  echo
  echo "=================================================================="
  echo "🚀  Computing ${LABEL}"
  echo "    LLM-Eval dir: ${EVAL_DIR}"
  echo "    MCQ dir     : ${MCQ_DIR}"
  echo "------------------------------------------------------------------"

  "$python_bin" eval.py \
      ${CMD_EVAL} "${EVAL_DIR}" \
      ${CMD_MCQ}  "${MCQ_DIR}"
done
