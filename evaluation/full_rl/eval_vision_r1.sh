#!/bin/bash
#
# Evaluate a full-finetuned vision_r1 checkpoint (CoT format).
#
# Usage:
#   bash evaluation/full_rl/eval_vision_r1.sh <CHECKPOINT_PATH>
#
# Examples:
#   # Evaluate baseline (no checkpoint, just the base model):
#   bash evaluation/full_rl/eval_vision_r1.sh
#
#   # Evaluate a training checkpoint:
#   bash evaluation/full_rl/eval_vision_r1.sh ./saves/7b_grpo_accuracy/global_step_15
#

set -x
set -euo pipefail
export PYTHONUNBUFFERED=1

CHECKPOINT_PATH="${1:-}"
MODEL_PATH="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"

# Derive save name from model path (e.g. "Qwen/Qwen2.5-VL-7B-Instruct" -> "Qwen2.5-VL-7B-Instruct")
MODEL_NAME="${MODEL_PATH##*/}"
SAVE_DIR="./evaluation/responses/vision_r1_full/${MODEL_NAME}"

DATASETS=(
  "zli12321/mmstar"
  "zli12321/mm-vet"
  "zli12321/MLLM_test"
  "zli12321/visnumbench"
  "zli12321/mmmu_pro_10options"
  "zli12321/mmmu-pro-vision"
  "zli12321/hallusionbench"
  "zli12321/MMMU"
  "zli12321/MMSI"
  "zli12321/mathverse"
  "zli12321/mathvision"
  "zli12321/mathvista"
  "zli12321/realWorldQA"
)

CKPT_ARGS=""
if [ -n "${CHECKPOINT_PATH}" ]; then
  CKPT_ARGS="trainer.load_checkpoint_path=${CHECKPOINT_PATH}"
  # Use checkpoint step in the save dir for clarity
  STEP_NAME="${CHECKPOINT_PATH##*/}"
  SAVE_DIR="./evaluation/responses/vision_r1_full/${MODEL_NAME}/${STEP_NAME}"
fi

for DS in "${DATASETS[@]}"; do
  SHORT_NAME="${DS##*/}"
  echo ">>> Evaluating ${SHORT_NAME} with vision_r1 (full RL)"

  python3 -m verl.trainer.main \
    config=evaluation/eval_config.yaml \
    data.train_files=hiyouga/geometry3k@test \
    data.val_files="${DS}@test" \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    data.format_prompt=./evaluation/format_prompt/cot_format.jinja \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.reward.reward_function=./evaluation/reward_function/eval_accuracy.py:compute_score \
    trainer.experiment_name="eval_vision_r1_full_${SHORT_NAME}" \
    trainer.response_path="${SAVE_DIR}/${SHORT_NAME}.jsonl" \
    trainer.n_gpus_per_node=8 \
    trainer.val_only=true \
    ${CKPT_ARGS}

  echo "------------------------------------------------------------"
done

echo "All responses saved under: ${SAVE_DIR}/"

# --- LLM Judge (extract \boxed{} then judge with LLM) ---
JUDGMENT_DIR="./evaluation/judgments/${SAVE_DIR#./evaluation/responses/}"
echo ">>> Running LLM Judge..."
python evaluation/llm_judge.py --response_dir "${SAVE_DIR}"
python evaluation/print_accuracy.py --judgment_dir "${JUDGMENT_DIR}"
