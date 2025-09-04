#!/bin/bash

set -x

set -euo pipefail

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct 
SAVE_PATH=7B-baseline

DATASETS=(
  "zli12321/mmlu_pro"
  "zli12321/SuperGPQA"
  "HuggingFaceH4/MATH-500"
  "zli12321/gsm8k"
)

# ------------------------------------------------------------------
# STATIC pieces of the command line (everything that never changes)
# ------------------------------------------------------------------
BASE_CMD="python3 -m verl.trainer.main \
  config=validation_examples/eval_config.yaml \
  data.train_files=hiyouga/geometry3k@test \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.image_key=images \
  worker.actor.model.model_path=${MODEL_PATH} \
  worker.rollout.max_model_len=25600 \
  worker.rollout.n=8 \
  trainer.total_epochs=20 \
  trainer.experiment_name=qwen2_5_vl_text_only_eval \
  trainer.save_checkpoint_path=./Evaluation/Raw-Outputs \
  trainer.n_gpus_per_node=8 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.actor.global_batch_size=8 \
  data.format_prompt=./validation_examples/format_prompt/cot_format.jinja \
  trainer.val_only=true"

# ------------------------------------------------------------------
# LOOP over datasets
# ------------------------------------------------------------------
for DS in "${DATASETS[@]}"; do
  # strip "owner/" prefix to make a clean filename, e.g. mmstar
  SHORT_NAME="${DS##*/}"

  echo ">>> Evaluating on ${DS}"
  CMD="${BASE_CMD} \
    data.val_files=${DS}@test \
    trainer.response_path=./validation_responses/${SAVE_PATH}/${SHORT_NAME}.jsonl"

  # show the command (optional)
  echo "$CMD" | sed 's/  */ /g'
  echo "------------------------------------------------------------"

  # run it
  eval $CMD
done

