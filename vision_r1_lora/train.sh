#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# MODEL_PATH=${1:-Qwen/Qwen2.5-VL-3B-Instruct}
MODEL_PATH=${1:-Qwen/Qwen2.5-VL-7B-Instruct}
# MODEL_PATH=${1:-Qwen/Qwen3-VL-8B-Instruct}

python3 -m vision_r1_lora.main \
    config=vision_r1_lora/config.yaml \
    data.train_files=LMMs-Lab-Turtle/Vision-SR1-47K@train \
    data.val_files=zli12321/mmstar@test \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=64 \
    worker.actor.optim.lr=1e-5 \
    worker.rollout.n=8 \
    trainer.total_epochs=1 \
    trainer.experiment_name=qwen2_5_vl_7b_visionR1_grpo_lora \
    trainer.save_checkpoint_path=./saves/7b_grpo_accuracy_lora \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=true \
    trainer.val_only=false
