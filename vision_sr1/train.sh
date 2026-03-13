#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# MODEL_PATH=${1:-Qwen/Qwen2.5-VL-3B-Instruct}
MODEL_PATH=${1:-Qwen/Qwen2.5-VL-7B-Instruct}
# MODEL_PATH=${1:-Qwen/Qwen3-VL-8B-Instruct}

python3 -m vision_sr1.main \
    config=vision_sr1/config.yaml \
    data.train_files=LMMs-Lab-Turtle/Vision-SR1-47K@train \
    data.val_files=zli12321/mmstar@test \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=8 \
    trainer.total_epochs=1 \
    trainer.experiment_name=qwen2_5_vl_7b_visionSR1_grpo \
    trainer.save_checkpoint_path=./saves/7b_grpo_self_reward \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=true \
    trainer.val_only=false
