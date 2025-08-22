#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
MODEL_PATH=LMMs-Lab-Turtle/Qwen-2.5VL-7B-Cold-Start


python3 -m verl.trainer.main \
    config=train_examples/selfReward_config.yaml \
    data.train_files=LMMs-Lab-Turtle/Vision-SR1-47K@train \
    data.val_files=zli12321/mmstar@test \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.max_model_len=8192 \
    worker.rollout.n=8 \
    trainer.total_epochs=1 \
    trainer.experiment_name=qwen2_5_vl_7b_selfReward_grpo \
    trainer.save_checkpoint_path=./saves/7b_grpo_selfReward \
    trainer.n_gpus_per_node=8 \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    trainer.val_before_train=true \
    trainer.val_only=false

