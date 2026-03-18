#!/bin/bash
# Train draft model script for Baseline EAGLE-3 (No LittleBit) with 4 GPUs
# Run this from the eagletrain3 directory or project root

cd "$(dirname "$0")"

# Set HuggingFace cache directories to the persistent project folder
export HF_DATASETS_CACHE="/group-volume/ym1012.kim/homepc/EAGLE/.cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="/group-volume/ym1012.kim/homepc/EAGLE/.cache/huggingface/hub"

python -m deepspeed.launcher.runner --num_gpus 4 main.py \
    --deepspeed_config ds_config.json \
    --basepath ../../Llama-3.1-8B-Instruct \
    --trainpath ../../sharegpt_train.jsonl \
    --testpath ../../sharegpt_test.jsonl \
    --savedir ./checkpoints_baseline \
    --disable_littlebit
