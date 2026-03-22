#!/bin/bash
# Train QAT draft model script for EAGLE-3 with 4 GPUs (eff_bit=0.3)
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
    --savedir ./checkpoints_qat_03 \
    --num_epochs 5 \
    --num_hidden_layers 1 \
    --draftpath ../../EAGLE3-LLaMA3.1-Instruct-8B \
    --eff_bit 0.3 \
    --lr 5e-5
