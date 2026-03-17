#!/bin/bash
# Train draft model script for EAGLE-3
# Run this from the eagletrain3 directory or project root

cd "$(dirname "$0")"

python -m deepspeed.launcher.runner main.py \
    --deepspeed_config ds_config.json \
    --basepath ../../Llama-3.1-8B-Instruct \
    --trainpath ../../sharegpt_train.jsonl \
    --testpath ../../sharegpt_test.jsonl \
    --savedir ./checkpoints
