#!/bin/bash
# Evaluate pretrained EAGLE3 draft model accuracy (no quantization) - 4 GPUs
# Measures position-wise (0~6) accuracy using the same evaluation logic as training.
cd "$(dirname "$0")"

export HF_DATASETS_CACHE="/group-volume/ym1012.kim/homepc/EAGLE/.cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="/group-volume/ym1012.kim/homepc/EAGLE/.cache/huggingface/hub"

python -m deepspeed.launcher.runner --num_gpus 4 eval_pretrained.py \
    --deepspeed_config ds_config.json \
    --basepath ../../Llama-3.1-8B-Instruct \
    --trainpath ../../sharegpt_train.jsonl \
    --testpath ../../sharegpt_test.jsonl \
    --draftpath ../../EAGLE3-LLaMA3.1-Instruct-8B \
    --num_hidden_layers 1 \
    --quant_method none
