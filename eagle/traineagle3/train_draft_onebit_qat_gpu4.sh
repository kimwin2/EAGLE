#!/bin/bash
# Train draft model with OneBit quantization QAT - 4 GPUs
cd "$(dirname "$0")"

export HF_DATASETS_CACHE="/group-volume/ym1012.kim/homepc/EAGLE/.cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="/group-volume/ym1012.kim/homepc/EAGLE/.cache/huggingface/hub"

python -m deepspeed.launcher.runner --num_gpus 4 main.py \
    --deepspeed_config ds_config.json \
    --basepath ../../Llama-3.1-8B-Instruct \
    --trainpath ../../sharegpt_train.jsonl \
    --testpath ../../sharegpt_test.jsonl \
    --savedir ./checkpoints_onebit_qat \
    --num_epochs 5 \
    --num_hidden_layers 1 \
    --draftpath ../../EAGLE3-LLaMA3.1-Instruct-8B \
    --quant_method onebit
