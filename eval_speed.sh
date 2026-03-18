#!/bin/bash
# eval_speed.sh
# Script to evaluate baseline vs. EAGLE3 generation speed
#
# Assumes models are located at:
# - Base Model: ../../Llama-3.1-8B-Instruct
# - EAGLE Model: ../../EAGLE3-LLAMA3.1-Instruct-8B

BASE_MODEL_PATH="../../Llama-3.1-8B-Instruct"
EAGLE_MODEL_PATH="../../EAGLE3-LLAMA3.1-Instruct-8B"

# Ensure the prompt uses the LLaMA 3.1 instruct compatible scripts
SCRIPT_BASELINE="eagle/evaluation/gen_baseline_answer_llama3chat.py"
SCRIPT_EAGLE="eagle/evaluation/gen_ea_answer_llama3chat.py"
SCRIPT_SPEED="eagle/evaluation/speed.py"

echo "=========================================================="
echo "Starting evaluation for Llama-3.1-8B-Instruct vs EAGLE-3"
echo "=========================================================="

echo "1) Generating answers using Baseline model..."
python $SCRIPT_BASELINE \
    --base-model-path $BASE_MODEL_PATH \
    --ea-model-path $EAGLE_MODEL_PATH \
    --model-id "llama3.1-8b-inst-baseline" \
    --bench-name "mt_bench" \
    --temperature 0.0

echo "2) Generating answers using EAGLE-3 model..."
python $SCRIPT_EAGLE \
    --base-model-path $BASE_MODEL_PATH \
    --ea-model-path $EAGLE_MODEL_PATH \
    --model-id "llama3.1-8b-inst-eagle3" \
    --bench-name "mt_bench" \
    --temperature 0.0 \
    --use_eagle3

echo "3) Calculating speedup..."
python $SCRIPT_SPEED

echo "Evaluation finished."
