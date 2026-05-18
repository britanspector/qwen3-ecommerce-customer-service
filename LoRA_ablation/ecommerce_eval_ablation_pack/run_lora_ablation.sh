#!/usr/bin/env bash
set -e

# Usage:
# bash run_lora_ablation.sh
#
# Before running, check these paths.

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

BASE_MODEL="/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"
TRAIN_FILE="/root/firstTunning/data/ecommerce_sft_messages_recommended_v3_train.jsonl"
EVAL_FILE="/root/firstTunning/data/ecommerce_sft_messages_recommended_v3_eval.jsonl"
OUTPUT_ROOT="/root/firstTunning/ablation_outputs"

# Conservative settings for 8B on one GPU.
EPOCHS=1
LR=5e-5
MAX_LEN=1024
BSZ=1
GAS=8

# 1) Regular LoRA with FP16 base weights: r = 4, 16, 64
for R in 4 16 64
do
  python train_lora_ablation.py \
    --base_model "$BASE_MODEL" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_root "$OUTPUT_ROOT" \
    --rank "$R" \
    --quantization none \
    --dtype fp16 \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --max_length "$MAX_LEN" \
    --per_device_train_batch_size "$BSZ" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GAS"
done

# 2) QLoRA with 4-bit NF4 base weights: r = 4, 16, 64
for R in 4 16 64
do
  python train_lora_ablation.py \
    --base_model "$BASE_MODEL" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_root "$OUTPUT_ROOT" \
    --rank "$R" \
    --quantization 4bit \
    --dtype bf16 \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --max_length "$MAX_LEN" \
    --per_device_train_batch_size "$BSZ" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GAS"
done

python summarize_ablation_results.py \
  --output_root "$OUTPUT_ROOT" \
  --output_csv "$OUTPUT_ROOT/ablation_summary.csv" \
  --output_md "$OUTPUT_ROOT/ablation_summary.md"
