#!/usr/bin/env bash
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

BASE_MODEL="/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"

# 4000 balanced train set + 400 balanced eval set
TRAIN_FILE="/root/firstTunning/data/ecommerce_sft_messages_recommended_v3_train_4000_balanced.jsonl"
EVAL_FILE="/root/firstTunning/data/ecommerce_sft_messages_recommended_v3_eval_400_balanced.jsonl"

OUTPUT_ROOT="/root/firstTunning/ablation_outputs_fast4000"

EPOCHS=1
LR=5e-5
MAX_LEN=768
BSZ=1
GAS=8

# Step 1: quick rank ablation under current q,k target.
for R in 4 16 64
do
  python train_lora_ablation_fast.py \
    --base_model "$BASE_MODEL" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_root "$OUTPUT_ROOT" \
    --rank "$R" \
    --target_preset qk \
    --quantization none \
    --dtype fp16 \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --max_length "$MAX_LEN" \
    --per_device_train_batch_size "$BSZ" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GAS"
done

# Step 2: compare QLoRA 4bit with the same q,k target.
for R in 4 16 64
do
  python train_lora_ablation_fast.py \
    --base_model "$BASE_MODEL" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_root "$OUTPUT_ROOT" \
    --rank "$R" \
    --target_preset qk \
    --quantization 4bit \
    --dtype bf16 \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --max_length "$MAX_LEN" \
    --per_device_train_batch_size "$BSZ" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GAS"
done

python summarize_ablation_results_fast.py \
  --output_root "$OUTPUT_ROOT" \
  --output_csv "$OUTPUT_ROOT/ablation_summary_fast4000.csv" \
  --output_md "$OUTPUT_ROOT/ablation_summary_fast4000.md"
