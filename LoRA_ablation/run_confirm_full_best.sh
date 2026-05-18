#!/usr/bin/env bash
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

BASE_MODEL="/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"
TRAIN_FILE="/root/firstTunning/data/ecommerce_sft_messages_recommended_v3_train.jsonl"
EVAL_FILE="/root/firstTunning/data/ecommerce_sft_messages_recommended_v3_eval.jsonl"
OUTPUT_ROOT="/root/firstTunning/ablation_outputs_confirm_full"

# Only run 1-2 selected configurations here after fast4000 results.
# Example: r=16, qk, FP16 LoRA.
python train_lora_ablation_fast.py \
  --base_model "$BASE_MODEL" \
  --train_file "$TRAIN_FILE" \
  --eval_file "$EVAL_FILE" \
  --output_root "$OUTPUT_ROOT" \
  --rank 16 \
  --target_preset qk \
  --quantization none \
  --dtype fp16 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --max_length 768 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8

python summarize_ablation_results_fast.py \
  --output_root "$OUTPUT_ROOT" \
  --output_csv "$OUTPUT_ROOT/confirm_full_summary.csv" \
  --output_md "$OUTPUT_ROOT/confirm_full_summary.md"
