#!/usr/bin/env bash
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

BASE_MODEL="/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"
TRAIN_FILE="/root/firstTunning/data/ecommerce_sft_messages_recommended_v3_train_4000_balanced.jsonl"
EVAL_FILE="/root/firstTunning/data/ecommerce_sft_messages_recommended_v3_eval_400_balanced.jsonl"
OUTPUT_ROOT="/root/firstTunning/ablation_outputs_target4000"

EPOCHS=1
LR=5e-5
MAX_LEN=768
BSZ=1
GAS=8
R=16

# Target-module ablation. This is optional and should be run after the r-grid.
# qk: current setting, fastest
# qv: common LoRA baseline, often stronger than qk
# attn: q/k/v/o attention projections
# all_linear: q/k/v/o + MLP projections, strongest but slowest
for TARGET in qk qv attn all_linear
do
  python train_lora_ablation_fast.py \
    --base_model "$BASE_MODEL" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_root "$OUTPUT_ROOT" \
    --rank "$R" \
    --target_preset "$TARGET" \
    --quantization none \
    --dtype fp16 \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --max_length "$MAX_LEN" \
    --per_device_train_batch_size "$BSZ" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GAS"
done

python summarize_ablation_results_fast.py \
  --output_root "$OUTPUT_ROOT" \
  --output_csv "$OUTPUT_ROOT/target_ablation_summary_4000.csv" \
  --output_md "$OUTPUT_ROOT/target_ablation_summary_4000.md"
