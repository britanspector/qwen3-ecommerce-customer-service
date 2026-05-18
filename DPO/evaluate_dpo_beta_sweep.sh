#!/usr/bin/env bash
set -e

cd /root/firstTunning

BASE_MODEL="/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"
EVAL_FILE="/root/firstTunning/SFT_eval/fixed_eval_set_100.jsonl"

for tag in beta005 beta01 beta03; do
  LORA_DIR="/root/firstTunning/outputs/dpo_lora_2000_${tag}_1epoch"
  OUT_FILE="/root/firstTunning/results/dpo_lora_2000_${tag}_eval.jsonl"
  echo "========================================"
  echo "Evaluating ${tag}"
  echo "LORA_DIR=${LORA_DIR}"
  echo "========================================"
  python evaluate_fixed_test_set.py \
    --base_model ${BASE_MODEL} \
    --lora_dir ${LORA_DIR} \
    --eval_file ${EVAL_FILE} \
    --output_file ${OUT_FILE} \
    --use_gold_category true
 done
