#!/bin/bash

# ==============================
# 打包 firstTunning，排除大文件
# 运行位置建议：/root/firstTunning
# 输出文件：/root/firstTunning_clean.tar.gz
# ==============================

set -e

PROJECT_DIR="/root/firstTunning"
OUTPUT_FILE="/root/firstTunning_clean.tar.gz"

echo "进入项目目录：$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "开始打包，排除 checkpoint 中的大文件和大模型权重..."

tar -czf "$OUTPUT_FILE" \
  --exclude='./outputs/dpo_lora_2000_beta03_1epoch/checkpoint-250/optimizer.pt' \
  --exclude='./outputs/dpo_lora_2000_beta01_1epoch/checkpoint-250/optimizer.pt' \
  --exclude='./outputs/dpo_lora_2000_beta005_1epoch/checkpoint-250/optimizer.pt' \
  --exclude='./outputs/dpo_lora_r16_1epoch/checkpoint-63/optimizer.pt' \
  --exclude='./ablation_outputs_fast4000_5runs/qlora4bit_qk_r16_bf16_alpha32/checkpoint-500/optimizer.pt' \
  --exclude='./ablation_outputs_fast4000_5runs/qlora4bit_qk_r16_bf16_alpha32/checkpoint-300/optimizer.pt' \
  --exclude='./ablation_outputs_fast4000_5runs/lora_qk_r16_fp16_alpha32/checkpoint-500/optimizer.pt' \
  --exclude='./ablation_outputs_fast4000_5runs/lora_qk_r16_fp16_alpha32/checkpoint-300/optimizer.pt' \
  --exclude='./outputs/qwen3_8b_ecommerce_sft_lora2/checkpoint-2709/optimizer.pt' \
  --exclude='./outputs/qwen3_8b_ecommerce_sft_lora2/checkpoint-2700/optimizer.pt' \
  --exclude='./outputs/qwen3_8b_ecommerce_sft_lora2/checkpoint-2400/optimizer.pt' \
  --exclude='./outputs/qwen3_8b_ecommerce_sft_lora2/checkpoint-2100/optimizer.pt' \
  --exclude='./outputs/qwen3_8b_ecommerce_sft_lora2/checkpoint-1800/optimizer.pt' \
  --exclude='./ablation_outputs_fast4000_5runs/qlora4bit_all_linear_r16_bf16_alpha32/checkpoint-500/optimizer.pt' \
  --exclude='./ablation_outputs_fast4000_5runs/qlora4bit_all_linear_r16_bf16_alpha32/checkpoint-300/optimizer.pt' \
  --exclude='./outputs/qwen3_8b_ecommerce_sft_lora/checkpoint-981/optimizer.pt' \
  --exclude='./outputs/qwen3_8b_ecommerce_sft_lora/checkpoint-900/optimizer.pt' \
  --exclude='./ablation_outputs_fast4000_5runs/lora_qk_r64_fp16_alpha128/checkpoint-500/optimizer.pt' \
  --exclude='./ablation_outputs_fast4000_5runs/lora_qk_r64_fp16_alpha128/checkpoint-300/optimizer.pt' \
  --exclude='./ablation_outputs_fast4000_5runs/qlora4bit_all_linear_r16_bf16_alpha32/adapter_model.safetensors' \
  --exclude='./ablation_outputs_fast4000_5runs/qlora4bit_all_linear_r16_bf16_alpha32/checkpoint-500/adapter_model.safetensors' \
  --exclude='./ablation_outputs_fast4000_5runs/qlora4bit_all_linear_r16_bf16_alpha32/checkpoint-300/adapter_model.safetensors' \
  --exclude='./outputs/qwen3_8b_ecommerce_sft_lora/checkpoint-981/adapter_model.safetensors' \
  --exclude='./outputs/qwen3_8b_ecommerce_sft_lora/adapter_model.safetensors' \
  --exclude='./outputs/qwen3_8b_ecommerce_sft_lora/checkpoint-900/adapter_model.safetensors' \
  --exclude='./ablation_outputs_fast4000_5runs/lora_qk_r64_fp16_alpha128/checkpoint-500/adapter_model.safetensors' \
  --exclude='./ablation_outputs_fast4000_5runs/lora_qk_r64_fp16_alpha128/adapter_model.safetensors' \
  --exclude='./ablation_outputs_fast4000_5runs/lora_qk_r64_fp16_alpha128/checkpoint-300/adapter_model.safetensors' \
  --exclude='./**/__pycache__' \
  --exclude='./**/.ipynb_checkpoints' \
  --exclude='./**/.cache' \
  --exclude='./**/wandb' \
  --exclude='./**/runs' \
  .

echo "打包完成：$OUTPUT_FILE"
echo "压缩包大小："
ls -lh "$OUTPUT_FILE"

echo "检查压缩包中是否仍包含大于 50M 的文件："
tar -tzf "$OUTPUT_FILE" | while read file; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        size=$(stat -c%s "$PROJECT_DIR/$file")
        if [ "$size" -gt $((50 * 1024 * 1024)) ]; then
            ls -lh "$PROJECT_DIR/$file"
        fi
    fi
done

echo "完成。"