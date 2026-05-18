#!/usr/bin/env bash
set -e

cd /root/firstTunning

# 第二轮 targeted DPO beta 对比实验
# 输出目录：
# /root/firstTunning/outputs/dpo_lora_2000_beta005_1epoch
# /root/firstTunning/outputs/dpo_lora_2000_beta01_1epoch
# /root/firstTunning/outputs/dpo_lora_2000_beta03_1epoch

for beta in 0.05 0.1 0.3; do
  echo "========================================"
  echo "Running DPO beta=${beta}"
  echo "========================================"
  python DPO/train_dpo_one_beta.py --beta ${beta}
done
