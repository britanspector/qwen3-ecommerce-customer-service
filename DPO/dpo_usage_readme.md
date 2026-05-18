# 电商客服 DPO 第一版使用说明

## 文件

- `ecommerce_dpo_500.jsonl`：500 条 DPO 偏好数据。
- `train_dpo_1epoch.py`：DPO 训练脚本，默认训练 1 epoch。
- `ecommerce_dpo_500_stats.json`：数据类别分布统计。

## 推荐放置路径

```bash
mkdir -p /root/firstTunning/data/dpo
cp ecommerce_dpo_500.jsonl /root/firstTunning/data/dpo/ecommerce_dpo_500.jsonl
cp train_dpo_1epoch.py /root/firstTunning/train_dpo_1epoch.py
```

## 训练

```bash
cd /root/firstTunning
python train_dpo_1epoch.py
```

默认路径配置在 `train_dpo_1epoch.py` 文件顶部：

```python
BASE_MODEL_PATH = "/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"
SFT_LORA_DIR = "/root/firstTunning/outputs/exp_lora_r16_fp16"
DPO_DATA_PATH = "/root/firstTunning/data/dpo/ecommerce_dpo_500.jsonl"
OUTPUT_DIR = "/root/firstTunning/outputs/dpo_lora_r16_1epoch"
```

## 评估

训练结束后重新跑固定测试集：

```bash
python evaluate_fixed_test_set.py \
  --base_model /root/autodl-tmp/modelscope/Qwen/Qwen3-8B \
  --lora_dir /root/firstTunning/outputs/dpo_lora_r16_1epoch \
  --eval_file /root/firstTunning/eval/fixed_eval_set_100.jsonl \
  --output_file /root/firstTunning/eval_results/dpo_lora_r16_eval.jsonl \
  --use_gold_category true
```

重点看：

- `avg_rule_score` 是否上升；
- `risky_response_rate` 是否仍保持低位；
- 商品参数、食品投诉、签收未收到、物流异常的人工样例是否更谨慎。

## 显存不足时

打开 `train_dpo_1epoch.py`，把：

```python
USE_4BIT = False
```

改成：

```python
USE_4BIT = True
```

如果当前环境 TRL/PEFT 版本过旧，建议升级：

```bash
pip install -U trl peft transformers datasets accelerate
```
