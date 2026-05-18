# 电商客服模型固定测试集 + LoRA/QLoRA 消融实验代码包

## 文件说明

- `fixed_eval_set_100.jsonl`  
  固定测试集，共 100 条，覆盖退换货、物流、签收未收到、错发漏发、商品参数、质量投诉、订单修改、价格优惠、其他咨询等场景。

- `evaluate_fixed_test_set.py`  
  固定测试集推理与规则评分脚本。可以评估：
  - 规则命中率 `avg_rule_score`
  - 风险回复率 `risky_response_rate`
  - 场景分类准确率 `category_accuracy`（当 `--use_gold_category false` 时）

- `train_lora_ablation.py`  
  LoRA / QLoRA 消融训练脚本。读取 TRL messages JSONL 格式，但不依赖 TRL。会输出：
  - `eval_loss`
  - `eval_mean_token_accuracy`
  - `max_memory_allocated_gb`
  - `max_memory_reserved_gb`
  - LoRA adapter

- `run_lora_ablation.sh`  
  一键运行 6 组实验：
  - LoRA FP16: r=4 / 16 / 64
  - QLoRA 4bit NF4: r=4 / 16 / 64

- `summarize_ablation_results.py`  
  汇总所有实验结果为 CSV 和 Markdown 表格。

## 推荐使用步骤

### 1. 上传到服务器

```bash
mkdir -p /root/firstTunning/eval
mkdir -p /root/firstTunning/ablation_code
```

把文件放到对应目录，或者直接解压本压缩包。

### 2. 检查训练数据路径

假设你的新数据在：

```bash
/root/firstTunning/data/ecommerce_sft_messages_recommended_v3_train.jsonl
/root/firstTunning/data/ecommerce_sft_messages_recommended_v3_eval.jsonl
```

如果路径不同，修改 `run_lora_ablation.sh` 中的 `TRAIN_FILE` 和 `EVAL_FILE`。

### 3. 运行消融实验

```bash
cd /root/firstTunning/ablation_code
bash run_lora_ablation.sh
```

### 4. 查看结果表

```bash
cat /root/firstTunning/ablation_outputs/ablation_summary.md
```

### 5. 固定测试集评估

```bash
python evaluate_fixed_test_set.py \
  --base_model /root/autodl-tmp/modelscope/Qwen/Qwen3-8B \
  --lora_dir /root/firstTunning/ablation_outputs/lora_r16_fp16_alpha32 \
  --eval_file /root/firstTunning/eval/fixed_eval_set_100.jsonl \
  --output_file /root/firstTunning/eval_results/lora_r16_fixed_eval.jsonl \
  --use_gold_category true
```

如果想测试模型是否能自己识别场景分类：

```bash
python evaluate_fixed_test_set.py \
  --base_model /root/autodl-tmp/modelscope/Qwen/Qwen3-8B \
  --lora_dir /root/firstTunning/ablation_outputs/lora_r16_fp16_alpha32 \
  --eval_file /root/firstTunning/eval/fixed_eval_set_100.jsonl \
  --output_file /root/firstTunning/eval_results/lora_r16_category_eval.jsonl \
  --use_gold_category false
```

## 如何判断当前训练是不是 QLoRA？

如果你的模型加载代码里有：

```python
BitsAndBytesConfig(load_in_4bit=True, ...)
```

或者：

```python
load_in_4bit=True
```

那就是 QLoRA。

如果只是：

```python
AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16)
```

或：

```python
AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)
```

再配合 `get_peft_model` / `PeftModel`，那是普通 LoRA，不是 QLoRA。

你之前日志里只有 `torch_dtype is deprecated`，没有看到 4bit 量化加载提示，所以大概率是普通 LoRA，dtype 可能是 FP16 或 BF16，取决于脚本里写的是 `torch.float16` 还是 `torch.bfloat16`。
