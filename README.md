# 电商客服场景大模型微调项目

本项目基于 **Qwen3-8B** 大语言模型，使用 **HuggingFace Transformers / TRL / PEFT** 框架，面向电商客服场景构建了一套从数据清洗、格式转换、SFT-LoRA 微调、DPO 偏好优化到多维评估的完整微调流程。

项目围绕退换货、物流查询、商品参数、优惠活动、投诉处理、售后协商等典型客服问题，构建高质量指令微调数据和偏好优化数据，使模型在电商客服场景下具备更稳定、礼貌、规范、可执行的回复能力。

---

## 1. 项目背景

通用大语言模型虽然具备较强的语言理解与生成能力，但在垂直电商客服场景中仍存在以下问题：

1. **业务表达不够贴合客服场景**原始模型回答往往偏泛化，缺少电商客服常见的安抚、解释、引导和规范表达。
2. **回答稳定性不足**在退换货、物流异常、优惠活动、投诉处理等场景下，模型可能出现回答不完整、承诺过度或处理建议不明确的问题。
3. **业务规范性不足**客服回复需要避免随意承诺退款、补发、赔偿等高风险表述，需要符合平台规则和实际处理流程。
4. **缺少可执行方案**
   好的客服回复不仅要礼貌，还需要给出清晰、具体、可执行的下一步处理建议。

因此，本项目通过监督微调和偏好优化，使模型更加适配电商客服任务。

---

## 2. 项目目标

本项目的核心目标是提升模型在电商客服场景下的：

- **回答准确性**：能够理解用户问题并给出符合业务场景的回答；
- **语言礼貌性**：能够先安抚用户情绪，保持客服语气；
- **业务规范性**：避免不合规承诺，如随意退款、赔偿、补发；
- **方案可执行性**：能够给出清晰的下一步处理建议；
- **回复稳定性**：在不同客服场景下保持一致、可靠的输出风格。

---

## 3. 技术路线

本项目整体流程如下：

```text
原始电商客服数据
        │
        ▼
数据清洗与质量筛选
        │
        ▼
构造 SFT 指令微调数据
        │
        ▼
Qwen3-8B + LoRA 监督微调
        │
        ▼
构造 DPO 偏好数据
        │
        ▼
DPO 偏好优化
        │
        ▼
自动指标 + 样本级胜率 + 规则命中率 + 模拟人工评分
        │
        ▼
模型效果分析与案例对比
```

---

## 4. 技术栈

| 模块         | 技术                                                  |
| ------------ | ----------------------------------------------------- |
| Base Model   | Qwen3-8B                                              |
| 训练框架     | HuggingFace Transformers, TRL                         |
| 参数高效微调 | PEFT / LoRA                                           |
| 监督微调     | SFT                                                   |
| 偏好优化     | DPO                                                   |
| 数据格式     | JSONL / messages format                               |
| 评估指标     | BLEU, BERTScore, 样本级胜率, 规则命中率, 模拟人工评分 |
| 运行环境     | Python, PyTorch, CUDA                                 |

---

## 5. 数据来源与处理

### 5.1 原始数据来源

本项目使用来自 GitHub `cooelf/DeepUtteranceAggregation` 的电商客服对话数据作为基础数据来源。原始数据包含多轮用户与客服对话，以及正负样本标签。

原始数据格式示例：

```text
1    嗯 嗯    因为 买二送 一是 坚果 类产品 呢 不 太 清楚 不好意思 客官    没事    客官 帮 您 看 了 下 哦 牛肉干 属于 打 95 折 的 商品 哦 不 参与 买二送 一 活动 呢 实在 抱歉 呢 不过 小店 牛肉干 口味 很 不错 呢 客官 喜欢 的话 可以 拍下 哦

0    嗯 嗯    因为 买二送 一是 坚果 类产品 呢 不 太 清楚 不好意思 客官    没事    放 尿布 里面 的 吗 没有 哦
```

其中：

- `1` 表示当前回复与上下文匹配；
- `0` 表示当前回复与上下文不匹配；
- 后续字段为对话上下文和候选客服回复。

### 5.2 SFT 数据清洗策略

由于原始数据并不能直接用于指令微调，因此本项目对数据进行了清洗和格式转换。主要清洗步骤包括：

1. **删除 SFT 阶段无效的负样本**原始数据中的 `0` 标签样本主要用于匹配或排序任务，不适合作为 SFT 的标准回答，因此在 SFT 阶段删除。
2. **删除重复问答样本**去除重复上下文和重复回答，避免模型过度学习模板化表达。
3. **删除过短回复样本**删除回复长度小于 20 字的样本，减少“好的”“没有哦”“不可以呢”等信息不足的回答。
4. **删除前后矛盾或场景不匹配样本**去除答案内部存在逻辑冲突、与上下文不一致或明显场景不匹配的样本。
5. **补充任务指令和系统角色约束**
   将原始对话转换为更适合大模型训练的 instruction / messages 格式。

经过清洗后，最终保留约 **5800 条高质量 SFT 样本**。

### 5.3 SFT 数据格式

本项目采用 HuggingFace TRL 支持的 `messages` 格式，示例如下：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一名专业、礼貌、遵守平台规则的电商客服助手。回答需要先安抚用户，再给出清晰、可执行的处理建议，不能随意承诺退款、赔偿或补发。"
    },
    {
      "role": "user",
      "content": "请根据以下电商客服对话上下文，生成一段合适、礼貌、可执行的客服回复。\n\n场景类别：商品参数\n对话上下文：\n用户：为啥只有两袋啊不是199排三代嘛"
    },
    {
      "role": "assistant",
      "content": "客官，小店宝贝是拍下 3 件后自动改价为 199 元哦。您这边目前拍了 2 件，所以价格显示为 198 元。您可以再确认一下购买数量，拍够 3 件后再查看优惠价格哦。"
    }
  ]
}
```

---

## 6. 场景类别设计

项目覆盖的典型电商客服场景包括：

| 场景类别  | 示例问题                         |
| --------- | -------------------------------- |
| 退换货    | “我买错尺码了可以换吗？”       |
| 物流查询  | “为什么物流三天没更新？”       |
| 商品参数  | “这个是纯棉的吗？”             |
| 优惠活动  | “不是说满减吗，为什么没便宜？” |
| 售后处理  | “收到货坏了怎么办？”           |
| 投诉安抚  | “你们客服怎么一直不处理？”     |
| 发票/订单 | “可以开发票吗？”               |
| 付款/改价 | “拍下后能不能改价格？”         |

---

## 7. SFT-LoRA 微调

### 7.1 微调方式

本项目采用 LoRA 进行参数高效微调，而不是全参数微调。LoRA 可以在显著降低显存占用和训练成本的同时，使模型学习电商客服领域的表达方式和业务规则。

### 7.2 SFT 阶段学习目标

SFT 阶段主要让模型学习：

1. 电商客服常用表达方式；
2. 不同业务场景下的回答结构；
3. 礼貌、安抚、规范的客服语气；
4. 基于上下文生成合适客服回复的能力；
5. 避免不必要的承诺和高风险话术。

### 7.3 训练产物

训练完成后，主要保存 LoRA adapter 权重，而不是完整的 Qwen3-8B 模型权重。典型输出文件包括：

```text
adapter_model.safetensors
adapter_config.json
tokenizer.json
tokenizer_config.json
trainer_state.json
training_args.bin
```

其中：

- `adapter_model.safetensors`：LoRA 权重文件；
- `adapter_config.json`：LoRA 配置文件；
- `trainer_state.json`：训练日志和 loss 记录；
- `tokenizer.json`：分词器相关文件。

---

## 8. DPO 偏好优化

在完成 SFT 后，本项目进一步构造了 DPO 偏好数据，对模型进行偏好优化。

### 8.1 DPO 数据构造目标

DPO 阶段重点解决 SFT 后仍可能存在的问题，例如：

1. 回答虽然通顺，但业务处理不够准确；
2. 有安抚语气，但缺少可执行方案；
3. 处理建议过于笼统；
4. 对用户投诉或负面情绪回应不足；
5. 存在不合规承诺风险。

### 8.2 DPO 偏好数据规模

本项目构造并扩充了约 **2000 对偏好数据**，每条数据包含：

- `prompt`：用户问题或客服上下文；
- `chosen`：更优客服回复；
- `rejected`：较差客服回复。

偏好判断重点围绕：

1. 回答准确性；
2. 情绪安抚能力；
3. 可执行方案生成能力；
4. 业务规范性；
5. 风险控制能力。

### 8.3 DPO 优化目标

DPO 阶段希望模型更偏向生成：

- 更准确的业务解释；
- 更完整的处理流程；
- 更符合客服规范的表达；
- 更能安抚用户情绪的回复；
- 更少出现随意承诺、模糊应付或答非所问。

---

## 9. 模型评估

本项目使用自动指标、样本级对比和规则评估相结合的方式进行评估。

### 9.1 SFT 阶段评估

SFT 阶段主要关注：

- 训练 loss 是否稳定下降；
- eval loss 是否趋于稳定；
- 输出是否更贴合电商客服语气；
- 是否能根据不同场景生成有效回复。

### 9.2 DPO 阶段评估

DPO 阶段构建了约 100 条客服评估集，并从多个维度评估模型效果。

使用的评估指标包括：

| 指标         | 含义                                                   |
| ------------ | ------------------------------------------------------ |
| BERTScore F1 | 衡量生成回复与参考答案的语义相似度                     |
| Char-BLEU    | 衡量生成回复与参考答案的字符级重合程度                 |
| 样本级胜率   | 比较两个模型输出时，新模型优于旧模型的样本比例         |
| 规则命中率   | 检查回答是否包含关键客服行为，如安抚、解释、行动建议等 |
| 模拟人工评分 | 从准确性、安抚能力、可执行性等维度进行打分             |

### 9.3 实验结果摘要

在 DPO 对比实验中，第二阶段 DPO 模型相比第一阶段模型表现更优：

| 指标         | 第一阶段 DPO | 第二阶段 DPO |
| ------------ | -----------: | -----------: |
| BERTScore F1 |       0.7097 |       0.7260 |
| 相对提升     |            - |       +2.30% |
| 样本级胜率   |            - |          65% |

实验结果表明，经过扩充偏好数据和 DPO 优化后，模型在语义匹配度和样本级偏好表现上均有提升。

---

## 10. 项目目录结构

项目目录结构如下：

```text
qwen3-ecommerce-customer-service/
│
├── README.md
├── project_introduction.md
├── references.txt
├── evaluate_fixed_test_set.py
├── evaluation.py
├── fixed_eval_set_100.jsonl
├── pack.sh
├── train_sft_qwen3_8b_lora.py
├── train_sft_qwen3_8b_lora2.py
│
├── data/
│   ├── check_data.py
│   ├── cleaning_report_40000.md
│   ├── cleaning_report2_60000.md
│   ├── ecommerce_sft_alpaca_strict20_5800.json
│   ├── ecommerce_sft_messages_recommended_v3_eval.jsonl
│   ├── ecommerce_sft_messages_recommended_v3_eval_400_balanced.jsonl
│   ├── ecommerce_sft_messages_recommended_v3_train.jsonl
│   ├── ecommerce_sft_messages_recommended_v3_train_4000_balanced.jsonl
│   ├── ecommerce_sft_trl_messages_strict20_5800_eval.jsonl
│   └── ecommerce_sft_trl_messages_strict20_5800_train.jsonl
│
├── DPO/
│   ├── dpo_2000_usage_readme.md
│   ├── dpo_usage_readme.md
│   ├── ecommerce_dpo_2000_quality.jsonl
│   ├── ecommerce_dpo_2000_quality_stats.json
│   ├── ecommerce_dpo_500.jsonl
│   ├── ecommerce_dpo_500_stats.json
│   ├── evaluate_dpo_beta_sweep.sh
│   ├── run_dpo_beta_sweep.sh
│   ├── train_dpo_1epoch.py
│   └── train_dpo_one_beta.py
│
├── LoRA_ablation/
│   ├── ecommerce_eval_ablation_pack/
│   ├── evaluate_fixed_test_set.py
│   ├── fixed_eval_set_100.jsonl
│   ├── README_4000_ABLATION.md
│   ├── run_confirm_full_best.sh
│   ├── run_lora_ablation_fast4000.sh
│   ├── run_lora_ablation_fast4000_5runs.sh
│   ├── run_target_ablation_4000.sh
│   ├── summarize_ablation_results_fast.py
│   └── train_lora_ablation_fast.py
│
├── ablation_outputs_fast4000_5runs/
│   ├── ablation_summary_fast4000_5runs.csv
│   ├── ablation_summary_fast4000_5runs.md
│   ├── lora_qk_r4_fp16_alpha8/
│   ├── lora_qk_r16_fp16_alpha32/
│   ├── lora_qk_r64_fp16_alpha128/
│   ├── qlora4bit_all_linear_r16_bf16_alpha32/
│   │   ├── checkpoint-300/
│   │   └── checkpoint-500/
│   └── qlora4bit_qk_r16_bf16_alpha32/
│       ├── checkpoint-300/
│       └── checkpoint-500/
│
├── outputs/
│   ├── qwen3_8b_ecommerce_sft_lora/
│   ├── qwen3_8b_ecommerce_sft_lora2/
│   ├── dpo_lora_2000_beta005_1epoch/
│   ├── dpo_lora_2000_beta01_1epoch/
│   ├── dpo_lora_2000_beta03_1epoch/
│   └── dpo_lora_r16_1epoch/
│
├── results/
│   ├── evaluation_comparison.png
│   ├── dpo_evaluation_comparison.png
│   ├── dpo_lora_2000_beta005_eval.jsonl
│   ├── dpo_lora_2000_beta005_eval.summary.json
│   ├── dpo_lora_2000_beta01_eval.jsonl
│   ├── dpo_lora_2000_beta01_eval.summary.json
│   ├── dpo_lora_2000_beta03_eval.jsonl
│   ├── dpo_lora_2000_beta03_eval.summary.json
│   ├── dpo_lora_eval.jsonl
│   ├── dpo_lora_eval.summary.json
│   ├── lora1/
│   ├── lora2/
│   └── lora_ablation/
│
├── SFT_eval/
└── train/
```

### 10.1 目录说明

- `README.md`：项目总说明文档。
- `project_introduction.md`：项目设计、任务目标与实验总结。
- `references.txt`：参考资料与链接。
- `evaluate_fixed_test_set.py`：固定 100 条高风险测试集评估脚本，兼容 LoRA 模型评估。
- `evaluation.py`：评估分析辅助脚本。
- `fixed_eval_set_100.jsonl`：固定评估集，用于对比 SFT/DPO 模型表现。
- `pack.sh`：打包发布脚本，排除大文件和中间优化器文件。
- `train_sft_qwen3_8b_lora.py`：SFT-LoRA 实验 1，使用 `data/ecommerce_sft_trl_messages_strict20_5800_train.jsonl`。
- `train_sft_qwen3_8b_lora2.py`：SFT-LoRA 实验 2，使用 `data/ecommerce_sft_messages_recommended_v3_train.jsonl`。
- `data/`：数据清洗、格式转换和训练集/验证集文件。
- `DPO/`：DPO 训练脚本、偏好数据和 beta 对比脚本。
- `LoRA_ablation/`：LoRA 消融实验代码、快速消融数据集和比较脚本。
- `ablation_outputs_fast4000_5runs/`：快速消融实验结果目录。
- `outputs/`：SFT / DPO 训练产生的 LoRA adapter 权重目录。
- `results/`：评估结果文件、对比图表和实验汇总。
- `SFT_eval/`：SFT 固定评估集目录。
- `train/`：训练相关辅助文件或历史脚本目录。

### 10.2 执行说明

#### SFT-LoRA

本项目共有 2 次 SFT-LoRA 实验：

1. `train_sft_qwen3_8b_lora.py`：第一组 SFT 实验，使用 `data/ecommerce_sft_trl_messages_strict20_5800_train.jsonl` 进行训练，输出目录为 `outputs/qwen3_8b_ecommerce_sft_lora`。
2. `train_sft_qwen3_8b_lora2.py`：第二组 SFT 实验，使用 `data/ecommerce_sft_messages_recommended_v3_train.jsonl` 进行训练，输出目录为 `outputs/qwen3_8b_ecommerce_sft_lora2`。

运行示例：

```bash
python train_sft_qwen3_8b_lora.py
python train_sft_qwen3_8b_lora2.py
```

如果需要评估某个 SFT LoRA 模型，可使用：

```bash
python evaluate_fixed_test_set.py \
  --base_model /root/autodl-tmp/modelscope/Qwen/Qwen3-8B \
  --lora_dir outputs/qwen3_8b_ecommerce_sft_lora \
  --eval_file SFT_eval/fixed_eval_set_100.jsonl \
  --output_file results/qwen3_8b_ecommerce_sft_lora_eval.jsonl \
  --use_gold_category true
```

#### LoRA 消融实验

项目中仅有 1 组 LoRA 消融实验，集中在 `LoRA_ablation/` 目录：

- `LoRA_ablation/train_lora_ablation_fast.py`：快速 LoRA / QLoRA 消融训练脚本。
- `LoRA_ablation/README_4000_ABLATION.md`：快速消融数据集说明。
- `LoRA_ablation/run_lora_ablation_fast4000.sh`：4000 样本快速消融对比。
- `LoRA_ablation/run_target_ablation_4000.sh`：target module 消融对比。
- `LoRA_ablation/run_confirm_full_best.sh`：确认最优配置的全量训练。

快速实验数据文件包括：

- `data/ecommerce_sft_messages_recommended_v3_train_4000_balanced.jsonl`
- `data/ecommerce_sft_messages_recommended_v3_eval_400_balanced.jsonl`

运行示例：

```bash
bash LoRA_ablation/run_lora_ablation_fast4000.sh
bash LoRA_ablation/run_target_ablation_4000.sh
bash LoRA_ablation/run_confirm_full_best.sh
```

消融结果会保存到 `ablation_outputs_fast4000_5runs/`，并汇总到 `ablation_outputs_fast4000_5runs/ablation_summary_fast4000_5runs.md`。

#### DPO 实验

本项目包含 2 次 DPO 实验：

1. `DPO/train_dpo_1epoch.py`：第一轮 DPO 实验，默认使用 `DPO/ecommerce_dpo_500.jsonl`，输出目录为 `outputs/dpo_lora_r16_1epoch`。
2. `DPO/train_dpo_one_beta.py`：第二轮 DPO beta 对比实验，基于 `DPO/ecommerce_dpo_2000_quality.jsonl`，可运行多个 beta 值，生成 `outputs/dpo_lora_2000_beta005_1epoch`、`outputs/dpo_lora_2000_beta01_1epoch`、`outputs/dpo_lora_2000_beta03_1epoch` 等结果目录。

运行示例：

```bash
python DPO/train_dpo_1epoch.py
python DPO/train_dpo_one_beta.py --beta 0.1
```

或批量运行 beta 对比：

```bash
bash DPO/run_dpo_beta_sweep.sh
```

DPO 评估示例：

```bash
python evaluate_fixed_test_set.py \
  --base_model /root/autodl-tmp/modelscope/Qwen/Qwen3-8B \
  --lora_dir outputs/dpo_lora_r16_1epoch \
  --eval_file SFT_eval/fixed_eval_set_100.jsonl \
  --output_file results/dpo_lora_r16_1epoch_eval.jsonl \
  --use_gold_category true
```

批量评估 beta 对比结果：

```bash
bash DPO/evaluate_dpo_beta_sweep.sh
```

如果运行环境与脚本中的默认 `/root/firstTunning` 路径不一致，请先调整 `DPO/train_dpo_1epoch.py`、`DPO/train_dpo_one_beta.py`、`train_sft_qwen3_8b_lora.py`、`train_sft_qwen3_8b_lora2.py` 中的路径配置。

---

## 11. 项目亮点

1. **完整微调流程**覆盖数据清洗、SFT-LoRA、DPO、推理测试和多维评估。
2. **面向真实电商客服场景**数据场景包括退换货、物流、商品参数、优惠活动和投诉处理等。
3. **强调业务规范性**通过 system prompt、数据筛选和偏好优化，减少随意承诺退款、赔偿、补发等风险表达。
4. **引入偏好优化**在 SFT 基础上进一步构建 2000 对偏好数据，通过 DPO 提升回答质量。
5. **评估方式多样**
   不仅使用 BLEU 和 BERTScore，也引入样本级胜率、规则命中率和模拟人工评分，更贴近客服任务需求。

---

## 12. 当前结果

本项目实验表明，SFT-LoRA 可以显著提升模型对电商客服话术和业务场景的适配能力；在进一步进行 DPO 优化后，模型在回答准确性、情绪安抚和可执行方案生成方面进一步提升。

其中，第二阶段 DPO 模型在 65% 的评估样本上优于第一阶段模型，BERTScore F1 从 0.7097 提升至 0.7260，相对提升 2.30%。

---

## 13. 后续优化方向

后续可以进一步从以下方向改进：

1. 扩充更高质量的真实客服数据；
2. 将客服场景类别划分得更细；
3. 引入 RAG，使模型能够查询商品规则、售后政策和物流说明；
4. 建立更严格的人工评估标准；
5. 增加高风险问题检测，如退款、赔偿、投诉升级等；
6. 尝试多组 LoRA rank、target modules 和 DPO beta 参数对比实验。
