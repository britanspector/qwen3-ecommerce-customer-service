# 4000 样本快速消融数据集与代码说明

## 数据来源

- 原推荐训练集：`ecommerce_sft_messages_recommended_v3_train.jsonl`
- 原推荐验证集：`ecommerce_sft_messages_recommended_v3_eval.jsonl`

## 4000 精选训练集

输出文件：

- `ecommerce_sft_messages_recommended_v3_train_4000_balanced.jsonl`

构造方式：

- 从原 14448 条训练集中按场景分层抽样；
- 保留全部/尽量保留高风险小类；
- 对质量投诉、退换货、错发漏发、签收未收到等第一次 LoRA 容易出错的场景做最低配额保护；
- 不做重复采样，不复制样本。

## 400 验证子集

输出文件：

- `ecommerce_sft_messages_recommended_v3_eval_400_balanced.jsonl`

用途：

- 快速消融时减少 eval 时间；
- 最终正式模型仍建议回到完整 eval 集和固定 100 条高风险测试集。

## 4000 训练集场景分布

- 物流查询场景：747
- 商品参数场景：663
- 退换货场景：600
- 质量投诉场景：500
- 订单修改场景：471
- 价格优惠场景：464
- 其他咨询场景：255
- 错发漏发场景：250
- 签收未收到场景：50

## 400 验证集场景分布

- 物流查询场景：89
- 商品参数场景：78
- 价格优惠场景：57
- 订单修改场景：50
- 退换货场景：49
- 质量投诉场景：33
- 其他咨询场景：28
- 错发漏发场景：12
- 签收未收到场景：4


## 推荐消融流程

### 第一阶段：4000 样本快速消融

运行：

```bash
cd /root/firstTunning/ablation_code_4000
bash run_lora_ablation_fast4000.sh
```

对比：

- LoRA q/k target, r=4
- LoRA q/k target, r=16
- LoRA q/k target, r=64
- QLoRA 4bit q/k target, r=4
- QLoRA 4bit q/k target, r=16
- QLoRA 4bit q/k target, r=64

输出：

```bash
/root/firstTunning/ablation_outputs_fast4000/ablation_summary_fast4000.md
```

### 第二阶段：可选 target 消融

运行：

```bash
bash run_target_ablation_4000.sh
```

对比：

- qk：q_proj, k_proj
- qv：q_proj, v_proj
- attn：q_proj, k_proj, v_proj, o_proj
- all_linear：q/k/v/o + gate/up/down

这一步不建议一开始就跑全量 14448 条。

### 第三阶段：全量确认

等 4000 样本确定最优方向后，只挑 1-2 组配置跑完整 14448 条。

运行示例：

```bash
bash run_confirm_full_best.sh
```

## 为什么不建议全组合都跑 14448 条？

因为你目前 q/k target 跑完整训练已经约 2 小时。若再加上：

- r=4/16/64
- LoRA/QLoRA
- qk/qv/attn/all_linear target

组合数会很快膨胀，时间成本过高。更合理的做法是：

1. 用 4000 条做快速相对比较；
2. 用固定 100 条高风险测试集看规则与风险；
3. 只把最有希望的 1-2 组配置放到 14448 条全量训练。
