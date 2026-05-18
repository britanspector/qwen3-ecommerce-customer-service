# 电商客服 SFT 数据清洗报告（前 40000 条 label=1）

## 数据来源

- `ecommerce_label1_20000.txt`：20000 行
- `ecommerce_label1_20001_40000.txt`：20000 行

## 合并去重

- 合并原始行数：40000
- 去重后唯一行数：37589
- 去除重复行数：2411
- 可解析 label=1 样本：37589

## 清洗版本

- 全量（回复不少于20个有效字符）：6112 条
- 精选（回复不少于20个有效字符，按质量分选前5800条）：5800 条
- 扩展（回复不少于12个有效字符，按质量分选前10000条）：10000 条

## 场景分布

- 物流查询：1936
- 投诉处理：1748
- 商品参数：1642
- 退换货：474

## 推荐使用

- TRL 正式 SFT：`ecommerce_sft_trl_messages_strict20_5800_train.jsonl` + `ecommerce_sft_trl_messages_strict20_5800_eval.jsonl`
- 备选全量严格训练：`ecommerce_sft_trl_messages_strict_all.jsonl`
- 扩大样本训练或 smoke test：`ecommerce_sft_trl_messages_relaxed12_10000.jsonl`
