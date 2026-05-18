#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed test-set evaluator for e-commerce LoRA/QLoRA models.

It supports two modes:
1) use_gold_category=True:
   The prompt includes the known category, testing whether the model can respond correctly under a given scene.
2) use_gold_category=False:
   The prompt asks the model to infer category + reply, testing category classification and reply quality.

Example:
python evaluate_fixed_test_set.py \
  --base_model /root/autodl-tmp/modelscope/Qwen/Qwen3-8B \
  --lora_dir /root/firstTunning/outputs/exp_lora_r16_fp16 \
  --eval_file /root/firstTunning/eval/fixed_eval_set_100.jsonl \
  --output_file /root/firstTunning/eval_results/lora_r16_eval.jsonl \
  --use_gold_category true
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


CATEGORIES = [
    "退换货场景", "物流查询场景", "签收未收到场景", "错发漏发场景",
    "商品参数场景", "质量投诉场景", "订单修改场景", "价格优惠场景", "其他咨询场景"
]


SYSTEM_PROMPT = """你是一个专业、谨慎、负责的电商客服助手。
你必须遵守以下原则：
1. 先识别场景，再给出客服回复。
2. 食品、母婴、投诉、退款、物流异常等高风险场景必须谨慎处理。
3. 不得编造商品参数、库存、快递时效、优惠金额、退款金额。
4. 不得未经核实直接承诺退款、补发、赔偿或判定丢件。
5. 食品疑似变质、有虫、发霉、异味时，应提醒用户先不要继续食用，并提供照片/视频等凭证以便售后核实。
6. 商品参数不确定时，应说明以详情页、包装标识或官方参数为准，并建议进一步核实。
输出格式必须为：
场景分类：xxx
客服回复：xxx
"""
# =========================
# 路径配置区：直接在这里修改
# =========================

# 基础模型路径
BASE_MODEL_PATH = "/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"

# LoRA / QLoRA 训练结果路径
# 如果只想测试原始基座模型，把这里改成 None
LORA_DIR = "/root/firstTunning/outputs/dpo_lora_r16_1epoch"

# 固定测试集路径
EVAL_FILE = "/root/firstTunning/SFT_eval/fixed_eval_set_100.jsonl"

# 输出结果路径
OUTPUT_FILE = "/root/firstTunning/results/dpo_lora_eval.jsonl"

# 是否使用已知类别进行测试
USE_GOLD_CATEGORY = True

# 精度设置：可选 "bf16" 或 "fp16"
DTYPE = "bf16"

# 最大生成 token 数
MAX_NEW_TOKENS = 256

# 是否只加载本地文件
LOCAL_FILES_ONLY = True

def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y"}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tokenizer, item, use_gold_category: bool):
    if use_gold_category:
        user_content = f"场景分类：{item['category']}\n用户问题：{item['question']}"
    else:
        user_content = (
            "请先从以下类别中选择最合适的一个场景分类，然后给出客服回复：\n"
            + "、".join(CATEGORIES)
            + f"\n用户问题：{item['question']}"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def extract_category(text: str):
    m = re.search(r"场景分类[:：]\s*([^\n\r]+)", text)
    if not m:
        return ""
    raw = m.group(1).strip()
    for cat in CATEGORIES:
        if cat in raw:
            return cat
    return raw[:20]


def contains_any(text, terms):
    return any(t and t in text for t in terms)


def score_output(item, output, use_gold_category: bool):
    output_norm = re.sub(r"\s+", "", output)

    # Each group in must_have_any requires at least one matched keyword.
    must_groups = item.get("must_have_any", [])
    must_hit_count = 0
    missed_groups = []
    for group in must_groups:
        if contains_any(output_norm, group):
            must_hit_count += 1
        else:
            missed_groups.append(group)

    forbidden = item.get("forbidden", [])
    forbidden_hits = [x for x in forbidden if x and x in output_norm]

    pred_cat = extract_category(output)
    category_correct = None
    if not use_gold_category:
        category_correct = int(pred_cat == item["category"])

    rule_score = must_hit_count / max(1, len(must_groups))
    risky = int(len(forbidden_hits) > 0)

    return {
        "pred_category": pred_cat,
        "category_correct": category_correct,
        "must_hit_count": must_hit_count,
        "must_total": len(must_groups),
        "rule_score": rule_score,
        "missed_groups": missed_groups,
        "forbidden_hits": forbidden_hits,
        "risky": risky,
    }


def generate(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    # =========================
    # 默认使用代码中定义的路径
    # 同时保留命令行参数覆盖能力
    # =========================
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", default=BASE_MODEL_PATH)
    parser.add_argument("--lora_dir", default=LORA_DIR)
    parser.add_argument("--eval_file", default=EVAL_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--use_gold_category", default=str(USE_GOLD_CATEGORY).lower())
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default=DTYPE)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--local_files_only", default=str(LOCAL_FILES_ONLY).lower())

    args = parser.parse_args()

    use_gold_category = str2bool(args.use_gold_category)
    local_files_only = str2bool(args.local_files_only)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.eval()

    if args.lora_dir:
        model = PeftModel.from_pretrained(model, args.lora_dir)
        model.eval()

    data = load_jsonl(args.eval_file)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    results = []

    for i, item in enumerate(data, 1):
        prompt = build_prompt(tokenizer, item, use_gold_category)
        output = generate(model, tokenizer, prompt, args.max_new_tokens)
        score = score_output(item, output, use_gold_category)

        row = {
            "id": item["id"],
            "category": item["category"],
            "risk_level": item.get("risk_level", ""),
            "question": item["question"],
            "output": output,
            **score,
        }

        results.append(row)

        print(
            f"[{i}/{len(data)}] "
            f"{item['id']} | "
            f"rule={score['rule_score']:.2f} | "
            f"risky={score['risky']}"
        )

    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results)
    avg_rule = sum(r["rule_score"] for r in results) / n
    risky_rate = sum(r["risky"] for r in results) / n

    summary = {
        "num_samples": n,
        "avg_rule_score": avg_rule,
        "risky_response_rate": risky_rate,
        "use_gold_category": use_gold_category,
    }

    if not use_gold_category:
        valid = [r for r in results if r["category_correct"] is not None]
        summary["category_accuracy"] = sum(r["category_correct"] for r in valid) / max(1, len(valid))

    summary_path = str(Path(args.output_file).with_suffix(".summary.json"))

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nEvaluation summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved details to: {args.output_file}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
