import json
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


base_model_path = "/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"
lora_adapter_path = "/root/firstTunning/outputs/qwen3_8b_ecommerce_sft_lora"

result_dir = Path("/root/firstTunning/results/lora1")
result_dir.mkdir(parents=True, exist_ok=True)

system_prompt = "你是一名专业、礼貌、遵守平台规则的电商客服助手。请根据用户问题给出清晰、礼貌、可执行的客服回复。不得编造商品参数，不得直接承诺退款、赔偿、补发，涉及食品安全、错发、未收到货，需要先核实证据，如果信息不足，要说明需要进一步确认"


test_cases = [
    {
        "category": "退换货",
        "question": "用户问：我买的食品已经拆封了，吃了一点觉得不喜欢，还能七天无理由退货吗？\n商品类别：食品；订单状态：已签收；特殊情况：已拆封"
    },
    {
        "category": "退换货",
        "question": "用户问：我买的衣服尺码不合适，吊牌还在，也没有洗过，可以换大一码吗？\n商品类别：服饰；订单状态：已签收；特殊情况：吊牌完整"
    },
    {
        "category": "物流查询",
        "question": "用户问：我的快递三天没有更新物流了，是不是丢件了？\n订单状态：已发货；物流状态：超过72小时未更新"
    },
    {
        "category": "物流查询",
        "question": "用户问：物流显示已经签收，但是我根本没有收到包裹，怎么办？\n订单状态：显示已签收；用户反馈：未收到商品"
    },
    {
        "category": "商品参数",
        "question": "用户问：这个婴儿湿巾是纯棉的吗？新生儿可以用来擦脸和擦屁股吗？\n商品类别：母婴用品；用户关注点：材质和适用人群"
    },
    {
        "category": "商品参数",
        "question": "用户问：这个充电器支持快充吗？可以给苹果手机用吗？\n商品类别：数码配件；用户关注点：快充协议和设备兼容"
    },
    {
        "category": "投诉处理",
        "question": "用户问：我收到的核桃很多都是坏的，还有虫子，你们必须马上给我退款，不然我就投诉。\n商品类别：食品；用户情绪：强烈不满；问题：疑似质量问题"
    },
    {
        "category": "投诉处理",
        "question": "用户问：你们发错货了，我买的是茶盒，结果收到的是别的东西，太耽误事了。\n订单状态：已签收；问题：错发商品；用户情绪：不满"
    }
]


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Qwen3 新版 tokenizer 可能支持 enable_thinking=False
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return text


def generate_answers(model, tokenizer, model_tag):
    model.eval()
    results = []

    for idx, item in enumerate(test_cases, start=1):
        prompt = build_prompt(tokenizer, item["question"])

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print("=" * 80)
        print(f"{model_tag} | 样本 {idx} | {item['category']}")
        print("问题：")
        print(item["question"])
        print("回答：")
        print(answer)

        results.append({
            "id": idx,
            "category": item["category"],
            "question": item["question"],
            "answer": answer,
        })

    return results


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    return model, tokenizer


def load_lora_model():
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        local_files_only=True,
    )

    return model, tokenizer


def release_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def save_results(base_results, lora_results):
    json_path = result_dir / "base_vs_lora_results.json"
    md_path = result_dir / "base_vs_lora_comparison.md"

    paired_results = []

    for base_item, lora_item in zip(base_results, lora_results):
        paired_results.append({
            "id": base_item["id"],
            "category": base_item["category"],
            "question": base_item["question"],
            "base_answer": base_item["answer"],
            "lora_answer": lora_item["answer"],
        })

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(paired_results, f, ensure_ascii=False, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Base Qwen3-8B vs LoRA SFT 推理对比\n\n")

        for item in paired_results:
            f.write(f"## 样本 {item['id']}：{item['category']}\n\n")
            f.write("### 用户问题\n\n")
            f.write(item["question"] + "\n\n")

            f.write("### Base Qwen3-8B 回答\n\n")
            f.write(item["base_answer"] + "\n\n")

            f.write("### LoRA SFT 后回答\n\n")
            f.write(item["lora_answer"] + "\n\n")

            f.write("---\n\n")

    print(f"JSON 结果已保存：{json_path}")
    print(f"Markdown 对比结果已保存：{md_path}")


def main():
    # print("开始加载 Base Qwen3-8B...")
    # base_model, base_tokenizer = load_base_model()
    # base_results = generate_answers(base_model, base_tokenizer, "Base Qwen3-8B")
    # release_model(base_model, base_tokenizer)

    print("\nBase 模型推理完成，开始加载 LoRA 模型...")
    lora_model, lora_tokenizer = load_lora_model()
    lora_results = generate_answers(lora_model, lora_tokenizer, "LoRA SFT")
    release_model(lora_model, lora_tokenizer)

    save_results(base_results, lora_results)


if __name__ == "__main__":
    main()