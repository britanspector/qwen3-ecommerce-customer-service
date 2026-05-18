#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DPO training script for e-commerce customer-service LoRA model.

This version is compatible with older TRL versions.

Key design:
1. Load base Qwen3-8B.
2. Load your SFT LoRA as the default trainable adapter.
3. Let old TRL create a reference adapter named "ref".
4. Copy weights from default adapter to ref adapter before training.
5. Train DPO for 1 epoch.
6. Save only the trained default policy adapter.
"""

import os
import json
import inspect
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    from peft import prepare_model_for_kbit_training
except Exception:
    prepare_model_for_kbit_training = None

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

from trl import DPOTrainer

try:
    from trl import DPOConfig
except Exception:
    from transformers import TrainingArguments as DPOConfig


# ============================================================
# 路径配置区
# ============================================================

BASE_MODEL_PATH = "/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"

# 推荐用你前面测试更稳的 checkpoint-2100 作为 DPO 起点
# 如果你想用最终 LoRA 目录，可以改成：
SFT_LORA_DIR = "/root/firstTunning/outputs/qwen3_8b_ecommerce_sft_lora2"
# SFT_LORA_DIR = "/root/firstTunning/outputs/qwen3_8b_ecommerce_sft_lora2/checkpoint-2100"

# DPO 数据路径
# 如果这个路径不存在，代码会自动尝试使用脚本同目录下的 ecommerce_dpo_500.jsonl
DPO_DATA_PATH = "/root/firstTunning/DPO/ecommerce_dpo_500.jsonl"

OUTPUT_DIR = "/root/firstTunning/outputs/dpo_lora_r16_1epoch"


# ============================================================
# 训练参数
# ============================================================

NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 5e-6
BETA = 0.1

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 768

BF16 = True
FP16 = False
USE_4BIT = False

LOCAL_FILES_ONLY = True
GRADIENT_CHECKPOINTING = True

LOGGING_STEPS = 5
SAVE_TOTAL_LIMIT = 2
SEED = 42


# ============================================================
# Prompt 配置
# ============================================================

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


def resolve_dpo_data_path() -> str:
    """Resolve DPO data path with fallback to script directory."""
    path = Path(DPO_DATA_PATH)

    if path.exists():
        return str(path)

    script_dir = Path(__file__).resolve().parent
    fallback = script_dir / "ecommerce_dpo_500.jsonl"

    if fallback.exists():
        print(f"[INFO] DPO_DATA_PATH 不存在，改用脚本同目录数据文件：{fallback}")
        return str(fallback)

    raise FileNotFoundError(
        f"DPO data file not found.\n"
        f"Checked:\n"
        f"1) {path}\n"
        f"2) {fallback}"
    )


def validate_lora_dir(lora_dir: str):
    """Check whether LoRA adapter files exist."""
    path = Path(lora_dir)

    if not path.exists():
        raise FileNotFoundError(f"SFT_LORA_DIR 不存在：{lora_dir}")

    config_file = path / "adapter_config.json"
    safetensors_file = path / "adapter_model.safetensors"
    bin_file = path / "adapter_model.bin"

    if not config_file.exists():
        raise FileNotFoundError(f"没有找到 adapter_config.json：{config_file}")

    if not safetensors_file.exists() and not bin_file.exists():
        raise FileNotFoundError(
            f"没有找到 adapter_model.safetensors 或 adapter_model.bin：{lora_dir}"
        )


def print_config():
    data_path = resolve_dpo_data_path()

    print("========== DPO Config ==========")
    print(f"BASE_MODEL_PATH = {BASE_MODEL_PATH}")
    print(f"SFT_LORA_DIR    = {SFT_LORA_DIR}")
    print(f"DPO_DATA_PATH   = {data_path}")
    print(f"OUTPUT_DIR      = {OUTPUT_DIR}")
    print(f"epochs          = {NUM_TRAIN_EPOCHS}")
    print(f"learning_rate   = {LEARNING_RATE}")
    print(f"beta            = {BETA}")
    print(f"bf16            = {BF16}")
    print(f"4bit            = {USE_4BIT}")
    print("================================")

    return data_path


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def apply_qwen_chat_template(tokenizer, user_content: str) -> str:
    """Apply Qwen chat template and disable thinking when possible."""
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


def normalize_user_prompt(raw_prompt: str) -> str:
    """Convert raw prompt into user content if needed."""
    raw_prompt = str(raw_prompt).strip()

    if "用户问题：" in raw_prompt:
        return raw_prompt

    return (
        "请先从以下类别中选择最合适的一个场景分类，然后给出客服回复：\n"
        + "、".join(CATEGORIES)
        + f"\n用户问题：{raw_prompt}"
    )


def format_one_example(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """
    Format one DPO example into:
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "..."
    }
    """

    chosen = str(example.get("chosen", "")).strip()
    rejected = str(example.get("rejected", "")).strip()

    if not chosen or not rejected:
        raise ValueError(f"Invalid DPO sample, chosen/rejected is empty: {example}")

    # Case 1: already has formatted chat prompt
    if "prompt" in example and example["prompt"]:
        raw_prompt = str(example["prompt"]).strip()

        if "<|im_start|>" in raw_prompt or "<|start_header_id|>" in raw_prompt:
            prompt = raw_prompt
        else:
            user_content = normalize_user_prompt(raw_prompt)
            prompt = apply_qwen_chat_template(tokenizer, user_content)

    # Case 2: generated DPO data usually uses user_prompt
    elif "user_prompt" in example and example["user_prompt"]:
        user_content = normalize_user_prompt(str(example["user_prompt"]).strip())
        prompt = apply_qwen_chat_template(tokenizer, user_content)

    # Case 3: fallback to question
    elif "question" in example and example["question"]:
        user_content = normalize_user_prompt(str(example["question"]).strip())
        prompt = apply_qwen_chat_template(tokenizer, user_content)

    else:
        raise ValueError(f"Invalid DPO sample, no prompt/user_prompt/question: {example}")

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def load_dpo_dataset(tokenizer, data_path: str):
    print(f"[INFO] Loading DPO data from: {data_path}")

    dataset = load_dataset(
        "json",
        data_files=data_path,
        split="train",
    )

    original_columns = dataset.column_names

    dataset = dataset.map(
        lambda x: format_one_example(x, tokenizer),
        remove_columns=original_columns,
        desc="Applying Qwen chat template",
    )

    print(f"[INFO] DPO samples: {len(dataset)}")
    print("[INFO] One formatted sample:")
    print(json.dumps(dataset[0], ensure_ascii=False, indent=2)[:3000])

    return dataset


def load_policy_model():
    validate_lora_dir(SFT_LORA_DIR)

    dtype = torch.bfloat16 if BF16 else torch.float16

    print(f"[INFO] Loading base model: {BASE_MODEL_PATH}")

    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": LOCAL_FILES_ONLY,
        "device_map": "auto",
        "torch_dtype": dtype,
    }

    if USE_4BIT:
        if BitsAndBytesConfig is None:
            raise ImportError("USE_4BIT=True 需要安装 bitsandbytes，并确保 BitsAndBytesConfig 可用。")

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        **model_kwargs,
    )

    base_model.config.use_cache = False

    if USE_4BIT:
        if prepare_model_for_kbit_training is None:
            raise ImportError("当前 peft 版本不支持 prepare_model_for_kbit_training。")
        base_model = prepare_model_for_kbit_training(base_model)

    if GRADIENT_CHECKPOINTING:
        try:
            base_model.gradient_checkpointing_enable()
        except Exception as e:
            print(f"[WARN] gradient_checkpointing_enable failed: {e}")

    print(f"[INFO] Loading trainable SFT adapter as default adapter: {SFT_LORA_DIR}")

    # 关键：不要传 adapter_name
    # 这样 PEFT 会把 SFT LoRA 加载为 default adapter
    # 旧版 TRL 正好需要 model.peft_config["default"]
    model = PeftModel.from_pretrained(
        base_model,
        SFT_LORA_DIR,
        is_trainable=True,
    )

    model.set_adapter("default")
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception as e:
            print(f"[WARN] enable_input_require_grads failed: {e}")

    model.print_trainable_parameters()
    return model


def filter_supported_kwargs(func, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Keep only kwargs supported by a function/class __init__ signature."""
    try:
        sig = inspect.signature(func)
        params = sig.parameters

        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )

        if has_var_kwargs:
            return kwargs, []

        supported = {k: v for k, v in kwargs.items() if k in params}
        ignored = [k for k in kwargs.keys() if k not in params]
        return supported, ignored

    except Exception:
        return kwargs, []


def make_dpo_config():
    candidate_kwargs = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": LOGGING_STEPS,
        "save_strategy": "epoch",
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "bf16": BF16,
        "fp16": FP16,
        "optim": "adamw_torch",
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "report_to": "none",
        "remove_unused_columns": False,
        "max_grad_norm": 1.0,

        # 新版 TRL/DPOConfig 可能支持这些
        # 旧版不支持时会自动忽略，然后尽量传给 DPOTrainer
        "beta": BETA,
        "max_length": MAX_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
    }

    supported_kwargs, ignored = filter_supported_kwargs(
        DPOConfig.__init__,
        candidate_kwargs,
    )

    if ignored:
        print(f"[WARN] 当前 TRL/DPOConfig 不识别这些参数，已忽略：{ignored}")

    args = DPOConfig(**supported_kwargs)
    return args, set(supported_kwargs.keys())


def make_trainer_kwargs(model, tokenizer, train_dataset, dpo_args, config_supported_keys):
    candidate_kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": train_dataset,
    }

    trainer_params = inspect.signature(DPOTrainer.__init__).parameters

    # 新版 TRL 可能叫 processing_class，旧版通常叫 tokenizer
    if "processing_class" in trainer_params:
        candidate_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        candidate_kwargs["tokenizer"] = tokenizer

    # 如果 DPOConfig 不支持 beta/max_length/max_prompt_length，
    # 但 DPOTrainer 支持，就传给 DPOTrainer
    if "beta" not in config_supported_keys and "beta" in trainer_params:
        candidate_kwargs["beta"] = BETA

    if "max_length" not in config_supported_keys and "max_length" in trainer_params:
        candidate_kwargs["max_length"] = MAX_LENGTH

    if "max_prompt_length" not in config_supported_keys and "max_prompt_length" in trainer_params:
        candidate_kwargs["max_prompt_length"] = MAX_PROMPT_LENGTH

    supported_kwargs, ignored = filter_supported_kwargs(
        DPOTrainer.__init__,
        candidate_kwargs,
    )

    if ignored:
        print(f"[WARN] 当前 TRL/DPOTrainer 不识别这些参数，已忽略：{ignored}")

    return supported_kwargs


def copy_default_adapter_to_ref_adapter(model):
    """
    Old TRL creates a reference adapter named "ref".
    To make reference = SFT LoRA instead of random LoRA,
    copy weights from default adapter to ref adapter after trainer initialization.
    """

    if not hasattr(model, "peft_config"):
        print("[WARN] model has no peft_config, skip adapter weight copy.")
        return

    adapter_names = list(model.peft_config.keys())

    if "default" not in adapter_names:
        print(f"[WARN] default adapter not found. Current adapters: {adapter_names}")
        return

    if "ref" not in adapter_names:
        print(f"[WARN] ref adapter not found. Current adapters: {adapter_names}")
        print("[WARN] 如果当前 TRL 没有创建 ref adapter，则跳过复制。")
        return

    print("[INFO] Copying adapter weights: default -> ref")

    state_dict = model.state_dict()
    copy_state = {}
    copied = 0

    for key, value in state_dict.items():
        if ".default." in key:
            ref_key = key.replace(".default.", ".ref.")
            if ref_key in state_dict:
                copy_state[ref_key] = value.detach().clone()
                copied += 1

    if copied == 0:
        print("[WARN] 没有找到可复制的 default -> ref adapter 权重。")
        return

    missing, unexpected = model.load_state_dict(copy_state, strict=False)

    print(f"[INFO] Copied {copied} tensors from default adapter to ref adapter.")

    # 冻结 ref adapter，确保只训练 default policy adapter
    frozen = 0
    for name, param in model.named_parameters():
        if ".ref." in name:
            param.requires_grad = False
            frozen += 1

    print(f"[INFO] Frozen ref adapter parameters: {frozen}")

    # 确保当前训练 adapter 是 default
    try:
        model.set_adapter("default")
    except Exception as e:
        print(f"[WARN] set_adapter('default') failed: {e}")


def save_policy_adapter_only(model, tokenizer):
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving trained policy adapter to: {OUTPUT_DIR}")

    try:
        model.set_adapter("default")
    except Exception:
        pass

    # 多 adapter 情况下，只保存 default，避免把 ref adapter 也保存进去
    try:
        model.save_pretrained(
            OUTPUT_DIR,
            selected_adapters=["default"],
            safe_serialization=True,
        )
    except TypeError:
        print("[WARN] 当前 PEFT 版本不支持 selected_adapters，改用普通 save_pretrained。")
        model.save_pretrained(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print("[INFO] Save finished.")
    print(f"[INFO] You can evaluate with lora_dir = {OUTPUT_DIR}")


def main():
    torch.manual_seed(SEED)

    data_path = print_config()

    tokenizer = load_tokenizer()
    train_dataset = load_dpo_dataset(tokenizer, data_path)

    model = load_policy_model()

    dpo_args, config_supported_keys = make_dpo_config()

    trainer_kwargs = make_trainer_kwargs(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dpo_args=dpo_args,
        config_supported_keys=config_supported_keys,
    )

    print("[INFO] Initializing DPOTrainer...")
    trainer = DPOTrainer(**trainer_kwargs)

    # 关键：旧版 TRL 初始化时会创建 ref adapter。
    # 这里把 SFT default adapter 权重复制给 ref adapter，保证 reference 也是 SFT。
    copy_default_adapter_to_ref_adapter(model)

    print("[INFO] Start DPO training...")
    trainer.train()

    save_policy_adapter_only(model, tokenizer)

    print("[INFO] DPO training completed.")


if __name__ == "__main__":
    main()