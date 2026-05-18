#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO training script for e-commerce customer-service LoRA model.
Compatible with older TRL versions.
Run one beta each time, e.g.:
python DPO/train_dpo_one_beta.py --beta 0.1
"""
import argparse
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

BASE_MODEL_PATH = "/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"
SFT_LORA_DIR = "/root/firstTunning/outputs/qwen3_8b_ecommerce_sft_lora2"
DPO_DATA_PATH = "/root/firstTunning/DPO/ecommerce_dpo_2000_quality.jsonl"
OUTPUT_ROOT = "/root/firstTunning/outputs"

NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 5e-6
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=True, choices=[0.05, 0.1, 0.3])
    parser.add_argument("--base_model", default=BASE_MODEL_PATH)
    parser.add_argument("--sft_lora_dir", default=SFT_LORA_DIR)
    parser.add_argument("--dpo_data_path", default=DPO_DATA_PATH)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=float, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--use_4bit", action="store_true")
    return parser.parse_args()

def beta_tag(beta: float) -> str:
    if abs(beta - 0.05) < 1e-9:
        return "beta005"
    if abs(beta - 0.1) < 1e-9:
        return "beta01"
    if abs(beta - 0.3) < 1e-9:
        return "beta03"
    return ("beta" + str(beta).replace(".", ""))

def resolve_dpo_data_path(path_str: str) -> str:
    path = Path(path_str)
    if path.exists():
        return str(path)
    script_dir = Path(__file__).resolve().parent
    fallback = script_dir / "ecommerce_dpo_2000_quality.jsonl"
    if fallback.exists():
        print(f"[INFO] DPO_DATA_PATH 不存在，改用脚本同目录数据文件：{fallback}")
        return str(fallback)
    raise FileNotFoundError(f"DPO data file not found: {path} or {fallback}")

def validate_lora_dir(lora_dir: str):
    path = Path(lora_dir)
    if not path.exists():
        raise FileNotFoundError(f"SFT_LORA_DIR 不存在：{lora_dir}")
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"没有找到 adapter_config.json：{path}")
    if not (path / "adapter_model.safetensors").exists() and not (path / "adapter_model.bin").exists():
        raise FileNotFoundError(f"没有找到 adapter_model.safetensors 或 adapter_model.bin：{path}")

def load_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=LOCAL_FILES_ONLY)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def apply_qwen_chat_template(tokenizer, user_content: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def normalize_user_prompt(raw_prompt: str) -> str:
    raw_prompt = str(raw_prompt).strip()
    if "用户问题：" in raw_prompt:
        return raw_prompt
    return "请先从以下类别中选择最合适的一个场景分类，然后给出客服回复：\n" + "、".join(CATEGORIES) + f"\n用户问题：{raw_prompt}"

def format_one_example(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    chosen = str(example.get("chosen", "")).strip()
    rejected = str(example.get("rejected", "")).strip()
    if not chosen or not rejected:
        raise ValueError(f"Invalid DPO sample: {example}")
    if "prompt" in example and example["prompt"]:
        raw_prompt = str(example["prompt"]).strip()
        prompt = raw_prompt if "<|im_start|>" in raw_prompt else apply_qwen_chat_template(tokenizer, normalize_user_prompt(raw_prompt))
    elif "user_prompt" in example and example["user_prompt"]:
        prompt = apply_qwen_chat_template(tokenizer, normalize_user_prompt(str(example["user_prompt"]).strip()))
    elif "question" in example and example["question"]:
        prompt = apply_qwen_chat_template(tokenizer, normalize_user_prompt(str(example["question"]).strip()))
    else:
        raise ValueError(f"No prompt/user_prompt/question: {example}")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

def load_dpo_dataset(tokenizer, data_path: str):
    print(f"[INFO] Loading DPO data from: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")
    original_columns = dataset.column_names
    dataset = dataset.map(lambda x: format_one_example(x, tokenizer), remove_columns=original_columns, desc="Applying Qwen chat template")
    print(f"[INFO] DPO samples: {len(dataset)}")
    print("[INFO] One formatted sample:")
    print(json.dumps(dataset[0], ensure_ascii=False, indent=2)[:2500])
    return dataset

def load_policy_model(args):
    validate_lora_dir(args.sft_lora_dir)
    dtype = torch.bfloat16 if BF16 else torch.float16
    print(f"[INFO] Loading base model: {args.base_model}")
    model_kwargs = {"trust_remote_code": True, "local_files_only": LOCAL_FILES_ONLY, "device_map": "auto", "torch_dtype": dtype}
    if args.use_4bit or USE_4BIT:
        if BitsAndBytesConfig is None:
            raise ImportError("USE_4BIT=True 需要安装 bitsandbytes。")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    base_model.config.use_cache = False
    if args.use_4bit or USE_4BIT:
        if prepare_model_for_kbit_training is None:
            raise ImportError("当前 peft 版本不支持 prepare_model_for_kbit_training。")
        base_model = prepare_model_for_kbit_training(base_model)
    if GRADIENT_CHECKPOINTING:
        try:
            base_model.gradient_checkpointing_enable()
        except Exception as e:
            print(f"[WARN] gradient_checkpointing_enable failed: {e}")
    print(f"[INFO] Loading trainable SFT adapter as default adapter: {args.sft_lora_dir}")
    model = PeftModel.from_pretrained(base_model, args.sft_lora_dir, is_trainable=True)
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
    try:
        sig = inspect.signature(func)
        params = sig.parameters
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_var_kwargs:
            return kwargs, []
        return {k: v for k, v in kwargs.items() if k in params}, [k for k in kwargs if k not in params]
    except Exception:
        return kwargs, []

def make_dpo_config(args, output_dir: str):
    candidate_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": args.learning_rate,
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
        "beta": args.beta,
        "max_length": MAX_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
    }
    supported_kwargs, ignored = filter_supported_kwargs(DPOConfig.__init__, candidate_kwargs)
    if ignored:
        print(f"[WARN] 当前 TRL/DPOConfig 不识别这些参数，已忽略：{ignored}")
    return DPOConfig(**supported_kwargs), set(supported_kwargs.keys())

def make_trainer_kwargs(model, tokenizer, train_dataset, dpo_args, config_supported_keys, beta):
    candidate_kwargs = {"model": model, "ref_model": None, "args": dpo_args, "train_dataset": train_dataset}
    trainer_params = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        candidate_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        candidate_kwargs["tokenizer"] = tokenizer
    if "beta" not in config_supported_keys and "beta" in trainer_params:
        candidate_kwargs["beta"] = beta
    if "max_length" not in config_supported_keys and "max_length" in trainer_params:
        candidate_kwargs["max_length"] = MAX_LENGTH
    if "max_prompt_length" not in config_supported_keys and "max_prompt_length" in trainer_params:
        candidate_kwargs["max_prompt_length"] = MAX_PROMPT_LENGTH
    supported_kwargs, ignored = filter_supported_kwargs(DPOTrainer.__init__, candidate_kwargs)
    if ignored:
        print(f"[WARN] 当前 TRL/DPOTrainer 不识别这些参数，已忽略：{ignored}")
    return supported_kwargs

def copy_default_adapter_to_ref_adapter(model):
    if not hasattr(model, "peft_config"):
        print("[WARN] model has no peft_config, skip adapter copy.")
        return
    names = list(model.peft_config.keys())
    if "default" not in names:
        print(f"[WARN] default adapter not found. Current adapters: {names}")
        return
    if "ref" not in names:
        print(f"[WARN] ref adapter not found. Current adapters: {names}")
        print("[WARN] 如果当前 TRL 没有创建 ref adapter，则跳过复制。")
        return
    print("[INFO] Copying adapter weights: default -> ref")
    state_dict = model.state_dict()
    copy_state = {}
    for key, value in state_dict.items():
        if ".default." in key:
            ref_key = key.replace(".default.", ".ref.")
            if ref_key in state_dict:
                copy_state[ref_key] = value.detach().clone()
    if not copy_state:
        print("[WARN] 没有找到可复制的 default -> ref adapter 权重。")
        return
    model.load_state_dict(copy_state, strict=False)
    frozen = 0
    for name, param in model.named_parameters():
        if ".ref." in name:
            param.requires_grad = False
            frozen += 1
    print(f"[INFO] Copied {len(copy_state)} tensors and frozen {frozen} ref adapter parameters.")
    try:
        model.set_adapter("default")
    except Exception as e:
        print(f"[WARN] set_adapter('default') failed: {e}")

def save_policy_adapter_only(model, tokenizer, output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving trained policy adapter to: {output_dir}")
    try:
        model.set_adapter("default")
    except Exception:
        pass
    try:
        model.save_pretrained(output_dir, selected_adapters=["default"], safe_serialization=True)
    except TypeError:
        print("[WARN] 当前 PEFT 版本不支持 selected_adapters，改用普通 save_pretrained。")
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Save finished. Evaluate with lora_dir = {output_dir}")

def main():
    args = parse_args()
    torch.manual_seed(SEED)
    data_path = resolve_dpo_data_path(args.dpo_data_path)
    tag = beta_tag(args.beta)
    output_dir = str(Path(args.output_root) / f"dpo_lora_2000_{tag}_1epoch")
    print("========== DPO Beta Experiment ==========")
    print(f"BASE_MODEL_PATH = {args.base_model}")
    print(f"SFT_LORA_DIR    = {args.sft_lora_dir}")
    print(f"DPO_DATA_PATH   = {data_path}")
    print(f"OUTPUT_DIR      = {output_dir}")
    print(f"beta            = {args.beta}")
    print(f"epochs          = {args.epochs}")
    print(f"learning_rate   = {args.learning_rate}")
    print("=========================================")
    tokenizer = load_tokenizer(args.base_model)
    train_dataset = load_dpo_dataset(tokenizer, data_path)
    model = load_policy_model(args)
    dpo_args, config_supported_keys = make_dpo_config(args, output_dir)
    trainer_kwargs = make_trainer_kwargs(model, tokenizer, train_dataset, dpo_args, config_supported_keys, args.beta)
    print("[INFO] Initializing DPOTrainer...")
    trainer = DPOTrainer(**trainer_kwargs)
    copy_default_adapter_to_ref_adapter(model)
    print("[INFO] Start DPO training...")
    trainer.train()
    save_policy_adapter_only(model, tokenizer, output_dir)
    print("[INFO] DPO training completed.")

if __name__ == "__main__":
    main()
