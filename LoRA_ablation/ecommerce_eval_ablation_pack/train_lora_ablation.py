#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA / QLoRA ablation training script for Qwen3-8B on TRL-style messages JSONL.

Supported experiments:
- LoRA rank r = 4 / 16 / 64
- Full-precision LoRA: FP16 or BF16 base model
- QLoRA: 4-bit NF4 quantized base model

This script does NOT require TRL. It reads the same `messages` JSONL format and masks
prompt tokens with -100, so only assistant responses contribute to loss.

Example:
python train_lora_ablation.py \
  --base_model /root/autodl-tmp/modelscope/Qwen/Qwen3-8B \
  --train_file /root/firstTunning/data/ecommerce_sft_messages_recommended_v3_train.jsonl \
  --eval_file /root/firstTunning/data/ecommerce_sft_messages_recommended_v3_eval.jsonl \
  --output_root /root/firstTunning/ablation_outputs \
  --rank 16 \
  --quantization none \
  --dtype fp16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8
"""

import argparse
import inspect
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y"}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def find_assistant_index(messages):
    idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not idxs:
        raise ValueError("No assistant message found.")
    return idxs[-1]


def apply_template(tokenizer, messages, add_generation_prompt=True):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def tokenize_messages(example, tokenizer, max_length: int):
    messages = example["messages"]
    assistant_idx = find_assistant_index(messages)

    prompt_messages = messages[:assistant_idx]
    answer = messages[assistant_idx]["content"]

    prompt_text = apply_template(tokenizer, prompt_messages, add_generation_prompt=True)
    answer_text = answer + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    # Keep the left prompt and truncate the tail if needed.
    # Most e-commerce samples are short, so this rarely affects data.
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "category": example.get("category", ""),
        "source": example.get("source", ""),
    }


@dataclass
class DataCollatorForCausalSFT:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        input_ids, attention_mask, labels = [], [], []
        pad_id = self.tokenizer.pad_token_id

        for f in features:
            cur_len = len(f["input_ids"])
            pad_len = max_len - cur_len
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class MemoryCallback(TrainerCallback):
    def __init__(self):
        self.max_allocated_gb = 0.0
        self.max_reserved_gb = 0.0

    def _update(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3
            self.max_allocated_gb = max(self.max_allocated_gb, allocated)
            self.max_reserved_gb = max(self.max_reserved_gb, reserved)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._update()
        if logs is not None:
            logs["max_memory_allocated_gb"] = round(self.max_allocated_gb, 4)
            logs["max_memory_reserved_gb"] = round(self.max_reserved_gb, 4)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._update()
        if metrics is not None:
            metrics["max_memory_allocated_gb"] = round(self.max_allocated_gb, 4)
            metrics["max_memory_reserved_gb"] = round(self.max_reserved_gb, 4)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # Causal LM loss shifts internally: logits[:, :-1] predict labels[:, 1:].
    pred_shift = preds[:, :-1]
    label_shift = labels[:, 1:]

    mask = label_shift != -100
    total = int(mask.sum())
    if total == 0:
        return {"mean_token_accuracy": 0.0}

    correct = (pred_shift[mask] == label_shift[mask]).sum()
    return {"mean_token_accuracy": float(correct / total)}


def build_training_args(args, output_dir):
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    if "save_strategy" in sig.parameters:
        kwargs["save_strategy"] = "steps"

    # Mixed precision mode
    if args.dtype == "bf16":
        kwargs["bf16"] = True
        kwargs["fp16"] = False
    elif args.dtype == "fp16":
        kwargs["bf16"] = False
        kwargs["fp16"] = True

    return TrainingArguments(**kwargs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_root", required=True)

    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=-1, help="Default -1 means alpha = 2 * rank.")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16")
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--local_files_only", type=str2bool, default=True)

    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1" if args.local_files_only else "0")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1" if args.local_files_only else "0")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    lora_alpha = args.lora_alpha if args.lora_alpha > 0 else args.rank * 2
    exp_name = f"{'qlora4bit' if args.quantization == '4bit' else 'lora'}_r{args.rank}_{args.dtype}_alpha{lora_alpha}"
    output_dir = Path(args.output_root) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )

    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_raw = read_jsonl(args.train_file)
    eval_raw = read_jsonl(args.eval_file)

    train_data = [tokenize_messages(x, tokenizer, args.max_length) for x in train_raw]
    eval_data = [tokenize_messages(x, tokenizer, args.max_length) for x in eval_raw]

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    collator = DataCollatorForCausalSFT(tokenizer)

    training_args = build_training_args(args, output_dir)
    mem_cb = MemoryCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[mem_cb],
    )

    start = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - start

    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if torch.cuda.is_available():
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3
    else:
        max_alloc = 0.0
        max_reserved = 0.0

    summary = {
        "experiment": exp_name,
        "base_model": args.base_model,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "rank": args.rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_length": args.max_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "train_runtime_seconds": train_seconds,
        "max_memory_allocated_gb": round(max_alloc, 4),
        "max_memory_reserved_gb": round(max_reserved, 4),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }

    with open(output_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nFinal ablation summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_dir / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
