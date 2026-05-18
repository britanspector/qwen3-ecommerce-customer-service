import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"

train_file = "data/ecommerce_sft_messages_recommended_v3_train.jsonl"
eval_file = "data/ecommerce_sft_messages_recommended_v3_eval.jsonl"

output_dir = "outputs/qwen3_8b_ecommerce_sft_lora2"


def main():
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "eval": eval_file,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=[
    #         "q_proj",
    #         "k_proj",
    #         "v_proj",
    #         "o_proj",
    #         "gate_proj",
    #         "up_proj",
    #         "down_proj",
    #     ],
    # )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=output_dir,

        # 数据与长度
        max_length=1024,
        packing=False,
        assistant_only_loss=True,

        # 训练轮数
        num_train_epochs=3,

        # batch 设置：单卡 24G/32G 推荐先这样
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,

        # 优化器
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # 精度
        bf16=True,

        # 日志和保存
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=300,
        save_total_limit=5,

        # 其他
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"训练完成，LoRA adapter 已保存到：{output_dir}")


if __name__ == "__main__":
    main()