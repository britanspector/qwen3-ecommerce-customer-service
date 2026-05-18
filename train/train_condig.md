# SFT Training Config

- Base model: Qwen3-8B
- Training framework: HuggingFace TRL SFTTrainer
- Fine-tuning method: LoRA
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: q_proj, v_proj
- Dataset: 5800 cleaned e-commerce customer service samples
- Epochs: 3
- Batch size: 1
- Gradient accumulation steps: 16
- Max length: 1024
- Evaluation strategy: every 100 steps
- Output: LoRA adapter