import json
from pathlib import Path
import matplotlib.pyplot as plt


output_dir = Path("/root/firstTunning/results/lora2")
trainer_state_path = Path("/root/firstTunning/outputs/qwen3_8b_ecommerce_sft_lora2/checkpoint-2709/trainer_state.json")

if not trainer_state_path.exists():
    raise FileNotFoundError(f"找不到文件：{trainer_state_path}")

with trainer_state_path.open("r", encoding="utf-8") as f:
    trainer_state = json.load(f)

log_history = trainer_state.get("log_history", [])

train_steps = []
train_losses = []

eval_steps = []
eval_losses = []

learning_rate_steps = []
learning_rates = []

for item in log_history:
    step = item.get("step")

    if step is None:
        continue

    if "loss" in item:
        train_steps.append(step)
        train_losses.append(item["loss"])

    if "eval_loss" in item:
        eval_steps.append(step)
        eval_losses.append(item["eval_loss"])

    if "learning_rate" in item:
        learning_rate_steps.append(step)
        learning_rates.append(item["learning_rate"])

print("========== 训练日志统计 ==========")
print(f"训练 loss 记录数：{len(train_losses)}")
print(f"验证 loss 记录数：{len(eval_losses)}")

if train_losses:
    print(f"初始 train loss：{train_losses[0]:.4f}")
    print(f"最终 train loss：{train_losses[-1]:.4f}")
    print(f"最低 train loss：{min(train_losses):.4f}")

if eval_losses:
    print(f"初始 eval loss：{eval_losses[0]:.4f}")
    print(f"最终 eval loss：{eval_losses[-1]:.4f}")
    print(f"最低 eval loss：{min(eval_losses):.4f}")

# 训练 loss 曲线
plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_losses, label="Train Loss")
if eval_losses:
    plt.plot(eval_steps, eval_losses, marker="o", label="Eval Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("SFT Training and Evaluation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_png = output_dir / "loss_curve.png"
loss_pdf = output_dir / "loss_curve.pdf"
plt.savefig(loss_png, dpi=300)
plt.savefig(loss_pdf)
plt.close()

print(f"loss 曲线已保存：{loss_png}")
print(f"loss 曲线 PDF 已保存：{loss_pdf}")

# 学习率曲线
if learning_rates:
    plt.figure(figsize=(8, 5))
    plt.plot(learning_rate_steps, learning_rates, label="Learning Rate")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    lr_png = output_dir / "learning_rate_curve.png"
    lr_pdf = output_dir / "learning_rate_curve.pdf"
    plt.savefig(lr_png, dpi=300)
    plt.savefig(lr_pdf)
    plt.close()

    print(f"学习率曲线已保存：{lr_png}")
    print(f"学习率曲线 PDF 已保存：{lr_pdf}")