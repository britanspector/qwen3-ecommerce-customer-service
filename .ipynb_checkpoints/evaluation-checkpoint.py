# evaluation_pipeline_better.py

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score

# =========================
# 1. 路径配置
# =========================

EVAL_SET_FILE = "/root/firstTunning/fixed_eval_set_100.jsonl"

SFT_EXP1_FILE = "/root/firstTunning/results/lora1/lora1_eval.jsonl"
SFT_EXP2_FILE = "/root/firstTunning/results/lora2/lora2_eval.jsonl"

DPO_EXP1_FILE = "/root/firstTunning/results/dpo_lora_eval.jsonl"
DPO_EXP2_FILE = "/root/firstTunning/results/dpo_lora_2000_beta01_eval.jsonl"

REFERENCE_FILE = "/root/firstTunning/references.txt"

LOCAL_BERT_MODEL_PATH = "/root/firstTunning/bert_models"

OUTPUT_PLOT_FILE = "/root/firstTunning/results/dpo_evaluation_comparison.png"


# =========================
# 2. 读取文件
# =========================

def load_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} 不存在，请检查路径！")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_references(ref_file):
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"{ref_file} 不存在，请检查路径！")

    with open(ref_file, "r", encoding="utf-8") as f:
        refs = [line.strip() for line in f if line.strip()]
    return refs


def extract_output(item):
    """
    兼容不同 eval 输出字段。
    你的文件目前应该是 item['output']。
    """
    for key in ["output", "prediction", "pred", "response", "answer", "generated_text"]:
        if key in item and isinstance(item[key], str):
            return item[key].strip()

    raise KeyError(f"找不到输出字段，当前字段包括：{list(item.keys())}")


def load_predictions(eval_file):
    data = load_jsonl(eval_file)
    return [extract_output(item) for item in data]


# =========================
# 3. 中文 BLEU：字符级
# =========================

def char_tokenize(text):
    text = re.sub(r"\s+", "", text)
    return list(text)


def compute_char_bleu(eval_file, reference_file):
    preds = load_predictions(eval_file)
    refs = load_references(reference_file)

    if len(preds) != len(refs):
        raise ValueError(f"数量不一致：preds={len(preds)}, refs={len(refs)}")

    predictions = [char_tokenize(pred) for pred in preds]
    references = [[char_tokenize(ref)] for ref in refs]

    smooth = SmoothingFunction().method4

    bleu = corpus_bleu(
        references,
        predictions,
        smoothing_function=smooth
    )

    return bleu


# =========================
# 4. BERTScore：返回平均值和逐样本结果
# =========================

def compute_bertscore(eval_file, reference_file):
    preds = load_predictions(eval_file)
    refs = load_references(reference_file)

    if len(preds) != len(refs):
        raise ValueError(f"数量不一致：preds={len(preds)}, refs={len(refs)}")

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    from bert_score import utils
    utils.model2layers[LOCAL_BERT_MODEL_PATH] = 12

    P, R, F1 = score(
        preds,
        refs,
        lang="zh",
        model_type=LOCAL_BERT_MODEL_PATH,
        rescale_with_baseline=False,
        batch_size=32,
        verbose=False
    )

    return {
        "precision_mean": P.mean().item(),
        "recall_mean": R.mean().item(),
        "f1_mean": F1.mean().item(),
        "f1_each": F1.cpu().numpy().tolist()
    }


# =========================
# 5. 规则指标：must-have / forbidden
# =========================

def compute_rule_metrics(eval_file, eval_set_file):
    preds = load_predictions(eval_file)
    samples = load_jsonl(eval_set_file)

    if len(preds) != len(samples):
        raise ValueError(f"数量不一致：preds={len(preds)}, samples={len(samples)}")

    must_pass_count = 0
    forbidden_hit_count = 0

    for pred, sample in zip(preds, samples):
        must_have_any = sample.get("must_have_any", [])
        forbidden = sample.get("forbidden", [])

        # must_have_any: 每一组关键词中命中任意一个即可
        if must_have_any:
            group_pass = []
            for group in must_have_any:
                ok = any(keyword in pred for keyword in group)
                group_pass.append(ok)
            must_pass = all(group_pass)
        else:
            must_pass = True

        forbidden_hit = any(keyword in pred for keyword in forbidden)

        if must_pass:
            must_pass_count += 1

        if forbidden_hit:
            forbidden_hit_count += 1

    total = len(preds)

    return {
        "must_have_pass_rate": must_pass_count / total,
        "forbidden_hit_rate": forbidden_hit_count / total
    }


# =========================
# 6. DPO 样本级胜率
# =========================

def compute_win_rate(f1_exp1, f1_exp2):
    f1_exp1 = np.array(f1_exp1)
    f1_exp2 = np.array(f1_exp2)

    win = np.sum(f1_exp2 > f1_exp1)
    tie = np.sum(f1_exp2 == f1_exp1)
    lose = np.sum(f1_exp2 < f1_exp1)
    total = len(f1_exp1)

    return {
        "win": int(win),
        "tie": int(tie),
        "lose": int(lose),
        "win_rate": win / total,
        "tie_rate": tie / total,
        "lose_rate": lose / total
    }


# =========================
# 7. 主程序
# =========================

def main():
    print("=" * 80)
    print("Start Evaluation")
    print("=" * 80)

    # SFT BLEU
    sft1_bleu = compute_char_bleu(SFT_EXP1_FILE, REFERENCE_FILE)
    sft2_bleu = compute_char_bleu(SFT_EXP2_FILE, REFERENCE_FILE)

    # DPO BERTScore
    dpo1_bert = compute_bertscore(DPO_EXP1_FILE, REFERENCE_FILE)
    dpo2_bert = compute_bertscore(DPO_EXP2_FILE, REFERENCE_FILE)

    dpo1_f1 = dpo1_bert["f1_mean"]
    dpo2_f1 = dpo2_bert["f1_mean"]

    abs_improve = dpo2_f1 - dpo1_f1
    rel_improve = abs_improve / dpo1_f1 * 100

    # DPO win rate
    win_stat = compute_win_rate(
        dpo1_bert["f1_each"],
        dpo2_bert["f1_each"]
    )

    # rule metrics
    dpo1_rule = compute_rule_metrics(DPO_EXP1_FILE, EVAL_SET_FILE)
    dpo2_rule = compute_rule_metrics(DPO_EXP2_FILE, EVAL_SET_FILE)

    print("\n[SFT BLEU]")
    print(f"SFT Exp1 Char-BLEU: {sft1_bleu:.4f}")
    print(f"SFT Exp2 Char-BLEU: {sft2_bleu:.4f}")

    print("\n[DPO BERTScore]")
    print(f"DPO Exp1 BERTScore F1: {dpo1_f1:.4f}")
    print(f"DPO Exp2 BERTScore F1: {dpo2_f1:.4f}")
    print(f"Absolute Improvement: +{abs_improve:.4f}")
    print(f"Relative Improvement: +{rel_improve:.2f}%")

    print("\n[DPO Pairwise Win Rate]")
    print(f"Exp2 Win : {win_stat['win']} / {len(dpo1_bert['f1_each'])}")
    print(f"Exp2 Tie : {win_stat['tie']} / {len(dpo1_bert['f1_each'])}")
    print(f"Exp2 Lose: {win_stat['lose']} / {len(dpo1_bert['f1_each'])}")
    print(f"Exp2 Win Rate: {win_stat['win_rate']:.4f}")

    print("\n[DPO Rule Metrics]")
    print(f"DPO Exp1 Must-have Pass Rate: {dpo1_rule['must_have_pass_rate']:.4f}")
    print(f"DPO Exp2 Must-have Pass Rate: {dpo2_rule['must_have_pass_rate']:.4f}")
    print(f"DPO Exp1 Forbidden Hit Rate : {dpo1_rule['forbidden_hit_rate']:.4f}")
    print(f"DPO Exp2 Forbidden Hit Rate : {dpo2_rule['forbidden_hit_rate']:.4f}")

    # =========================
    # 可视化：只画 DPO，更干净
    # =========================

    labels = ["BERTScore F1", "Must-have Pass", "1 - Forbidden Hit", "Win Rate"]
    exp1_scores = [
        dpo1_f1,
        dpo1_rule["must_have_pass_rate"],
        1 - dpo1_rule["forbidden_hit_rate"],
        0.5
    ]

    exp2_scores = [
        dpo2_f1,
        dpo2_rule["must_have_pass_rate"],
        1 - dpo2_rule["forbidden_hit_rate"],
        win_stat["win_rate"]
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars1 = ax.bar(x - width / 2, exp1_scores, width, label="DPO Exp1")
    bars2 = ax.bar(x + width / 2, exp2_scores, width, label="DPO Exp2")

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("DPO Evaluation Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300)
    plt.show()

    print("\nPlot saved to:", OUTPUT_PLOT_FILE)


if __name__ == "__main__":
    main()