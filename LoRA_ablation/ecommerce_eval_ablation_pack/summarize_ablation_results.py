#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
from pathlib import Path


def get_nested(d, path, default=""):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    files = sorted(root.glob("*/ablation_summary.json"))

    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            s = json.load(f)

        row = {
            "experiment": s.get("experiment", fp.parent.name),
            "quantization": s.get("quantization", ""),
            "dtype": s.get("dtype", ""),
            "rank": s.get("rank", ""),
            "lora_alpha": s.get("lora_alpha", ""),
            "learning_rate": s.get("learning_rate", ""),
            "epochs": s.get("num_train_epochs", ""),
            "train_samples": s.get("train_samples", ""),
            "eval_samples": s.get("eval_samples", ""),
            "eval_loss": get_nested(s, ["eval_metrics", "eval_loss"]),
            "eval_mean_token_accuracy": get_nested(s, ["eval_metrics", "eval_mean_token_accuracy"]),
            "train_runtime_seconds": s.get("train_runtime_seconds", ""),
            "max_memory_allocated_gb": s.get("max_memory_allocated_gb", ""),
            "max_memory_reserved_gb": s.get("max_memory_reserved_gb", ""),
            "path": str(fp.parent),
        }
        rows.append(row)

    # Sort: regular LoRA first, then QLoRA, by rank
    rows.sort(key=lambda x: (x["quantization"], int(x["rank"]) if str(x["rank"]).isdigit() else 999))

    fieldnames = [
        "experiment", "quantization", "dtype", "rank", "lora_alpha",
        "learning_rate", "epochs", "train_samples", "eval_samples",
        "eval_loss", "eval_mean_token_accuracy", "train_runtime_seconds",
        "max_memory_allocated_gb", "max_memory_reserved_gb", "path"
    ]

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    md_lines = []
    md_lines.append("| Experiment | Quant | DType | r | alpha | Eval Loss | Eval Acc | Max Alloc GB | Max Reserved GB |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md_lines.append(
            f"| {r['experiment']} | {r['quantization']} | {r['dtype']} | {r['rank']} | {r['lora_alpha']} | "
            f"{fmt(r['eval_loss'])} | {fmt(r['eval_mean_token_accuracy'])} | "
            f"{fmt(r['max_memory_allocated_gb'])} | {fmt(r['max_memory_reserved_gb'])} |"
        )

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Found {len(rows)} experiment summaries.")
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved MD: {args.output_md}")


if __name__ == "__main__":
    main()
