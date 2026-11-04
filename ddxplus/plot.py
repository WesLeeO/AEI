"""Figures for the report.

  python plot.py   # -> results/pareto.png, results/rl_curves.png
"""
import glob
import json
import os

import matplotlib.pyplot as plt


def pareto():
    pts = []
    with open("results/baselines_test.json") as f:
        for name, m in json.load(f).items():
            if "upper bound" not in name:
                pts.append((name, m["n_questions"], m["accuracy"], "#9aa0a6"))
    for path in sorted(glob.glob("results/*.json")):
        if "baselines" in path:
            continue
        with open(path) as f:
            m = json.load(f)["metrics"]
        pts.append((os.path.basename(path)[:-5], m["n_questions"], m["accuracy"], "#1a73e8"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, x, y, c in pts:
        ax.scatter(x, y, color=c, s=30)
        ax.annotate(name, (x, y), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("average number of questions")
    ax.set_ylabel("ICD-10 top-1 accuracy")
    ax.set_title("DDXPlus test: accuracy vs. questions asked (grey = non-LLM baselines)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/pareto.png", dpi=200)


def rl_curves():
    logs = sorted(glob.glob("ckpt/*/train_log.jsonl"))
    logs = [p for p in logs if "sft" not in p]
    if not logs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for path in logs:
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        label = path.split("/")[1]
        for ax, key in zip(axes, ["return", "accuracy", "n_questions"]):
            ax.plot([r["iteration"] for r in rows], [r[key] for r in rows], label=label, alpha=0.8)
            val = [(r["iteration"], r[f"val_{key}"]) for r in rows if f"val_{key}" in r]
            if val:
                ax.plot(*zip(*val), "o--", ms=3, label=f"{label} (val)")
            ax.set_title(key)
            ax.set_xlabel("iteration")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/rl_curves.png", dpi=200)


if __name__ == "__main__":
    pareto()
    rl_curves()
