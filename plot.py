"""Figures for the report.

  python plot.py   # -> results/pareto<setting>.png, results/rl_curves<setting>.png

LLM results (results/<name>.json) are drawn against the non-LLM baselines of the same setting,
inferred from the name suffix (_binary, _ddx, _binary_ddx; none = easy). RL curves are read from
results/logs/<run>.jsonl (copies of ckpt/<run>/train_log.jsonl) and ckpt/*/train_log.jsonl.
"""
import glob
import json
import os

import matplotlib.pyplot as plt

SETTINGS = ["_binary_ddx", "_binary", "_ddx", ""]  # longest suffix first


def setting_of(name):
    return next(s for s in SETTINGS if name.endswith(s))


def pareto():
    runs = {}
    for path in sorted(glob.glob("results/*.json")):
        name = os.path.basename(path)[:-5]
        if name.startswith("baselines"):
            continue
        with open(path) as f:
            runs.setdefault(setting_of(name), []).append((name, json.load(f)["metrics"]))
    for setting, llm in runs.items():
        pts = []
        base = f"results/baselines_test{setting}.json"
        if os.path.exists(base):
            with open(base) as f:
                for name, m in json.load(f).items():
                    if "upper bound" not in name:
                        pts.append((name, m["n_questions"], m["accuracy"], "#9aa0a6"))
        for name, m in llm:
            pts.append((name[: len(name) - len(setting)] if setting else name, m["n_questions"], m["accuracy"], "#1a73e8"))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, x, y, c in pts:
            ax.scatter(x, y, color=c, s=30)
            ax.annotate(name, (x, y), fontsize=7, xytext=(4, 2), textcoords="offset points")
        ax.set_xlabel("average number of questions")
        ax.set_ylabel("ICD-10 top-1 accuracy")
        ax.set_title(f"DDXPlus test{setting or ' (easy)'}: accuracy vs. questions (grey = non-LLM baselines)", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"results/pareto{setting}.png", dpi=200)
        plt.close(fig)


def rl_curves():
    logs = {}
    for path in glob.glob("ckpt/*/train_log.jsonl") + glob.glob("results/logs/*.jsonl"):
        run = os.path.basename(path)[:-6] if path.startswith("results") else path.split("/")[1]
        if not run.startswith("sft"):
            logs[run] = path  # results/logs wins over ckpt (globbed later)
    by_setting = {}
    for run, path in sorted(logs.items()):
        by_setting.setdefault(setting_of(run), []).append((run, path))
    for setting, items in by_setting.items():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for i, (run, path) in enumerate(items):
            with open(path) as f:
                rows = [json.loads(line) for line in f]
            label = run[: len(run) - len(setting)] if setting else run
            color = f"C{i}"
            for ax, key in zip(axes, ["return", "accuracy", "n_questions"]):
                ax.plot([r["iteration"] for r in rows], [r[key] for r in rows], color=color, alpha=0.5,
                        label=f"{label} (train, sampled)")
                val = [(r["iteration"], r[f"val_{key}"]) for r in rows if f"val_{key}" in r]
                if val:
                    ax.plot(*zip(*val), "o--", color=color, ms=4, label=f"{label} (val, greedy)")
                ax.set_title(key)
                ax.set_xlabel("iteration")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"results/rl_curves{setting}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    pareto()
    rl_curves()
