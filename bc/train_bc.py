"""Behaviour cloning (imitation of the asker) — PyTorch port of LMRL-Gym guess_city BC.

Trains a LoRA policy to predict the asker's questions with next-token cross-entropy,
masked to the question (action) tokens. Optionally %BC-filters the data (keep only
correct conversations, or the top-p% by return) before training.

Run:  python bc/train_bc.py --mode train
      python bc/train_bc.py --mode eval --adapter bc/bc_model
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.environment import TokenHistory  # noqa: E402
from common.guess_city import (  # noqa: E402
    GuessCityEnv, Phi3Oracle, conversation_to_text_trajectory, load_conversations,
)
from common.evaluate import evaluate  # noqa: E402
from common.models import load_policy  # noqa: E402
from common.rollout import sample_question  # noqa: E402
from common.utils import pad_sequences  # noqa: E402
from config import BCConfig  # noqa: E402


# ----------------------------------------------------------------------------- data utils


def trajectory_return(convo) -> float:
    """Sum of rewards = -(#questions) (or -(#questions-1) if the final guess is correct)."""
    traj = conversation_to_text_trajectory(convo)
    return float(sum(traj.reward))


def filter_conversations(convos, cfg: BCConfig):
    if cfg.data_filter == "all":
        return convos
    if cfg.data_filter == "correct":
        return [c for c in convos if c.get("correct", False)]
    if cfg.data_filter == "top_p":
        scores = np.array([trajectory_return(c) for c in convos])
        threshold = np.percentile(scores, 100 - cfg.take_top_p)
        return [c for c, s in zip(convos, scores) if s >= threshold]
    raise ValueError(f"unknown data_filter {cfg.data_filter}")


class BCDataset(Dataset):
    def __init__(self, convos, tokenizer, max_length: int):
        histories = []
        for convo in convos:
            traj = conversation_to_text_trajectory(convo)
            histories.append(TokenHistory.from_text_history(traj.text_history, tokenizer))
        self.tokens = pad_sequences([h.tokens for h in histories], tokenizer.pad_token_id, np.int64, max_length)
        self.is_action = pad_sequences([h.is_action for h in histories], False, np.bool_, max_length)
        self.attention = (self.tokens != tokenizer.pad_token_id).astype(np.int64)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.tokens[i], dtype=torch.long),
            "is_action": torch.tensor(self.is_action[i], dtype=torch.bool),
            "attention_mask": torch.tensor(self.attention[i], dtype=torch.long),
        }


def bc_loss(logits, input_ids, attention_mask, is_action, non_action_weight: float):
    """Next-token CE, masked to action tokens (answers weighted by non_action_weight)."""
    token_losses = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(input_ids[:, 1:].shape)
    attn = attention_mask[:, 1:].float()
    act = is_action[:, 1:].float()
    weight = (act + (1.0 - act) * non_action_weight) * attn
    return (token_losses * weight).sum() / attn.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------------- train


def build_lora_policy(base_model_path: str):
    from peft import LoraConfig, get_peft_model
    base = load_policy(base_model_path)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["c_attn"], task_type="CAUSAL_LM")
    return get_peft_model(base, lora)


def train(cfg: BCConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    convos = filter_conversations(load_conversations(cfg.train_json), cfg)
    print(f"Training on {len(convos)} conversations (filter={cfg.data_filter}).")
    loader = DataLoader(BCDataset(convos, tokenizer, cfg.max_length),
                        batch_size=cfg.batch_size, shuffle=True)

    model = build_lora_policy(cfg.base_model_path) if cfg.use_lora else load_policy(cfg.base_model_path)
    model.to(device)
    model.train()

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    total_steps = len(loader) * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup_frac * total_steps), total_steps)

    for epoch in range(cfg.epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = bc_loss(logits, batch["input_ids"], batch["attention_mask"],
                           batch["is_action"], cfg.non_action_weight)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} | loss {loss.item():.4f}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Saved BC policy to {cfg.output_dir}")


def evaluate_policy(cfg: BCConfig, adapter_path: str, num_cities: int = 30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_policy(cfg.base_model_path, adapter_path).to(device).eval()

    env = GuessCityEnv(Phi3Oracle(device=device))
    gen_kwargs = {"do_sample": True, "temperature": 0.7, "top_p": 0.95}
    metrics = evaluate(env, lambda p: sample_question(model, tokenizer, p, device, gen_kwargs=gen_kwargs),
                       num_cities=num_cities, verbose=True)
    print(f"accuracy={metrics['accuracy']:.3f}  avg_questions={metrics['avg_questions']:.2f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--adapter", default="bc/bc_model")
    parser.add_argument("--num-cities", type=int, default=30)
    args = parser.parse_args()

    cfg = BCConfig()
    if args.mode == "train":
        train(cfg)
    else:
        evaluate_policy(cfg, args.adapter, args.num_cities)
