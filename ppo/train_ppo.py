"""PPO training for the GuessMyCity asker — PyTorch port of LMRL-Gym guess_city PPO.

Each iteration: roll out episodes with the current policy + Phi-3 oracle, snapshot
old log-probs / values / reference log-probs, fold the KL penalty into the reward, compute
GAE over action tokens, then take several clipped PPO steps.

Run:  python ppo/train_ppo.py --mode train --num-iterations 100
      python ppo/train_ppo.py --mode eval --adapter ppo/ppo_model
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.environment import TokenTrajectory  # noqa: E402
from common.evaluate import evaluate  # noqa: E402
from common.guess_city import (  # noqa: E402
    GuessCityEnv, Phi3Oracle, conversation_to_text_trajectory,
)
from common.models import GPT2WithValueHead, load_policy  # noqa: E402
from common.rollout import run_episode, sample_question  # noqa: E402
from common.utils import pad_sequences, token_logprobs_from_logits  # noqa: E402
from config import PPOConfig  # noqa: E402
from ppo import compute_advantages_returns, ppo_loss  # noqa: E402


# ------------------------------------------------------------------------- rollout / batch


def collect_episode(env, model, tokenizer, device, cfg: PPOConfig):
    gen_kwargs = {"do_sample": True, "temperature": cfg.temperature, "top_p": cfg.top_p}
    generate = lambda p: sample_question(model.policy, tokenizer, p, device, cfg.max_new_tokens, gen_kwargs)
    result = run_episode(env, generate)
    # rebuild token-level arrays through the SAME path as the training data (consistent reward)
    convo = {
        "lines": [f"{q} {a}" for q, a in zip(result["questions"], result["answers"])],
        "correct": result["correct"],
    }
    traj = conversation_to_text_trajectory(convo, max_conversation_len=cfg.max_questions)
    tt = TokenTrajectory.from_text_trajectory(traj, tokenizer)
    return tt.tokens, tt.is_action.astype(np.int64), tt.reward, result


def collect_batch(env, model, tokenizer, device, cfg: PPOConfig, n_episodes: int):
    tok, act, rew, results = [], [], [], []
    for _ in range(n_episodes):
        t, a, r, res = collect_episode(env, model, tokenizer, device, cfg)
        tok.append(t); act.append(a); rew.append(r); results.append(res)
    pad = tokenizer.pad_token_id
    input_ids = pad_sequences(tok, pad, np.int64, cfg.max_length)
    action_mask = pad_sequences(act, 0, np.int64, cfg.max_length)
    rewards = pad_sequences(rew, 0.0, np.float32, cfg.max_length)
    attention = (input_ids != pad).astype(np.int64)
    to = lambda x, d: torch.tensor(x, dtype=d, device=device)
    return {
        "input_ids": to(input_ids, torch.long),
        "attention_mask": to(attention, torch.long),
        "action_mask": to(action_mask, torch.float32),
        "rewards": to(rewards, torch.float32),
        "results": results,
    }


# ---------------------------------------------------------------------------------- train


def train(cfg: PPOConfig, num_iterations: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2WithValueHead(
        load_policy(cfg.base_model_path, cfg.sft_adapter_path, trainable=True)).to(device)
    ref_model = load_policy(cfg.base_model_path, cfg.sft_adapter_path).to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    env = GuessCityEnv(Phi3Oracle(device=device), max_questions=cfg.max_questions)

    policy_optim = AdamW(model.policy_trainable_parameters(), lr=cfg.policy_lr)
    value_optim = AdamW(model.value_parameters(), lr=cfg.value_lr)
    kl_coef = cfg.init_kl_coef

    for it in range(num_iterations):
        batch = collect_batch(env, model, tokenizer, device, cfg, cfg.episodes_per_iter)
        input_ids, attention = batch["input_ids"], batch["attention_mask"]
        action_mask, rewards = batch["action_mask"], batch["rewards"]

        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, attention_mask=attention).logits
            ref_logprobs = token_logprobs_from_logits(ref_logits, input_ids)
            old_logits, old_values, _ = model(input_ids, attention)
            old_logprobs = token_logprobs_from_logits(old_logits, input_ids)

        advantages, returns, mean_kl = compute_advantages_returns(
            rewards, old_values, old_logprobs, ref_logprobs, action_mask, attention,
            kl_coef=kl_coef, gamma=cfg.gamma, lam=cfg.lam)
        old_state_values = old_values[:, :-1].detach()
        mask = action_mask[:, 1:] * attention[:, 1:].float()

        for _ in range(cfg.ppo_epochs):
            logits, values, _ = model(input_ids, attention)
            new_logprobs = token_logprobs_from_logits(logits, input_ids)
            new_values = values[:, :-1]
            loss, logs = ppo_loss(
                new_logprobs, old_logprobs.detach(), advantages.detach(), returns.detach(),
                new_values, old_state_values, mask,
                cliprange=cfg.cliprange, cliprange_value=cfg.cliprange_value,
                value_loss_coef=cfg.value_loss_coef)
            policy_optim.zero_grad(); value_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy_trainable_parameters(), cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.value_parameters(), cfg.max_grad_norm)
            policy_optim.step(); value_optim.step()

        if it % cfg.log_every == 0:
            results = batch["results"]
            acc = np.mean([r["correct"] for r in results])
            avg_q = np.mean([r["asked"] for r in results])
            print(f"iter {it} | loss {loss.item():.4f} | policy {logs['policy_loss']:.4f} "
                  f"| value {logs['value_loss']:.4f} | kl {mean_kl:.4f} "
                  f"| acc {acc:.2f} | avgQ {avg_q:.1f}")
        if it % cfg.save_every == 0 and it > 0:
            save(model, tokenizer, cfg, f"{cfg.output_dir}_iter{it}")

    save(model, tokenizer, cfg, cfg.output_dir)
    print("Training complete.")


def save(model, tokenizer, cfg, path):
    os.makedirs(path, exist_ok=True)
    model.policy.save_pretrained(path)
    torch.save(model.value_head.state_dict(), os.path.join(path, "value_head.pth"))
    tokenizer.save_pretrained(path)
    print(f"Saved PPO policy to {path}")


def evaluate_policy(cfg: PPOConfig, adapter_path: str, num_cities: int = 30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = load_policy(cfg.base_model_path, adapter_path).to(device).eval()
    env = GuessCityEnv(Phi3Oracle(device=device), max_questions=cfg.max_questions)
    gen_kwargs = {"do_sample": True, "temperature": cfg.temperature, "top_p": cfg.top_p}
    metrics = evaluate(env, lambda p: sample_question(policy, tokenizer, p, device, cfg.max_new_tokens, gen_kwargs),
                       num_cities=num_cities, verbose=True)
    print(f"accuracy={metrics['accuracy']:.3f} avg_questions={metrics['avg_questions']:.2f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--adapter", default="ppo/ppo_model")
    parser.add_argument("--num-cities", type=int, default=30)
    args = parser.parse_args()

    cfg = PPOConfig()
    if args.mode == "train":
        train(cfg, args.num_iterations)
    else:
        evaluate_policy(cfg, args.adapter, args.num_cities)
