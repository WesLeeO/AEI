"""Monte-Carlo returns training + value-weighted decoding — PyTorch port of LMRL-Gym.

Trains a per-token Q head on top of the (frozen) SFT policy by regressing reward-to-go,
then decodes with the faithful value-weighted rule:

    final_logits = sft_logits + beta * Q          (value_rl_base/gpt2/generation.py:116)

Run:  python mc/train_mc.py --mode train
      python mc/train_mc.py --mode eval --beta 16
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.environment import TextTrajectoryChain, TokenTrajectoryChain  # noqa: E402
from common.evaluate import evaluate  # noqa: E402
from common.guess_city import (  # noqa: E402
    GuessCityEnv, Phi3Oracle, conversation_to_text_trajectory, load_conversations,
)
from common.models import GPT2WithQHead, load_policy  # noqa: E402
from common.rollout import postprocess_question  # noqa: E402
from config import MCConfig  # noqa: E402
from mc_returns import MCData, MCDataset, mc_loss  # noqa: E402


# ----------------------------------------------------------------------------- data build


def build_mc_dataset(convos, tokenizer, cfg: MCConfig) -> MCDataset:
    if cfg.data_filter == "correct":
        convos = [c for c in convos if c.get("correct", False)]
    mc_data = []
    for convo in convos:
        traj = conversation_to_text_trajectory(convo)
        chain = TokenTrajectoryChain.from_text_trajectory_chain(
            TextTrajectoryChain(traj, None), tokenizer)
        mc_data.append(MCData.from_token_trajectory_chain(chain, gamma=cfg.gamma))
    return MCDataset(mc_data, tokenizer.pad_token_id, cfg.max_length)


# ---------------------------------------------------------------------- value-weighted decode


def make_mc_generate(model_q: GPT2WithQHead, tokenizer, device, beta: float,
                     max_new_tokens: int = 30, temperature: float = 1.0):
    policy = model_q.policy

    @torch.no_grad()
    def generate(prompt: str) -> str:
        ids = tokenizer.encode(
            prompt, return_tensors="pt", truncation=True,
            max_length=tokenizer.model_max_length - max_new_tokens,
        ).to(device)
        out = policy(input_ids=ids, use_cache=True, output_hidden_states=True, return_dict=True)
        past = out.past_key_values
        generated = []
        for _ in range(max_new_tokens):
            hidden = out.hidden_states[-1][:, -1, :]
            q = model_q.q_head(hidden)                         # [1, V]
            logits = out.logits[:, -1, :] + beta * q           # faithful additive rule
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            if nxt.item() == tokenizer.eos_token_id:
                break
            generated.append(nxt.item())
            out = policy(input_ids=nxt, past_key_values=past, use_cache=True,
                         output_hidden_states=True, return_dict=True)
            past = out.past_key_values
        return postprocess_question(tokenizer.decode(generated, skip_special_tokens=True))

    return generate


# ---------------------------------------------------------------------------------- train


def load_model(cfg: MCConfig, tokenizer):
    policy = load_policy(cfg.base_model_path, cfg.sft_adapter_path)
    return GPT2WithQHead(policy, tokenizer)


def train(cfg: MCConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    convos = load_conversations(cfg.train_json)
    loader = DataLoader(build_mc_dataset(convos, tokenizer, cfg),
                        batch_size=cfg.batch_size, shuffle=True)

    model = load_model(cfg, tokenizer).to(device)
    # freeze the policy; train only the Q head (offline RL on a fixed pi_beta)
    for p in model.policy.parameters():
        p.requires_grad = False
    model.q_head.train()

    optim = Adam(model.q_head.parameters(), lr=cfg.lr)
    total_steps = len(loader) * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.05 * total_steps), total_steps)

    for epoch in range(cfg.epochs):
        optim.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, q = model(batch["input_ids"], batch["attention_mask"])
            loss, logs = mc_loss(q, logits, batch["input_ids"], batch["attention_mask"],
                                 batch["should_take_action"], batch["returns"], cfg.cql_weight)
            (loss / cfg.accum_steps).backward()
            if (step + 1) % cfg.accum_steps == 0:
                optim.step()
                sched.step()
                optim.zero_grad()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} | loss {loss.item():.4f} "
                      f"| q {logs['q_loss']:.4f} | cql {logs['cql_loss']:.4f}")

    os.makedirs(os.path.dirname(cfg.q_function_out), exist_ok=True)
    torch.save(model.q_head.state_dict(), cfg.q_function_out)
    print(f"Saved Q head to {cfg.q_function_out}")


def evaluate_mc(cfg: MCConfig, beta: float, num_cities: int = 30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(cfg, tokenizer).to(device).eval()
    model.q_head.load_state_dict(torch.load(cfg.q_function_out, map_location=device))

    env = GuessCityEnv(Phi3Oracle(device=device))
    generate = make_mc_generate(model, tokenizer, device, beta=beta)
    metrics = evaluate(env, generate, num_cities=num_cities, verbose=True)
    print(f"beta={beta} accuracy={metrics['accuracy']:.3f} avg_questions={metrics['avg_questions']:.2f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--num-cities", type=int, default=30)
    args = parser.parse_args()

    cfg = MCConfig()
    if args.mode == "train":
        train(cfg)
    else:
        evaluate_mc(cfg, args.beta if args.beta is not None else cfg.beta, args.num_cities)
