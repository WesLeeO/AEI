"""Online RL fine-tuning of the LLM doctor: PPO (turn-level GAE + value head) or GRPO.

  PPO : per-turn rewards (-cost per question, +1 correct diagnosis), KL to the reference
        folded into each turn's reward, GAE over turns (gamma=1, lambda=0.95), whitened
        advantages broadcast to the turn's tokens, clipped policy + clipped value loss.
  GRPO: G rollouts per patient, advantage = (R - mean_group) / std_group for every token of
        the episode, clipped policy loss + k3 KL penalty to the reference.
The reference policy is the initial model (base + initial adapter), obtained by keeping a
frozen copy of the initial LoRA weights.

  python train_rl.py --algo ppo  --model Qwen/Qwen2.5-1.5B-Instruct --adapter ckpt/sft --out ckpt/ppo
  python train_rl.py --algo grpo --model Qwen/Qwen2.5-1.5B-Instruct --adapter ckpt/sft --out ckpt/grpo
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from peft import PeftModel

from data import load_patients
from env import add_env_args, aggregate, build_env
from llm import (ValueHead, action_logprobs, collate, load_policy, load_tokenizer, tokenize_dialogue,
                 turn_start_states)
from rollout import evaluate, run_episodes


def gae(rewards, values, gamma, lam):
    adv, last = np.zeros(len(rewards)), 0.0
    for k in reversed(range(len(rewards))):
        nxt = values[k + 1] if k + 1 < len(values) else 0.0
        delta = rewards[k] + gamma * nxt - values[k]
        last = delta + gamma * lam * last
        adv[k] = last
    return adv, adv + np.array(values)


def split_by_row(flat, mask):
    return list(torch.split(flat, mask.sum(1).tolist()))


class Trainer:
    def __init__(self, args):
        self.args = args
        self.tok = load_tokenizer(args.model)
        self.model = load_policy(args.model, adapter=args.adapter, lora_r=args.lora_r,
                                 gradient_checkpointing=args.grad_ckpt)
        self.device = next(self.model.parameters()).device
        for m in self.model.modules():  # no dropout: keeps old/new log-prob ratios exact
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        # Frozen reference = initial adapter (or the base model when starting from scratch)
        self.model.add_adapter("ref", self.model.peft_config["default"])
        ref_weights = {k.replace(".default.", ".ref."): v.clone()
                       for k, v in self.model.state_dict().items() if ".default." in k}
        missing = set(ref_weights) - set(self.model.state_dict())
        assert ref_weights and not missing, f"could not copy reference adapter: {sorted(missing)[:3]}"
        self.model.load_state_dict(ref_weights, strict=False)
        for n, p in self.model.named_parameters():
            if ".ref." in n:
                p.requires_grad_(False)
        self.model.set_adapter("default")
        params = [p for p in self.model.parameters() if p.requires_grad]
        groups = [{"params": params, "lr": args.lr}]
        if args.algo == "ppo":
            self.value_head = ValueHead(self.model.config.hidden_size).to(self.device)
            groups.append({"params": self.value_head.parameters(), "lr": args.value_lr})
        self.opt = torch.optim.AdamW(groups, weight_decay=0.0)

        self.make_env = build_env(args)
        self.train_pool = load_patients("train", args.train_pool, seed=args.pool_seed)
        self.val_patients = load_patients("validate", args.eval_n, seed=0)

    # ---- helpers ----------------------------------------------------------------------------

    def micro_batches(self, seqs):
        for i in range(0, len(seqs), self.args.micro_batch):
            chunk = seqs[i: i + self.args.micro_batch]
            yield i, [x.to(self.device) for x in collate(chunk, self.tok.pad_token_id)]

    def set_adapter(self, name):
        # set_adapter re-enables grads on the selected adapter; keep ref frozen
        self.model.set_adapter(name)
        for n, p in self.model.named_parameters():
            if ".ref." in n:
                p.requires_grad_(False)

    @torch.no_grad()
    def score(self, seqs):
        """Old log-probs, reference log-probs and (PPO) old turn values per sequence."""
        old, ref, vals = [], [], []
        self.model.eval()
        for _, (ids, attn, turn) in self.micro_batches(seqs):
            act = turn[:, 1:] >= 0
            logp, hidden = action_logprobs(self.model, ids, attn, turn, return_hidden=True)
            old += split_by_row(logp, act)
            if self.args.algo == "ppo":
                first = act & (turn[:, 1:] != turn[:, :-1])
                vals += split_by_row(self.value_head(turn_start_states(hidden, turn)), first)
            self.set_adapter("ref")
            ref += split_by_row(action_logprobs(self.model, ids, attn, turn), act)
            self.set_adapter("default")
        self.model.train()
        return old, ref, vals

    # ---- one iteration ----------------------------------------------------------------------

    def iteration(self):
        a = self.args
        G = a.group_size if a.algo == "grpo" else 1
        patients = random.sample(self.train_pool, a.patients_per_iter)
        envs = run_episodes(self.model, self.tok, self.make_env, [p for p in patients for _ in range(G)],
                            temperature=a.temperature, top_p=a.top_p, batch_size=a.gen_batch)
        seqs = [tokenize_dialogue(self.tok, e.messages) for e in envs]
        old, ref, vals = self.score(seqs)

        # per-token advantages (and PPO value targets) for every sequence
        token_adv, returns_t, old_vals_t = [], [], []
        if a.algo == "grpo":
            R = np.array([sum(e.rewards) for e in envs]).reshape(-1, G)
            A = ((R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-4)).ravel()
            for (_, tr), adv in zip(seqs, A):
                n = sum(t >= 0 for t in tr[1:])
                token_adv.append(torch.full((n,), float(adv), device=self.device))
        else:
            advs, rets = [], []
            for e, (_, tr), lo, lr_, v in zip(envs, seqs, old, ref, vals):
                turns = torch.tensor([t for t in tr[1:] if t >= 0], device=self.device)
                kl_turn = torch.zeros(len(e.rewards), device=self.device).index_add_(0, turns, lo - lr_)
                r = np.array(e.rewards) - a.kl_coef * kl_turn.cpu().numpy()
                adv, ret = gae(r, v.cpu().numpy(), a.gamma, a.lam)
                advs.append(adv)
                rets.append(ret)
            flat = np.concatenate(advs)
            mu, sd = flat.mean(), flat.std() + 1e-8
            for (_, tr), adv, ret, v in zip(seqs, advs, rets, vals):
                turns = torch.tensor([t for t in tr[1:] if t >= 0], device=self.device)
                adv_t = torch.tensor((adv - mu) / sd, device=self.device, dtype=torch.float32)
                token_adv.append(adv_t[turns])
                returns_t.append(torch.tensor(ret, device=self.device, dtype=torch.float32))
                old_vals_t.append(v)

        stats = {"clipfrac": [], "kl": [], "policy_loss": [], "value_loss": []}
        order = list(range(len(seqs)))
        for _ in range(a.ppo_epochs):
            random.shuffle(order)
            for s in range(0, len(order), a.minibatch):
                idx = order[s: s + a.minibatch]
                mb = [seqs[i] for i in idx]
                n_tok = sum(len(old[i]) for i in idx)
                n_turn = sum(len(envs[i].rewards) for i in idx)
                for off, (ids, attn, turn) in self.micro_batches(mb):
                    rows = idx[off: off + a.micro_batch]
                    act = turn[:, 1:] >= 0
                    logp, hidden = action_logprobs(self.model, ids, attn, turn, return_hidden=True)
                    lo = torch.cat([old[i] for i in rows])
                    lr_ = torch.cat([ref[i] for i in rows])
                    adv = torch.cat([token_adv[i] for i in rows])
                    ratio = torch.exp(logp - lo)
                    pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - a.clip, 1 + a.clip))
                    loss = pg.sum() / n_tok
                    log_ratio_ref = lr_ - logp
                    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1  # k3 estimator
                    if a.algo == "grpo":
                        loss = loss + a.kl_coef * kl.sum() / n_tok
                    else:
                        first = act & (turn[:, 1:] != turn[:, :-1])
                        v = self.value_head(turn_start_states(hidden, turn))
                        v_old = torch.cat([old_vals_t[i] for i in rows])
                        ret = torch.cat([returns_t[i] for i in rows])
                        v_clip = v_old + (v - v_old).clamp(-a.clip_value, a.clip_value)
                        vl = 0.5 * torch.max((v - ret) ** 2, (v_clip - ret) ** 2)
                        loss = loss + a.vf_coef * vl.sum() / n_turn
                        stats["value_loss"].append(vl.mean().item())
                    loss.backward()
                    stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
                    stats["kl"].append(kl.mean().item())
                    stats["policy_loss"].append(pg.mean().item())
                torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], 1.0)
                self.opt.step()
                self.opt.zero_grad()

        out = aggregate([e.summary() for e in envs])
        out.update({k: float(np.mean(v)) for k, v in stats.items() if v})
        return out

    def save(self, path):
        self.model.save_pretrained(path, selected_adapters=["default"])
        self.tok.save_pretrained(path)
        if self.args.algo == "ppo":
            torch.save(self.value_head.state_dict(), os.path.join(path, "value_head.pt"))

    def train(self):
        a = self.args
        os.makedirs(a.out, exist_ok=True)
        log = open(os.path.join(a.out, "train_log.jsonl"), "w")
        best = -float("inf")
        for it in range(1, a.iterations + 1):
            m = self.iteration()
            m["iteration"] = it
            if it % a.eval_every == 0 or it == a.iterations:
                val = evaluate(self.model, self.tok, self.make_env, self.val_patients, batch_size=a.gen_batch)
                m.update({f"val_{k}": v for k, v in val.items()})
                if val["return"] > best:
                    best = val["return"]
                    self.save(os.path.join(a.out, "best"))
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
            log.write(json.dumps(m) + "\n")
            log.flush()
        self.save(os.path.join(a.out, "last"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "grpo"], default="ppo")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="initial LoRA (e.g. the SFT checkpoint); None = RL from scratch")
    ap.add_argument("--out", default="ckpt/ppo")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--patients_per_iter", type=int, default=64)
    ap.add_argument("--group_size", type=int, default=8, help="GRPO rollouts per patient")
    ap.add_argument("--ppo_epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--gen_batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--value_lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--clip_value", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--kl_coef", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    add_env_args(ap)
    ap.add_argument("--train_pool", type=int, default=50_000)
    ap.add_argument("--pool_seed", type=int, default=2)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    Trainer(args).train()


if __name__ == "__main__":
    main()
