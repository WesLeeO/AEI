"""Supervised fine-tuning (filtered behaviour cloning) on expert dialogues.

Next-token cross-entropy on the doctor's turns only (patient / prompt tokens are masked).

  python train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --data data/sft_expert.jsonl --out ckpt/sft
"""
import argparse
import json
import math
import os
import random

import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data import KnowledgeBase, load_patients
from env import EnvConfig, EnvFactory
from llm import action_logprobs, collate, load_policy, load_tokenizer, tokenize_dialogue
from rollout import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/sft_expert.jsonl")
    ap.add_argument("--out", default="ckpt/sft")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--eval_n", type=int, default=200, help="validation patients evaluated at the end")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = load_tokenizer(args.model)
    model = load_policy(args.model, lora_r=args.lora_r, gradient_checkpointing=args.grad_ckpt)
    model.print_trainable_parameters()
    device = next(model.parameters()).device

    with open(args.data) as f:
        rows = [json.loads(line) for line in f][: args.max_examples]
    seqs = [tokenize_dialogue(tok, r["messages"]) for r in rows]
    print(f"{len(seqs)} dialogues, mean length {sum(len(s[0]) for s in seqs) / len(seqs):.0f} tokens")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(seqs) / args.batch_size)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "w")

    model.train()
    bar = tqdm(total=steps)
    for epoch in range(args.epochs):
        random.shuffle(seqs)
        for i in range(0, len(seqs), args.batch_size):
            batch = seqs[i: i + args.batch_size]
            n_tok = sum(sum(t >= 0 for t in tr[1:]) for _, tr in batch)
            total = 0.0
            for j in range(0, len(batch), args.micro_batch):
                ids, attn, turn = (x.to(device) for x in collate(batch[j: j + args.micro_batch], tok.pad_token_id))
                loss = -action_logprobs(model, ids, attn, turn).sum() / n_tok
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            bar.update(1)
            bar.set_postfix(loss=f"{total:.4f}")
            log.write(json.dumps({"epoch": epoch, "step": bar.n, "loss": total}) + "\n")
    log.close()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    if args.eval_n:
        kb = KnowledgeBase()
        make_env = EnvFactory(kb, EnvConfig())
        metrics = evaluate(model, tok, make_env, load_patients("validate", args.eval_n, 0),
                           out_path=os.path.join(args.out, "val_eval.json"))
        print("validation:", json.dumps(metrics))


if __name__ == "__main__":
    main()
