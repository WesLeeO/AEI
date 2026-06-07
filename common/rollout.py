"""Shared episode rollout: drive a policy against the GuessCityEnv.

`run_episode` is decoder-agnostic — it takes a `generate_question(prompt) -> str` callable,
so the same loop serves PPO sampling, MC value-weighted decoding, and plain evaluation.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from .guess_city import GuessCityEnv, render_prompt


def postprocess_question(text: str) -> str:
    """Keep only the first question: cut at the first '?'."""
    text = text.strip()
    if "?" in text:
        text = text.split("?")[0] + "?"
    return text.strip()


@torch.no_grad()
def sample_question(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 30,
                    gen_kwargs: Optional[Dict] = None) -> str:
    """Generate one question from the policy via HF .generate()."""
    gen_kwargs = gen_kwargs or {}
    tokenizer.truncation_side = "left"
    ids = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True,
        max_length=tokenizer.model_max_length - max_new_tokens,
    ).to(device)
    out = model.generate(
        input_ids=ids, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, **gen_kwargs,
    )
    text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return postprocess_question(text)


def run_episode(
    env: GuessCityEnv,
    generate_question: Callable[[str], str],
    city: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Run one full episode. Returns the questions asked and the final outcome."""
    env.reset(seed=seed, city=city)
    if verbose:
        print(f"[target] {env.target}")
    infos: List[Dict] = []
    while True:
        prompt = render_prompt(env.history)
        question = generate_question(prompt)
        info = env.step_question(question)
        infos.append(info)
        if verbose:
            print(f"  Q{info['asked']}: {question}\n       A: {info['answer']}")
        if info["done"]:
            break
    return {
        "target": env.target,
        "questions": [q for q, _ in env.history],
        "answers": [a for _, a in env.history],
        "asked": env.asked,
        "correct": infos[-1]["correct"],
        "infos": infos,
    }
