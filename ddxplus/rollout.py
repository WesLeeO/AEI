"""Batched multi-turn rollouts of an LLM doctor against simulated DDXPlus patients."""
import json
import os

from env import aggregate
from llm import generate


def run_episodes(model, tok, make_env, patients, greedy=False, temperature=1.0, top_p=1.0,
                 max_new_tokens=64, batch_size=32):
    """Plays all patients in parallel, one doctor turn at a time. Returns finished envs."""
    was_training = model.training
    model.eval()
    envs = [make_env(p) for p in patients]
    while True:
        active = [e for e in envs if not e.done]
        if not active:
            break
        actions = generate(model, tok, [e.messages for e in active], max_new_tokens=max_new_tokens,
                           temperature=temperature, top_p=top_p, greedy=greedy, batch_size=batch_size)
        for env, a in zip(active, actions):
            env.step(a)
    model.train(was_training)
    return envs


def evaluate(model, tok, make_env, patients, out_path=None, n_examples=20, **gen):
    envs = run_episodes(model, tok, make_env, patients, greedy=True, **gen)
    summaries = [e.summary() for e in envs]
    metrics = aggregate(summaries)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        examples = [{"summary": s, "dialogue": e.messages[1:]} for e, s in zip(envs[:n_examples], summaries)]
        with open(out_path, "w") as f:
            json.dump({"metrics": metrics, "examples": examples}, f, indent=2)
    return metrics
