"""Evaluate an LLM doctor (zero-shot, SFT or RL adapter) on held-out DDXPlus patients.

  python evaluate.py --model Qwen/Qwen2.5-7B-Instruct --question_menu --name zeroshot_7b
  python evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter ckpt/ppo/best --name ppo_1.5b
"""
import argparse
import json
import os

from data import KnowledgeBase, load_patients
from env import EnvConfig, EnvFactory
from llm import load_policy, load_tokenizer
from rollout import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True, help="results/<name>.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_questions", type=int, default=20)
    ap.add_argument("--question_cost", type=float, default=0.01)
    ap.add_argument("--question_menu", action="store_true", help="list askable questions in the prompt")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    tok = load_tokenizer(args.adapter or args.model)
    model = load_policy(args.model, adapter=args.adapter, trainable=False)
    kb = KnowledgeBase()
    make_env = EnvFactory(kb, EnvConfig(max_questions=args.max_questions, question_cost=args.question_cost,
                                        question_menu=args.question_menu))
    patients = load_patients(args.split, args.n, args.seed)
    metrics = evaluate(model, tok, make_env, patients, out_path=os.path.join("results", f"{args.name}.json"),
                       batch_size=args.batch_size)
    metrics.update(model=args.model, adapter=args.adapter, split=args.split)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
