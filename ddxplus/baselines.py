"""Non-LLM baselines + expert dialogue generation for SFT.

Every baseline is played through the same text environment as the LLM agents (questions are
sent as their DDXPlus question text, diagnoses as 'Diagnosis: ... (ICD-10: ...)'), so all
metrics are directly comparable.

  python baselines.py --mode eval --n 2000
  python baselines.py --mode make_sft --n 20000 --out data/sft_expert.jsonl
"""
import argparse
import json
import os
import random

import numpy as np
from tqdm import tqdm

from bayes import InfoGainExpert, NaiveBayes
from data import KnowledgeBase, load_patients
from env import EnvConfig, EnvFactory, aggregate


def diagnosis_text(kb: KnowledgeBase, cond: str) -> str:
    return f"Diagnosis: {cond} (ICD-10: {kb.icd_label[cond]})"


def play(make_env, patient, questions: list[str], diagnosis: str):
    env = make_env(patient)
    for e in questions:
        env.step(env.kb.question[e])
    if not env.done:
        env.step(diagnosis)
    return env


def run_infogain(kb, nb, make_env, patients, threshold, max_q, epsilon=0.0):
    expert = InfoGainExpert(nb, threshold, max_q, epsilon)
    envs = []
    for p in patients:
        asked, pred = expert.run(p)
        envs.append(play(make_env, p, asked, diagnosis_text(kb, kb.conditions[pred])))
    return envs


def run_random(kb, nb, make_env, patients, k, rng):
    envs = []
    for p in patients:
        pool = [e for e in kb.evidence_names if e != p.initial_evidence]
        asked = rng.sample(pool, k)
        pred = kb.conditions[nb.predict(p, asked)]
        envs.append(play(make_env, p, asked, diagnosis_text(kb, pred)))
    return envs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "make_sft"], default="eval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_questions", type=int, default=20)
    ap.add_argument("--question_cost", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.9, help="expert stopping threshold (make_sft)")
    ap.add_argument("--epsilon", type=float, default=0.0, help="expert random-question rate (make_sft)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kb = KnowledgeBase()
    nb = NaiveBayes.load_or_fit(kb)
    make_env = EnvFactory(kb, EnvConfig(max_questions=args.max_questions, question_cost=args.question_cost))

    if args.mode == "make_sft":
        # seed+1 so the dialogues are mostly disjoint from the NB fitting sample
        patients = load_patients("train", args.n, args.seed + 1)
        out = args.out or os.path.join("data", "sft_expert.jsonl")
        kept = 0
        with open(out, "w") as f:
            for env in tqdm(run_infogain(kb, nb, make_env, patients, args.threshold, args.max_questions, args.epsilon)):
                s = env.summary()
                if s["correct"]:  # filtered BC: imitate successful dialogues only
                    f.write(json.dumps({"messages": env.messages, **s}) + "\n")
                    kept += 1
        print(f"wrote {kept}/{len(patients)} correct expert dialogues to {out}")
        return

    patients = load_patients(args.split, args.n, args.seed)
    rng = random.Random(args.seed)
    results = {}

    results["nb_all_evidence (upper bound, no dialogue)"] = {
        "accuracy": float(np.mean([kb.conditions[nb.predict(p, p.evidences)] == p.pathology for p in patients])),
        "n_questions": float(np.mean([len(p.evidences) - 1 for p in patients])),
    }
    results["nb_no_questions"] = aggregate([play(make_env, p, [], diagnosis_text(kb, kb.conditions[nb.predict(p, [])])).summary() for p in patients])
    for k in [5, 10, 20]:
        results[f"random_{k}q + nb"] = aggregate([e.summary() for e in run_random(kb, nb, make_env, patients, k, rng)])
    for tau in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        envs = run_infogain(kb, nb, make_env, patients, tau, args.max_questions)
        results[f"infogain_expert tau={tau}"] = aggregate([e.summary() for e in envs])
    for eps in [0.3, 0.6]:
        envs = run_infogain(kb, nb, make_env, patients, 0.9, args.max_questions, eps)
        results[f"noisy_expert eps={eps} tau=0.9"] = aggregate([e.summary() for e in envs])

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"baselines_{args.split}.json"), "w") as f:
        json.dump(results, f, indent=2)
    for name, r in results.items():
        print(f"{name:45s} acc={r['accuracy']:.3f}  questions={r['n_questions']:.1f}"
              + (f"  recall={r['evidence_recall']:.2f}  return={r['return']:.3f}" if "return" in r else ""))


if __name__ == "__main__":
    main()
