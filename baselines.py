"""Non-LLM baselines + expert dialogue generation for SFT.

Every baseline is played through the same text environment as the LLM agents (questions are
sent as their DDXPlus question text, diagnoses as 'Diagnosis: ... (ICD-10: ...)'), so all
metrics are directly comparable.

  python baselines.py --mode eval --n 2000 [--binary_questions] [--differential]
  python baselines.py --mode make_sft --n 20000 [--binary_questions] [--differential]
"""
import argparse
import json
import os
import random

import numpy as np
from tqdm import tqdm

from bayes import InfoGainExpert, NaiveBayes
from data import load_patients
from env import DIFFERENTIAL_K, add_env_args, aggregate, build_env, setting_tag


def diagnosis_text(make_env, log_post: np.ndarray) -> str:
    """'Diagnosis: A (ICD-10: ..)' or, in differential mode, the top-3 ranking separated by ';'."""
    kb = make_env.kb
    k = DIFFERENTIAL_K if make_env.cfg.differential else 1
    top = [kb.conditions[i] for i in np.argsort(-log_post)[:k]]
    return "Diagnosis: " + "; ".join(f"{c} (ICD-10: {kb.icd_label[c]})" for c in top)


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
        asked, log_post = expert.run(p)
        envs.append(play(make_env, p, asked, diagnosis_text(make_env, log_post)))
    return envs


def run_random(kb, nb, make_env, patients, k, rng):
    envs = []
    for p in patients:
        pool = [e for e in kb.evidence_names if e != p.initial_evidence]
        asked = rng.sample(pool, k)
        envs.append(play(make_env, p, asked, diagnosis_text(make_env, nb.log_posterior(p, asked))))
    return envs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "make_sft"], default="eval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    add_env_args(ap)
    ap.add_argument("--threshold", type=float, default=0.9, help="expert stopping threshold (make_sft)")
    ap.add_argument("--epsilon", type=float, default=0.0, help="expert random-question rate (make_sft)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    make_env = build_env(args)
    kb = make_env.kb
    nb = NaiveBayes.load_or_fit(kb)

    if args.mode == "make_sft":
        # seed+1 so the dialogues are mostly disjoint from the NB fitting sample
        patients = load_patients("train", args.n, args.seed + 1)
        out = args.out or os.path.join("data", f"sft_expert{setting_tag(args)}.jsonl")
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

    all_q = [e for e in kb.evidence_names]
    results["nb_all_questions_answered (upper bound, no dialogue)"] = {
        "accuracy": float(np.mean([kb.conditions[nb.predict(p, all_q)] == p.pathology for p in patients])),
        "n_questions": float(len(all_q)),
    }
    results["nb_no_questions"] = aggregate(
        [play(make_env, p, [], diagnosis_text(make_env, nb.log_posterior(p, []))).summary() for p in patients])
    for k in [5, 10, 20]:
        results[f"random_{k}q + nb"] = aggregate([e.summary() for e in run_random(kb, nb, make_env, patients, k, rng)])
    for tau in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        envs = run_infogain(kb, nb, make_env, patients, tau, args.max_questions)
        results[f"infogain_expert tau={tau}"] = aggregate([e.summary() for e in envs])
    for eps in [0.3, 0.6]:
        envs = run_infogain(kb, nb, make_env, patients, 0.9, args.max_questions, eps)
        results[f"noisy_expert eps={eps} tau=0.9"] = aggregate([e.summary() for e in envs])

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"baselines_{args.split}{setting_tag(args)}.json"), "w") as f:
        json.dump(results, f, indent=2)
    for name, r in results.items():
        print(f"{name:52s} acc={r['accuracy']:.3f}  questions={r['n_questions']:.1f}"
              + (f"  top3={r['top3']:.3f}  ddx_recall={r['ddx_recall']:.2f}  return={r['return']:.3f}"
                 if "return" in r else ""))


if __name__ == "__main__":
    main()
