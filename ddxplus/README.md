# Learning to Ask the Right Questions: Diagnostic History-Taking on DDXPlus

An LLM "doctor" talks to a simulated patient over several turns. It asks one question per turn, then
commits to a diagnosis written as an **ICD-10 code**. It is trained with **SFT** (filtered behaviour cloning
on expert dialogues) followed by online RL (**PPO** or **GRPO**). The goal is to be correct while asking as few
questions as possible.

This folder is the medical half of the project. The GuessMyCity half (BC / MC / PPO on LMRL-Gym) lives in the
repo root. MIMIC access was not available, so the medical task uses the public **DDXPlus** dataset
(Fansi Tchango et al., NeurIPS 2022 D&B): 1.3M synthetic patients, 49 pathologies, each with an ICD-10 code,
223 evidences (symptoms and antecedents) with natural-language questions.

## Environment

- **Patient:** age, sex, chief complaint (DDXPlus `INITIAL_EVIDENCE`) and a hidden set of evidences.
- **Doctor turn:** a free-text question, or `Diagnosis: <condition> (ICD-10: <code>)`.
- **Question → evidence:** exact match, otherwise nearest DDXPlus question by sentence-embedding cosine
  (`bge-small-en-v1.5`, threshold 0.75). Below the threshold the patient answers "Sorry, I don't understand the question."
- **Answers** are rendered deterministically from the ground truth ("Yes.", "No.", "Cheek(r), forehead.",
  "4 (on a scale from 0 to 10)."). There is **no LLM oracle**, so the environment is cheap, exact and reproducible.
- **Reward:** `+1` for the correct pathology / ICD-10 code, `-0.01` per question; at most 20 questions.
- **Metrics:** ICD-10 top-1 accuracy, average number of questions, return, evidence recall (fraction of the patient's
  positive evidences that were asked), unmatched / repeated question counts, and whether the prediction is in the
  patient's DDXPlus differential.

## Methods

| Method | What it is |
|---|---|
| `nb_no_questions` | Naive Bayes on age, sex and chief complaint only |
| `random_kq + nb` | k random questions, then Naive Bayes diagnosis |
| `infogain_expert` | Greedy expected-information-gain questioner over a Naive-Bayes posterior; stops when max posterior ≥ τ |
| zero-shot LLM | Instruct model, optionally with the list of askable questions in the prompt (`--question_menu`) |
| **SFT** | LoRA fine-tuning on correct expert dialogues (doctor tokens only) |
| **PPO** | Turn-level PPO: value head on the state before each doctor turn, KL-in-reward, GAE (γ=1, λ=0.95), clipped policy and value losses |
| **GRPO** | 8 rollouts per patient, group-normalised episode returns, clipped loss and k3 KL to the SFT reference |

The non-LLM baselines are played through the same text environment, so all numbers are directly comparable.

## Results

Non-LLM baselines, DDXPlus test (2,000 patients):

<!-- filled from results/baselines_test.json -->
| Method | ICD-10 acc. | # questions | Return |
|---|---|---|---|
| nb_all_evidence (upper bound, no dialogue) | 98.6% | 14.6 | – |
| nb_no_questions | 34.5% | 0.0 | 0.345 |
| random_5q + nb | 45.3% | 5.0 | 0.403 |
| random_10q + nb | 54.1% | 10.0 | 0.442 |
| random_20q + nb | 69.8% | 20.0 | 0.498 |
| infogain_expert tau=0.5 | 84.3% | 1.0 | 0.833 |
| infogain_expert tau=0.7 | 93.5% | 1.3 | 0.923 |
| infogain_expert tau=0.8 | 95.5% | 1.4 | 0.941 |
| infogain_expert tau=0.9 | 96.9% | 1.7 | 0.952 |
| infogain_expert tau=0.95 | 97.8% | 1.9 | 0.959 |
| infogain_expert tau=0.99 | 98.8% | 2.5 | 0.963 |
| noisy_expert eps=0.3 tau=0.9 | 96.8% | 2.2 | 0.946 |
| noisy_expert eps=0.6 tau=0.9 | 96.5% | 3.6 | 0.930 |

LLM agents: **to be filled after running on the cluster** (`results/*.json`, `results/pareto.png`).

| Agent | Model | ICD-10 acc. | # questions | Return |
|---|---|---|---|---|
| zero-shot | | | | |
| zero-shot + question list | | | | |
| SFT | | | | |
| SFT → PPO | | | | |
| SFT → GRPO | | | | |

Note: DDXPlus patients are sampled from a generator in which evidences are (close to) conditionally independent
given the pathology. That makes Naive Bayes a near-optimal model, and the info-gain expert a very strong
reference. The interesting questions are how close LLM agents get, and how much RL improves over imitation,
especially when imitating a weaker demonstrator (`--epsilon` / lower `--threshold`).

## Models

The GuessMyCity experiments use GPT-2-medium. The DDXPlus code takes any Hugging Face chat model through
`--model` (LoRA everywhere, bf16 on GPU), so it can be scaled to whatever the cluster allows:

| Model | GPU memory (SFT / RL) | Notes |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | ~16 GB | debugging only |
| `Qwen/Qwen2.5-1.5B-Instruct` | 24–40 GB | default; fast iteration |
| `Qwen/Qwen2.5-3B-Instruct` | 40 GB | |
| `Qwen/Qwen2.5-7B-Instruct` / `meta-llama/Llama-3.1-8B-Instruct` | 80 GB (`--grad_ckpt`, smaller `--micro_batch`) | strongest zero-shot baseline |

Qwen3 chat templates (thinking mode) are not supported by the action-mask tokenisation; use Qwen2.5 or Llama-3.

## Usage

```bash
pip install -r ../requirements.txt sentence-transformers
# data (~180 MB zipped) from https://figshare.com/articles/dataset/20043374 into ddxplus/data/, then unzip

python baselines.py --mode eval --n 2000              # results/baselines_test.json
python baselines.py --mode make_sft --n 20000         # data/sft_expert.jsonl

python evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --name zeroshot_1.5b --question_menu
python train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --out ckpt/sft_1.5b
python train_rl.py --algo ppo  --model Qwen/Qwen2.5-1.5B-Instruct --adapter ckpt/sft_1.5b --out ckpt/ppo_1.5b
python train_rl.py --algo grpo --model Qwen/Qwen2.5-1.5B-Instruct --adapter ckpt/sft_1.5b --out ckpt/grpo_1.5b
python evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter ckpt/ppo_1.5b/best --name ppo_1.5b
python plot.py

# everything at once on a SLURM cluster (ETH Euler):
sbatch --export=MODEL=Qwen/Qwen2.5-1.5B-Instruct,TAG=1.5b slurm/pipeline.sbatch
```

## Files

```
data.py        knowledge base (questions, ICD-10 codes) + patient loading
env.py         text environment: question matching, patient answers, diagnosis parsing, reward
bayes.py       Naive Bayes + information-gain expert
baselines.py   non-LLM baselines, expert SFT dialogues
llm.py         model/LoRA loading, chat tokenisation with action masks, batched generation
rollout.py     batched multi-turn rollouts + evaluation
train_sft.py   SFT / filtered BC
train_rl.py    PPO and GRPO
evaluate.py    held-out evaluation -> results/<name>.json
plot.py        accuracy-vs-questions plot, RL curves
slurm/         cluster job script
```
