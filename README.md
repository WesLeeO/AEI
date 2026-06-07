# GuessMyCity — PyTorch port of LMRL-Gym

A clean PyTorch reimplementation of three finetuning methods for the **GuessMyCity** task
from [LMRL-Gym](https://github.com/abdulhaim/LMRL-Gym) (original is JAX). An "asker" policy
(GPT-2-medium + LoRA) learns to identify a target city in as few questions as possible; a
freeform **Phi-3** oracle answers questions (the analog of the GPT-3 oracle that generated
the official dataset).

Reward (all methods, matching the benchmark): **−1 per question, 0 per answer, 0 for a
correct final guess**.

## Layout

```
common/            shared core (used by all three methods)
  environment.py   Text / TokenTrajectory data structures (port of LLM_RL/environment.py)
  guess_city.py    conversation parsing, Phi-3 oracle, live env, data loading
  models.py        GPT2WithValueHead (PPO), GPT2WithQHead (MC)
  utils.py         get_rtg (RTG), get_advantages_and_returns (GAE), whiten, padding
  rollout.py       run one episode (policy + oracle)
  evaluate.py      accuracy + avg-#questions over a set of cities
config.py          hyperparameters (BCConfig / MCConfig / PPOConfig), benchmark defaults
bc/  train_bc.py   behaviour cloning (filtered imitation of the asker)
mc/  mc_returns.py + train_mc.py   Monte-Carlo returns (Q-regression + CQL, beta decoding)
ppo/ ppo.py + train_ppo.py         PPO (KL-in-reward, GAE, clipped loss)
```

Expected local assets (not in this repo): `./gpt2-medium-offline` (base model),
`bc/fine_tuned_gpt2_medium_lora_filtered` (SFT adapter), `train.json` / `eval.json`
(official RAIL dataset), and the `microsoft/Phi-3-mini-4k-instruct` oracle.

## Methods (what each replicates)

| Method | Idea | Key faithful detail |
|---|---|---|
| **BC** (`bc/`) | imitate the asker | next-token CE masked to question tokens (`non_action_weight=0`) |
| **MC** (`mc/`) | offline value-weighted decoding | per-token RTG over action tokens (γ=0.99) + CQL; decode `logits = sft + β·Q`, β=16 |
| **PPO** (`ppo/`) | on-policy RL | KL folded into reward, GAE (γ=1.0, λ=0.95) + whitening, clipped policy/value loss |

## Usage

```bash
pip install -r requirements.txt

# Behaviour cloning
python bc/train_bc.py --mode train
python bc/train_bc.py --mode eval --adapter bc/bc_model

# Monte-Carlo returns (trains a Q head on the frozen SFT policy)
python mc/train_mc.py --mode train
python mc/train_mc.py --mode eval --beta 16

# PPO
python ppo/train_ppo.py --mode train --num-iterations 100
python ppo/train_ppo.py --mode eval --adapter ppo/ppo_model
```

Hyperparameters live in `config.py` and default to the LMRL-Gym guess_city settings.
