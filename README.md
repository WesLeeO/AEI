# Semester Project, LAS LAB Active Elicitation Information

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

## Loss functions

**BC** — masked next-token cross-entropy (answer tokens weighted by $w=$ `non_action_weight`, default $0$):

$$\mathcal{L}_{\text{BC}}=-\frac{\sum_t m_t\,[a_t+(1-a_t)\,w]\,\log\pi_\theta(x_{t+1}\mid x_{\le t})}{\sum_t m_t}$$

**MC** — Q-regression onto reward-to-go + CQL anchor; decoded with $\ell=\ell_\pi+\beta\,Q$ ($\beta=16$):

$$\mathcal{L}_{\text{MC}}=\frac{1}{N}\sum_t m_t\Big[\tfrac12\big(Q_\phi(s_t,a_t)-\mathrm{sg}[R_t]\big)^2-\lambda_{\text{CQL}}\,\log\pi_\theta(a_t\mid s_t)\Big],\qquad R_t=\sum_{k\ge t}\gamma^{k-t}r_k$$

**PPO** — clipped policy + clipped value, with the KL folded into the reward before GAE:

$$\mathcal{L}_{\text{PPO}}=\frac{1}{N}\sum_t m_t\Big[\max\!\big(-A_t\rho_t,\,-A_t\,\mathrm{clip}(\rho_t,1\!-\!\epsilon,1\!+\!\epsilon)\big)+\tfrac{c_v}{2}\max\!\big((V_\psi-\hat R_t)^2,(V^{\text{clip}}-\hat R_t)^2\big)\Big]$$

$$\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}$$

$$\hat r_t=r_t-\beta_{\text{KL}}\big(\log\pi_{\text{old}}(a_t\mid s_t)-\log\pi_{\text{ref}}(a_t\mid s_t)\big)$$

$$\delta_t=\hat r_t+\gamma V(s_{t+1})-V(s_t)$$

$$A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$$

$$\hat R_t=A_t+V(s_t)$$

**Symbols.**

- $\pi_\theta$ = policy LM (params $\theta$)
- $Q_\phi$ = Q head (params $\phi$, separate from the LM)
- $a_t$ = action (question-token) indicator at step $t$
- $m_t$ = action mask $\wedge$ attention
- $N=\sum_t m_t$
- $\mathrm{sg}[\cdot]$ = stop-gradient
- $R_t$ = (discounted) reward-to-go
- $A_t$ = whitened GAE advantage
- $\hat R_t$ = value target
- $V_\psi$ = value head (params $\psi$, separate from the LM), $V^{\text{clip}}=V_{\text{old}}+\mathrm{clip}(V_\psi-V_{\text{old}},-\epsilon_v,\epsilon_v)$
- $\beta$ = MC decode weight
- $\beta_{\text{KL}}$ = `init_kl_coef`
- $\lambda_{\text{CQL}}$ = `cql_weight`
- $\epsilon,\epsilon_v$ = `cliprange`, `cliprange_value`
- $c_v$ = `value_loss_coef`

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
