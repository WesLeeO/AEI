# Learning to Ask the Right Questions: Diagnostic History-Taking with RL

Semester project, LAS Lab, ETH Zürich.

An LLM "doctor" talks to a simulated patient over several turns. It asks one question per turn, then commits to a
diagnosis written as an **ICD-10 code**. It is trained with **SFT** (filtered behaviour cloning on expert dialogues),
followed by online RL with **PPO** or **GRPO**. The goal is to be correct while asking as few questions as possible.

Patients come from the public **DDXPlus** dataset (Fansi Tchango et al., NeurIPS 2022 Datasets & Benchmarks):
1.3M synthetic patients, 49 pathologies, each with an ICD-10 code, and 223 evidences (symptoms and antecedents),
each with a natural-language question.

## Environment

- **Patient:** age, sex, chief complaint (DDXPlus `INITIAL_EVIDENCE`) and a hidden set of evidences.
- **Doctor turn:** a free-text question, or `Diagnosis: <condition> (ICD-10: <code>)`.
- **Question → evidence:** exact match, otherwise nearest DDXPlus question by sentence-embedding cosine
  (`bge-small-en-v1.5`, threshold 0.75). Below the threshold the patient answers "Sorry, I don't understand the question."
- **Answers** are rendered deterministically from the ground truth ("Yes.", "No.", "Cheek(r), forehead.",
  "4 (on a scale from 0 to 10)."). There is **no LLM oracle**, so the environment is cheap, exact and reproducible.
- **Reward per doctor turn:** `−0.01` for a question; `+1` for a correct diagnosis (pathology / ICD-10 code), else `0`.
  After 20 questions the next turn is forced to be the diagnosis. Episode return: $R = \mathbb{1}[\text{correct}] - 0.01\cdot\#\text{questions}$.
- **Metrics:** ICD-10 top-1 and top-3 accuracy, average number of questions, return, differential recall / precision
  (see below), evidence recall (fraction of the patient's positive evidences that were asked), and unmatched /
  repeated question counts.

### Settings

| Flag | Effect |
|---|---|
| *(none)* — **easy** | Original DDXPlus questions. Multi-choice questions ("Do you feel pain somewhere?") reveal several facts in one answer ("Cheek(r), forehead."). |
| `--binary_questions` | **Yes/no only.** Every multi-choice question is split into one yes/no question per option ("Do you feel pain in the left cheek?"), 210 new questions in total (428 askable questions overall). The scale and categorical questions (e.g. pain intensity 0–10) stay as they are. |
| `--differential` | The doctor answers with a **ranked differential** of up to 3 conditions: `Diagnosis: A (ICD-10: ..); B (ICD-10: ..); C (ICD-10: ..)`. Top-1 is the first entry. |
| `--ddx_reward_weight w` | Diagnosis reward becomes $(1-w)\cdot\mathbb{1}[\text{top-1 correct}] + w\cdot\text{ddx recall}$ (default $w=0$). |

The **hard** setting is `--binary_questions --differential`. **Differential recall** is the fraction of the DDXPlus
ground-truth top-3 differential (each patient comes with a ranked list of plausible pathologies) that appears in
the predicted ranking. **Differential precision** is the fraction of predicted conditions that appear anywhere in
the ground-truth differential.

## Methods

| Method | What it is |
|---|---|
| zero-shot LLM | Instruct model, optionally with the list of askable questions in the prompt (`--question_menu`) |
| **SFT** | LoRA fine-tuning on correct expert dialogues (doctor tokens only) |
| **PPO** | Turn-level PPO: value head on the state before each doctor turn, KL-in-reward, GAE over turns, clipped policy and value losses |
| **GRPO** | G rollouts per patient, group-normalised episode returns, clipped loss and KL to the SFT reference |

RL starts from the SFT adapter, which is also the frozen reference policy for the KL terms.

### Loss functions

Notation: $\pi_\theta$ is the policy LM (LoRA), $\pi_{\text{old}}$ the policy that generated the rollouts, $\pi_{\text{ref}}$
the frozen SFT policy. Token $t$ belongs to doctor turn $k(t)$; $m_t=1$ for doctor (action) tokens only, and
$N=\sum_t m_t$.

**SFT** — next-token cross-entropy on the doctor's tokens only (patient and prompt tokens are masked):

$$\mathcal{L}_{\text{SFT}}=-\frac{1}{N}\sum_t m_t\log\pi_\theta(x_t\mid x_{<t})$$

**PPO** — rewards, values and advantages are per **turn** $k$; the ratio and clipping are per **token**:

$$\hat r_k=r_k-\beta_{\text{KL}}\sum_{t\in k}\big(\log\pi_{\text{old}}(x_t\mid x_{<t})-\log\pi_{\text{ref}}(x_t\mid x_{<t})\big)$$

$$\delta_k=\hat r_k+\gamma V(s_{k+1})-V(s_k),\qquad A_k=\sum_{l\ge0}(\gamma\lambda)^l\delta_{k+l},\qquad \hat R_k=A_k+V(s_k)$$

$$\rho_t=\frac{\pi_\theta(x_t\mid x_{<t})}{\pi_{\text{old}}(x_t\mid x_{<t})}$$

$$\mathcal{L}_{\text{PPO}}=\frac{1}{N}\sum_t m_t\max\big(-A_{k(t)}\rho_t,\,-A_{k(t)}\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\big)+\frac{c_v}{K}\sum_k\tfrac12\max\big((V_\psi(s_k)-\hat R_k)^2,(V^{\text{clip}}_k-\hat R_k)^2\big)$$

- $s_k$ is the dialogue before doctor turn $k$. $V_\psi$ is a value head on the hidden state that predicts that turn's first token.
- Advantages are whitened over the batch and held fixed during the update. $\hat R_k$ is the λ-return, the critic's
  regression target. $V^{\text{clip}}_k=V_{\text{old}}+\mathrm{clip}(V_\psi-V_{\text{old}},-\epsilon_v,\epsilon_v)$.
- Why per turn: rewards only arrive once the patient answers a whole message, so credit is assigned per question,
  not per token. Defaults: γ=1, λ=0.95, ε=ε_v=0.2, β_KL=0.02, c_v=0.5.

**GRPO** — no critic. Each patient is played $G=8$ times with returns $R_1,\dots,R_G$, and every doctor token of
episode $i$ gets the same advantage:

$$A_i=\frac{R_i-\mathrm{mean}(R_{1:G})}{\mathrm{std}(R_{1:G})},\qquad \Delta_t=\log\pi_{\text{ref}}(x_t\mid x_{<t})-\log\pi_\theta(x_t\mid x_{<t})$$

$$\mathcal{L}_{\text{GRPO}}=\frac{1}{N}\sum_t m_t\Big[\max\big(-A_i\rho_t,\,-A_i\,\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\big)+\beta_{\text{KL}}\big(e^{\Delta_t}-\Delta_t-1\big)\Big]$$

## Bayesian baselines and the SFT expert

`bayes.py` holds a non-LLM reference agent. It serves two purposes:

1. a **baseline** that shows what good questioning looks like on DDXPlus;
2. the **expert** whose successful dialogues make up the SFT data.

**Naive Bayes model.** From 200k training patients, estimate the prior $P(c)$ over the 49 pathologies and the
likelihood of each answer given the pathology: $P(\text{yes}\mid c)$ for yes/no evidences, $P(\text{value}\mid c)$ for
categorical ones (e.g. pain intensity 0–10), and $P(\text{value}\in\text{answer}\mid c)$ for multi-choice ones (e.g.
pain location), all with Laplace smoothing. Given the answers so far, the posterior is

$$P(c\mid \text{answers})\;\propto\;P(c)\,P(\text{age}\mid c)\,P(\text{sex}\mid c)\prod_{\text{answers } e}P(e\mid c).$$

**Information-gain expert.** At each turn, pick the question whose answer is expected to reduce the posterior's
uncertainty the most:

$$e^\star=\arg\max_e\Big[H\big(P(c)\big)-\mathbb{E}_{\text{answer}}\,H\big(P(c\mid\text{answer})\big)\Big]$$

Stop when $\max_c P(c\mid\cdot)\ge\tau$ (or after 20 questions) and diagnose $\arg\max_c$. Expected entropies are exact
for yes/no and categorical questions, and Monte-Carlo estimated for multi-choice ones. `--epsilon` makes the expert
ask a random question with probability ε, which gives a weaker demonstrator that RL can improve on.

DDXPlus patients are sampled from a generator in which evidences are (close to) conditionally independent given the
pathology. That makes Naive Bayes nearly the right model, so the expert is a very strong reference. The interesting
questions are how close LLM agents get, and how much RL improves over imitation.

## Results

Non-LLM baselines, DDXPlus test (2,000 patients), played through the same text environment. The full sweeps
(τ ∈ {0.5…0.99}, 5/10/20 random questions) are in `results/baselines_test*.json`.

| Setting | Method | Top-1 acc. | Top-3 acc. | DDx recall | # questions |
|---|---|---|---|---|---|
| easy | NB, no questions (age, sex, complaint) | 34.5% | 34.5% | 0.18 | 0.0 |
| easy | 20 random questions + NB | 69.8% | 69.8% | 0.28 | 20.0 |
| easy | Info-gain expert, τ=0.9 | 96.9% | 96.9% | 0.35 | 1.7 |
| easy | Noisy expert, ε=0.6 | 96.5% | 96.5% | 0.35 | 3.6 |
| yes/no | 20 random questions + NB | 59.9% | 59.9% | 0.25 | 20.0 |
| yes/no | Info-gain expert, τ=0.9 | 97.2% | 97.2% | 0.35 | 4.1 |
| yes/no | Noisy expert, ε=0.6 | 93.3% | 93.3% | 0.34 | 7.7 |
| **hard** (yes/no + differential) | NB, no questions | 34.5% | 57.9% | 0.35 | 0.0 |
| **hard** | 20 random questions + NB | 59.9% | 80.3% | 0.41 | 20.0 |
| **hard** | Info-gain expert, τ=0.9 | 97.2% | 99.8% | 0.48 | 4.1 |
| **hard** | Noisy expert, ε=0.6 | 93.3% | 97.8% | 0.47 | 7.7 |

Takeaways:

- **Multi-choice questions make DDXPlus easy.** One "where is your pain?" question lifts accuracy from 35% to about 85%.
  With yes/no questions only, the expert needs about 2.4× more questions (4.1 vs 1.7), and random questioning
  drops by 10 points.
- **The differential leaves headroom.** The expert is almost always right in its top 3, but it recovers only about
  half of the DDXPlus ground-truth differential (recall ≈ 0.48). Naive Bayes is overconfident and ranks the
  runners-up poorly. This is a natural target for RL with `--ddx_reward_weight > 0`.
- With every question answered, Naive Bayes reaches 99.6% top-1, so the remaining challenge is choosing *which*
  few questions to ask.

LLM agents: **to be filled after running on the cluster** (`results/*.json`, `results/pareto.png`).

| Setting | Agent | Model | Top-1 acc. | Top-3 acc. | DDx recall | # questions | Return |
|---|---|---|---|---|---|---|---|
| easy | zero-shot + question list | | | | | | |
| easy | SFT | | | | | | |
| easy | SFT → PPO | | | | | | |
| easy | SFT → GRPO | | | | | | |
| hard | zero-shot + question list | | | | | | |
| hard | SFT | | | | | | |
| hard | SFT → PPO | | | | | | |
| hard | SFT → GRPO | | | | | | |

## Models

Any Hugging Face chat model works through `--model` (LoRA everywhere, bf16 on GPU):

| Model | GPU memory (SFT / RL) | Notes |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | ~16 GB | debugging only |
| `Qwen/Qwen2.5-1.5B-Instruct` | 24–40 GB | default; fast iteration |
| `Qwen/Qwen2.5-3B-Instruct` | 40 GB | |
| `Qwen/Qwen2.5-7B-Instruct` / `meta-llama/Llama-3.1-8B-Instruct` | 80 GB (`--grad_ckpt`, smaller `--micro_batch`) | strongest zero-shot baseline |

Qwen3 chat templates (thinking mode) are not supported by the action-mask tokenisation; use Qwen2.5 or Llama-3.

## Usage

```bash
pip install -r requirements.txt
# DDXPlus (~180 MB zipped) from https://figshare.com/articles/dataset/20043374 into data/, then unzip

python baselines.py --mode eval --n 2000              # results/baselines_test.json
python baselines.py --mode make_sft --n 20000         # data/sft_expert.jsonl

python evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --name zeroshot_1.5b --question_menu
python train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --out ckpt/sft_1.5b
python train_rl.py --algo ppo  --model Qwen/Qwen2.5-1.5B-Instruct --adapter ckpt/sft_1.5b --out ckpt/ppo_1.5b
python train_rl.py --algo grpo --model Qwen/Qwen2.5-1.5B-Instruct --adapter ckpt/sft_1.5b --out ckpt/grpo_1.5b
python evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter ckpt/ppo_1.5b/best --name ppo_1.5b
python plot.py

# hard setting: add the same flags to every command (baselines, SFT data, training and evaluation)
python baselines.py --mode make_sft --n 20000 --binary_questions --differential   # data/sft_expert_binary_ddx.jsonl

# everything at once on a SLURM cluster (ETH Euler), submitted from the repo root:
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-1.5B-Instruct,TAG=1.5b slurm/pipeline.sbatch
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-1.5B-Instruct,TAG=1.5b,SETTING=hard slurm/pipeline.sbatch
```

## Files

```
data.py        knowledge base (questions, ICD-10 codes) + patient loading
env.py         text environment: question matching, patient answers, diagnosis parsing, reward
bayes.py       Naive Bayes + information-gain expert
baselines.py   non-LLM baselines, expert SFT dialogues
llm.py         model/LoRA loading, chat tokenisation with action masks, batched generation
rollout.py     batched multi-turn rollouts + evaluation
train_sft.py   SFT (filtered behaviour cloning)
train_rl.py    PPO and GRPO
evaluate.py    held-out evaluation -> results/<name>.json
plot.py        accuracy-vs-questions plot, RL curves
slurm/         cluster job script
```
