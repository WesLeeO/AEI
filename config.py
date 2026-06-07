"""Hyperparameters for BC / MC-returns / PPO.

Defaults match the LMRL-Gym guess_city configs:
  - BC  : llm_rl_scripts/guess_city/bc/train_bc.py
  - MC  : llm_rl_scripts/guess_city/mc/train_mc_returns.py   (beta=16, gamma=0.99, cql=0.01)
  - PPO : llm_rl_scripts/guess_city/ppo/train_ppo.py         (gamma=1.0, lam=0.95, kl=0.001)
"""
from dataclasses import dataclass


# Shared model/data paths
BASE_MODEL_PATH = "./gpt2-medium-offline"
SFT_ADAPTER_PATH = "bc/fine_tuned_gpt2_medium_lora_filtered"
ORACLE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
TRAIN_JSON = "train.json"
EVAL_JSON = "eval.json"
MAX_LENGTH = 1024
MAX_QUESTIONS = 20


@dataclass
class BCConfig:
    base_model_path: str = BASE_MODEL_PATH
    train_json: str = TRAIN_JSON
    output_dir: str = "bc/bc_model"
    max_length: int = MAX_LENGTH
    epochs: int = 1
    lr: float = 1e-4
    batch_size: int = 4
    warmup_frac: float = 0.05
    non_action_weight: float = 0.0   # 0 = train only on the asker's questions
    data_filter: str = "correct"      # "all" | "correct" | "top_p"
    take_top_p: float = 100.0         # used when data_filter == "top_p"
    use_lora: bool = True


@dataclass
class MCConfig:
    base_model_path: str = BASE_MODEL_PATH
    sft_adapter_path: str = SFT_ADAPTER_PATH
    train_json: str = TRAIN_JSON
    q_function_out: str = "mc/q_function.pth"
    mc_adapter_out: str = "mc/mc_model"   # the LoRA policy (theta) fine-tuned during MC
    max_length: int = MAX_LENGTH
    epochs: int = 1
    lr: float = 1e-5
    batch_size: int = 4
    accum_steps: int = 8
    gamma: float = 0.99
    cql_weight: float = 0.01
    beta: float = 16.0               # value-weighted decoding strength (additive in logit space)
    data_filter: str = "correct"      # BC-style filtering of training conversations


@dataclass
class PPOConfig:
    base_model_path: str = BASE_MODEL_PATH
    sft_adapter_path: str = SFT_ADAPTER_PATH
    output_dir: str = "ppo/ppo_model"
    oracle_model: str = ORACLE_MODEL
    max_length: int = MAX_LENGTH
    max_questions: int = MAX_QUESTIONS
    # optimisation
    policy_lr: float = 1e-5
    value_lr: float = 1e-5
    ppo_epochs: int = 4
    episodes_per_iter: int = 8
    max_grad_norm: float = 1.0
    # PPO / GAE
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    value_loss_coef: float = 1.0
    init_kl_coef: float = 0.001
    # generation
    max_new_tokens: int = 30
    temperature: float = 0.7
    top_p: float = 0.95
    # logging
    log_every: int = 1
    save_every: int = 50
