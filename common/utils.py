"""Numeric helpers shared by BC / MC / PPO — faithful PyTorch ports of LMRL-Gym.

  - get_rtg                    : discounted reward-to-go    (mc_returns/data.py:get_rtg)
  - get_advantages_and_returns : GAE + returns              (ppo/base_interface.py:253)
  - whiten                     : advantage normalisation    (ppo/base_interface.py:245)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ------------------------------------------------------------------------ padding helpers


def pad_sequences(seqs: List[np.ndarray], pad_value, dtype, max_length: int = None) -> np.ndarray:
    """Right-pad (and right-truncate) a list of 1-D arrays into a [B, L] matrix."""
    lengths = [len(s) for s in seqs]
    L = max(lengths) if max_length is None else min(max(lengths), max_length)
    out = np.full((len(seqs), L), pad_value, dtype=dtype)
    for i, s in enumerate(seqs):
        s = s[:L]
        out[i, : len(s)] = s
    return out


def token_logprobs_from_logits(logits, token_ids):
    """log p(token_{t+1} | <=t) for each position. logits [B,L,V], token_ids [B,L] -> [B,L-1]."""
    import torch.nn.functional as F  # lazy: keeps the numpy helpers importable without torch
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    return logp.gather(-1, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


# ----------------------------------------------------------------------- reward-to-go (MC)


def get_rtg(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted reward-to-go: rtg[t] = sum_{k>=t} gamma^(k-t) * rewards[k]."""
    rewards = np.asarray(rewards, dtype=np.float32)
    rtg = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + gamma * running
        rtg[t] = running
    return rtg


# ----------------------------------------------------------------------------- GAE (PPO)


def whiten(xs: np.ndarray, shift_mean: bool = True) -> np.ndarray:
    """Normalise to zero mean / unit variance (advantage whitening)."""
    mean, var = xs.mean(), xs.var()
    out = (xs - mean) / np.sqrt(var + 1e-8)
    if not shift_mean:
        out = out + mean
    return out


def get_advantages_and_returns(
    state_values: np.ndarray,       # [B, T]  V(s_t)
    next_state_values: np.ndarray,  # [B, T]  V(s_{t+1}), 0 at terminal
    action_rewards: np.ndarray,     # [B, T]  reward for taking action at t (may include KL term)
    *,
    gamma: float,
    lam: float,
    use_whitening: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation, exactly as in LMRL-Gym / the PPO paper."""
    assert state_values.shape == next_state_values.shape == action_rewards.shape
    T = state_values.shape[1]
    lastgaelam = 0.0
    advs_rev = []
    for t in reversed(range(T)):
        delta = action_rewards[:, t] + gamma * next_state_values[:, t] - state_values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advs_rev.append(lastgaelam)
    advantages = np.stack(advs_rev[::-1], axis=1)
    returns = advantages + state_values
    if use_whitening:
        advantages = whiten(advantages)
    return advantages, returns
