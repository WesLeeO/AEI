"""PPO objective + advantage assembly — PyTorch port of LMRL-Gym ppo/base_interface.py.

Two faithful details vs. a naive implementation:
  - **KL-in-reward**: the per-token KL (logpi - logpi_ref) on the policy's own (action)
    tokens is subtracted from the reward *before* GAE (trlx / LMRL style), instead of being
    a separate full-vocab KL loss term.
  - GAE runs over the **action tokens** of each episode (answer tokens are skipped), with the
    next-state value bootstrapped from the following action token (0 at the terminal action).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import get_advantages_and_returns, whiten  # noqa: E402


def compute_advantages_returns(
    token_rewards: torch.Tensor,   # [B, L]  sparse reward (on last question token)
    old_values: torch.Tensor,      # [B, L]  V(s) per token
    old_logprobs: torch.Tensor,    # [B, L-1]
    ref_logprobs: torch.Tensor,    # [B, L-1]
    action_mask: torch.Tensor,     # [B, L]  1 on question tokens
    attention_mask: torch.Tensor,  # [B, L]
    *,
    kl_coef: float,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Returns advantages [B,L-1], returns [B,L-1] (in the 'predict token i+1' frame), mean_kl."""
    device = old_values.device
    B, L = action_mask.shape

    # shifted (i -> produces token i+1) frame, length L-1
    sta = (action_mask[:, 1:] * attention_mask[:, 1:]).bool().cpu().numpy()
    kl = (old_logprobs - ref_logprobs).cpu().numpy()                  # [B, L-1]
    reward = token_rewards[:, 1:].cpu().numpy() - kl_coef * kl        # KL folded into reward
    state_values = old_values[:, :-1].cpu().numpy()                   # V(s_i)

    advantages = np.zeros_like(reward, dtype=np.float32)
    returns = np.zeros_like(reward, dtype=np.float32)

    for b in range(B):
        idx = np.where(sta[b])[0]
        if idx.size == 0:
            continue
        sv = state_values[b, idx]
        nsv = np.concatenate([sv[1:], [0.0]]).astype(np.float32)      # next action's value, 0 at end
        rew = reward[b, idx].astype(np.float32)
        adv, ret = get_advantages_and_returns(
            sv[None], nsv[None], rew[None], gamma=gamma, lam=lam, use_whitening=False)
        advantages[b, idx] = adv[0]
        returns[b, idx] = ret[0]

    # whiten advantages over all action tokens in the batch
    flat_mask = sta.reshape(-1)
    flat_adv = advantages.reshape(-1)
    if flat_mask.sum() > 1:
        flat_adv[flat_mask] = whiten(flat_adv[flat_mask])
        advantages = flat_adv.reshape(advantages.shape)

    mean_kl = float(kl[sta].mean()) if sta.any() else 0.0
    return (torch.tensor(advantages, device=device),
            torch.tensor(returns, device=device), mean_kl)


def ppo_loss(
    new_logprobs: torch.Tensor,   # [B, L-1]
    old_logprobs: torch.Tensor,   # [B, L-1]
    advantages: torch.Tensor,     # [B, L-1]
    returns: torch.Tensor,        # [B, L-1]
    new_values: torch.Tensor,     # [B, L-1]  V(s_i)
    old_values: torch.Tensor,     # [B, L-1]  V(s_i) (old)
    mask: torch.Tensor,           # [B, L-1]  action & attention
    *,
    cliprange: float,
    cliprange_value: float,
    value_loss_coef: float,
) -> Tuple[torch.Tensor, Dict]:
    n = mask.sum().clamp(min=1.0)

    # clipped value loss
    values_clipped = old_values + torch.clamp(new_values - old_values, -cliprange_value, cliprange_value)
    vf1 = (new_values - returns) ** 2
    vf2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * (torch.max(vf1, vf2) * mask).sum() / n

    # clipped policy loss
    log_ratio = (new_logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = (torch.max(pg1, pg2) * mask).sum() / n

    loss = pg_loss + value_loss_coef * vf_loss
    approx_kl = ((ratio - 1) - log_ratio).sum().item() / n.item()
    return loss, {"policy_loss": pg_loss.item(), "value_loss": vf_loss.item(), "approx_kl": approx_kl}
