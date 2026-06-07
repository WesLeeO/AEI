"""Monte-Carlo returns: data pipeline + loss — PyTorch port of LMRL-Gym mc_returns.

  - MCData.from_token_trajectory_chain : per-token reward-to-go over action tokens
    (mc_returns/data.py)
  - mc_loss : Q-regression onto RTG + CQL cross-entropy (mc_returns/base_interface.py)
"""
from __future__ import annotations

import os
import sys
from typing import List, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.environment import TokenTrajectoryChain  # noqa: E402
from common.utils import get_rtg, pad_sequences  # noqa: E402


class MCData(NamedTuple):
    input_ids: np.ndarray        # [t]
    should_take_action: np.ndarray  # [t-1] bool
    returns: np.ndarray          # [t-1] float

    @classmethod
    def from_token_trajectory_chain(cls, chain: TokenTrajectoryChain, gamma: float) -> "MCData":
        # gather rewards on action tokens across the (single-link) chain, shifted x[1:]
        filtered_rewards, stas = [], []
        for tt in chain.to_list():
            sta = tt.is_action[1:]
            filtered_rewards.append(tt.reward[1:][sta])
            stas.append(sta)
        filtered_rewards = np.concatenate(filtered_rewards, axis=0)

        rtgs = get_rtg(filtered_rewards, gamma=gamma)

        should_take_action = chain.token_trajectory.is_action[1:]
        returns = np.zeros_like(should_take_action, dtype=np.float32)
        returns[should_take_action] = rtgs[: should_take_action.sum()]
        return cls(
            input_ids=chain.token_trajectory.tokens,
            should_take_action=should_take_action,
            returns=returns,
        )


class MCDataset(torch.utils.data.Dataset):
    """Pads a list of MCData into fixed-length tensors (input_ids [L], the rest [L-1])."""

    def __init__(self, mc_data: List[MCData], pad_token_id: int, max_length: int):
        self.input_ids = pad_sequences([d.input_ids for d in mc_data], pad_token_id, np.int64, max_length)
        self.should_take_action = pad_sequences(
            [d.should_take_action for d in mc_data], False, np.bool_, max_length - 1)
        self.returns = pad_sequences([d.returns for d in mc_data], 0.0, np.float32, max_length - 1)
        self.attention = (self.input_ids != pad_token_id).astype(np.int64)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.input_ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention[i], dtype=torch.long),
            "should_take_action": torch.tensor(self.should_take_action[i], dtype=torch.bool),
            "returns": torch.tensor(self.returns[i], dtype=torch.float32),
        }


def mc_loss(q_values, q_logits, input_ids, attention_mask, should_take_action, returns, cql_weight):
    """
    q_values, q_logits : [B, L, V] from the model on input_ids
    input_ids, attention_mask : [B, L]
    should_take_action, returns : [B, L-1]  (already shifted, from MCDataset)
    """
    q = q_values[:, :-1, :]           # value of state t, choosing action t+1
    logits = q_logits[:, :-1, :]
    target = input_ids[:, 1:]
    attn = attention_mask[:, 1:].float()

    mask = should_take_action.float() * attn
    n = mask.sum().clamp(min=1.0)

    q_sa = q.gather(-1, target.unsqueeze(-1)).squeeze(-1)        # [B, L-1]
    q_loss = (0.5 * (q_sa - returns.detach()) ** 2 * mask).sum() / n

    cql = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none"
    ).reshape(target.shape)
    cql_loss = (cql * mask).sum() / n

    loss = q_loss + cql_weight * cql_loss
    return loss, {"q_loss": q_loss.item(), "cql_loss": cql_loss.item()}
