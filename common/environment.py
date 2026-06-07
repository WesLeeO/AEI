"""Trajectory data structures — PyTorch/NumPy port of LMRL-Gym's LLM_RL/environment.py.

A conversation is represented as a `TextHistory`: an ordered tuple of `Text` segments,
each flagged `is_action` (True for the asker's questions, False for the oracle's answers
and scaffolding). Rewards live on `TextTrajectory` (one float per Text segment). These are
tokenized into `TokenTrajectory` objects where the per-segment reward is placed on the
*last token* of the segment — exactly as the benchmark does.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------- text level


@dataclass(frozen=True)
class Text:
    text: str
    is_action: bool


TextHistory = Tuple[Text, ...]


def text_history_to_str(text_history: TextHistory) -> str:
    return "".join(t.text for t in text_history)


@dataclass(frozen=True)
class TextTrajectory:
    text_history: TextHistory
    reward: Tuple[float, ...]
    done: bool

    def __post_init__(self):
        assert len(self.reward) == len(self.text_history), "need one reward per Text segment"
        assert all(
            r == 0.0 for r, t in zip(self.reward, self.text_history) if not t.is_action
        ), "reward on non-action (answer) segments must be 0.0"


@dataclass(frozen=True)
class TextTrajectoryChain:
    text_trajectory: TextTrajectory
    next: Optional["TextTrajectoryChain"] = None


# ---------------------------------------------------------------------------- token level


@dataclass(frozen=True)
class TokenHistory:
    tokens: np.ndarray   # [t] int32
    is_action: np.ndarray  # [t] bool

    def __post_init__(self):
        assert self.tokens.ndim == 1 and self.is_action.ndim == 1
        assert self.tokens.shape == self.is_action.shape

    @classmethod
    def from_text_history(
        cls,
        text_history: TextHistory,
        tokenizer,
        token_process: Optional[Callable[[List[int]], List[int]]] = None,
    ) -> "TokenHistory":
        token_process = token_process or (lambda x: x)
        tokens, is_action = [], []
        for item in text_history:
            new_tokens = token_process(tokenizer.encode(item.text))
            tokens.extend(new_tokens)
            is_action.extend([item.is_action] * len(new_tokens))
        return cls(np.array(tokens, dtype=np.int32), np.array(is_action, dtype=np.bool_))


@dataclass(frozen=True)
class TokenTrajectory:
    tokens: np.ndarray     # [t] int32
    is_action: np.ndarray  # [t] bool
    reward: np.ndarray     # [t] float32 — segment reward on the segment's last token
    done: np.ndarray       # scalar bool

    def __post_init__(self):
        assert self.tokens.ndim == 1 and self.is_action.ndim == 1 and self.reward.ndim == 1
        assert self.done.ndim == 0
        assert self.is_action.shape == self.tokens.shape == self.reward.shape
        assert not np.any((~self.is_action) & (self.reward != 0.0)), "reward must be 0 on non-actions"

    @classmethod
    def from_text_trajectory(
        cls,
        text_trajectory: TextTrajectory,
        tokenizer,
        token_process: Optional[Callable[[List[int]], List[int]]] = None,
    ) -> "TokenTrajectory":
        token_process = token_process or (lambda x: x)
        tokens, is_action, reward = [], [], []
        for i, item in enumerate(text_trajectory.text_history):
            new_tokens = token_process(tokenizer.encode(item.text))
            tokens.extend(new_tokens)
            is_action.extend([item.is_action] * len(new_tokens))
            # segment reward goes on the segment's last token
            reward.extend([0.0] * (len(new_tokens) - 1) + [text_trajectory.reward[i]])
        return cls(
            np.array(tokens, dtype=np.int32),
            np.array(is_action, dtype=np.bool_),
            np.array(reward, dtype=np.float32),
            np.array(text_trajectory.done, dtype=np.bool_),
        )


@dataclass(frozen=True)
class TokenTrajectoryChain:
    token_trajectory: TokenTrajectory
    next: Optional["TokenTrajectoryChain"] = None

    def to_list(self) -> List[TokenTrajectory]:
        curr, out = self, []
        while curr is not None:
            out.append(curr.token_trajectory)
            curr = curr.next
        return out

    @classmethod
    def from_text_trajectory_chain(
        cls,
        text_trajectory_chain: TextTrajectoryChain,
        tokenizer,
        token_process: Optional[Callable[[List[int]], List[int]]] = None,
    ) -> "TokenTrajectoryChain":
        nxt = (
            cls.from_text_trajectory_chain(text_trajectory_chain.next, tokenizer, token_process)
            if text_trajectory_chain.next is not None
            else None
        )
        return cls(
            TokenTrajectory.from_text_trajectory(
                text_trajectory_chain.text_trajectory, tokenizer, token_process
            ),
            nxt,
        )


# ------------------------------------------------------------------------------ env ABC


class TextEnv(ABC):
    """Minimal text-in / text-out environment (port of LLM_RL.environment.TextEnv)."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> TextHistory:
        ...

    @abstractmethod
    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        ...
