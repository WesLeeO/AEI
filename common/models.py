"""GPT-2 policy wrappers with the heads each algorithm needs.

  - GPT2WithValueHead : policy LM + scalar value head V(s)   (PPO)
  - GPT2WithQHead     : policy LM + per-token Q head Q(s, .)  (MC returns / ILQL-style)

Both wrap a HuggingFace causal LM (optionally a PEFT/LoRA adapter) and expose the base
LM's logits plus the head output, computed from the final hidden states.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import PeftModel


def load_policy(base_model_path: str, adapter_path: Optional[str] = None):
    """Load a base causal LM and (optionally) wrap it with a LoRA adapter."""
    base = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True)
    if adapter_path is not None:
        return PeftModel.from_pretrained(base, adapter_path)
    return base


class GPT2WithValueHead(nn.Module):
    """Policy + value head V(s) for PPO."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        self.policy.config.output_hidden_states = True
        self.hidden_size = self.policy.config.hidden_size
        self.value_head = nn.Linear(self.hidden_size, 1)

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.policy(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.hidden_states[-1]                 # [B, L, H]
        values = self.value_head(hidden).squeeze(-1)   # [B, L]
        return out.logits, values, hidden

    def value_parameters(self):
        return self.value_head.parameters()

    def policy_trainable_parameters(self):
        return [p for n, p in self.policy.named_parameters() if p.requires_grad]


class GPT2WithQHead(nn.Module):
    """Policy + per-token Q head Q(s, a) for Monte-Carlo returns.

    The Q head is zero-initialised so that, before training, it does not perturb the
    base policy's logits during value-weighted decoding.
    """

    def __init__(self, policy: nn.Module, tokenizer):
        super().__init__()
        self.policy = policy
        self.policy.config.output_hidden_states = True
        self.tokenizer = tokenizer
        self.hidden_size = self.policy.config.hidden_size
        self.vocab_size = len(tokenizer)
        self.q_head = nn.Linear(self.hidden_size, self.vocab_size)
        nn.init.zeros_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.policy(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        q = self.q_head(out.hidden_states[-1])  # [B, L, V]
        return out.logits, q
