"""LLM policy utilities: loading (+LoRA, value head), chat tokenisation with action masks,
batched generation and memory-light log-prob / value computation."""
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Minimal template for base models that ship without one (e.g. GPT-2 / tiny smoke-test models).
FALLBACK_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}{{ eos_token }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy(name: str, adapter: str | None = None, lora_r: int = 16, trainable: bool = True,
                gradient_checkpointing: bool = False):
    """Base model + LoRA. With `adapter`, loads an existing LoRA (e.g. the SFT one) and keeps
    training it; the base model with the adapter disabled serves as the frozen reference."""
    device, dtype = device_and_dtype()
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                         target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    return model


class ValueHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h):
        return self.v(h.float()).squeeze(-1)


# ---- tokenisation ------------------------------------------------------------------------

def render(tok, messages, add_generation_prompt=False) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def tokenize_dialogue(tok, messages, max_length: int = 4096):
    """Token ids of a full dialogue plus, per token, the index of the assistant turn it
    belongs to (-1 for prompt / patient tokens). Built from rendered-prefix differences so it
    works with any chat template whose renderings are prefix-consistent."""
    ids, turn = [], []
    prev = ""
    k = 0
    for j, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        before = render(tok, messages[:j], add_generation_prompt=True)
        after = render(tok, messages[: j + 1])
        assert before.startswith(prev) and after.startswith(before), "chat template is not prefix-consistent"
        seg = tok(before[len(prev):], add_special_tokens=False)["input_ids"]
        act = tok(after[len(before):], add_special_tokens=False)["input_ids"]
        ids += seg + act
        turn += [-1] * len(seg) + [k] * len(act)
        prev, k = after, k + 1
    if len(ids) > max_length:  # keep the end of the dialogue (it carries the reward)
        ids, turn = ids[-max_length:], turn[-max_length:]
    return ids, turn


def collate(batch, pad_id):
    """batch: list of (ids, turn) -> padded tensors (right padding)."""
    T = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), T), pad_id)
    attn = torch.zeros((len(batch), T), dtype=torch.long)
    turn = torch.full((len(batch), T), -1)
    for i, (ids, tr) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attn[i, : len(ids)] = 1
        turn[i, : len(tr)] = torch.tensor(tr)
    return input_ids, attn, turn


# ---- forward passes ----------------------------------------------------------------------

def _backbone_and_head(model):
    hf = model.get_base_model() if isinstance(model, PeftModel) else model
    return hf.base_model, hf.get_output_embeddings()


def action_logprobs(model, input_ids, attn, turn, return_hidden=False):
    """Log-probs of action tokens only (lm_head is applied to action positions only, which
    avoids materialising (B, T, vocab) logits). Returns flat tensors over action tokens."""
    backbone, head = _backbone_and_head(model)
    hidden = backbone(input_ids=input_ids, attention_mask=attn).last_hidden_state
    act = turn[:, 1:] >= 0  # token t+1 is an action -> use hidden state at t
    h = hidden[:, :-1][act]
    logits = head(h).float()
    targets = input_ids[:, 1:][act]
    logp = torch.gather(logits.log_softmax(-1), 1, targets[:, None]).squeeze(1)
    return (logp, hidden) if return_hidden else logp


def turn_start_states(hidden, turn):
    """Hidden state that predicts the first token of every assistant turn (the 'state' s_k).
    Returns (n_turns_total, H) ordered by (sequence, turn)."""
    first = (turn[:, 1:] >= 0) & (turn[:, 1:] != turn[:, :-1])
    return hidden[:, :-1][first]


@torch.no_grad()
def generate(model, tok, prompts: list[list[dict]], max_new_tokens=64, temperature=1.0, top_p=1.0,
             greedy=False, batch_size=32):
    """Batched generation of the next assistant message for each dialogue."""
    tok.padding_side = "left"
    device = next(model.parameters()).device
    outs = []
    for i in range(0, len(prompts), batch_size):
        texts = [render(tok, m, add_generation_prompt=True) for m in prompts[i: i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=not greedy,
            temperature=temperature if not greedy else None, top_p=top_p if not greedy else None,
            pad_token_id=tok.pad_token_id,
        )
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip().split("\n")[0])
    tok.padding_side = "right"
    return outs
