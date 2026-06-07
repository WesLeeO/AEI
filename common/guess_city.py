"""GuessMyCity task: data parsing, Phi-3 oracle, live environment.

Faithful PyTorch port of llm_rl_scripts/guess_city/env/{data,env,oracle}.py for the
*freeform* variant (descriptive answers), which is what the official RAIL dataset uses.

Conversation schema (train.json / eval.json):
    {"lines": ["Question? Answer", ...], "correct": bool, "word": "city,country"}

Reward convention (identical to the benchmark): -1 per question, 0 per answer, and 0 for
the final question if it correctly guessed the city.
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Tuple

from .environment import Text, TextEnv, TextHistory, TextTrajectory

MAX_CONVERSATION_LENGTH = 20

CITIES = [
    'Guayaquil, Ecuador', 'Taipei, China', 'Zibo, China', 'Jinan, China', 'Alexandria, Egypt',
    'Berlin, Germany', 'Sydney, Australia', 'Istanbul, Turkey', 'Osaka, Japan', 'Hong Kong, China',
    'Bogota, Colombia', 'Jakarta, Indonesia', 'Bogor, Indonesia', 'Bandung, Indonesia', 'Kolkata, India',
    'Tashkent, Uzbekistan', 'Chengdu, China', 'Giza, Egypt', 'Semarang, Indonesia', 'Lima, Peru',
    'Hyderabad, India', 'Havana, Cuba', 'Harbin, China', 'Izmir, Turkey', 'Brasilia, Brazil',
    'Shenyang, China', 'Delhi, India', 'Baghdad, Iraq', 'Rio de Janeiro, Brazil', 'London, UK',
    'Rome, Italy', 'Los Angeles, USA', 'Mexico City, Mexico', 'Bucharest, Romania', 'Ho Chi Minh City, Vietnam',
    'Daegu, South Korea', 'Toronto, Canada', 'Surabaya, Indonesia', 'Bangalore, India', 'Fortaleza, Brazil',
    'Yokohama, Japan', 'Salvador, Brazil', 'St. Petersburg, Russia', 'Beijing, China', 'Wuhan, China',
    'Karachi, Pakistan', 'Cirebon, Indonesia', 'Dhaka, Bangladesh', 'Chicago, USA', 'Mumbai, India',
    'Guangzhou, China', 'Santiago, Chile', 'Budapest, Hungary', 'Tehran, Iran', 'Houston, USA',
    'Casablanca, Morocco', 'Kinshasa, Congo', 'Malang, Indonesia', 'Qingdao, China', 'Xi’an, China',
    'Caracas, Venezuela', 'Abidjan, Côte d’Ivoire', 'Medellin, Colombia', 'Tokyo, Japan', 'Chennai, India',
    'Kanpur, India', 'Bangkok, Thailand', 'Addis Ababa, Ethiopia', 'Busan, South Korea', 'Dalian, China',
    'Tianjin, China', 'Mashhad, Iran', 'Yangon, Myanmar', 'Sukabumi, Indonesia', 'Moscow, Russia',
    'Incheon, South Korea', 'Buenos Aires, Argentina', 'Cali, Colombia', 'New York, USA', 'Lahore, Pakistan',
    'Ahmedabad, India', 'Chongqing, China', 'Changchun, China', 'Nanjing, China', 'Madrid, Spain',
    'Taiyuan, China', 'Shanghai, China', 'Cairo, Egypt', 'Medan, Indonesia', 'Belo Horizonte, Brazil',
    'Paris, France', 'Nagoya, Japan', 'São Paulo, Brazil', 'Singapore, Singapore', 'Kiev, Ukraine',
    'Pyongyang, North Korea', 'Faisalabad, Pakistan', 'Ankara, Turkey', 'Quezon City, Philippines',
]


# ----------------------------------------------------------------- text rendering helpers
# A conversation is rendered as:  "Q:{question} A:{answer}\n" ... and the prompt for the
# next question is the rendered history followed by "Q:". The question text is the action;
# the "Q:", " A:{answer}\n" scaffolding are context (non-actions).


def render_prompt(history: List[Tuple[str, str]]) -> str:
    """Render (question, answer) pairs into the policy prompt, ending in 'Q:'."""
    s = "".join(f"Q:{q} A:{a}\n" for q, a in history)
    return s + "Q:"


def city_name(word: str) -> str:
    """'hyderabad,india' / 'Hyderabad, India' -> 'hyderabad' (the part before the comma)."""
    return word.split(",")[0].strip().lower()


def is_correct_guess(word: str, question: str) -> bool:
    """Freeform correctness: the target city name appears in the question."""
    return city_name(word) in question.lower()


def split_question_answer(line: str) -> Tuple[str, str]:
    """Split a data line 'Question? Answer' into (question_with_?, answer)."""
    if "? " in line:
        *q, a = line.split("? ")
        return "? ".join(q) + "?", a.strip()
    # fallback: split on first '?'
    q, _, a = line.partition("?")
    return q.strip() + "?", a.strip()


def conversation_to_text_trajectory(
    convo: Dict, max_conversation_len: int = MAX_CONVERSATION_LENGTH
) -> TextTrajectory:
    """Build a TextTrajectory (with benchmark rewards) from a raw conversation dict."""
    texts: List[Text] = []
    rewards: List[float] = []
    for line in convo["lines"]:
        question, answer = split_question_answer(line)
        texts.extend([Text("Q:", False), Text(question, True), Text(f" A:{answer}\n", False)])
        rewards.extend([0.0, -1.0, 0.0])

    if convo.get("correct", False) and any(t.is_action for t in texts):
        last_action = max(i for i, t in enumerate(texts) if t.is_action)
        rewards[last_action] = 0.0  # correct final guess is not penalised

    done = bool(convo.get("correct", False)) or len(convo["lines"]) >= max_conversation_len
    return TextTrajectory(tuple(texts), tuple(rewards), done)


def load_conversations(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- the oracle


class Phi3Oracle:
    """Freeform question-answering oracle (analog of the GPT-3 data-generation oracle)."""

    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading oracle model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _prompt(self, question: str, city: str) -> str:
        return (
            f" Answer the following question about {city}.\n"
            f"    Do not reveal the name of {city} in your answer.\n"
            f"    Do not generate more than 50 words.\n"
            f"    Q:{question}\n    A:"
        )

    def generate_answer(self, city: str, question: str) -> str:
        import torch
        inputs = self.tokenizer(self._prompt(question, city), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=100, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        for cue in ("\n", "Question", "Answer"):
            idx = answer.find(cue)
            if idx != -1:
                answer = answer[:idx].strip()
                break
        return answer


# ------------------------------------------------------------------------- live environment


class GuessCityEnv(TextEnv):
    """Live environment: a target city + the oracle. Reward = -1/question, 0 on a correct
    guess; episode ends on a correct guess or after `max_questions` (benchmark semantics).
    """

    def __init__(self, oracle: Phi3Oracle, cities: List[str] = CITIES,
                 max_questions: int = MAX_CONVERSATION_LENGTH):
        self.oracle = oracle
        self.cities = cities
        self.max_questions = max_questions
        self.target: Optional[str] = None
        self.history: List[Tuple[str, str]] = []
        self.asked = 0

    def reset(self, seed: Optional[int] = None, city: Optional[str] = None) -> TextHistory:
        rng = random.Random(seed)
        self.target = city if city is not None else rng.choice(self.cities)
        self.history = []
        self.asked = 0
        return (Text("Q:", False),)

    def step_question(self, question: str) -> Dict:
        """String-level convenience step used by the rollout loop."""
        assert self.target is not None, "call reset() first"
        question = question.strip()
        self.asked += 1
        answer = self.oracle.generate_answer(self.target, question)
        correct = is_correct_guess(self.target, question)
        done = correct or self.asked >= self.max_questions
        reward = 0.0 if correct else -1.0
        self.history.append((question, answer))
        return {"question": question, "answer": answer, "reward": reward,
                "done": done, "correct": correct, "asked": self.asked}

    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        assert text_history[-1].is_action, "last segment must be the asker's question"
        info = self.step_question(text_history[-1].text)
        new_history = text_history + (Text(f" A:{info['answer']}\n", is_action=False),)
        return new_history, info["reward"], info["done"]
