"""Multi-turn history-taking environment over DDXPlus patients.

The doctor (policy) writes one message per turn: either a free-text question or a final
"Diagnosis: <condition> (ICD-10: <code>)". Questions are mapped to the closest DDXPlus
evidence question (exact match, else sentence-embedding cosine); the simulated patient answers
deterministically from its ground-truth evidences. No LLM oracle is involved, so the
environment is cheap and noise-free.

Settings:
  binary_questions  multi-choice questions are split into yes/no questions per option
  differential      the doctor gives a ranked differential of up to 3 conditions

Reward: +1 for a correct (top-1) diagnosis, -question_cost per question. With
ddx_reward_weight=w, the diagnosis reward is (1-w)*top1 + w*recall of the DDXPlus top-3 differential.
"""
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data import KnowledgeBase, Patient

SYSTEM_PROMPT = (
    "You are a doctor taking a patient's history. Each turn, ask the patient exactly one short "
    "{question_kind} question. Use as few questions as possible (at most {max_q}). "
    "When you are confident, answer exactly in the form {diagnosis_format} using these conditions:\n{conditions}"
)
SINGLE_FORMAT = "'Diagnosis: <condition> (ICD-10: <code>)'"
DIFFERENTIAL_FORMAT = (
    "'Diagnosis: <most likely condition> (ICD-10: <code>); <second> (ICD-10: <code>); <third> (ICD-10: <code>)', "
    "ranked from most to least likely,"
)
DIFFERENTIAL_K = 3
MENU_PROMPT = "\n\nYou may ask any of these questions:\n{questions}"
LIMIT_NOTE = " [You have reached the question limit. Give your diagnosis now.]"
NOT_UNDERSTOOD = "Sorry, I don't understand the question."
DIAG_RE = re.compile(r"diagnos[ie]s\s*[:\-]", re.I)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


class QuestionMatcher:
    """Maps a free-text question to a DDXPlus evidence (or None).

    Exact (normalised) match first, else nearest question by sentence-embedding cosine
    (BAAI/bge-small-en-v1.5; falls back to word TF-IDF if sentence-transformers is missing).
    """

    def __init__(self, kb: KnowledgeBase, threshold: float = 0.75, embed_model: str = "BAAI/bge-small-en-v1.5"):
        self.kb = kb
        self.threshold = threshold
        self.exact = {_norm(q): e for q, e in kb.question_to_evidence.items()}
        self.names = kb.evidence_names
        self.cache = {}
        questions = [kb.question[e] for e in self.names]
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(embed_model, device="cpu")
            self.mat = self.model.encode(questions, normalize_embeddings=True)
        except ImportError:
            self.model = None
            self.threshold = min(threshold, 0.5)
            self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
            self.mat = self.vec.fit_transform(questions)

    def _similarities(self, q: str) -> np.ndarray:
        if self.model is not None:
            return self.mat @ self.model.encode([q], normalize_embeddings=True)[0]
        return (self.mat @ self.vec.transform([q]).T).toarray().ravel()

    def __call__(self, question: str) -> tuple[str | None, float]:
        q = re.sub(r"^\s*(question\s*\d*\s*:)\s*", "", question, flags=re.I).strip()
        if _norm(q) in self.exact:
            return self.exact[_norm(q)], 1.0
        if q not in self.cache:
            sims = self._similarities(q)
            i = int(np.argmax(sims))
            self.cache[q] = (self.names[i] if sims[i] >= self.threshold else None, float(sims[i]))
        return self.cache[q]


class DiagnosisParser:
    """Extracts the ranked list of conditions in a 'Diagnosis: ...' message, in order of
    appearance, from ICD-10 codes or condition names."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.code_to_cond = {code: c for c in kb.conditions for code in kb.icd[c]}
        # longest names first so "Unstable angina" wins over "Stable angina"
        self.names = sorted(kb.conditions, key=len, reverse=True)

    def __call__(self, text: str) -> list[str]:
        hits, taken = [], [False] * len(text)
        for m in re.finditer(r"[A-Za-z]\d{2}(?:\.\d+)?", text):
            if m.group().upper() in self.code_to_cond:
                hits.append((m.start(), self.code_to_cond[m.group().upper()]))
                taken[m.start():m.end()] = [True] * (m.end() - m.start())
        low = text.lower()
        for name in self.names:
            for m in re.finditer(re.escape(name.lower()), low):
                if not any(taken[m.start():m.end()]):
                    hits.append((m.start(), name))
                    taken[m.start():m.end()] = [True] * (m.end() - m.start())
        ranked = []
        for _, c in sorted(hits):
            if c not in ranked:
                ranked.append(c)
        return ranked


@dataclass
class EnvConfig:
    max_questions: int = 20
    question_cost: float = 0.01
    match_threshold: float = 0.75
    question_menu: bool = False  # list every askable question in the system prompt (zero-shot)
    binary_questions: bool = False  # split multi-choice questions into yes/no per option
    differential: bool = False  # doctor gives a ranked differential (up to 3 conditions)
    ddx_reward_weight: float = 0.0  # weight of differential recall in the diagnosis reward


def intro_text(kb: KnowledgeBase, p: Patient) -> str:
    sex = "man" if p.sex == "M" else "woman"
    return (
        f"Patient: {p.age}-year-old {sex}. Reason for consultation: when asked "
        f"\"{kb.question[p.initial_evidence]}\" the patient answered yes."
    )


@dataclass
class DDXEnv:
    kb: KnowledgeBase
    patient: Patient
    cfg: EnvConfig
    matcher: QuestionMatcher
    parser: DiagnosisParser
    messages: list = field(default_factory=list)
    asked: list = field(default_factory=list)  # matched evidence per question (None if unmatched)
    rewards: list = field(default_factory=list)  # one reward per doctor turn
    done: bool = False
    ranking: list = field(default_factory=list)  # predicted conditions, most likely first

    def __post_init__(self):
        system = SYSTEM_PROMPT.format(
            question_kind="yes/no" if self.cfg.binary_questions else "yes/no or short-answer",
            max_q=self.cfg.max_questions,
            diagnosis_format=DIFFERENTIAL_FORMAT if self.cfg.differential else SINGLE_FORMAT,
            conditions=self.kb.condition_list_text())
        if self.cfg.question_menu:
            system += MENU_PROMPT.format(questions="\n".join(f"- {self.kb.question[e]}" for e in self.kb.evidence_names))
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": intro_text(self.kb, self.patient)},
        ]

    @property
    def n_questions(self) -> int:
        return len(self.asked)

    @property
    def prediction(self) -> str | None:
        return self.ranking[0] if self.ranking else None

    def ddx_recall(self) -> float:
        """Fraction of the DDXPlus top-3 differential that appears in the predicted ranking."""
        gold = [c for c, _ in sorted(self.patient.differential, key=lambda x: -x[1])[:DIFFERENTIAL_K]]
        return len(set(gold) & set(self.ranking[:DIFFERENTIAL_K])) / max(len(gold), 1)

    def step(self, action: str) -> float:
        """Apply one doctor message; returns that turn's reward."""
        assert not self.done
        action = action.strip()
        self.messages.append({"role": "assistant", "content": action})
        forced = self.n_questions >= self.cfg.max_questions
        if forced or DIAG_RE.search(action):
            self.ranking = self.parser(action)[: DIFFERENTIAL_K if self.cfg.differential else 1]
            self.done = True
            w = self.cfg.ddx_reward_weight
            reward = (1 - w) * float(self.prediction == self.patient.pathology) + w * self.ddx_recall()
        else:
            evidence, _ = self.matcher(action) if action else (None, 0.0)
            self.asked.append(evidence)
            reply = self.patient.answer_text(self.kb, evidence) if evidence else NOT_UNDERSTOOD
            if self.n_questions >= self.cfg.max_questions:
                reply += LIMIT_NOTE
            self.messages.append({"role": "user", "content": reply})
            reward = -self.cfg.question_cost
        self.rewards.append(reward)
        return reward

    def summary(self) -> dict:
        positives = self.kb.positive_evidences(self.patient) - {self.patient.initial_evidence}
        asked = {e for e in self.asked if e}
        gold_ddx = {c for c, _ in self.patient.differential}
        return {
            "pathology": self.patient.pathology,
            "icd10": self.kb.icd_label[self.patient.pathology],
            "prediction": self.prediction,
            "ranking": self.ranking,
            "correct": self.prediction == self.patient.pathology,
            "top3": self.patient.pathology in self.ranking[:3],
            "in_differential": self.prediction in gold_ddx,
            "ddx_precision": len(set(self.ranking) & gold_ddx) / max(len(self.ranking), 1),
            "ddx_recall": self.ddx_recall(),
            "n_questions": self.n_questions,
            "n_unmatched": sum(e is None for e in self.asked),
            "n_repeated": len(self.asked) - len(set(self.asked)),
            "evidence_recall": len(asked & positives) / max(len(positives), 1),
            "return": float(sum(self.rewards)),
        }


class EnvFactory:
    """Shares the knowledge base / matcher / parser across many episodes."""

    def __init__(self, kb: KnowledgeBase, cfg: EnvConfig):
        self.kb, self.cfg = kb, cfg
        self.matcher = QuestionMatcher(kb, cfg.match_threshold)
        self.parser = DiagnosisParser(kb)

    def __call__(self, patient: Patient) -> DDXEnv:
        return DDXEnv(self.kb, patient, self.cfg, self.matcher, self.parser)


def add_env_args(ap):
    """Environment flags shared by every script."""
    ap.add_argument("--max_questions", type=int, default=20)
    ap.add_argument("--question_cost", type=float, default=0.01)
    ap.add_argument("--binary_questions", action="store_true", help="only yes/no questions (split multi-choice)")
    ap.add_argument("--differential", action="store_true", help="ranked differential of up to 3 conditions")
    ap.add_argument("--ddx_reward_weight", type=float, default=0.0)


def setting_tag(args) -> str:
    """File-name suffix for the environment setting, e.g. '_binary_ddx'."""
    return ("_binary" if args.binary_questions else "") + ("_ddx" if args.differential else "")


def build_env(args, question_menu: bool = False) -> EnvFactory:
    kb = KnowledgeBase(binary_questions=args.binary_questions)
    cfg = EnvConfig(max_questions=args.max_questions, question_cost=args.question_cost, question_menu=question_menu,
                    binary_questions=args.binary_questions, differential=args.differential,
                    ddx_reward_weight=args.ddx_reward_weight)
    return EnvFactory(kb, cfg)


def aggregate(summaries: list[dict]) -> dict:
    keys = ["correct", "top3", "in_differential", "ddx_precision", "ddx_recall", "n_questions", "n_unmatched",
            "n_repeated", "evidence_recall", "return"]
    out = {k: float(np.mean([s[k] for s in summaries])) for k in keys}
    out["accuracy"] = out.pop("correct")
    out["n"] = len(summaries)
    return out
