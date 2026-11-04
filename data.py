"""DDXPlus knowledge base + patient loading.

DDXPlus (Fansi Tchango et al., NeurIPS 2022 Datasets & Benchmarks) is a synthetic dataset of
~1.3M patients over 49 pathologies, each with an ICD-10 code. Every patient has an age, a sex,
an initial evidence (chief complaint), the full list of evidences (symptoms + antecedents)
and a ground-truth pathology. Files come from https://figshare.com/articles/dataset/20043374
"""
import ast
import json
import os
import pickle
import re
from dataclasses import dataclass, field

import pandas as pd

DATA_DIR = os.environ.get("DDX_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
SPLIT_FILES = {
    "train": "release_train_patients",
    "validate": "release_validate_patients",
    "test": "release_test_patients",
}
NO_ANSWER = {"NA": "Not applicable.", "N": "No."}


def normalize_icd(code: str) -> list[str]:
    """'j17, j18' -> ['J17', 'J18']; 'g44.009' -> ['G44.009']."""
    return [c.strip().upper() for c in re.split(r"[,/]", code) if c.strip()]


LOCATION_QUESTIONS = {  # yes/no templates for multi-choice evidences (binary_questions mode)
    "E_55": "Do you feel pain in the {}?",
    "E_57": "Does the pain radiate to the {}?",
    "E_133": "Is the affected region located on the {}?",
    "E_152": "Is there swelling in the {}?",
    "E_54": "Would you describe your pain as {}?",
}


def _side(text: str) -> str:
    """'iliac wing(R)' -> 'right iliac wing'."""
    m = re.match(r"(.*)\((R|L)\)\s*$", text)
    return f"{'right' if m.group(2) == 'R' else 'left'} {m.group(1).strip()}" if m else text


class KnowledgeBase:
    """Evidence (question) and condition metadata.

    With `binary_questions=True`, every multi-choice question (pain location, pain character, ...) is
    replaced by one yes/no question per option, e.g. "Do you feel pain in the left cheek?". These
    sub-evidences are named '<evidence>@<value>' and behave like binary evidences everywhere.
    """

    def __init__(self, data_dir: str = DATA_DIR, binary_questions: bool = False):
        self.binary_questions = binary_questions
        with open(os.path.join(data_dir, "release_evidences.json")) as f:
            self.evidences = json.load(f)
        with open(os.path.join(data_dir, "release_conditions.json")) as f:
            raw = json.load(f)
        # Patients reference pathologies by their English name.
        self.conditions = sorted(v["cond-name-eng"] for v in raw.values())
        self.cond_index = {c: i for i, c in enumerate(self.conditions)}
        by_eng = {v["cond-name-eng"]: v for v in raw.values()}
        self.icd = {c: normalize_icd(by_eng[c]["icd10-id"]) for c in self.conditions}
        self.icd_label = {c: "/".join(self.icd[c]) for c in self.conditions}
        self.severity = {c: by_eng[c]["severity"] for c in self.conditions}

        self.evidence_names = sorted(self.evidences, key=lambda e: int(e.split("_")[1]))
        self.question = {e: self.evidences[e]["question_en"] for e in self.evidence_names}
        if binary_questions:
            subs = []
            for e, vals in self._occurring_multi_values(data_dir).items():
                for v in vals:
                    sub = f"{e}@{v}"
                    self.question[sub] = LOCATION_QUESTIONS[e].format(_side(self.value_text(e, v)))
                    subs.append(sub)
                del self.question[e]
            self.evidence_names = [e for e in self.evidence_names if self.dtype(e) != "M"] + subs
        self.question_to_evidence = {q: e for e, q in self.question.items()}

    def _occurring_multi_values(self, data_dir: str) -> dict[str, list[str]]:
        """Options of each multi-choice evidence that occur in training patients (cached)."""
        path = os.path.join(data_dir, "multi_values.json")
        if not os.path.exists(path):
            occ = {e: set() for e in self.evidences if self.evidences[e]["data_type"] == "M"}
            for p in load_patients("train", 200_000, 0, data_dir):
                for e in occ:
                    occ[e] |= p.evidences.get(e, set())
            skip = {"nowhere", "NA"}
            occ = {e: sorted(v for v in vals if v != str(self.evidences[e]["default_value"])
                             and self.evidences[e]["value_meaning"][v]["en"] not in skip)
                   for e, vals in occ.items()}
            with open(path, "w") as f:
                json.dump(occ, f, indent=1)
        with open(path) as f:
            return json.load(f)

    def dtype(self, e: str) -> str:
        return "B" if "@" in e else self.evidences[e]["data_type"]

    def positive_evidences(self, p: "Patient") -> set[str]:
        """The patient's positive evidences, in this knowledge base's question space."""
        out = set()
        for e, v in p.evidences.items():
            if isinstance(v, set) and self.binary_questions:
                out |= {f"{e}@{x}" for x in v if f"{e}@{x}" in self.question}
            else:
                out.add(e)
        return out

    def default(self, e: str) -> str:
        return str(self.evidences[e]["default_value"])

    def values(self, e: str) -> list[str]:
        """Possible values for categorical (C) / multi-choice (M) evidences, as strings."""
        return [str(v) for v in self.evidences[e]["possible-values"]]

    def value_text(self, e: str, v: str) -> str:
        meaning = self.evidences[e]["value_meaning"].get(v)
        if meaning is None:  # numeric scale
            return f"{v} (on a scale from 0 to 10)"
        return meaning["en"]

    def condition_list_text(self) -> str:
        return "\n".join(f"- {c} (ICD-10: {self.icd_label[c]})" for c in self.conditions)


@dataclass
class Patient:
    age: int
    sex: str  # "M" / "F"
    pathology: str  # English condition name
    initial_evidence: str
    # B evidence -> True, C evidence -> value, M evidence -> set of values (all as strings)
    evidences: dict = field(default_factory=dict)
    differential: list = field(default_factory=list)  # [(condition, prob), ...]

    def observation(self, kb: KnowledgeBase, e: str):
        """Ground-truth answer to evidence `e` (default value if the patient doesn't have it)."""
        t = kb.dtype(e)
        if "@" in e:  # yes/no sub-question of a multi-choice evidence
            base, v = e.split("@")
            return v in self.evidences.get(base, ())
        if t == "B":
            return e in self.evidences
        if t == "C":
            return self.evidences.get(e, kb.default(e))
        return self.evidences.get(e, {kb.default(e)})

    def answer_text(self, kb: KnowledgeBase, e: str) -> str:
        obs = self.observation(kb, e)
        t = kb.dtype(e)
        if t == "B":
            return "Yes." if obs else "No."
        vals = [obs] if t == "C" else sorted(obs)
        texts = [NO_ANSWER.get(kb.value_text(e, v), kb.value_text(e, v)) for v in vals]
        if len(texts) == 1 and texts[0] in NO_ANSWER.values():
            return texts[0]
        return ", ".join(t.rstrip(".") for t in texts if t not in NO_ANSWER.values()).capitalize() + "."


def _parse_row(row) -> Patient:
    evs = {}
    for item in ast.literal_eval(row.EVIDENCES):
        if "_@_" in item:
            name, val = item.split("_@_")
            evs.setdefault(name, set()).add(val)
        else:
            evs[item] = True
    return Patient(
        age=int(row.AGE),
        sex=row.SEX,
        pathology=row.PATHOLOGY,
        initial_evidence=row.INITIAL_EVIDENCE,
        evidences=evs,
        differential=[tuple(x) for x in ast.literal_eval(row.DIFFERENTIAL_DIAGNOSIS)],
    )


def load_patients(split: str, n: int | None = None, seed: int = 0, data_dir: str = DATA_DIR) -> list[Patient]:
    """Load (a seeded random subset of) a split. Parsed subsets are cached as pickles."""
    cache = os.path.join(data_dir, f"cache_{split}_{n}_{seed}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    df = pd.read_csv(os.path.join(data_dir, SPLIT_FILES[split]))
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    kb = KnowledgeBase(data_dir)
    patients = []
    for row in df.itertuples(index=False):
        p = _parse_row(row)
        # C evidences hold a single value
        for e, v in p.evidences.items():
            if v is not True and kb.dtype(e) == "C":
                p.evidences[e] = next(iter(v))
        patients.append(p)
    with open(cache, "wb") as f:
        pickle.dump(patients, f)
    return patients
