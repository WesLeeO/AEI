"""Naive-Bayes diagnostic model + greedy information-gain questioner.

Used as (i) non-LLM baselines and (ii) the expert that generates SFT dialogues.
Likelihoods are estimated from training patients with Laplace smoothing:
  B evidence: Bernoulli P(present | c)
  C evidence: categorical P(value | c)
  M evidence: independent Bernoulli P(value in answer | c) per possible value
"""
import os
import pickle

import numpy as np

from data import DATA_DIR, KnowledgeBase, Patient

AGE_BINS = [0, 2, 5, 15, 30, 45, 60, 75, 200]


def entropy(p: np.ndarray, axis=-1) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis)


class NaiveBayes:
    def __init__(self, kb: KnowledgeBase, patients: list[Patient], alpha: float = 1.0):
        self.kb = kb
        C = len(kb.conditions)
        y = np.array([kb.cond_index[p.pathology] for p in patients])
        counts = np.bincount(y, minlength=C).astype(float)
        self.log_prior = np.log((counts + alpha) / (counts.sum() + alpha * C))

        def cond_mean(feat: np.ndarray) -> np.ndarray:
            """feat: (n, k) 0/1 -> P(feat | c): (k, C)."""
            sums = np.zeros((C, feat.shape[1]))
            np.add.at(sums, y, feat)
            return ((sums + alpha) / (counts[:, None] + 2 * alpha)).T

        def onehot(idx: np.ndarray, k: int) -> np.ndarray:
            out = np.zeros((len(idx), k))
            out[np.arange(len(idx)), idx] = 1
            return out

        def categorical(idx: np.ndarray, k: int) -> np.ndarray:
            """-> P(value | c): (k, C)."""
            sums = np.zeros((C, k))
            np.add.at(sums, (y, idx), 1)
            return ((sums + alpha) / (counts[:, None] + alpha * k)).T

        age_idx = np.digitize([p.age for p in patients], AGE_BINS) - 1
        self.age = categorical(age_idx, len(AGE_BINS) - 1)
        self.sex = categorical(np.array([p.sex == "F" for p in patients], dtype=int), 2)

        self.binary = [e for e in kb.evidence_names if kb.dtype(e) == "B"]
        self.b_index = {e: i for i, e in enumerate(self.binary)}
        feats = np.array([[e in p.evidences for e in self.binary] for p in patients], dtype=float)
        self.b_prob = cond_mean(feats)  # (nB, C)

        self.cat, self.multi = {}, {}
        for e in kb.evidence_names:
            vals = kb.values(e)
            vidx = {v: i for i, v in enumerate(vals)}
            if kb.dtype(e) == "C":
                idx = np.array([vidx[str(p.observation(kb, e))] for p in patients])
                self.cat[e] = (vidx, categorical(idx, len(vals)))
            elif kb.dtype(e) == "M":
                feat = np.zeros((len(patients), len(vals)))
                for i, p in enumerate(patients):
                    for v in p.observation(kb, e):
                        feat[i, vidx[v]] = 1
                self.multi[e] = (vidx, cond_mean(feat))

    @classmethod
    def load_or_fit(cls, kb: KnowledgeBase, n_train: int = 200_000, seed: int = 0) -> "NaiveBayes":
        from data import load_patients

        path = os.path.join(DATA_DIR, f"nb_{n_train}_{seed}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        nb = cls(kb, load_patients("train", n_train, seed))
        with open(path, "wb") as f:
            pickle.dump(nb, f)
        return nb

    # ---- inference -------------------------------------------------------------------------

    def log_lik(self, e: str, obs) -> np.ndarray:
        """log P(observation of evidence e | c) for every condition: (C,)."""
        t = self.kb.dtype(e)
        if t == "B":
            q = self.b_prob[self.b_index[e]]
            return np.log(q if obs else 1 - q)
        if t == "C":
            vidx, table = self.cat[e]
            return np.log(table[vidx[str(obs)]])
        vidx, prob = self.multi[e]
        present = np.zeros(len(vidx), dtype=bool)
        present[[vidx[v] for v in obs]] = True
        return np.where(present[:, None], np.log(prob), np.log(1 - prob)).sum(0)

    def initial_log_post(self, p: Patient) -> np.ndarray:
        age_bin = np.digitize(p.age, AGE_BINS) - 1
        lp = self.log_prior + np.log(self.age[age_bin]) + np.log(self.sex[int(p.sex == "F")])
        return lp + self.log_lik(p.initial_evidence, True)

    @staticmethod
    def normalize(log_post: np.ndarray) -> np.ndarray:
        p = np.exp(log_post - log_post.max())
        return p / p.sum()

    def info_gain(self, post: np.ndarray, n_samples: int = 128) -> dict[str, float]:
        """Expected entropy reduction of the posterior for every evidence question."""
        h = entropy(post)
        gains = {}
        # Binary: vectorised over all B evidences
        py = self.b_prob @ post  # (nB,)
        post_y = self.b_prob * post / np.clip(py[:, None], 1e-12, None)
        post_n = (1 - self.b_prob) * post / np.clip(1 - py[:, None], 1e-12, None)
        exp_h = py * entropy(post_y) + (1 - py) * entropy(post_n)
        gains.update(zip(self.binary, h - exp_h))
        for e, (_, table) in self.cat.items():
            pv = table @ post  # (k,)
            post_v = table * post / np.clip(pv[:, None], 1e-12, None)
            gains[e] = h - (pv * entropy(post_v)).sum()
        # Multi-choice: Monte-Carlo estimate over sampled answer sets (2^k outcomes otherwise)
        rng = np.random.default_rng(0)
        cs = rng.choice(len(post), size=n_samples, p=post)
        for e, (_, prob) in self.multi.items():
            sets = rng.random((n_samples, prob.shape[0])) < prob[:, cs].T  # (S, k)
            log_post = np.log(np.clip(post, 1e-12, None)) + sets @ np.log(prob) + (~sets) @ np.log(1 - prob)
            gains[e] = h - float(entropy(np.apply_along_axis(self.normalize, 1, log_post)).mean())
        return gains

    def predict(self, p: Patient, evidences) -> int:
        lp = self.initial_log_post(p)
        for e in evidences:
            lp = lp + self.log_lik(e, p.observation(self.kb, e))
        return int(np.argmax(lp))


class InfoGainExpert:
    """Greedy questioner: ask the max-information-gain question until max posterior >= threshold."""

    def __init__(self, nb: NaiveBayes, threshold: float = 0.9, max_questions: int = 20, epsilon: float = 0.0, seed: int = 0):
        self.nb, self.threshold, self.max_questions = nb, threshold, max_questions
        self.epsilon = epsilon  # prob. of asking a random question instead (weaker demonstrator)
        self.rng = np.random.default_rng(seed)

    def run(self, p: Patient) -> tuple[list[str], int]:
        """Returns (asked evidences, predicted condition index)."""
        lp = self.nb.initial_log_post(p)
        asked = [p.initial_evidence]
        while len(asked) - 1 < self.max_questions:
            post = self.nb.normalize(lp)
            if post.max() >= self.threshold:
                break
            if self.rng.random() < self.epsilon:
                e = self.rng.choice([e for e in self.nb.kb.evidence_names if e not in asked])
            else:
                gains = self.nb.info_gain(post)
                e = max((g, e) for e, g in gains.items() if e not in asked)[1]
            asked.append(e)
            lp = lp + self.nb.log_lik(e, p.observation(self.nb.kb, e))
        return asked[1:], int(np.argmax(lp))
