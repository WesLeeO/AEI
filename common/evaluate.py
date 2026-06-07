"""Evaluation: run episodes over a fixed set of cities and report accuracy / efficiency.

Mirrors the metrics used throughout the repo: success rate (correct guess) and the average
number of questions asked.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

from .guess_city import CITIES, GuessCityEnv
from .rollout import run_episode


def evaluate(
    env: GuessCityEnv,
    generate_question: Callable[[str], str],
    num_cities: int = 30,
    cities: Optional[List[str]] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict:
    """Evaluate a decoder over `num_cities` sampled cities."""
    cities = cities or CITIES
    rng = random.Random(seed)
    sample = rng.sample(cities, min(num_cities, len(cities)))

    correct, total_asked, trajectories = 0, 0, []
    for city in sample:
        result = run_episode(env, generate_question, city=city, verbose=verbose)
        correct += int(result["correct"])
        total_asked += result["asked"]
        trajectories.append({"target": result["target"], "questions": result["questions"],
                             "correct": result["correct"], "asked": result["asked"]})

    n = len(sample)
    return {
        "accuracy": correct / n,
        "avg_questions": total_asked / n,
        "n": n,
        "trajectories": trajectories,
    }
