"""Shared PyTorch core for the GuessMyCity task.

A faithful PyTorch port of the LMRL-Gym (JAX) GuessMyCity pipeline:
  - environment.py : Text / TokenTrajectory data structures (port of LLM_RL/environment.py)
  - guess_city.py  : conversation parsing, Phi-3 oracle, live environment, data loading
  - models.py      : GPT-2 + value head (PPO) / + Q head (MC)
  - utils.py       : padding, log-probs, reward-to-go, GAE
  - rollout.py     : run one episode (policy + oracle)
  - evaluate.py    : accuracy / avg-#questions over a set of cities
"""
