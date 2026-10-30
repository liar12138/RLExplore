"""Simple controls only. These are NOT full NBV or ActPerMoMa reproductions."""
import numpy as np
from .env import VIEWS


def choose_action(name, observation, step):
    candidates = observation["candidates"]
    valid = observation["action_mask"][len(VIEWS):-1]
    if name == "scan" and step < len(VIEWS):
        return step
    if name not in {"scan", "geometry", "reactive"}:
        raise ValueError(f"unknown baseline: {name}")
    if valid.any():
        # Observed support fraction - known obstacles - uncertainty.
        scores = candidates[:, 4] - 2*candidates[:, 6] - .5*candidates[:, 7]
        scores[~valid] = -np.inf
        choice = int(scores.argmax())
        if name == "reactive" or step >= len(VIEWS) or scores[choice] > .7:
            return len(VIEWS)+choice
    return step % len(VIEWS)
