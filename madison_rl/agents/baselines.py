"""Baseline orchestrator policies for comparison against the learned PPO agent.

All baselines expose the same interface as SB3 policies:
    action, _state = policy.predict(obs, deterministic=True)

This lets eval code treat them uniformly.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from madison_rl.env import (
    EMBED_DIM,
    FINISH_ACTION,
    NUM_ACTIONS,
    NUM_CAPABILITIES,
    NUM_SPECIALISTS,
)


# Observation layout (must match intelligence_env._get_obs):
#   [0:16)  task_embedding
#   [16]    partial_quality
#   [17]    normalized_step
#   [18]    normalized_cost_used
#   [19:23) specialists_used_mask
#   [23]    last_confidence
#   [24]    last_quality_gain
_EMB = slice(0, EMBED_DIM)
_QUAL = EMBED_DIM
_USED = slice(EMBED_DIM + 3, EMBED_DIM + 3 + NUM_SPECIALISTS)


class RandomPolicy:
    """Picks a uniformly random action each step."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = True) -> Tuple[int, None]:
        return int(self.rng.integers(0, NUM_ACTIONS)), None


class RoundRobinPolicy:
    """Dispatches specialists in fixed order 0,1,2,3 then FINISH."""

    def __init__(self, seed: int = 0):
        self._counter = 0

    def predict(self, obs, deterministic: bool = True) -> Tuple[int, None]:
        obs = np.asarray(obs)
        used = obs[_USED]
        # Pick the smallest-index unused specialist; if all used, FINISH
        for i in range(NUM_SPECIALISTS):
            if used[i] < 0.5:
                return i, None
        return FINISH_ACTION, None


class GreedyCapabilityPolicy:
    """Oracle-lite: uses the task embedding plus knowledge of the capability
    prototypes to route specialists. Uses deep effort (via a sidechannel —
    see ``wrap_env_for_oracle`` below) when dispatched.

    Scoring: cosine similarity between the (normalized) task embedding and
    each capability prototype. Dispatches every capability above threshold,
    re-dispatching the top one if total dispatches < 3 and the top score is
    very high (indicating a high-weight required capability).

    This is the best purely analytical baseline we could construct.
    """

    def __init__(self, prototypes: np.ndarray, threshold: float = 0.20, seed: int = 0):
        self.prototypes = prototypes
        self.threshold = threshold
        self._dispatched_counts = np.zeros(NUM_SPECIALISTS, dtype=int)

    def _reset_counts(self):
        self._dispatched_counts[:] = 0

    def predict(self, obs, deterministic: bool = True) -> Tuple[int, None]:
        obs = np.asarray(obs, dtype=np.float32)
        emb = obs[_EMB]
        used = obs[_USED]
        qual = float(obs[_QUAL])

        # If this is a fresh episode (no specialists used yet), reset counts
        if used.sum() < 0.5:
            self._reset_counts()

        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        scores = self.prototypes @ emb_norm  # (NUM_CAPABILITIES,)
        order = np.argsort(-scores)

        # If quality already high enough, finish
        if qual >= 0.75:
            return FINISH_ACTION, None

        # Prefer unused, above-threshold capabilities
        for cap_idx in order:
            cap_idx = int(cap_idx)
            if scores[cap_idx] < self.threshold:
                break
            if used[cap_idx] < 0.5:
                self._dispatched_counts[cap_idx] += 1
                return cap_idx, None

        # All needed caps used once — if the top score is high, double down
        top = int(order[0])
        if scores[top] > 0.35 and self._dispatched_counts[top] < 2:
            self._dispatched_counts[top] += 1
            return top, None

        return FINISH_ACTION, None
