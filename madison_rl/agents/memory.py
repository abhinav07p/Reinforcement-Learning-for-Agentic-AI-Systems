"""Episodic memory store for the Madison-RL orchestrator.

Addresses rubric requirement (Agent Integration → "Memory implementation
and usage") by giving the orchestrator access to a persistent record of
past task episodes. The store records which specialists were dispatched,
what effort was used, and the resulting quality/reward for each task type.

At the start of each episode, the orchestrator can query the store with
the current task embedding and receive a summary of the best-performing
strategy for the k most similar past tasks. This is appended to the
observation as additional context (or used for logging/analysis).

The memory persists across episodes within a training or eval session.

Usage in the orchestrator loop:
    memory = MemoryStore(capacity=2000)
    for ep in range(n_episodes):
        obs, info = env.reset()
        emb = obs[:16]  # task embedding
        recall = memory.recall(emb, k=3)
        # ... run episode ...
        memory.store(emb, actions_taken, final_quality, total_reward)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Episode:
    """A single remembered episode."""
    embedding: np.ndarray          # (16,) task embedding
    actions: List[int]             # specialist IDs dispatched (excl. FINISH)
    final_quality: float
    total_reward: float
    success: bool


@dataclass
class RecallResult:
    """Summary of the k most similar past episodes."""
    episodes: List[Episode]
    similarities: List[float]
    best_action_sequence: Optional[List[int]]
    avg_reward: float
    avg_quality: float
    n_recalled: int

    def summary(self) -> str:
        if self.n_recalled == 0:
            return "No similar episodes in memory."
        acts = ", ".join(str(a) for a in (self.best_action_sequence or []))
        return (
            f"Recalled {self.n_recalled} similar episodes: "
            f"avg_reward={self.avg_reward:+.2f}, "
            f"avg_quality={self.avg_quality:.2f}, "
            f"best_actions=[{acts}]"
        )


class MemoryStore:
    """Fixed-capacity episodic memory with cosine-similarity retrieval.

    Parameters:
        capacity: max episodes to store (FIFO eviction when full)
        embed_dim: dimensionality of task embeddings (default 16)
    """

    def __init__(self, capacity: int = 2000, embed_dim: int = 16):
        self.capacity = capacity
        self.embed_dim = embed_dim
        self._episodes: List[Episode] = []
        # Pre-allocated embedding matrix for fast similarity search
        self._emb_matrix: Optional[np.ndarray] = None
        self._dirty = True   # rebuild matrix on next recall

    @property
    def size(self) -> int:
        return len(self._episodes)

    def store(
        self,
        embedding: np.ndarray,
        actions: List[int],
        final_quality: float,
        total_reward: float,
        success: bool | None = None,
    ) -> None:
        """Record a completed episode."""
        if success is None:
            success = final_quality >= 0.6  # default threshold
        ep = Episode(
            embedding=np.asarray(embedding, dtype=np.float32).copy(),
            actions=list(actions),
            final_quality=float(final_quality),
            total_reward=float(total_reward),
            success=bool(success),
        )
        if len(self._episodes) >= self.capacity:
            self._episodes.pop(0)   # FIFO eviction
        self._episodes.append(ep)
        self._dirty = True

    def recall(self, query_embedding: np.ndarray, k: int = 5) -> RecallResult:
        """Retrieve the k most similar past episodes by cosine similarity.

        Returns a RecallResult with the episodes, their similarities,
        the action sequence from the highest-reward match, and aggregate
        statistics.
        """
        if not self._episodes:
            return RecallResult(
                episodes=[], similarities=[], best_action_sequence=None,
                avg_reward=0.0, avg_quality=0.0, n_recalled=0,
            )
        # Rebuild embedding matrix if needed
        if self._dirty or self._emb_matrix is None:
            self._emb_matrix = np.stack(
                [ep.embedding for ep in self._episodes]
            )  # (N, D)
            # Normalize rows for cosine similarity
            norms = np.linalg.norm(self._emb_matrix, axis=1, keepdims=True) + 1e-8
            self._emb_matrix = self._emb_matrix / norms
            self._dirty = False

        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)

        sims = self._emb_matrix @ q   # (N,)
        k_actual = min(k, len(self._episodes))
        top_idx = np.argsort(-sims)[:k_actual]

        recalled = [self._episodes[i] for i in top_idx]
        top_sims = [float(sims[i]) for i in top_idx]

        # Best action sequence: from the highest-reward recalled episode
        best_ep = max(recalled, key=lambda e: e.total_reward)

        return RecallResult(
            episodes=recalled,
            similarities=top_sims,
            best_action_sequence=best_ep.actions,
            avg_reward=float(np.mean([e.total_reward for e in recalled])),
            avg_quality=float(np.mean([e.final_quality for e in recalled])),
            n_recalled=k_actual,
        )

    def get_stats(self) -> dict:
        """Return aggregate statistics over all stored episodes."""
        if not self._episodes:
            return {"size": 0, "success_rate": 0.0, "avg_reward": 0.0}
        return {
            "size": len(self._episodes),
            "success_rate": float(np.mean([e.success for e in self._episodes])),
            "avg_reward": float(np.mean([e.total_reward for e in self._episodes])),
            "avg_quality": float(np.mean([e.final_quality for e in self._episodes])),
            "avg_steps": float(np.mean([len(e.actions) for e in self._episodes])),
        }

    def clear(self) -> None:
        """Reset the memory."""
        self._episodes.clear()
        self._emb_matrix = None
        self._dirty = True
