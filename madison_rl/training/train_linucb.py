"""LinUCB contextual bandit policy.

A Category-4 (Exploration Strategies) implementation using the LinUCB
algorithm (Li et al. 2010). Unlike DQN/PPO which learn a full MDP policy,
LinUCB treats each orchestrator step as a contextual bandit problem:
    - Context: the current observation (task embedding + state features)
    - Arms: the 5 discrete actions (4 specialists + FINISH)
    - Reward: the per-step shaped reward received after taking the action

For each arm a, LinUCB maintains a ridge-regression estimate of
    θ_a = argmin_θ Σ_t (r_t - x_t^T θ)^2 + λ||θ||^2
via incremental updates to A_a = λI + Σ x_t x_t^T and b_a = Σ r_t x_t.
The UCB action is
    a* = argmax_a  x^T θ_a  +  α * sqrt(x^T A_a^{-1} x)
where α controls the exploration bonus.

This is fully online (no replay buffer) and trains in a single pass
through the environment. It ignores temporal structure (does not
bootstrap future returns), which is a known limitation — but that's
also the point of including it: it gives us a clean baseline showing
how much PPO's MDP reasoning buys over pure bandit reasoning.

Usage:
    python -m madison_rl.training.train_linucb --seed 0 --episodes 3000
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from madison_rl.env import IntelligenceTaskEnv, NUM_ACTIONS, OBS_DIM


RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class LinUCB:
    """Contextual bandit with one linear model per arm."""

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0,
                 ridge: float = 1.0):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        # A_a = ridge * I  (d x d)  for each arm
        self.A = np.stack([ridge * np.eye(context_dim) for _ in range(n_arms)])
        # A_inv cached for fast inference
        self.A_inv = np.stack([np.eye(context_dim) / ridge for _ in range(n_arms)])
        # b_a = zero vector for each arm
        self.b = np.zeros((n_arms, context_dim))

    def _theta(self, a: int) -> np.ndarray:
        return self.A_inv[a] @ self.b[a]

    def score(self, context: np.ndarray) -> np.ndarray:
        """UCB score for every arm."""
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta = self._theta(a)
            mean = float(context @ theta)
            explore = self.alpha * float(np.sqrt(context @ self.A_inv[a] @ context))
            scores[a] = mean + explore
        return scores

    def predict(self, obs, deterministic: bool = True) -> Tuple[int, None]:
        """SB3-compatible interface for eval."""
        scores = self.score(np.asarray(obs, dtype=np.float64))
        return int(np.argmax(scores)), None

    def update(self, context: np.ndarray, arm: int, reward: float):
        """Incremental update via Sherman-Morrison for A_inv."""
        x = context.reshape(-1, 1)
        # A <- A + x x^T
        self.A[arm] += x @ x.T
        # Sherman-Morrison: A_inv_new = A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)
        A_inv = self.A_inv[arm]
        Ax = A_inv @ x                          # (d, 1)
        denom = 1.0 + float((x.T @ Ax).item())  # scalar
        self.A_inv[arm] = A_inv - (Ax @ Ax.T) / denom
        # b <- b + r x
        self.b[arm] += reward * context


def train(seed: int = 0, episodes: int = 3_000, alpha: float = 0.5,
          verbose: int = 0) -> dict:
    print(f"Training LinUCB  seed={seed}  episodes={episodes}  alpha={alpha}")
    env = IntelligenceTaskEnv(seed=seed)
    policy = LinUCB(n_arms=NUM_ACTIONS, context_dim=OBS_DIM, alpha=alpha, ridge=1.0)

    rng = np.random.default_rng(seed)
    curve_episodes: List[int] = []
    curve_reward: List[float] = []
    curve_success: List[float] = []
    window_r: List[float] = []
    window_s: List[float] = []

    for ep in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 10_000_000)))
        done = False
        ep_reward = 0.0
        while not done:
            a, _ = policy.predict(obs)
            new_obs, r, term, trunc, info = env.step(a)
            policy.update(obs.astype(np.float64), a, float(r))
            obs = new_obs
            ep_reward += r
            done = term or trunc
        window_r.append(ep_reward)
        window_s.append(
            1.0 if info["partial_quality"] >= info["threshold"] else 0.0
        )
        if len(window_r) > 100:
            window_r = window_r[-100:]
            window_s = window_s[-100:]

        if (ep + 1) % 50 == 0:
            curve_episodes.append(ep + 1)
            curve_reward.append(float(np.mean(window_r)))
            curve_success.append(float(np.mean(window_s)))
            if verbose and (ep + 1) % 200 == 0:
                print(
                    f"  [ep {ep+1:>5}] mean_reward={curve_reward[-1]:+.3f} "
                    f"success={curve_success[-1]:.2%}"
                )

    # Save policy
    out = RESULTS_DIR / f"linucb_seed{seed}.npz"
    np.savez(
        out,
        A=policy.A,
        A_inv=policy.A_inv,
        b=policy.b,
        alpha=policy.alpha,
        context_dim=policy.context_dim,
        n_arms=policy.n_arms,
        curve_episodes=np.array(curve_episodes),
        curve_reward=np.array(curve_reward),
        curve_success=np.array(curve_success),
    )
    print(f"  saved: {out.name}")
    if curve_reward:
        print(
            f"  final mean_reward={curve_reward[-1]:+.3f} "
            f"success={curve_success[-1]:.2%}"
        )
    return {
        "timesteps": curve_episodes,
        "mean_reward": curve_reward,
        "mean_success": curve_success,
    }


def load(seed: int) -> LinUCB:
    path = RESULTS_DIR / f"linucb_seed{seed}.npz"
    data = np.load(path)
    policy = LinUCB(
        n_arms=int(data["n_arms"]),
        context_dim=int(data["context_dim"]),
        alpha=float(data["alpha"]),
    )
    policy.A = data["A"]
    policy.A_inv = data["A_inv"]
    policy.b = data["b"]
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3_000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    train(seed=args.seed, episodes=args.episodes, alpha=args.alpha,
          verbose=args.verbose)
