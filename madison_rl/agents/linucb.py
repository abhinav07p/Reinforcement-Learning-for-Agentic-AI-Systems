"""LinUCB — Linear Upper Confidence Bound contextual bandit.

This is the Category 4 (Exploration Strategies) implementation from the
rubric. It treats each specialist dispatch as an "arm" and learns a linear
reward model per arm, conditioned on the task embedding (the "context").

Why a bandit is an interesting baseline here:
    Unlike PPO, LinUCB has no concept of sequential state transitions —
    it picks one specialist and is done. Running it inside our episodic
    env means we use it as a *single-shot* dispatch policy: pick one
    specialist, execute it once, then FINISH. This is deliberately
    hamstrung compared to PPO (which can multi-dispatch) and is exactly
    the comparison the report needs: "bandit exploration vs full RL."

LinUCB algorithm (Li et al. 2010, disjoint linear model):

    For each arm a:
        A_a ← I_d  (d×d identity)
        b_a ← 0    (d-dim zeros)

    At each round t, given context x_t:
        For each arm a:
            θ_a  ← A_a⁻¹ b_a
            p_a  ← θ_aᵀ x_t + α · sqrt(x_tᵀ A_a⁻¹ x_t)
        a_t = argmax_a p_a
        observe reward r_t
        A_{a_t} ← A_{a_t} + x_t x_tᵀ
        b_{a_t} ← b_{a_t} + r_t x_t

The bonus term α · sqrt(xᵀ A⁻¹ x) is the UCB exploration bonus — it
shrinks as we see more data for that arm in contexts similar to x.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from madison_rl.env import (
    EMBED_DIM,
    FINISH_ACTION,
    IntelligenceTaskEnv,
    NUM_SPECIALISTS,
)


_EMB = slice(0, EMBED_DIM)


class LinUCBPolicy:
    """LinUCB over the 4 specialists; context = task embedding only.

    We allow LinUCB to dispatch up to `max_dispatches` specialists per
    episode, finishing immediately after. Each dispatch is an independent
    bandit round — the bandit does not model sequential state. This is
    the "contextual bandit with k independent rounds" framing and gives
    LinUCB a fair shot on tasks that need multi-dispatch.
    """

    def __init__(self, alpha: float = 1.0, max_dispatches: int = 3, seed: int = 0):
        self.alpha = alpha
        self.max_dispatches = max_dispatches
        self.d = EMBED_DIM
        self.n_arms = NUM_SPECIALISTS
        self.A = [np.eye(self.d) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.d) for _ in range(self.n_arms)]
        self.rng = np.random.default_rng(seed)
        self._dispatch_count = 0

    # --------------------------------------------- SB3-like predict API
    def predict(self, obs, deterministic: bool = True) -> Tuple[int, None]:
        obs = np.asarray(obs, dtype=np.float32)
        # Detect fresh episode: no specialists used yet
        used = obs[EMBED_DIM + 3 : EMBED_DIM + 3 + NUM_SPECIALISTS]
        if used.sum() < 0.5:
            self._dispatch_count = 0

        if self._dispatch_count >= self.max_dispatches:
            return FINISH_ACTION, None

        x = obs[_EMB].astype(np.float64)
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
        action = int(np.argmax(p))
        self._dispatch_count += 1
        return action, None

    # --------------------------------------------- training update
    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        x = np.asarray(context, dtype=np.float64)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    # --------------------------------------------- serialization
    def save(self, path: str) -> None:
        np.savez(
            path,
            alpha=self.alpha,
            A=np.stack(self.A),
            b=np.stack(self.b),
        )

    @classmethod
    def load(cls, path: str) -> "LinUCBPolicy":
        data = np.load(path)
        obj = cls(alpha=float(data["alpha"]))
        obj.A = [data["A"][i] for i in range(NUM_SPECIALISTS)]
        obj.b = [data["b"][i] for i in range(NUM_SPECIALISTS)]
        return obj


def train_linucb(seed: int = 0, n_rounds: int = 5000,
                 alpha: float = 1.0, max_dispatches: int = 3,
                 verbose: int = 0) -> dict:
    """Train a LinUCB policy on multi-dispatch episodes.

    Each round is a full episode. LinUCB picks up to ``max_dispatches``
    specialists, then the episode ends. Each specialist dispatch is
    credited with the final episode reward divided equally across its
    dispatches (simple equal-credit assignment — bandits can't do GAE).
    """
    from pathlib import Path
    RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training LinUCB  seed={seed}  rounds={n_rounds}  alpha={alpha}  "
          f"max_dispatches={max_dispatches}")
    env = IntelligenceTaskEnv(seed=seed + 50)
    policy = LinUCBPolicy(alpha=alpha, max_dispatches=max_dispatches, seed=seed)
    rng = np.random.default_rng(seed + 50)

    curve_rounds, curve_reward, curve_success = [], [], []
    window, window_s = [], []

    for t in range(n_rounds):
        obs, info = env.reset(seed=int(rng.integers(0, 10_000_000)))
        context = obs[_EMB].copy()
        done = False
        dispatch_log = []   # (arm,) list
        ep_reward = 0.0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            action = int(action)
            if action < NUM_SPECIALISTS:
                dispatch_log.append(action)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            done = term or trunc

        # Credit-assignment: share ep_reward equally among dispatches
        if dispatch_log:
            per_arm_reward = ep_reward / len(dispatch_log)
            for a in dispatch_log:
                policy.update(context, a, per_arm_reward)

        window.append(ep_reward)
        success = 1.0 if info.get("partial_quality", 0) >= info.get("threshold", 1) else 0.0
        window_s.append(success)
        if len(window) > 200:
            window = window[-200:]
            window_s = window_s[-200:]
        if (t + 1) % 200 == 0:
            curve_rounds.append(t + 1)
            curve_reward.append(float(np.mean(window)))
            curve_success.append(float(np.mean(window_s)))
            if verbose:
                print(
                    f"  [round {t+1:>5}] mean_reward={curve_reward[-1]:+.3f} "
                    f"success={curve_success[-1]:.2%}"
                )

    out_path = RESULTS_DIR / f"linucb_seed{seed}.npz"
    policy.save(str(out_path))
    curve_path = RESULTS_DIR / f"linucb_seed{seed}_curve.npz"
    np.savez(curve_path,
             rounds=np.array(curve_rounds),
             reward=np.array(curve_reward),
             success=np.array(curve_success))
    print(f"  saved: {out_path.name}")
    if curve_reward:
        print(f"  final: reward={curve_reward[-1]:+.3f} success={curve_success[-1]:.2%}")
    return {
        "rounds": curve_rounds,
        "reward": curve_reward,
        "success": curve_success,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-dispatches", type=int, default=3)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    train_linucb(seed=args.seed, n_rounds=args.rounds, alpha=args.alpha,
                 max_dispatches=args.max_dispatches, verbose=args.verbose)
