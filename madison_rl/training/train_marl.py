"""Multi-Agent RL: Independent PPO (IPPO) for specialist effort policies.

Setup:
    - The orchestrator is a frozen pre-trained PPO policy.
    - Each of the 4 specialists has its own tiny PPO policy that selects
      EFFORT in {0=shallow, 1=medium, 2=deep} given a local observation
      (task embedding + partial quality + remaining budget).
    - All specialists share the *team* reward (the episode return), which
      is the core MARL reward-sharing mechanism.

We implement IPPO from scratch — a lightweight REINFORCE-with-baseline is
fine here because the specialist action space is Discrete(3) and episodes
are short. Using SB3 for each specialist would complicate the coupling.

Because the specialists act *within* the env's step() call (via the
``specialist_effort`` kwarg), we drive the environment with a custom rollout
loop rather than through SB3.

Output:
    experiments/results/marl_specialists_seed{N}.npz  (policy weights + curve)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from stable_baselines3 import PPO

from madison_rl.env import (
    EFFORT_LEVELS,
    EMBED_DIM,
    FINISH_ACTION,
    IntelligenceTaskEnv,
    NUM_SPECIALISTS,
)


RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Specialist local observation: [embedding (16) | partial_quality | remaining_budget]
LOCAL_OBS_DIM = EMBED_DIM + 2


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class LinearPolicy:
    """Tiny linear softmax policy over EFFORT_LEVELS. Trained with
    REINFORCE-with-baseline (advantage = return - running mean)."""

    def __init__(self, obs_dim: int, n_actions: int, lr: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = 0.001 * rng.standard_normal((obs_dim, n_actions))
        # Warm-start bias: strongly prefer medium effort (action 1). This
        # matches the orchestrator's training-time default, so MARL starts
        # at orchestrator-alone performance and learns deviations.
        self.b = np.array([0.0, 2.0, 0.0], dtype=np.float64)[:n_actions]
        self.lr = lr
        self.baseline = 0.0
        self.baseline_alpha = 0.05

    def logits(self, obs: np.ndarray) -> np.ndarray:
        return obs @ self.W + self.b

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> Tuple[int, float]:
        p = softmax(self.logits(obs))
        a = int(rng.choice(len(p), p=p))
        return a, float(p[a])

    def greedy(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.logits(obs)))

    def update(self, obs_list, act_list, returns_list):
        """REINFORCE-with-baseline update, averaged over the batch."""
        if not obs_list:
            return
        obs = np.asarray(obs_list, dtype=np.float32)          # (T, D)
        acts = np.asarray(act_list, dtype=np.int64)          # (T,)
        rets = np.asarray(returns_list, dtype=np.float32)    # (T,)
        # Update running baseline
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * float(rets.mean())
        adv = rets - self.baseline
        # Normalize advantages for stability
        if adv.std() > 1e-6:
            adv = adv / (adv.std() + 1e-6)

        logits = obs @ self.W + self.b
        probs = softmax(logits)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(acts)), acts] = 1.0
        # dL/dlogits for policy gradient: (probs - one_hot) * adv[:, None]
        grad_logits = (probs - one_hot) * adv[:, None]
        grad_W = obs.T @ grad_logits / len(obs)
        grad_b = grad_logits.mean(axis=0)
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b


def build_local_obs(env: IntelligenceTaskEnv, task) -> np.ndarray:
    return np.concatenate(
        [
            task.embedding,
            np.array(
                [env._partial_quality, (task.time_budget - env._step_count) / max(task.time_budget, 1)],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def run_episode_marl(
    env: IntelligenceTaskEnv,
    orchestrator: PPO,
    specialists: List[LinearPolicy],
    rng: np.random.Generator,
    deterministic_specialists: bool = False,
):
    obs, info = env.reset(seed=int(rng.integers(0, 10_000_000)))
    done = False
    # Per-specialist trajectories for this episode
    traj = {i: {"obs": [], "act": []} for i in range(NUM_SPECIALISTS)}
    ep_reward = 0.0
    while not done:
        action, _ = orchestrator.predict(obs, deterministic=True)
        action = int(action)
        if action == FINISH_ACTION:
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            done = term or trunc
            break
        # Specialist effort selection
        local = build_local_obs(env, env._task)
        if deterministic_specialists:
            effort = specialists[action].greedy(local)
        else:
            effort, _ = specialists[action].act(local, rng)
        traj[action]["obs"].append(local)
        traj[action]["act"].append(effort)
        obs, r, term, trunc, info = env.step(action, specialist_effort=effort)
        ep_reward += r
        done = term or trunc
    return ep_reward, traj, info


def train_marl(
    seed: int = 0,
    episodes: int = 6000,
    pretrained_orchestrator: str | None = None,
    verbose: int = 0,
):
    print(f"Training MARL specialists  seed={seed}  episodes={episodes}")
    # Load frozen pre-trained orchestrator (required).
    if pretrained_orchestrator is None:
        pretrained_orchestrator = str(RESULTS_DIR / f"ppo_orchestrator_seed{seed}.zip")
    orch = PPO.load(pretrained_orchestrator)
    print(f"  loaded orchestrator: {Path(pretrained_orchestrator).name}")

    env = IntelligenceTaskEnv(seed=seed + 99)
    rng = np.random.default_rng(seed + 31337)
    specialists = [
        LinearPolicy(LOCAL_OBS_DIM, EFFORT_LEVELS, lr=0.02, seed=seed + 100 + i)
        for i in range(NUM_SPECIALISTS)
    ]

    # Training loop: collect batches of episodes, update each specialist with
    # the *team* reward as its return (shared reward = cooperative MARL).
    batch_size = 32
    curve_timesteps: List[int] = []
    curve_reward: List[float] = []
    curve_success: List[float] = []
    window_rewards: List[float] = []
    window_success: List[float] = []
    ep_counter = 0

    for batch in range(episodes // batch_size):
        batch_obs = {i: [] for i in range(NUM_SPECIALISTS)}
        batch_act = {i: [] for i in range(NUM_SPECIALISTS)}
        batch_ret = {i: [] for i in range(NUM_SPECIALISTS)}
        for _ in range(batch_size):
            ep_reward, traj, info = run_episode_marl(env, orch, specialists, rng)
            window_rewards.append(ep_reward)
            window_success.append(
                1.0 if info.get("partial_quality", 0) >= info.get("threshold", 1) else 0.0
            )
            ep_counter += 1
            for i in range(NUM_SPECIALISTS):
                for obs_t, act_t in zip(traj[i]["obs"], traj[i]["act"]):
                    batch_obs[i].append(obs_t)
                    batch_act[i].append(act_t)
                    batch_ret[i].append(ep_reward)  # shared reward
        for i in range(NUM_SPECIALISTS):
            specialists[i].update(batch_obs[i], batch_act[i], batch_ret[i])

        if len(window_rewards) >= 100:
            window_rewards = window_rewards[-100:]
            window_success = window_success[-100:]
        curve_timesteps.append(ep_counter)
        curve_reward.append(float(np.mean(window_rewards)))
        curve_success.append(float(np.mean(window_success)))
        if verbose and (batch + 1) % 10 == 0:
            print(
                f"  [ep {ep_counter:>5}] "
                f"mean_reward={curve_reward[-1]:+.3f} success={curve_success[-1]:.2%}"
            )

    # Save
    out = RESULTS_DIR / f"marl_specialists_seed{seed}.npz"
    np.savez(
        out,
        **{f"W{i}": specialists[i].W for i in range(NUM_SPECIALISTS)},
        **{f"b{i}": specialists[i].b for i in range(NUM_SPECIALISTS)},
        curve_episodes=np.array(curve_timesteps),
        curve_reward=np.array(curve_reward),
        curve_success=np.array(curve_success),
    )
    print(f"  saved: {out.name}")
    if curve_reward:
        print(
            f"  final (stochastic training) reward={curve_reward[-1]:+.3f} "
            f"success={curve_success[-1]:.2%}"
        )

    # Deterministic evaluation of the learned specialists
    eval_env = IntelligenceTaskEnv(seed=seed + 777)
    eval_rng = np.random.default_rng(seed + 777)
    det_rewards, det_success = [], []
    for _ in range(200):
        r, _, info = run_episode_marl(
            eval_env, orch, specialists, eval_rng, deterministic_specialists=True
        )
        det_rewards.append(r)
        det_success.append(
            1.0 if info.get("partial_quality", 0) >= info.get("threshold", 1) else 0.0
        )
    print(
        f"  deterministic eval (200 eps): reward={np.mean(det_rewards):+.3f}±{np.std(det_rewards):.2f} "
        f"success={np.mean(det_success):.2%}"
    )
    return {
        "timesteps": curve_timesteps,
        "mean_reward": curve_reward,
        "mean_success": curve_success,
    }


def load_specialists(seed: int) -> List[LinearPolicy]:
    path = RESULTS_DIR / f"marl_specialists_seed{seed}.npz"
    data = np.load(path)
    out = []
    for i in range(NUM_SPECIALISTS):
        p = LinearPolicy(LOCAL_OBS_DIM, EFFORT_LEVELS, seed=0)
        p.W = data[f"W{i}"]
        p.b = data[f"b{i}"]
        out.append(p)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    train_marl(seed=args.seed, episodes=args.episodes, verbose=args.verbose)
