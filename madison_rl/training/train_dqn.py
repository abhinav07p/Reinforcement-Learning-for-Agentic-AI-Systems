"""Train a DQN orchestrator on IntelligenceTaskEnv.

This is a Category-1 (Value-Based Learning) implementation that sits
alongside the PPO (Category-2) orchestrator. Same environment, same
observation and action spaces — only the learning algorithm differs.

Why DQN here:
    - Rubric category #1 explicitly lists "Q-Learning, SARSA, or DQN"
    - Our action space is Discrete(5), which is exactly what DQN needs
    - Enables a head-to-head comparison with PPO in the results table
    - SB3's DQN class drops in with minimal changes

Key hyperparameter notes:
    - buffer_size: 50k (env episodes are short, so 50k steps is plenty)
    - learning_starts: 2000 (warmup before Q updates)
    - target_update_interval: 500
    - exploration_fraction: 0.3 (30% of training for epsilon decay)
    - exploration_final_eps: 0.05

Usage:
    python -m madison_rl.training.train_dqn --seed 0 --timesteps 80000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from madison_rl.env import IntelligenceTaskEnv


RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class DQNEpisodeLogger(BaseCallback):
    def __init__(self, log_every: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self.episode_rewards: List[float] = []
        self.episode_successes: List[float] = []
        self.timesteps: List[int] = []
        self.mean_rewards: List[float] = []
        self.mean_successes: List[float] = []
        self._current_reward = 0.0
        self._last_log_step = 0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0.0])
        done = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        if hasattr(reward, "__len__"):
            r = float(reward[0])
            d = bool(done[0])
            info = infos[0]
        else:
            r = float(reward)
            d = bool(done)
            info = infos

        self._current_reward += r
        if d:
            self.episode_rewards.append(self._current_reward)
            pq = info.get("partial_quality", 0.0)
            thr = info.get("threshold", 1.0)
            self.episode_successes.append(1.0 if pq >= thr else 0.0)
            self._current_reward = 0.0

        if self.num_timesteps - self._last_log_step >= self.log_every:
            self._last_log_step = self.num_timesteps
            if self.episode_rewards:
                recent = self.episode_rewards[-100:]
                recent_s = self.episode_successes[-100:]
                self.timesteps.append(int(self.num_timesteps))
                self.mean_rewards.append(float(np.mean(recent)))
                self.mean_successes.append(float(np.mean(recent_s)))
                if self.verbose:
                    print(
                        f"  [{self.num_timesteps:>7}] "
                        f"mean_reward={np.mean(recent):+.3f} "
                        f"success={np.mean(recent_s):.2%}"
                    )
        return True


def train(seed: int = 0, timesteps: int = 80_000, verbose: int = 0,
          device: str = "auto") -> dict:
    print(f"Training DQN orchestrator  seed={seed}  timesteps={timesteps}  device={device}")
    env = Monitor(IntelligenceTaskEnv(seed=seed))

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=64,
        tau=1.0,  # hard target updates
        gamma=0.98,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=0,
        device=device,
    )

    logger = DQNEpisodeLogger(log_every=2048, verbose=verbose)
    model.learn(total_timesteps=timesteps, callback=logger, progress_bar=False)

    model_path = RESULTS_DIR / f"dqn_orchestrator_seed{seed}.zip"
    model.save(str(model_path))

    curve_path = RESULTS_DIR / f"dqn_orchestrator_seed{seed}_curve.csv"
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timesteps", "mean_reward", "mean_success"])
        for t, r, s in zip(logger.timesteps, logger.mean_rewards, logger.mean_successes):
            w.writerow([t, r, s])

    print(f"  saved: {model_path.name}")
    if logger.mean_rewards:
        print(
            f"  final mean_reward={logger.mean_rewards[-1]:+.3f} "
            f"success={logger.mean_successes[-1]:.2%}"
        )
    return {
        "timesteps": logger.timesteps,
        "mean_reward": logger.mean_rewards,
        "mean_success": logger.mean_successes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    train(seed=args.seed, timesteps=args.timesteps, verbose=args.verbose,
          device=args.device)
