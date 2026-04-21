"""Train the PPO orchestrator on IntelligenceTaskEnv.

Usage:
    python -m madison_rl.training.train_orchestrator --seed 0 --timesteps 150000

Produces:
    experiments/results/ppo_orchestrator_seed{N}.zip           (model)
    experiments/results/ppo_orchestrator_seed{N}_curve.csv     (learning curve)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from madison_rl.env import IntelligenceTaskEnv


RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class EpisodeLogger(BaseCallback):
    """Logs per-episode reward and success at a fixed interval."""

    def __init__(self, log_every: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self.episode_rewards: List[float] = []
        self.episode_successes: List[float] = []
        self.timesteps: List[int] = []
        self.mean_rewards: List[float] = []
        self.mean_successes: List[float] = []
        self._current_rewards = np.zeros(1)

    def _on_step(self) -> bool:
        # SB3 gives us infos and rewards per env in self.locals
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if rewards is None or dones is None or infos is None:
            return True

        if self._current_rewards.shape[0] != len(rewards):
            self._current_rewards = np.zeros(len(rewards))

        self._current_rewards += np.asarray(rewards)
        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(float(self._current_rewards[i]))
                info = infos[i]
                pq = info.get("partial_quality", 0.0)
                thr = info.get("threshold", 1.0)
                self.episode_successes.append(1.0 if pq >= thr else 0.0)
                self._current_rewards[i] = 0.0

        if self.num_timesteps % self.log_every < len(rewards):
            if len(self.episode_rewards) > 0:
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

    def to_dict(self):
        return {
            "timesteps": self.timesteps,
            "mean_reward": self.mean_rewards,
            "mean_success": self.mean_successes,
        }


def make_env_fn(seed: int):
    def _make():
        env = IntelligenceTaskEnv(seed=seed)
        env = Monitor(env)
        return env
    return _make


def train(seed: int = 0, timesteps: int = 150_000, verbose: int = 0,
          device: str = "auto") -> dict:
    print(f"Training PPO orchestrator  seed={seed}  timesteps={timesteps}  device={device}")
    env = make_vec_env(make_env_fn(seed), n_envs=4, seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=0,
        device=device,
    )

    logger = EpisodeLogger(log_every=2048, verbose=verbose)
    model.learn(total_timesteps=timesteps, callback=logger, progress_bar=False)

    model_path = RESULTS_DIR / f"ppo_orchestrator_seed{seed}.zip"
    model.save(str(model_path))
    curve_path = RESULTS_DIR / f"ppo_orchestrator_seed{seed}_curve.csv"
    import csv
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timesteps", "mean_reward", "mean_success"])
        for t, r, s in zip(logger.timesteps, logger.mean_rewards, logger.mean_successes):
            w.writerow([t, r, s])

    print(f"  saved: {model_path.name}")
    print(f"  saved: {curve_path.name}")
    if logger.mean_rewards:
        print(
            f"  final mean_reward={logger.mean_rewards[-1]:+.3f} "
            f"success={logger.mean_successes[-1]:.2%}"
        )
    return logger.to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Torch device. 'auto' uses GPU if available. "
                             "Note: for the small MLP in this project, CPU is "
                             "often faster than GPU due to per-step transfer overhead.")
    args = parser.parse_args()
    train(seed=args.seed, timesteps=args.timesteps, verbose=args.verbose,
          device=args.device)
