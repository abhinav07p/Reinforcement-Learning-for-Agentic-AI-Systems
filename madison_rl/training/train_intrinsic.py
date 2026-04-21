"""Train PPO with intrinsic motivation (count-based novelty bonus).

A Category-4 (Exploration Strategies) enhancement. The environment is
configured with ``intrinsic_reward=True``, which adds a count-based
novelty bonus to each step's reward:

    r_intrinsic = coef / sqrt(visit_count(task_bucket, action) + 1)

This encourages the orchestrator to try underused (task-type, specialist)
combinations early in training, which can accelerate exploration of
the action space on harder task distributions.

The intrinsic bonus fades as counts grow, so the asymptotic policy is
governed by the extrinsic reward only.

Usage:
    python -m madison_rl.training.train_intrinsic --seed 0 --timesteps 60000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from madison_rl.env import IntelligenceTaskEnv
from madison_rl.training.train_orchestrator import EpisodeLogger


RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_env_fn(seed: int, intrinsic_coef: float):
    def _make():
        env = IntelligenceTaskEnv(
            seed=seed, intrinsic_reward=True, intrinsic_coef=intrinsic_coef,
        )
        env = Monitor(env)
        return env
    return _make


def train(seed: int = 0, timesteps: int = 60_000, intrinsic_coef: float = 0.15,
          verbose: int = 0, device: str = "auto") -> dict:
    print(f"Training PPO+Intrinsic  seed={seed}  timesteps={timesteps}  "
          f"coef={intrinsic_coef}  device={device}")
    env = make_vec_env(make_env_fn(seed, intrinsic_coef), n_envs=4, seed=seed)

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

    model_path = RESULTS_DIR / f"ppo_intrinsic_seed{seed}.zip"
    model.save(str(model_path))

    curve_path = RESULTS_DIR / f"ppo_intrinsic_seed{seed}_curve.csv"
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
    return logger.to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=60_000)
    parser.add_argument("--intrinsic-coef", type=float, default=0.15)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    train(
        seed=args.seed,
        timesteps=args.timesteps,
        intrinsic_coef=args.intrinsic_coef,
        verbose=args.verbose,
        device=args.device,
    )
