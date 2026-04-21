"""Transfer learning experiment for the orchestrator.

A Category-5 (Meta-Learning / Transfer Learning) implementation.

Protocol:
    Phase A — Source task (easy): train PPO from scratch on tasks
              that require only 1-2 capabilities (min_caps=1, max_caps=2).
    Phase B — Target task (hard): either
              (i) fine-tune the Phase-A model on tasks requiring 2-4
                  capabilities (min_caps=2, max_caps=4), or
              (ii) train a fresh PPO from scratch on the target tasks.
    Compare learning curves: the fine-tuned model should reach high
    performance faster than the from-scratch model.

We report the *number of env steps needed to cross 90% success* as the
headline transfer-learning metric, plus side-by-side learning curves.

Usage:
    python -m madison_rl.training.train_transfer --seed 0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from madison_rl.env import IntelligenceTaskEnv
from madison_rl.training.train_orchestrator import EpisodeLogger


RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PPO_KWARGS = dict(
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
    verbose=0,
)


def make_env_fn(seed: int, min_caps: int, max_caps: int):
    def _make():
        env = IntelligenceTaskEnv(
            seed=seed, task_min_caps=min_caps, task_max_caps=max_caps,
        )
        env = Monitor(env)
        return env
    return _make


def _save_curve(path: Path, logger: EpisodeLogger):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timesteps", "mean_reward", "mean_success"])
        for t, r, s in zip(logger.timesteps, logger.mean_rewards, logger.mean_successes):
            w.writerow([t, r, s])


def _steps_to_threshold(logger: EpisodeLogger, success_threshold: float = 0.9) -> int:
    for t, s in zip(logger.timesteps, logger.mean_successes):
        if s >= success_threshold:
            return int(t)
    return -1  # never crossed


def train(seed: int = 0, source_steps: int = 40_000,
          target_steps: int = 40_000, device: str = "auto") -> dict:
    print(f"\n=== Transfer experiment  seed={seed} ===")

    # ------------------------------------------------------------ Phase A
    print(f"Phase A (source, easy 1-2 caps)  steps={source_steps}")
    env_a = make_vec_env(make_env_fn(seed, 1, 2), n_envs=4, seed=seed)
    model_a = PPO("MlpPolicy", env_a, seed=seed, device=device, **PPO_KWARGS)
    log_a = EpisodeLogger(log_every=2048, verbose=0)
    model_a.learn(total_timesteps=source_steps, callback=log_a, progress_bar=False)
    model_a_path = RESULTS_DIR / f"ppo_transfer_source_seed{seed}.zip"
    model_a.save(str(model_a_path))
    _save_curve(RESULTS_DIR / f"ppo_transfer_source_seed{seed}_curve.csv", log_a)
    print(f"  source final success={log_a.mean_successes[-1]:.2%}")

    # ------------------------------------------------------------ Phase B1: fine-tune
    print(f"Phase B1 (target, hard 2-4 caps, FINE-TUNED)  steps={target_steps}")
    env_b = make_vec_env(make_env_fn(seed + 500, 2, 4), n_envs=4, seed=seed + 500)
    model_ft = PPO.load(str(model_a_path), env=env_b, device=device)
    log_ft = EpisodeLogger(log_every=2048, verbose=0)
    model_ft.learn(total_timesteps=target_steps, callback=log_ft, progress_bar=False)
    model_ft.save(str(RESULTS_DIR / f"ppo_transfer_finetune_seed{seed}.zip"))
    _save_curve(RESULTS_DIR / f"ppo_transfer_finetune_seed{seed}_curve.csv", log_ft)
    ft_threshold = _steps_to_threshold(log_ft, 0.9)
    print(f"  fine-tuned final success={log_ft.mean_successes[-1]:.2%}  "
          f"(reached 90% at {ft_threshold} steps)")

    # ------------------------------------------------------------ Phase B2: scratch
    print(f"Phase B2 (target, hard 2-4 caps, FROM SCRATCH)  steps={target_steps}")
    env_c = make_vec_env(make_env_fn(seed + 500, 2, 4), n_envs=4, seed=seed + 500)
    model_sc = PPO("MlpPolicy", env_c, seed=seed, device=device, **PPO_KWARGS)
    log_sc = EpisodeLogger(log_every=2048, verbose=0)
    model_sc.learn(total_timesteps=target_steps, callback=log_sc, progress_bar=False)
    model_sc.save(str(RESULTS_DIR / f"ppo_transfer_scratch_seed{seed}.zip"))
    _save_curve(RESULTS_DIR / f"ppo_transfer_scratch_seed{seed}_curve.csv", log_sc)
    sc_threshold = _steps_to_threshold(log_sc, 0.9)
    print(f"  scratch final success={log_sc.mean_successes[-1]:.2%}  "
          f"(reached 90% at {sc_threshold} steps)")

    # Speedup metric
    if ft_threshold > 0 and sc_threshold > 0:
        speedup = sc_threshold / ft_threshold
        print(f"  TRANSFER SPEEDUP: {speedup:.2f}x  "
              f"(scratch {sc_threshold} steps / fine-tuned {ft_threshold} steps)")
    elif ft_threshold > 0 and sc_threshold < 0:
        print(f"  TRANSFER SPEEDUP: infinite (scratch never reached 90%)")

    return {
        "source_final": log_a.mean_successes[-1] if log_a.mean_successes else None,
        "finetune_final": log_ft.mean_successes[-1] if log_ft.mean_successes else None,
        "scratch_final": log_sc.mean_successes[-1] if log_sc.mean_successes else None,
        "finetune_threshold_steps": ft_threshold,
        "scratch_threshold_steps": sc_threshold,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-steps", type=int, default=40_000)
    parser.add_argument("--target-steps", type=int, default=40_000)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    train(
        seed=args.seed,
        source_steps=args.source_steps,
        target_steps=args.target_steps,
        device=args.device,
    )
