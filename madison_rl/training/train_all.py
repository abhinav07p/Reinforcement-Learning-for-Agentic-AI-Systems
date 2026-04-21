"""Train PPO and MARL for all seeds. This is the main experiment driver.

Usage:
    python -m madison_rl.training.train_all --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import time

from madison_rl.training.train_orchestrator import train as train_orch
from madison_rl.training.train_marl import train_marl


def main(seeds, orch_steps: int, marl_episodes: int, device: str = "auto"):
    t0 = time.time()
    for seed in seeds:
        print(f"\n========== SEED {seed} ==========")
        train_orch(seed=seed, timesteps=orch_steps, verbose=0, device=device)
        train_marl(seed=seed, episodes=marl_episodes, verbose=0)
    print(f"\nAll seeds trained in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--orch-steps", type=int, default=60_000)
    parser.add_argument("--marl-episodes", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    main(args.seeds, args.orch_steps, args.marl_episodes, args.device)
