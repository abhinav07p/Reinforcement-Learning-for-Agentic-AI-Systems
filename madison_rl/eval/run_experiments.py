"""Run the full experimental suite.

Conditions (all 9 are optional — each is skipped if its trained model
file is missing, so you can run partial experiments):
    1. Random                — uniform action selection
    2. RoundRobin            — fixed-order dispatch
    3. Oracle                — informed heuristic with ground-truth prototypes
    4. LinUCB                — contextual bandit (Category 4)
    5. DQN                   — value-based RL (Category 1)
    6. PPO                   — policy-gradient RL (Category 2)
    7. PPO+MARL              — hierarchical multi-agent RL (Category 3)
    8. PPO+Intrinsic         — PPO with count-based novelty bonus (Category 4)
    9. PPO-Transfer          — PPO fine-tuned from an easy-task pretrain
                               (Category 5: meta-learning / transfer)

For each condition, we evaluate over N_EVAL_EPISODES held-out tasks and
repeat across seeds for statistical validation.

Output:
    experiments/results/all_conditions.csv   (per-episode records)
    experiments/results/summary.csv          (per-condition × seed means)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from stable_baselines3 import DQN, PPO

from madison_rl.agents.baselines import (
    GreedyCapabilityPolicy,
    RandomPolicy,
    RoundRobinPolicy,
)
from madison_rl.agents.linucb import LinUCBPolicy
from madison_rl.env import FINISH_ACTION, IntelligenceTaskEnv
from madison_rl.env.tasks import TaskGenerator
from madison_rl.training.train_marl import load_specialists, run_episode_marl


RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_EVAL_EPISODES = 200
EVAL_SEED_OFFSET = 9000


def _eval_policy(policy, env: IntelligenceTaskEnv, rng: np.random.Generator,
                 n_episodes: int) -> List[dict]:
    """Evaluate any SB3-like policy (has .predict)."""
    rows = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 10_000_000)))
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            a, _ = policy.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(a))
            ep_reward += r
            steps += 1
            done = term or trunc
        success = 1.0 if info["partial_quality"] >= info["threshold"] else 0.0
        rows.append(
            {
                "episode": ep,
                "reward": ep_reward,
                "success": success,
                "steps": steps,
                "final_quality": info["partial_quality"],
            }
        )
    return rows


def _eval_marl(orch, specialists, env: IntelligenceTaskEnv,
               rng: np.random.Generator, n_episodes: int) -> List[dict]:
    rows = []
    for ep in range(n_episodes):
        r, traj, info = run_episode_marl(
            env, orch, specialists, rng, deterministic_specialists=True
        )
        success = 1.0 if info["partial_quality"] >= info["threshold"] else 0.0
        steps = sum(len(traj[i]["obs"]) for i in range(len(traj))) + 1
        rows.append(
            {
                "episode": ep,
                "reward": r,
                "success": success,
                "steps": steps,
                "final_quality": info["partial_quality"],
            }
        )
    return rows


def run_all(seeds: List[int]) -> None:
    all_records: List[dict] = []
    summary: List[dict] = []

    # Shared prototypes (needed for oracle baseline)
    prototypes = TaskGenerator(seed=0)._prototypes

    conditions = [
        "Random", "RoundRobin", "Oracle",
        "LinUCB", "DQN",
        "PPO", "PPO+MARL", "PPO+Intrinsic", "PPO-Transfer",
    ]

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        eval_env = IntelligenceTaskEnv(seed=seed + EVAL_SEED_OFFSET)
        rng_seed = seed + EVAL_SEED_OFFSET

        orch_path = RESULTS_DIR / f"ppo_orchestrator_seed{seed}.zip"
        marl_path = RESULTS_DIR / f"marl_specialists_seed{seed}.npz"
        dqn_path = RESULTS_DIR / f"dqn_orchestrator_seed{seed}.zip"
        linucb_path = RESULTS_DIR / f"linucb_seed{seed}.npz"
        intrinsic_path = RESULTS_DIR / f"ppo_intrinsic_seed{seed}.zip"
        transfer_path = RESULTS_DIR / f"ppo_transfer_finetune_seed{seed}.zip"

        if not orch_path.exists():
            print(f"  [skip seed] missing {orch_path.name} — train PPO first")
            continue

        orch = PPO.load(str(orch_path))
        marl = load_specialists(seed) if marl_path.exists() else None
        dqn_model = DQN.load(str(dqn_path)) if dqn_path.exists() else None
        linucb_model = LinUCBPolicy.load(str(linucb_path)) if linucb_path.exists() else None
        intrinsic_model = PPO.load(str(intrinsic_path)) if intrinsic_path.exists() else None
        transfer_model = PPO.load(str(transfer_path)) if transfer_path.exists() else None

        policies: Dict[str, Callable] = {
            "Random": lambda: _eval_policy(
                RandomPolicy(seed=seed), eval_env, np.random.default_rng(rng_seed), N_EVAL_EPISODES
            ),
            "RoundRobin": lambda: _eval_policy(
                RoundRobinPolicy(), eval_env, np.random.default_rng(rng_seed), N_EVAL_EPISODES
            ),
            "Oracle": lambda: _eval_policy(
                GreedyCapabilityPolicy(prototypes=prototypes),
                eval_env, np.random.default_rng(rng_seed), N_EVAL_EPISODES,
            ),
            "PPO": lambda: _eval_policy(
                orch, eval_env, np.random.default_rng(rng_seed), N_EVAL_EPISODES
            ),
        }
        if linucb_model is not None:
            policies["LinUCB"] = lambda m=linucb_model: _eval_policy(
                m, eval_env, np.random.default_rng(rng_seed), N_EVAL_EPISODES
            )
        if dqn_model is not None:
            policies["DQN"] = lambda m=dqn_model: _eval_policy(
                m, eval_env, np.random.default_rng(rng_seed), N_EVAL_EPISODES
            )
        if marl is not None:
            policies["PPO+MARL"] = lambda: _eval_marl(
                orch, marl, eval_env, np.random.default_rng(rng_seed), N_EVAL_EPISODES
            )
        if intrinsic_model is not None:
            policies["PPO+Intrinsic"] = lambda m=intrinsic_model: _eval_policy(
                m, eval_env, np.random.default_rng(rng_seed), N_EVAL_EPISODES
            )
        if transfer_model is not None:
            policies["PPO-Transfer"] = lambda m=transfer_model: _eval_policy(
                m, eval_env, np.random.default_rng(rng_seed), N_EVAL_EPISODES
            )

        for cond_name in conditions:
            if cond_name not in policies:
                continue
            rows = policies[cond_name]()
            rewards = [r["reward"] for r in rows]
            successes = [r["success"] for r in rows]
            steps = [r["steps"] for r in rows]
            print(
                f"  {cond_name:<10}  reward={np.mean(rewards):+.3f}±{np.std(rewards):.2f}"
                f"  success={np.mean(successes):.2%}"
                f"  steps={np.mean(steps):.1f}"
            )
            for row in rows:
                all_records.append({"condition": cond_name, "seed": seed, **row})
            summary.append(
                {
                    "condition": cond_name,
                    "seed": seed,
                    "mean_reward": float(np.mean(rewards)),
                    "std_reward": float(np.std(rewards)),
                    "success_rate": float(np.mean(successes)),
                    "mean_steps": float(np.mean(steps)),
                    "n": len(rows),
                }
            )

    # Write per-episode records
    records_path = RESULTS_DIR / "all_conditions.csv"
    with open(records_path, "w", newline="") as f:
        if all_records:
            w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            w.writeheader()
            w.writerows(all_records)
    print(f"\nSaved: {records_path.name}  ({len(all_records)} rows)")

    # Write summary
    summary_path = RESULTS_DIR / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    print(f"Saved: {summary_path.name}  ({len(summary)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    run_all(seeds=args.seeds)
