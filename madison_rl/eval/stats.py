"""Statistical analysis of experiment results.

Loads experiments/results/all_conditions.csv and computes:
    - Per-condition mean, std, 95% bootstrap CI
    - Welch's t-test between PPO and each baseline
    - Cohen's d effect size
    - Per-seed means for statistical validation across seeds

Writes: experiments/results/stats_report.txt and stats_table.csv
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if s < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / s)


def analyze() -> None:
    records_path = RESULTS_DIR / "all_conditions.csv"
    if not records_path.exists():
        raise FileNotFoundError(
            f"{records_path} not found. Run `python -m madison_rl.eval.run_experiments` first."
        )
    df = pd.read_csv(records_path)
    conditions = df["condition"].unique().tolist()
    print(f"Loaded {len(df)} records across {len(conditions)} conditions")

    # Per-condition aggregate (pooled over seeds)
    agg_rows = []
    per_cond_reward: Dict[str, np.ndarray] = {}
    for c in conditions:
        sub = df[df["condition"] == c]
        rewards = sub["reward"].values.astype(float)
        successes = sub["success"].values.astype(float)
        per_cond_reward[c] = rewards
        lo, hi = bootstrap_ci(rewards)
        agg_rows.append(
            {
                "condition": c,
                "n": len(rewards),
                "mean_reward": float(rewards.mean()),
                "std_reward": float(rewards.std(ddof=1)),
                "ci95_low": lo,
                "ci95_high": hi,
                "success_rate": float(successes.mean()),
                "mean_steps": float(sub["steps"].mean()),
            }
        )

    table_path = RESULTS_DIR / "stats_table.csv"
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    # Pairwise comparisons vs PPO (pooled, per-episode level)
    pair_rows = []
    if "PPO" in per_cond_reward:
        ppo = per_cond_reward["PPO"]
        for c in conditions:
            if c == "PPO":
                continue
            other = per_cond_reward[c]
            t, p = stats.ttest_ind(ppo, other, equal_var=False)  # Welch
            d = cohens_d(ppo, other)
            pair_rows.append(
                {
                    "comparison": f"PPO vs {c}",
                    "mean_diff": float(ppo.mean() - other.mean()),
                    "t_stat": float(t),
                    "p_value": float(p),
                    "cohens_d": d,
                }
            )

    # Per-seed means — enables across-seed Welch's t-test too
    per_seed = (
        df.groupby(["condition", "seed"])["reward"]
        .mean()
        .reset_index()
        .pivot(index="seed", columns="condition", values="reward")
    )
    seed_table_path = RESULTS_DIR / "per_seed_means.csv"
    per_seed.to_csv(seed_table_path)

    # Text report
    report_path = RESULTS_DIR / "stats_report.txt"
    with open(report_path, "w") as f:
        f.write("MADISON-RL — STATISTICAL REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-condition summary (pooled across seeds)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'condition':<12}{'n':>6}{'mean':>10}{'std':>10}"
                f"{'CI95_low':>12}{'CI95_high':>12}{'succ%':>10}\n")
        for r in agg_rows:
            f.write(
                f"{r['condition']:<12}{r['n']:>6}{r['mean_reward']:>10.3f}"
                f"{r['std_reward']:>10.3f}{r['ci95_low']:>12.3f}"
                f"{r['ci95_high']:>12.3f}{100*r['success_rate']:>9.1f}%\n"
            )
        f.write("\n")

        if pair_rows:
            f.write("Pairwise tests (Welch's t, pooled episodes) vs PPO\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'comparison':<22}{'Δmean':>10}{'t':>10}{'p':>12}{'d':>10}\n")
            for r in pair_rows:
                sig = "***" if r["p_value"] < 0.001 else (
                    "**" if r["p_value"] < 0.01 else (
                        "*" if r["p_value"] < 0.05 else "ns"))
                f.write(
                    f"{r['comparison']:<22}{r['mean_diff']:>10.3f}"
                    f"{r['t_stat']:>10.2f}{r['p_value']:>12.2e}"
                    f"{r['cohens_d']:>10.2f}  {sig}\n"
                )
            f.write("\nSignificance: *** p<.001  ** p<.01  * p<.05  ns not significant\n")
            f.write("Cohen's d:    0.2 small  0.5 medium  0.8 large\n\n")

        f.write("Per-seed mean reward (tests across independent seeds)\n")
        f.write("-" * 60 + "\n")
        f.write(per_seed.to_string())
        f.write("\n")

    print(f"Saved: {table_path.name}")
    print(f"Saved: {seed_table_path.name}")
    print(f"Saved: {report_path.name}")

    # Echo to console
    print("\n" + open(report_path).read())


if __name__ == "__main__":
    analyze()
