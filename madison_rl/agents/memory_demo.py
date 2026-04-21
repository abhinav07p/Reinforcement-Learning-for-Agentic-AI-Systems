"""Demonstrate the MemoryStore with a trained PPO orchestrator.

Shows that as the memory fills with past episodes, the recall system
finds similar tasks and surfaces the previously-best action sequence.
This is a diagnostic/analysis tool more than a performance booster —
it lets operators understand how the orchestrator has been behaving
and whether it's consistent on similar tasks.

Usage:
    python -m madison_rl.agents.memory_demo
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from stable_baselines3 import PPO

from madison_rl.agents.memory import MemoryStore
from madison_rl.env import (
    IntelligenceTaskEnv,
    SPECIALIST_NAMES,
    NUM_SPECIALISTS,
    FINISH_ACTION,
    EMBED_DIM,
)


RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"


def run_demo(n_episodes: int = 100, seed: int = 42):
    model_path = RESULTS_DIR / "ppo_orchestrator_seed0.zip"
    if not model_path.exists():
        print(f"Missing {model_path} — train with: python -m madison_rl.training.train_orchestrator")
        return

    model = PPO.load(str(model_path))
    env = IntelligenceTaskEnv(seed=seed)
    memory = MemoryStore(capacity=500)
    rng = np.random.default_rng(seed)

    print("=" * 70)
    print("MEMORY STORE DEMO — Running PPO orchestrator with episodic memory")
    print("=" * 70)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 10_000_000)))
        embedding = obs[:EMBED_DIM].copy()

        # Recall from memory before acting
        recall = memory.recall(embedding, k=3)

        actions_taken = []
        done = False
        ep_reward = 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            if a < NUM_SPECIALISTS:
                actions_taken.append(a)
            obs, r, term, trunc, info = env.step(a)
            ep_reward += r
            done = term or trunc

        final_q = info.get("partial_quality", 0.0)
        threshold = info.get("threshold", 0.0)
        success = final_q >= threshold

        # Store in memory
        memory.store(
            embedding=embedding,
            actions=actions_taken,
            final_quality=final_q,
            total_reward=ep_reward,
            success=success,
        )

        # Print first 5 and last 5 episodes with memory recall info
        if ep < 5 or ep >= n_episodes - 5:
            act_names = [SPECIALIST_NAMES[a] for a in actions_taken]
            status = "SUCCESS" if success else "FAIL"
            print(f"\n[Episode {ep+1:>3}] {status}  quality={final_q:.2f}  "
                  f"reward={ep_reward:+.2f}")
            print(f"  Actions: {act_names}")
            print(f"  Memory recall: {recall.summary()}")
        elif ep == 5:
            print(f"\n  ... ({n_episodes - 10} episodes omitted) ...\n")

    print("\n" + "=" * 70)
    print("FINAL MEMORY STATISTICS")
    print("=" * 70)
    stats = memory.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Final recall demonstration: find most similar task to a random query
    print("\nRecall demo: querying memory with a fresh task embedding...")
    obs, info = env.reset(seed=99999)
    recall = memory.recall(obs[:EMBED_DIM], k=5)
    print(f"  Query task required capabilities: {info['required_caps']}")
    print(f"  {recall.summary()}")
    print(f"  Top-3 similarities: {[f'{s:.3f}' for s in recall.similarities[:3]]}")


if __name__ == "__main__":
    run_demo()
