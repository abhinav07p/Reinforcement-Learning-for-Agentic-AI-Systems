"""Smoke tests for the Madison-RL environment.

Run:  python -m tests.test_env
"""
import numpy as np

from madison_rl.env import (
    IntelligenceTaskEnv,
    NUM_ACTIONS,
    OBS_DIM,
    FINISH_ACTION,
    SPECIALIST_NAMES,
)


def test_reset_obs_shape():
    env = IntelligenceTaskEnv(seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_DIM,), f"bad obs shape: {obs.shape}"
    assert obs.dtype == np.float32
    assert "task_id" in info
    print(f"[OK] reset obs shape = {obs.shape}")


def test_action_space():
    env = IntelligenceTaskEnv(seed=0)
    assert env.action_space.n == NUM_ACTIONS == 5
    print(f"[OK] action space = Discrete({NUM_ACTIONS})")


def test_deterministic_seed():
    env1 = IntelligenceTaskEnv(seed=123)
    env2 = IntelligenceTaskEnv(seed=123)
    o1, _ = env1.reset(seed=123)
    o2, _ = env2.reset(seed=123)
    assert np.allclose(o1, o2), "same seed produced different obs"
    # Run identical action sequence
    for a in [0, 1, 2, 3, FINISH_ACTION]:
        s1 = env1.step(a)
        s2 = env2.step(a)
        assert np.allclose(s1[0], s2[0]), f"obs diverged at action {a}"
        assert abs(s1[1] - s2[1]) < 1e-6, f"reward diverged at action {a}"
        if s1[2] or s1[3]:
            break
    print("[OK] deterministic under seed")


def test_random_rollout():
    env = IntelligenceTaskEnv(seed=7)
    rng = np.random.default_rng(7)
    successes = 0
    total_reward = 0.0
    n_episodes = 50
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=7 + ep)
        done = False
        ep_reward = 0.0
        while not done:
            a = int(rng.integers(0, NUM_ACTIONS))
            obs, r, term, trunc, info = env.step(a)
            ep_reward += r
            done = term or trunc
        total_reward += ep_reward
        if info.get("partial_quality", 0) >= info.get("threshold", 1):
            successes += 1
    avg = total_reward / n_episodes
    print(
        f"[OK] random policy over {n_episodes} episodes: "
        f"avg_reward={avg:.3f}  success_rate={successes/n_episodes:.2%}"
    )


def test_greedy_perfect_routing():
    """Dispatch exactly the specialists the task needs with medium effort,
    then finish. Should succeed more often than random."""
    env = IntelligenceTaskEnv(seed=11)
    successes = 0
    rewards = []
    n_episodes = 50
    for ep in range(n_episodes):
        obs, info = env.reset(seed=11 + ep)
        req = np.array(info["required_caps"])
        # Dispatch all specialists whose capability is required (weight > 0.1)
        needed = [i for i in range(4) if req[i] > 0.1]
        ep_reward = 0.0
        for sp in needed:
            obs, r, term, trunc, info = env.step(sp, specialist_effort=2)  # deep
            ep_reward += r
            if term or trunc:
                break
        else:
            obs, r, term, trunc, info = env.step(FINISH_ACTION)
            ep_reward += r
        rewards.append(ep_reward)
        if info.get("partial_quality", 0) >= info.get("threshold", 1):
            successes += 1
    print(
        f"[OK] oracle policy: avg_reward={np.mean(rewards):.3f}  "
        f"success_rate={successes/n_episodes:.2%}"
    )


if __name__ == "__main__":
    test_reset_obs_shape()
    test_action_space()
    test_deterministic_seed()
    test_random_rollout()
    test_greedy_perfect_routing()
    print("\nAll smoke tests passed.")
