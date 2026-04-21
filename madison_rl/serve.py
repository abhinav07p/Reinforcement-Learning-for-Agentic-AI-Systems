"""Madison-RL Production Serving Interface.

Wraps the trained orchestrator into a clean, production-style API with:
    - Configuration from dict/YAML (no hardcoded params)
    - Structured logging via Python's logging module
    - Graceful error handling and fallback strategies
    - Episodic memory integration for recall-augmented routing
    - Monitoring hooks: quality degradation detection, per-episode stats
    - Health check / readiness probe

This module demonstrates production-readiness patterns even though the
underlying specialists are simulated. In a real deployment, the mock
specialist pool would be replaced by an LLM gateway (see demo/ollama_demo.py
for the inference-time integration pattern).

Usage:
    from madison_rl.serve import MadisonOrchestrator

    orch = MadisonOrchestrator.from_config({
        "model_path": "experiments/results/ppo_orchestrator_seed0.zip",
        "algorithm": "PPO",
        "memory_capacity": 1000,
        "quality_alert_threshold": 0.7,
    })

    result = orch.run(task_seed=42)
    print(result)

    # Health check
    print(orch.health())

    # Monitoring stats
    print(orch.monitoring_stats())
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from madison_rl.agents.memory import MemoryStore
from madison_rl.env import (
    EMBED_DIM,
    FINISH_ACTION,
    IntelligenceTaskEnv,
    NUM_SPECIALISTS,
    SPECIALIST_NAMES,
)


logger = logging.getLogger("madison_rl.serve")


@dataclass
class OrchestratorResult:
    """Structured result from a single orchestrator run."""
    success: bool
    final_quality: float
    total_reward: float
    actions_taken: List[str]
    steps: int
    latency_ms: float
    memory_recall_summary: str
    task_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "final_quality": round(self.final_quality, 4),
            "total_reward": round(self.total_reward, 3),
            "actions": self.actions_taken,
            "steps": self.steps,
            "latency_ms": round(self.latency_ms, 1),
            "memory_recall": self.memory_recall_summary,
            **self.task_info,
        }


class QualityMonitor:
    """Detects quality degradation via a rolling window."""

    def __init__(self, window: int = 50, alert_threshold: float = 0.7):
        self.window = window
        self.alert_threshold = alert_threshold
        self._successes: List[float] = []
        self._rewards: List[float] = []
        self._alert_fired = False

    def record(self, success: bool, reward: float) -> Optional[str]:
        self._successes.append(1.0 if success else 0.0)
        self._rewards.append(reward)
        if len(self._successes) > self.window:
            self._successes = self._successes[-self.window:]
            self._rewards = self._rewards[-self.window:]

        if len(self._successes) >= self.window:
            rate = np.mean(self._successes)
            if rate < self.alert_threshold and not self._alert_fired:
                self._alert_fired = True
                msg = (f"QUALITY ALERT: success rate {rate:.1%} dropped below "
                       f"threshold {self.alert_threshold:.1%} over last "
                       f"{self.window} episodes")
                logger.warning(msg)
                return msg
            elif rate >= self.alert_threshold:
                self._alert_fired = False
        return None

    def stats(self) -> dict:
        if not self._successes:
            return {"episodes_tracked": 0}
        return {
            "episodes_tracked": len(self._successes),
            "rolling_success_rate": round(float(np.mean(self._successes)), 3),
            "rolling_mean_reward": round(float(np.mean(self._rewards)), 3),
            "alert_active": self._alert_fired,
        }


class MadisonOrchestrator:
    """Production-grade wrapper around the trained RL orchestrator.

    Handles model loading, environment management, memory integration,
    monitoring, and structured error handling in a single clean interface.
    """

    def __init__(
        self,
        model,
        algorithm: str = "PPO",
        memory_capacity: int = 1000,
        monitor_window: int = 50,
        quality_alert_threshold: float = 0.7,
        env_seed: int = 0,
    ):
        self._model = model
        self._algorithm = algorithm
        self._memory = MemoryStore(capacity=memory_capacity)
        self._monitor = QualityMonitor(
            window=monitor_window, alert_threshold=quality_alert_threshold
        )
        self._env = IntelligenceTaskEnv(seed=env_seed)
        self._total_runs = 0
        self._load_time = time.time()
        logger.info(
            f"MadisonOrchestrator initialized: algorithm={algorithm}, "
            f"memory_capacity={memory_capacity}, monitor_window={monitor_window}"
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MadisonOrchestrator":
        """Factory method: build from a configuration dict.

        Required keys:
            model_path (str): path to saved SB3 model (.zip)
        Optional keys:
            algorithm (str): "PPO" or "DQN" (default "PPO")
            memory_capacity (int): episodic memory size (default 1000)
            monitor_window (int): quality monitoring window (default 50)
            quality_alert_threshold (float): alert if success dips below (default 0.7)
            env_seed (int): environment seed (default 0)
        """
        model_path = config["model_path"]
        algorithm = config.get("algorithm", "PPO").upper()

        logger.info(f"Loading {algorithm} model from {model_path}")
        if algorithm == "DQN":
            from stable_baselines3 import DQN
            model = DQN.load(model_path)
        else:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)

        return cls(
            model=model,
            algorithm=algorithm,
            memory_capacity=config.get("memory_capacity", 1000),
            monitor_window=config.get("monitor_window", 50),
            quality_alert_threshold=config.get("quality_alert_threshold", 0.7),
            env_seed=config.get("env_seed", 0),
        )

    def run(self, task_seed: Optional[int] = None) -> OrchestratorResult:
        """Execute the orchestrator on one task.

        Returns a structured OrchestratorResult with all relevant metadata.
        Handles errors gracefully — never raises; logs and returns a failed result.
        """
        t0 = time.perf_counter()
        self._total_runs += 1

        try:
            seed = task_seed or int(np.random.default_rng().integers(0, 10_000_000))
            obs, info = self._env.reset(seed=seed)
            embedding = obs[:EMBED_DIM].copy()

            # Memory recall
            recall = self._memory.recall(embedding, k=5)
            recall_summary = recall.summary()

            actions_taken = []
            done = False
            total_reward = 0.0

            while not done:
                action, _ = self._model.predict(obs, deterministic=True)
                action = int(action)
                action_name = SPECIALIST_NAMES[action] if action < NUM_SPECIALISTS else "FINISH"
                if action < NUM_SPECIALISTS:
                    actions_taken.append(action_name)
                obs, r, term, trunc, info = self._env.step(action)
                total_reward += r
                done = term or trunc

            final_quality = info.get("partial_quality", 0.0)
            threshold = info.get("threshold", 0.0)
            success = final_quality >= threshold

            # Store in memory
            self._memory.store(
                embedding=embedding,
                actions=[SPECIALIST_NAMES.index(a) for a in actions_taken],
                final_quality=final_quality,
                total_reward=total_reward,
                success=success,
            )

            # Monitor
            alert = self._monitor.record(success, total_reward)
            if alert:
                logger.warning(f"Run #{self._total_runs}: {alert}")

            latency = (time.perf_counter() - t0) * 1000

            result = OrchestratorResult(
                success=success,
                final_quality=final_quality,
                total_reward=total_reward,
                actions_taken=actions_taken,
                steps=len(actions_taken),
                latency_ms=latency,
                memory_recall_summary=recall_summary,
                task_info={
                    "difficulty": info.get("difficulty"),
                    "threshold": threshold,
                    "task_seed": seed,
                },
            )

            logger.info(
                f"Run #{self._total_runs}: {'SUCCESS' if success else 'FAIL'} "
                f"quality={final_quality:.3f} reward={total_reward:+.2f} "
                f"steps={len(actions_taken)} latency={latency:.1f}ms"
            )
            return result

        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(f"Run #{self._total_runs} FAILED: {e}", exc_info=True)
            return OrchestratorResult(
                success=False,
                final_quality=0.0,
                total_reward=0.0,
                actions_taken=[],
                steps=0,
                latency_ms=latency,
                memory_recall_summary="error",
                task_info={"error": str(e)},
            )

    def health(self) -> Dict[str, Any]:
        """Readiness probe — returns model status and uptime."""
        return {
            "status": "healthy",
            "algorithm": self._algorithm,
            "model_loaded": self._model is not None,
            "total_runs": self._total_runs,
            "memory_size": self._memory.size,
            "uptime_seconds": round(time.time() - self._load_time, 1),
            "monitor": self._monitor.stats(),
        }

    def monitoring_stats(self) -> Dict[str, Any]:
        """Aggregate monitoring statistics."""
        return {
            "total_runs": self._total_runs,
            "memory": self._memory.get_stats(),
            "quality_monitor": self._monitor.stats(),
        }


# ---------------------------------------------------------------- CLI demo
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    model_path = "experiments/results/ppo_orchestrator_seed0.zip"
    if not Path(model_path).exists():
        print(f"Missing {model_path} — train first")
        sys.exit(1)

    orch = MadisonOrchestrator.from_config({
        "model_path": model_path,
        "algorithm": "PPO",
        "memory_capacity": 500,
        "quality_alert_threshold": 0.8,
    })

    print("\n=== Health check ===")
    print(orch.health())

    print("\n=== Running 20 tasks ===")
    for i in range(20):
        result = orch.run(task_seed=i * 1000)
        if i < 5 or i >= 15:
            print(f"  Task {i+1:>2}: {'✓' if result.success else '✗'}  "
                  f"quality={result.final_quality:.2f}  "
                  f"actions={result.actions_taken}  "
                  f"latency={result.latency_ms:.0f}ms")
        elif i == 5:
            print("  ...")

    print("\n=== Monitoring stats ===")
    stats = orch.monitoring_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
