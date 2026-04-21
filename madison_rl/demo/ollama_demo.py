"""Real-LLM demo for the Madison-RL orchestrator.

This script is for the *qualitative demo only* — it shows that the trained
orchestrator policy can drive a team of real LLM specialists, not just mock
ones. Training still happens exclusively in simulation (for speed and
reproducibility); the LLMs only appear at inference time.

How it works:
    1. Load a trained PPO orchestrator.
    2. Replace the mock specialist pool with an OllamaSpecialistPool that
       calls local LLMs via the Ollama HTTP API (http://localhost:11434).
    3. Pose an intelligence query as a natural-language prompt.
    4. Let the orchestrator pick specialists; each specialist's "response"
       is the text completion from a local model (e.g. llama3:8b).
    5. The orchestrator's observation uses a hash-based pseudo-embedding
       derived from the query text, plus a quality estimate from a simple
       text-overlap heuristic.

Requirements:
    pip install requests
    Ollama running locally with at least one model pulled, e.g.:
        ollama pull llama3
        ollama serve

If Ollama is not available, the script falls back to a pure mock run so
you can still record a demo video without a working LLM stack.

Usage:
    python -m madison_rl.demo.ollama_demo \
        --model experiments/results/ppo_orchestrator_seed0.zip \
        --query "Assess the impact of recent AI chip export restrictions on TSMC"
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO

from madison_rl.env import (
    EMBED_DIM,
    FINISH_ACTION,
    IntelligenceTaskEnv,
    NUM_SPECIALISTS,
    SPECIALIST_NAMES,
)


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


SPECIALIST_PROMPTS = {
    0: "You are a research specialist. Given the following intelligence query, "
       "list 3-5 key facts, named entities, and recent developments relevant to it. "
       "Be concise.\n\nQuery: {query}",
    1: "You are a quantitative analyst. Given the following intelligence query and "
       "prior findings, identify numerical signals, trends, and risk factors.\n\n"
       "Query: {query}\nPrior findings:\n{prior}",
    2: "You are a synthesizer. Given the following query and prior findings, "
       "produce a 3-sentence executive summary.\n\nQuery: {query}\nPrior findings:\n{prior}",
    3: "You are a fact-checker. Given the following claims, flag any that are "
       "unverified, speculative, or potentially inaccurate. Be brief.\n\nClaims:\n{prior}",
}


def text_to_embedding(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic pseudo-embedding from a text string via SHA256 bytes.
    Good enough for a qualitative demo; not a real semantic embedding.
    """
    h = hashlib.sha256(text.encode()).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    arr = arr / 255.0 - 0.5
    if len(arr) < dim:
        arr = np.tile(arr, (dim // len(arr)) + 1)
    return arr[:dim].astype(np.float32)


def ollama_complete(prompt: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    try:
        import requests
    except ImportError:
        return None
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception as e:
        print(f"  [ollama unreachable: {e}]")
    return None


def estimate_quality(findings: List[str]) -> float:
    """Toy quality estimator: reward diversity and length.
    In a real system, this would be an LLM-as-judge or rubric-based scorer.
    """
    if not findings:
        return 0.0
    total_len = sum(len(f) for f in findings)
    unique_bigrams = set()
    for f in findings:
        words = f.lower().split()
        unique_bigrams.update(zip(words, words[1:]))
    # Normalize: diversity + volume, capped at 1
    q = min(1.0, 0.4 * min(len(unique_bigrams) / 60, 1.0) + 0.6 * min(total_len / 800, 1.0))
    return float(q)


def build_obs(query_embedding, partial_quality, step_count,
              max_steps, cost_used, used_mask, last_conf, last_gain):
    obs = np.concatenate(
        [
            query_embedding,
            np.array([partial_quality], dtype=np.float32),
            np.array([step_count / max(max_steps, 1)], dtype=np.float32),
            np.array([cost_used / 20.0], dtype=np.float32),
            used_mask.astype(np.float32),
            np.array([last_conf], dtype=np.float32),
            np.array([last_gain], dtype=np.float32),
        ]
    )
    return obs.astype(np.float32)


def run_demo(model_path: str, query: str, llm_model: str = DEFAULT_MODEL,
             max_steps: int = 8):
    print("=" * 70)
    print("MADISON-RL REAL-LLM DEMO")
    print("=" * 70)
    print(f"Query: {query}\n")

    ppo = PPO.load(model_path)
    query_emb = text_to_embedding(query)

    findings: List[str] = []
    used_mask = np.zeros(NUM_SPECIALISTS, dtype=np.float32)
    partial_quality = 0.0
    cost_used = 0.0
    last_conf = 0.0
    last_gain = 0.0

    for step in range(max_steps):
        obs = build_obs(
            query_emb, partial_quality, step, max_steps, cost_used,
            used_mask, last_conf, last_gain,
        )
        action, _ = ppo.predict(obs, deterministic=True)
        action = int(action)

        if action == FINISH_ACTION:
            print(f"\n[step {step}] ORCHESTRATOR DECISION: FINISH")
            break

        name = SPECIALIST_NAMES[action]
        print(f"\n[step {step}] ORCHESTRATOR DISPATCHES: {name}")
        prompt = SPECIALIST_PROMPTS[action].format(
            query=query,
            prior="\n".join(findings) if findings else "(none yet)",
        )
        print(f"  prompt: {prompt[:120]}...")
        response = ollama_complete(prompt, model=llm_model)
        if response is None:
            response = (
                f"[MOCK {name} response — Ollama unavailable. "
                "Replace with real LLM call to enable live demo.]"
            )
            print(f"  [falling back to mock specialist]")
        else:
            response = response[:400]
        print(f"  response: {response[:200]}...")
        findings.append(response)

        old_q = partial_quality
        partial_quality = estimate_quality(findings)
        last_gain = partial_quality - old_q
        last_conf = min(1.0, last_gain * 2 + 0.3)
        cost_used += 2.0
        used_mask[action] = 1.0

    print("\n" + "=" * 70)
    print(f"FINAL QUALITY: {partial_quality:.3f}")
    print(f"SPECIALISTS USED: {[SPECIALIST_NAMES[i] for i in range(NUM_SPECIALISTS) if used_mask[i] > 0.5]}")
    print(f"TOTAL FINDINGS: {len(findings)}")
    print("=" * 70)
    return {"findings": findings, "final_quality": partial_quality}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--query",
        type=str,
        default="Assess the impact of recent AI chip export restrictions on TSMC.",
    )
    parser.add_argument("--llm", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_demo(args.model, args.query, args.llm)
