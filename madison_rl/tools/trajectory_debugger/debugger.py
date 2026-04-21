"""Trajectory Replay & Credit Assignment Debugger.

A custom tool for inspecting and diagnosing what a trained orchestrator
policy is doing episode-by-episode. This is useful because:

    - PPO policies are black boxes; you want to see which decisions
      were pivotal (high advantage) vs routine.
    - When MARL specialists are added, you want to see which team member
      contributed most to the episode outcome (credit assignment).
    - When policies fail, you want a replay you can step through.

Features:
    1. Rolls out N episodes with a trained PPO orchestrator.
    2. For each step, records:
         - observation vector
         - action chosen + its policy probability
         - value function estimate V(s)
         - immediate reward
         - quality gain produced by the specialist
    3. Computes GAE-based advantages per step.
    4. Flags "pivotal" decisions as those with |advantage| > mean + std.
    5. Ranks specialists by total contributed quality-gain × action-prob.
    6. Exports a step-by-step HTML report.

CLI usage:
    python -m madison_rl.tools.trajectory_debugger.debugger \
        --model experiments/results/ppo_orchestrator_seed0.zip \
        --episodes 5 \
        --output experiments/results/trajectory_report.html
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from stable_baselines3 import PPO

from madison_rl.env import (
    FINISH_ACTION,
    IntelligenceTaskEnv,
    NUM_SPECIALISTS,
    SPECIALIST_NAMES,
)


def compute_gae(rewards: List[float], values: List[float], gamma: float = 0.98,
                lam: float = 0.95) -> List[float]:
    """Standard Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = [0.0] * T
    last_gae = 0.0
    # bootstrap value is 0 at terminal
    next_value = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    return advantages


def rollout_with_policy_info(model: PPO, env: IntelligenceTaskEnv,
                             seed: int) -> Dict:
    obs, info = env.reset(seed=seed)
    steps: List[Dict] = []
    rewards: List[float] = []
    values: List[float] = []
    done = False
    while not done:
        obs_t = torch.as_tensor(obs).unsqueeze(0).float()
        with torch.no_grad():
            # Get distribution + value from the policy
            dist = model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.cpu().numpy()[0]
            value = float(model.policy.predict_values(obs_t).cpu().numpy()[0][0])
            action = int(np.argmax(probs))  # deterministic
        new_obs, r, term, trunc, info = env.step(action)
        steps.append(
            {
                "t": len(steps),
                "action": action,
                "action_name": SPECIALIST_NAMES[action] if action < NUM_SPECIALISTS else "FINISH",
                "action_probs": probs.tolist(),
                "picked_prob": float(probs[action]),
                "value": value,
                "reward": float(r),
                "partial_quality": info.get("partial_quality", 0.0),
                "last_quality_gain": info.get("quality_gain", 0.0) if action < NUM_SPECIALISTS else 0.0,
            }
        )
        rewards.append(float(r))
        values.append(value)
        obs = new_obs
        done = term or trunc

    advantages = compute_gae(rewards, values)
    for s, a in zip(steps, advantages):
        s["advantage"] = float(a)

    # Pivotal decision flagging
    adv_arr = np.array(advantages)
    if len(adv_arr) > 1:
        thr = adv_arr.mean() + adv_arr.std()
        for s in steps:
            s["pivotal"] = s["advantage"] > thr
    else:
        for s in steps:
            s["pivotal"] = False

    # Specialist credit = quality gain produced × picked prob
    credit = {name: 0.0 for name in SPECIALIST_NAMES}
    for s in steps:
        if s["action"] < NUM_SPECIALISTS:
            credit[SPECIALIST_NAMES[s["action"]]] += s["last_quality_gain"] * s["picked_prob"]

    return {
        "steps": steps,
        "final_quality": float(info.get("partial_quality", 0.0)),
        "threshold": float(info.get("threshold", 0.0)),
        "success": float(info.get("partial_quality", 0.0)) >= float(info.get("threshold", 0.0)),
        "total_reward": float(sum(rewards)),
        "credit": credit,
        "task_info": {
            "difficulty": info.get("difficulty"),
            "required_caps": info.get("required_caps"),
        },
    }


def render_html(episodes: List[Dict], out_path: Path) -> None:
    css = """
    .report-container { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 30px auto; padding: 0 20px; }
    .report-container h1 { border-bottom: 2px solid #3366cc; padding-bottom: 8px; }
    .report-container h2 { color: #3366cc; margin-top: 40px; }
    .report-container .episode { background: rgba(128, 128, 128, 0.05); border-radius: 8px; padding: 16px 20px; margin-bottom: 24px; border: 1px solid rgba(128, 128, 128, 0.2); }
    .report-container .episode.success { border-left: 5px solid #2a9; }
    .report-container .episode.fail { border-left: 5px solid #c44; }
    .report-container table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }
    .report-container th, .report-container td { border: 1px solid rgba(128, 128, 128, 0.3); padding: 6px 10px; text-align: left; }
    .report-container th { background: rgba(128, 128, 128, 0.1); }
    .report-container tr.pivotal { background: rgba(255, 243, 205, 0.2); font-weight: bold; }
    .report-container .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .report-container .badge.success { background: #d4edda; color: #155724; }
    .report-container .badge.fail { background: #f8d7da; color: #721c24; }
    .report-container .credit { background: rgba(128, 128, 128, 0.1); padding: 10px; border-radius: 4px; margin-top: 10px; }
    .report-container .credit-bar { height: 18px; background: #3366cc; display: inline-block; vertical-align: middle; margin-right: 6px; border-radius: 2px; }
    """
    parts = [
        "<html><head><meta charset='utf-8'><title>Madison-RL Trajectory Report</title>",
        f"<style>{css}</style></head><body><div class='report-container'>",
        "<h1>Madison-RL — Trajectory Replay & Credit Assignment Report</h1>",
        f"<p>Rolled out <b>{len(episodes)}</b> episodes with a trained PPO orchestrator. "
        "Pivotal decisions (advantage &gt; mean+σ) are highlighted. "
        "Credit assignment shows specialist contribution to final task quality.</p>",
    ]

    for i, ep in enumerate(episodes):
        cls = "success" if ep["success"] else "fail"
        badge = "<span class='badge success'>SUCCESS</span>" if ep["success"] else "<span class='badge fail'>FAIL</span>"
        parts.append(f"<div class='episode {cls}'>")
        parts.append(f"<h2>Episode {i + 1} &nbsp; {badge}</h2>")
        req = ep["task_info"]["required_caps"] or []
        req_str = ", ".join(
            f"{SPECIALIST_NAMES[j]}:{req[j]:.2f}" for j in range(len(req)) if req[j] > 0.1
        )
        parts.append(
            f"<p>Difficulty: <b>{ep['task_info']['difficulty']:.2f}</b> &nbsp; "
            f"Threshold: <b>{ep['threshold']:.2f}</b> &nbsp; "
            f"Final quality: <b>{ep['final_quality']:.3f}</b> &nbsp; "
            f"Total reward: <b>{ep['total_reward']:+.2f}</b><br>"
            f"Required capabilities: {html.escape(req_str)}</p>"
        )

        # Step table
        parts.append("<table><tr><th>t</th><th>action</th><th>π(a)</th>"
                     "<th>V(s)</th><th>reward</th><th>advantage</th>"
                     "<th>quality</th><th>pivotal</th></tr>")
        for s in ep["steps"]:
            tr_class = "pivotal" if s["pivotal"] else ""
            parts.append(
                f"<tr class='{tr_class}'>"
                f"<td>{s['t']}</td>"
                f"<td>{html.escape(s['action_name'])}</td>"
                f"<td>{s['picked_prob']:.2f}</td>"
                f"<td>{s['value']:+.2f}</td>"
                f"<td>{s['reward']:+.3f}</td>"
                f"<td>{s['advantage']:+.3f}</td>"
                f"<td>{s['partial_quality']:.2f}</td>"
                f"<td>{'⭐' if s['pivotal'] else ''}</td>"
                f"</tr>"
            )
        parts.append("</table>")

        # Credit assignment
        credit = ep["credit"]
        max_c = max(credit.values()) if credit and max(credit.values()) > 0 else 1.0
        parts.append("<div class='credit'><b>Specialist credit (quality × action prob):</b><br>")
        for name, c in credit.items():
            w = int(300 * max(c, 0) / max_c)
            parts.append(
                f"{html.escape(name):<12} "
                f"<span class='credit-bar' style='width:{w}px'></span> "
                f"{c:+.3f}<br>"
            )
        parts.append("</div>")
        parts.append("</div>")

    parts.append("</div></body></html>")
    out_path.write_text("".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/trajectory_report.html",
    )
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = IntelligenceTaskEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    episodes = []
    for _ in range(args.episodes):
        ep = rollout_with_policy_info(model, env, int(rng.integers(0, 10_000_000)))
        episodes.append(ep)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_html(episodes, out)
    print(f"Wrote {out}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Success rate: {np.mean([e['success'] for e in episodes]):.1%}")


if __name__ == "__main__":
    main()
