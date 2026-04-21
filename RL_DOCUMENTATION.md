# Reinforcement Learning Implementation Documentation

This document provides a detailed breakdown of the Reinforcement Learning (RL) approach used in the Madison-RL project, as required by the project deliverables.

---

## 1. Problem Formulation (MDP)

The task is modeled as a **Markov Decision Process (MDP)** where a central "Orchestrator" learns to coordinate a team of specialists.

### 1.1 State Space (Observation)
The observation vector has **25 dimensions**, providing the agent with a complete high-level view of the task and its own progress:

| Component | Dimensions | Description |
|-----------|------------|-------------|
| **Task Embedding** | 16 | A static semantic representation of the intelligence task. |
| **Current Quality** | 1 | The partial quality score achieved so far ($0.0$ to $1.0$). |
| **Normalized Step** | 1 | Percentage of the time budget used ($t / T_{max}$ ). |
| **Normalized Cost**| 1 | Total effort cost incurred so far. |
| **Usage Mask** | 4 | Binary flags indicating which specialists have already been used. |
| **Last Confidence**| 1 | The confidence score from the most recent specialist dispatch. |
| **Last Gain** | 1 | The quality improvement from the last step. |

### 1.2 Action Space
The model uses a **Discrete(5)** action space:
*   **0-3:** Dispatch a specific specialist (**Researcher, Analyst, Synthesizer, Validator**).
*   **4:** **FINISH** — Manually terminate the episode and submit the current result.

### 1.3 Reward Function
The reward is designed to handle the "sparse reward" problem while ensuring the agent remains efficient:
*   **Step-wise Shaping:** `+0.5 * quality_gain - 0.02 * cost` (encourages progress, penalizes waste).
*   **Terminal Reward:** 
    *   **Success:** `+10.0 * final_quality` (if quality ≥ threshold).
    *   **Failure:** `-2.0` (if finished below threshold).
    *   **Efficiency Bonus:** `+1.5 * (remaining_budget / total_budget)` if successful.

---

## 2. Implemented RL Algorithms

We implemented **six variants** across **five distinct RL categories** to demonstrate breadth:

### 2.1 Policy Gradient: PPO (Proximal Policy Optimization)
*   **Category:** Policy Gradient / "Full" RL.
*   **Architecture:** Multi-Layer Perceptron (MLP) with two hidden layers of 64 neurons each.
*   **Optimization:** Uses the clipped surrogate objective to ensure stable updates. This is the primary orchestrator for the project.

### 2.2 Value-Based: DQN (Deep Q-Network)
*   **Category:** Value-Based Learning.
*   **Implementation:** Trained against the same MDP as PPO. Uses a Replay Buffer to improve sample efficiency (reaching 95% success ~3x faster than PPO).

### 2.3 Multi-Agent RL: IPPO (Independent PPO)
*   **Category:** Cooperation & Competition.
*   **Logic:** The specialists are agents themselves. They learn to adjust their "Effort Level" (Shallow, Medium, Deep) using shared team rewards, co-adapting to the orchestrator's routing style.

### 2.4 Exploration: LinUCB & Intrinsic Motivation
*   **Category:** Exploration Strategies.
*   **LinUCB:** A contextual bandit that uses Upper Confidence Bounds to explore which specialist matches which task embedding.
*   **Intrinsic Motivation:** Augments the PPO reward with a **count-based novelty bonus** ($1 / \sqrt{n}$), forcing the agent to try rare specialist-task combinations.

### 2.5 Transfer Learning: Pre-training & Fine-tuning
*   **Category:** Meta-Learning.
*   **Approach:** The model was pre-trained on "Easy" tasks (1-2 required skills) and then fine-tuned on "Hard" tasks (3-4 required skills), demonstrating massive speedup in adaptation.

---

## 3. Training & Simulation Framework

### 3.1 Specialist Simulation
During training, we use a **Specialist Response Model** that calculates quality gain based on:
1.  **Capability Match:** The alignment between the task's required skills and the specialist's expertise.
2.  **Diminishing Returns:** Gains are higher early in the episode and harder to get as quality approaches 1.0.

### 3.2 Hyperparameters
*   **Learning Rate:** 3e-4 (PPO) / 1e-4 (DQN)
*   **Gamma (Discount):** 0.98
*   **Batch Size:** 64
*   **Timesteps:** 150,000 for convergence.
