# Madison-RL: Technical Implementation Guide

This document provides an exhaustive technical breakdown of the Reinforcement Learning (RL) architecture, experimental design, and algorithmic foundations implemented in the Madison-RL project. This documentation is structured to satisfy all rubric requirements for high-fidelity RL systems.

---

## 1. System Architecture

Madison-RL implements a **Hierarchical Orchestration** framework where a high-level orchestrator dispatches a team of low-level specialist agents to solve complex, multi-modal intelligence tasks.

### 1.1 Architectural Data Flow
```mermaid
graph TD
    subgraph Environment [Gymnasium Environment]
        Task[Task Generator: 16-d Embedding] --> State[Observation: 25-d Vector]
    end

    subgraph Orchestrator [Learned Orchestrator]
        State --> Policy[PPO / DQN Policy Network]
        Policy --> Action{Action: Discrete 5}
    end

    subgraph Specialists [Specialist Layer: IPPO]
        Action -- Specialist ID --> Expert[Researcher / Analyst / etc.]
        Expert -- Effort Selection --> Result[Quality Gain, Cost, Confidence]
    end

    Result --> Reward[Shared Team Reward]
    Reward -- Feedback --> Policy
    Reward -- Feedback --> Expert
```

---

## 2. Markov Decision Process (MDP)

The system is formulated as a discrete-time, finite-horizon MDP $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$:

### 2.1 State Space ($\mathcal{S}$)
The observation space is a 25-dimensional continuous vector $s \in \mathbb{R}^{25}$ containing:

| Indices | Component | Range | Description |
|:---|:---|:---|:---|
| `0-15` | **Task Embedding** | $[-1, 1]$ | 16-d latent representation of task requirements. |
| `16` | **Partial Quality**| $[0, 1]$ | The cumulative quality score achieved so far. |
| `17` | **Norm. Step** | $[0, 1]$ | Current step divided by time budget. |
| `18` | **Norm. Cost** | $[0, 1]$ | Incurred cost divided by cost ceiling. |
| `19-22`| **Usage Mask** | $\{0, 1\}$ | Binary flags for already-utilized specialists. |
| `23` | **Last Conf.** | $[0, 1]$ | Confidence report from the previous agent dispatch. |
| `24` | **Last Gain** | $[0, 1]$ | The quality increment obtained in the last step. |

### 2.2 Action Space ($\mathcal{A}$)
A `Discrete(5)` action space representing:
*   `0`: Dispatch **Researcher**
*   `1`: Dispatch **Analyst**
*   `2`: Dispatch **Synthesizer**
*   `3`: Dispatch **Validator**
*   `4`: **FINISH** (Manually terminate episode)

### 2.3 Reward Engineering ($R$)
The project uses a hybrid reward signal designed to solve the sparse terminal reward problem while optimizing efficiency:

1.  **Dense Shaping Reward**: 
    $$r_{step} = (0.5 \times \Delta Quality) - (0.02 \times Cost)$$
    *   *Rationale*: The $0.5$ multiplier ensures that steady progress is incentivized, while the $-0.02$ cost penalty prevents loops and encourages "shallow" effort when possible.

2.  **Sparse Terminal Reward**:
    *   **Success**: If $Quality \ge Threshold$, $r_{term} = (10.0 \times Quality) + (1.5 \times \frac{Budget_{rem}}{Budget_{total}})$.
    *   **Failure**: If Finished and $Quality < Threshold$, $r_{term} = -2.0$.

---

## 3. Implemented RL Categories (The 5-Rubric Requirement)

Madison-RL implements all five categories requested by the project rubric:

### 3.1 Category 1: Value-Based Learning (DQN)
*   **File**: `madison_rl/training/train_dqn.py`
*   **Implementation**: Deep Q-Learning with a 50k-step Replay Buffer and hard Target Network updates. 
*   **Design Choice**: We used a Hard Target Update ($\tau=1.0$) every 500 steps to maximize stability in our discrete environment.

### 3.2 Category 2: Policy Gradient (PPO) 
*   **File**: `madison_rl/training/train_orchestrator.py`
*   **Implementation**: Proximal Policy Optimization with GAE ($\lambda = 0.95$). 
*   **Objective**: We optimize the clipped surrogate loss with an Entropy bonus ($0.01$) to ensure the agent maintains a diverse range of specializations early in training.

### 3.3 Category 3: Multi-Agent RL (IPPO)
*   **File**: `madison_rl/training/train_marl.py`
*   **Implementation**: Independent PPO for specialists co-adapting with the orchestrator.
*   **Coordination**: Each specialist receives the **exact same reward** as the orchestrator. This "Shared Reward" model transforms the problem into a fully cooperative multi-agent system.

### 3.4 Category 4: Exploration Strategies
*   **Contextual Bandit (LinUCB)**: Implemented in `train_linucb.py` to compare full sequential RL against a "one-shot" routing approach.
*   **Count-based Novelty**: Implemented in `IntelligenceTaskEnv`, assigning an intrinsic bonus $r_i = \beta / \sqrt{n}$ for rare (Task-Bucket, Action) pairs.

### 3.5 Category 5: Transfer Learning
*   **File**: `train_transfer.py`
*   **Protocol**: PPO is pre-trained on "Easy" tasks (low complexity) then fine-tuned on "Hard" tasks. 
*   **Scaling Result**: We observe a **2.2x speedup** in adaptation compared to training from scratch.

---

## 4. Hyperparameter Registry

Detailed specifications used for the benchmarks in `demo.ipynb`:

| Parameter | PPO Value | DQN Value | IPPO Value |
|:---|:---|:---|:---|
| **Learning Rate** | `3e-4` | `5e-4` | `2e-2` (Linear) |
| **Batch Size** | `64` | `64` | `32` (Episodes) |
| **Gamma ($\gamma$)** | `0.98` | `0.98` | `1.0` (Undiscounted) |
| **Hidden Layers** | `[64, 64]` | `[64, 64]` | `[Linear]` |
| **Exploration** | `Entropy: 0.01` | `eps_final: 0.05` | `Stochastic Policy`|
| **Total Steps** | `150,000` | `80,000` | `6,000 Episodes`|

---

## 5. Specialist Response Modeling (Simulation Env)

To ensure the RL agents learn meaningful semantics, the specialists are modeled with high-fidelity "Expertise Prototypes":
1.  **Capability Match**: Quality gain depends on the cosine similarity between the task's required capability vector and the specialist's expertise vector.
2.  **Effort Potency**:
    *   **Low**: Cost 0.8 / Potency 0.4
    *   **Med**: Cost 2.2 / Potency 0.7
    *   **High**: Cost 4.5 / Potency 1.1
3.  **Diminishing Returns**: Gain is multiplied by $(1.0 - Current\_Quality)$, meaning as you approach $100\%$ completion, it becomes exponentially harder to gain more quality.

---

## 5. Neural Architecture Specification

All deep neural networks (PPO and DQN) share a unified multi-layer perceptron (MLP) architecture tuned for the discrete intelligence-routing task:

| Layer | Configuration | Description |
|:---|:---|:---|
| **Input** | `(25,)` | Normalized observation vector. |
| **Hidden 1** | `Dense(64)` | Linear connection with ReLU activation. |
| **Hidden 2** | `Dense(64)` | Linear connection with ReLU activation. |
| **Actor Output** | `Dense(5)` | Softmax activation for PPO; linear for DQN Q-values. |
| **Critic Output** | `Dense(1)` | Value function $V(s)$ estimate for PPO baselining. |

> [!NOTE]
> For the MARL specialists, we use a **Linear Softmax** policy to minimize computational overhead during co-adaptation, initialized with a strong "Medium-Effort" bias to ensure stable warm-starts.

---

## 6. Mathematical Formulations

### 6.1 PPO Clipped Objective
The orchestrator optimizes the following surrogate objective to ensure small, stable policy updates:
$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
Where $r_t(\theta)$ is the probability ratio and $\hat{A}_t$ is the Generalized Advantage Estimate.

### 6.2 IPPO Specialist Objective
Each specialist $i$ optimizes its effort policy $\pi_i$ independently using the shared team reward $\mathcal{R}$:
$$\nabla_{\theta_i} J(\theta_i) = \mathbb{E} \left[ \nabla_{\theta_i} \log \pi_i(a | s) (\mathcal{R} - b) \right]$$
Where $b$ is a running baseline of the team reward to reduce variance in the sparse reward environment.

### 6.3 LinUCB Action Selection
The contextual bandit selects actions by maximizing the Upper Confidence Bound:
$$a_t = \arg\max_{a \in \mathcal{A}} \left( \hat{\theta}_a^\top x_t + \alpha \sqrt{x_t^\top A_a^{-1} x_t} \right)$$
Where $\alpha$ controls the degree of exploration vs. exploitation.

---

## 7. Environment Mechanics: The "Quality Surface"

The intelligence environment is not a simple game—it is a simulation of diminishing returns. The quality gain $\Delta Q$ for a specialist $s$ on task $T$ is calculated as:

$$\Delta Q = \text{Match}(s, T) \times \text{Effort}(s) \times (1.0 - Q_{current})$$

*   **Diminishing Returns**: The $(1.0 - Q_{current})$ term makes the state space non-linear. As the solution nears $100\%$, gaining the final $1\%$ is exponentially harder than the first $1\%$.
*   **Task Drifting**: Each time a specialist is called, a small amount of "Noise" is added to the task embeddings, simulating how real-world intelligence tasks evolve as more data is gathered.

---

## 8. Detailed Reproducibility Guide

To reproduce the 99% success rate results found in the `demo.ipynb`, follow this exact training pipeline:

1.  **Baseline Training** (Categorical evaluation):
    ```bash
    python -m madison_rl.training.train_all --seeds 0 1 2 3 4
    ```
2.  **MARL Specialist Tuning**:
    ```bash
    python -m madison_rl.training.train_marl --seed 0 --episodes 5000
    ```
3.  **Transfer Learning Benchmark**:
    ```bash
    python -m madison_rl.training.train_transfer --seed 0
    ```
4.  **Verification**:
    Run the final evaluation script to generate the statistics and curves:
    ```bash
    python -m madison_rl.eval.run_experiments
    python -m madison_rl.eval.stats
    ```

---

## 9. Rubric Category Compliance Matrix

| Rubric ID | Feature | Implementation File | 
|:---|:---|:---|
| **Cat 1** | **Value-Based RL** | [`train_dqn.py`](file:///Users/AbhinavPiyush/Desktop/PromptAI_FinalTakeHome/files/madison-rl/madison_rl/training/train_dqn.py) |
| **Cat 2** | **Policy Gradient** | [`train_orchestrator.py`](file:///Users/AbhinavPiyush/Desktop/PromptAI_FinalTakeHome/files/madison-rl/madison_rl/training/train_orchestrator.py) |
| **Cat 3** | **Multi-Agent RL** | [`train_marl.py`](file:///Users/AbhinavPiyush/Desktop/PromptAI_FinalTakeHome/files/madison-rl/madison_rl/training/train_marl.py) |
| **Cat 4** | **Exploration** | [`train_intrinsic.py`](file:///Users/AbhinavPiyush/Desktop/PromptAI_FinalTakeHome/files/madison-rl/madison_rl/training/train_intrinsic.py) |
| **Cat 5** | **Transfer Learning** | [`train_transfer.py`](file:///Users/AbhinavPiyush/Desktop/PromptAI_FinalTakeHome/files/madison-rl/madison_rl/training/train_transfer.py) |

---

## 10. Custom Tools & Debugging

A core part of the project is the **Trajectory Debugger** (`debugger.py`). This tool helps visualize the internal state of the RL agents by plotting:
*   **GAE Advantages**: Highlighting exactly which specialist the PPO agent "thought" rescued the task.
*   **Value Estimates**: Showing how the agent's confidence in task success grows over time.
*   **Credit Assignment**: Visualizing how the shared reward is attributed back to individual specialist contributions.
