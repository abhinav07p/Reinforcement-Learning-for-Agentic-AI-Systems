# Madison-RL: Learned Orchestration for Multi-Agent Intelligence Gathering

> Reinforcement learning for agentic AI systems — an extension of the Humanitarians.AI
> **Madison Intelligence Agent** framework. A PPO-trained orchestrator learns to
> dispatch specialist agents (Researcher, Analyst, Synthesizer, Validator) to
> complete intelligence-gathering tasks, and specialists co-adapt their effort
> levels via independent PPO (IPPO) with shared team reward.

## Highlights

- **All 5 rubric RL categories implemented** (the assignment required only 2):
  - **Category 1 (Value-Based):** DQN orchestrator via Stable-Baselines3
  - **Category 2 (Policy Gradient):** PPO orchestrator via Stable-Baselines3 + REINFORCE-with-baseline for IPPO specialists
  - **Category 3 (Multi-Agent RL):** Independent PPO (IPPO) specialists with shared team reward
  - **Category 4 (Exploration Strategies):** LinUCB contextual bandit + count-based intrinsic motivation on PPO
  - **Category 5 (Meta-Learning / Transfer):** PPO pretrained on easy tasks then fine-tuned on hard tasks
- **Three agentic-system categories covered** in one coherent frame: orchestration (primary), workflow (task DAG), research (the domain)
- **Custom tool:** Trajectory Replay & Credit Assignment Debugger — HTML reports visualizing per-step action probabilities, value estimates, GAE advantages, and per-specialist credit
- **Rigorous evaluation:** 9 conditions evaluated with Welch's t-tests, bootstrap 95% CIs, Cohen's d
- **Real-LLM demo** via Ollama (optional) — the trained policy drives a team of local LLM specialists at inference time

## Results preview

Held-out evaluation over 200 episodes (seed 0 shown; see `stats_report.txt` for full across-seed stats):

| Condition | Rubric Category | Mean reward | Success rate |
|-----------|----------------|-------------|--------------|
| Random          | baseline  | −0.05 | 18.5% |
| RoundRobin      | baseline  | +0.51 | 25.0% |
| Oracle          | baseline  | +4.72 | 63.5% |
| **LinUCB**      | **Cat 4 (bandit)**  | +5.69 | 70.5% |
| **DQN**         | **Cat 1 (value-based)**  | +9.52 | 98.5% |
| **PPO**         | **Cat 2 (policy gradient)** | **+9.68** | **99.0%** |
| **PPO + MARL**  | **Cat 3 (multi-agent)**     | **+9.68** | **99.0%** |
| **PPO + Intrinsic** | **Cat 4 (novelty bonus)** | +9.59 | 98.5% |
| **PPO Transfer** | **Cat 5 (transfer learning)** | +9.66 | 99.0% |

**Key findings:**
- All four full-RL methods saturate at 98.5–99% on held-out tasks (they are statistically indistinguishable from each other)
- All four crush every non-RL baseline at p < 10⁻²² (Cohen's d ≥ 1.1)
- **DQN is ~3× more sample-efficient than PPO** but noisier — reaches 95% at ~8000 steps vs PPO's ~25000
- **Transfer learning gives a dramatic early-phase advantage**: fine-tuned model reaches 90% success at ~4000 steps on hard tasks; from-scratch never crosses 90% in the same budget
- **LinUCB beats the informed Oracle (70.5% vs 63.5%)** but is ~28 points behind full RL — contextual bandits learn task→specialist routing but cannot optimize multi-step decisions

![Success rates](experiments/results/figures/fig3_success_rates.png)

## Install

```bash
bash setup.sh
source .venv/bin/activate
```

Or manually:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### 🚀 Quick demo (no training required)
The fastest way to see the project is to open **`demo.ipynb`** in Jupyter or VS Code. Every cell has been pre-executed, so you can scroll through outputs without running anything. To re-run it yourself:

```bash
pip install jupyter
jupyter notebook demo.ipynb
```

The notebook walks through the environment, loads all six pre-trained RL models, runs a live episode, displays the 9-condition comparison table and significance tests, embeds all 7 figures, and runs the custom trajectory debugger — all from pre-computed artifacts already in the zip. No training needed.

### 1. Smoke test the environment
```bash
python -m tests.test_env
```

### 2. Train PPO orchestrators (one seed ≈ 60s on CPU)
```bash
python -m madison_rl.training.train_orchestrator --seed 0 --timesteps 60000
python -m madison_rl.training.train_orchestrator --seed 1 --timesteps 60000
# ... or
python -m madison_rl.training.train_all --seeds 0 1 2 3 4
```

**GPU vs CPU.** Training auto-detects CUDA and uses it if available
(`--device auto`, the default). You can force it with `--device cpu` or
`--device cuda`. However, for the small MLP policy used here (`[64, 64]`),
CPU is typically *as fast or faster* than GPU because of per-step tensor
transfer overhead. Stable-Baselines3 officially recommends CPU for
MlpPolicy. The MARL specialist training is pure NumPy and always runs
on CPU. The code will run on either without changes.

### 2b. Train all other RL methods (each ~30-90s on CPU)
```bash
# Value-Based RL (Category 1): DQN orchestrator
python -m madison_rl.training.train_dqn --seed 0 --timesteps 80000

# Exploration: contextual bandit (Category 4)
python -m madison_rl.agents.linucb --seed 0 --rounds 5000

# Exploration: intrinsic motivation on PPO (Category 4)
python -m madison_rl.training.train_intrinsic --seed 0 --timesteps 60000

# Meta/Transfer Learning (Category 5): pretrain on easy, fine-tune on hard
python -m madison_rl.training.train_transfer --seed 0

### 3. Train MARL specialists on top (requires a trained orchestrator)
```bash
python -m madison_rl.training.train_marl --seed 0 --episodes 2000
```

### 4. Run the full experimental suite
```bash
python -m madison_rl.eval.run_experiments --seeds 0 1 2 3 4
python -m madison_rl.eval.stats
python -m madison_rl.eval.plots
```

Outputs: `experiments/results/stats_report.txt`, `experiments/results/figures/`

### 5. Custom tool — trajectory debugger
```bash
python -m madison_rl.tools.trajectory_debugger.debugger \
    --model experiments/results/ppo_orchestrator_seed0.zip \
    --episodes 5 \
    --output experiments/results/trajectory_report.html
```

Open `trajectory_report.html` in a browser to see per-step action probabilities,
value estimates, advantages, and credit assignment.

### 6. Real-LLM demo (optional — requires Ollama)
```bash
# Install Ollama, pull a model, start the server:
#   ollama pull llama3
#   ollama serve
python -m madison_rl.demo.ollama_demo \
    --model experiments/results/ppo_orchestrator_seed0.zip \
    --query "Assess the impact of recent AI chip export restrictions on TSMC"
```

If Ollama isn't running, the script falls back to mock specialist outputs so
the demo still runs.

## Repo layout

```
madison-rl/
├── README.md
├── requirements.txt
├── setup.sh
├── demo.ipynb                     # ⭐ Pre-executed demo notebook
├── build_report_pdf.py            # Regenerates the PDF from figures
├── build_demo_notebook.py         # Regenerates demo.ipynb
├── madison_rl/
│   ├── env/
│   │   ├── tasks.py               # Synthetic intelligence task generator
│   │   ├── mock_llm.py            # Deterministic specialist simulators
│   │   └── intelligence_env.py    # Gymnasium environment
│   ├── agents/
│   │   └── baselines.py           # Random, RoundRobin, Oracle policies
│   ├── training/
│   │   ├── train_orchestrator.py  # PPO via Stable-Baselines3
│   │   ├── train_marl.py          # IPPO specialists (shared reward)
│   │   └── train_all.py           # Convenience driver
│   ├── eval/
│   │   ├── run_experiments.py     # 5-condition × N-seed evaluation
│   │   ├── stats.py               # Welch t-test + bootstrap CIs + Cohen's d
│   │   └── plots.py               # All figures
│   ├── tools/
│   │   └── trajectory_debugger/   # Custom tool
│   └── demo/
│       └── ollama_demo.py         # Real-LLM inference-time demo
├── experiments/
│   └── results/                   # Models, curves, CSVs, figures
├── tests/
│   └── test_env.py                # Environment smoke tests
└── report/
    └── technical_report.md        # Full writeup
```

## License

MIT — educational project.
