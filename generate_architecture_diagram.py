"""Generate a proper architecture diagram for the technical report.

Creates experiments/results/figures/architecture_diagram.png using
matplotlib patches and annotations. Replaces the ASCII art in the report.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

# Color scheme
C_ENV    = "#E8EAF6"
C_ORCH   = "#BBDEFB"
C_SPEC   = "#C8E6C9"
C_MARL   = "#FFF9C4"
C_MEM    = "#F3E5F5"
C_REWARD = "#FFCCBC"
C_TEXT   = "#1a1a1a"
FONT     = {"fontsize": 9, "ha": "center", "va": "center", "color": C_TEXT, "fontweight": "bold"}
FONTSMALL = {"fontsize": 7.5, "ha": "center", "va": "center", "color": "#444"}

def box(x, y, w, h, color, label, sublabel=None):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor="#666", linewidth=1.2)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.15, label, **FONT)
        ax.text(x + w/2, y + h/2 - 0.2, sublabel, **FONTSMALL)
    else:
        ax.text(x + w/2, y + h/2, label, **FONT)

def arrow(x1, y1, x2, y2, label=None, color="#666"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my + 0.15, label, fontsize=7, color="#666", ha="center")

# Title
ax.text(6, 6.7, "Madison-RL: System Architecture", fontsize=14,
        ha="center", va="center", fontweight="bold", color="#1f3a78")

# Environment box (top)
box(3.5, 5.4, 5, 0.9, C_ENV, "IntelligenceTaskEnv (Gymnasium)",
    "task embedding (16-d) + difficulty + budget + threshold")

# Orchestrator
box(4, 3.8, 4, 0.9, C_ORCH, "PPO / DQN Orchestrator",
    "MlpPolicy [64,64]  •  Discrete(5) actions")

# Memory store
box(9.0, 3.8, 2.2, 0.9, C_MEM, "MemoryStore",
    "episodic recall (k-NN)")

# Specialists row
spec_names = ["Researcher\n(SEARCH)", "Analyst\n(ANALYSIS)", "Synthesizer\n(SYNTHESIS)", "Validator\n(FACTCHECK)"]
spec_x = [0.5, 3.0, 5.5, 8.0]
for i, (sx, name) in enumerate(zip(spec_x, spec_names)):
    box(sx, 1.8, 2.2, 0.9, C_SPEC, name)
    # MARL sub-label
    ax.text(sx + 1.1, 1.55, f"IPPO π{i+1}  effort∈{{S,M,D}}", fontsize=6.5,
            ha="center", color="#666", style="italic")

# FINISH action
box(10.5, 1.8, 1.2, 0.9, "#CFD8DC", "FINISH")

# Reward / quality update
box(3.5, 0.3, 5, 0.8, C_REWARD, "Shared Team Reward",
    "r = 0.5·Δquality − 0.02·cost + terminal ± intrinsic novelty")

# Arrows: env → orchestrator
arrow(6, 5.4, 6, 4.7, "obs (25-d)")

# Arrows: orchestrator → specialists
for sx in spec_x:
    arrow(6, 3.8, sx + 1.1, 2.7, color="#3366cc")
arrow(6, 3.8, 11.1, 2.7, color="#888")  # FINISH

# Arrows: orchestrator → memory
arrow(8.0, 4.25, 9.0, 4.25, "recall")
arrow(9.0, 4.0, 8.0, 4.0, "context")

# Arrows: specialists → reward
for sx in spec_x:
    arrow(sx + 1.1, 1.8, 6, 1.1, color="#44aa77")

# Arrows: reward → env (loop back)
arrow(6, 0.3, 2.5, 0.3, color="#cc3366")
ax.annotate("", xy=(2.5, 5.9), xytext=(2.5, 0.3),
            arrowprops=dict(arrowstyle="-|>", color="#cc3366", lw=1.5,
                           connectionstyle="arc3,rad=0.0"))
ax.text(1.8, 3.1, "next\nstep", fontsize=7, color="#cc3366", ha="center")

# RL method labels
ax.text(0.4, 4.35, "Category 2:\nPPO (Policy Grad)", fontsize=7,
        ha="left", color="#3366cc", fontweight="bold")
ax.text(0.4, 4.0, "Category 1:\nDQN (Value-Based)", fontsize=7,
        ha="left", color="#884488", fontweight="bold")
ax.text(0.4, 0.9, "Category 3:\nIPPO (Multi-Agent)", fontsize=7,
        ha="left", color="#44aa77", fontweight="bold")
ax.text(10.7, 0.9, "Cat 4: LinUCB\n+ Intrinsic", fontsize=7,
        ha="center", color="#aa8844", fontweight="bold")
ax.text(10.7, 0.5, "Cat 5: Transfer\nLearning", fontsize=7,
        ha="center", color="#339966", fontweight="bold")

# Legend-like note
ax.text(6, 6.3, "All 5 rubric RL categories integrated into a single hierarchical system",
        fontsize=8, ha="center", va="center", color="#666", style="italic")

plt.tight_layout()
out = "experiments/results/figures/architecture_diagram.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
