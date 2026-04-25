"""
UMBRA Pictorial Analysis — generates 6 charts from local log files.
Run locally:  python analysis.py
Run in Colab: %run analysis.py
"""

import json, math, collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path("logs/reward_graphs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DARK   = "#0d1117"
PANEL  = "#161b22"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
AMBER  = "#d29922"
RED    = "#f85149"
WHITE  = "#e6edf3"
GREY   = "#8b949e"

plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GREY,
    "axes.labelcolor":   WHITE,
    "xtick.color":       WHITE,
    "ytick.color":       WHITE,
    "text.color":        WHITE,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.8,
    "font.family":       "monospace",
})

# ── Load data ─────────────────────────────────────────────────────────────────
episodes = []
with open("logs/episodes.jsonl") as f:
    for line in f:
        episodes.append(json.loads(line.strip()))

rollouts = []
with open("logs/rollout_samples.jsonl") as f:
    for line in f:
        rollouts.append(json.loads(line.strip()))

with open("results.json") as f:
    results = json.load(f)

# ── Chart 1: Reward progression over episodes (rollout checkpoints) ───────────
def chart_reward_progression():
    eps    = [r["ep"]           for r in rollouts]
    rews   = [r["total_reward"] for r in rollouts]

    # EMA smoothing
    alpha, ema, emas = 0.4, rews[0], []
    for r in rews:
        ema = alpha * r + (1 - alpha) * ema
        emas.append(ema)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK)

    ax.fill_between(eps, rews, alpha=0.18, color=ACCENT)
    ax.plot(eps, rews,  "o--", color=ACCENT, alpha=0.55, linewidth=1.2,
            markersize=6, label="Episode reward")
    ax.plot(eps, emas,  "-",   color=GREEN,  linewidth=2.5,
            label="EMA trend (α=0.4)")

    # Stage boundary annotations
    for x, label, col in [(167, "Stage 2", AMBER), (333, "Stage 3", RED)]:
        ax.axvline(x, color=col, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(x + 5, max(rews) * 0.92, label, color=col, fontsize=9)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Reward", fontsize=11)
    ax.set_title("Reward Progression Across 500 Episodes", fontsize=13,
                 fontweight="bold", color=WHITE, pad=12)
    ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GREY)
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = OUT_DIR / "1_reward_progression.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Chart 2: Performance metrics by difficulty (Easy / Medium / Hard) ─────────
def chart_metrics_by_difficulty():
    metrics = [
        "Sycophancy rate",
        "Deception catch rate",
        "Calibration score",
        "Coalition detection rate",
        "Suboptimal action rate",
    ]
    difficulties = ["easy", "medium", "hard"]
    colors = [ACCENT, GREEN, AMBER]

    x      = np.arange(len(metrics))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(DARK)

    for i, (diff, col) in enumerate(zip(difficulties, colors)):
        vals = [results[diff][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=diff.capitalize(),
                      color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color=WHITE)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace(" ", "\n") for m in metrics], fontsize=9)
    ax.set_ylabel("Score (0–1)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Performance Metrics by Difficulty Level", fontsize=13,
                 fontweight="bold", color=WHITE, pad=12)
    ax.legend(facecolor=PANEL, edgecolor=GREY)
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = OUT_DIR / "2_metrics_by_difficulty.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Chart 3: Reward distribution by curriculum stage ──────────────────────────
def chart_reward_by_stage():
    by_stage = collections.defaultdict(list)
    for ep in episodes:
        by_stage[ep["stage"]].append(ep["total_reward"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.patch.set_facecolor(DARK)
    stage_colors = {1: ACCENT, 2: AMBER, 3: RED}
    stage_names  = {1: "Stage 1 — Agreeable NPCs",
                    2: "Stage 2 — Coalition NPCs",
                    3: "Stage 3 — Manipulator NPCs"}

    for ax, stage in zip(axes, [1, 2, 3]):
        data = by_stage[stage]
        col  = stage_colors[stage]
        ax.hist(data, bins=18, color=col, alpha=0.8, edgecolor=DARK)
        ax.axvline(np.mean(data), color=WHITE, linestyle="--", linewidth=1.5,
                   label=f"Mean={np.mean(data):.2f}")
        ax.set_title(stage_names[stage], fontsize=10, color=WHITE, pad=8)
        ax.set_xlabel("Episode Reward", fontsize=9)
        ax.set_ylabel("Count" if stage == 1 else "", fontsize=9)
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GREY)
        ax.grid(True, axis="y")

    fig.suptitle("Reward Distribution by Curriculum Stage", fontsize=13,
                 fontweight="bold", color=WHITE, y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "3_reward_by_stage.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Chart 4: NPC encounter frequency ──────────────────────────────────────────
def chart_npc_frequency():
    npc_counts = collections.Counter()
    for ep in episodes:
        for npc in ep["active_npcs"]:
            npc_counts[npc] += 1

    npcs   = list(npc_counts.keys())
    counts = [npc_counts[n] for n in npcs]
    cols   = [ACCENT, GREEN, AMBER, RED, "#bc8cff", "#ff7b72"][:len(npcs)]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK)
    bars = ax.barh(npcs, counts, color=cols, alpha=0.85)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(c), va="center", fontsize=9, color=WHITE)

    ax.set_xlabel("Episodes", fontsize=11)
    ax.set_title("NPC Encounter Frequency (538 episodes)", fontsize=13,
                 fontweight="bold", color=WHITE, pad=12)
    ax.grid(True, axis="x")
    plt.tight_layout()
    path = OUT_DIR / "4_npc_frequency.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Chart 5: Action diversity (unique actions per rollout episode) ─────────────
def chart_action_diversity():
    eps      = [r["ep"]                      for r in rollouts]
    n_unique = [len(set(r["actions"]))        for r in rollouts]
    n_total  = [len(r["actions"])             for r in rollouts]
    entropy  = []
    for r in rollouts:
        acts = r["actions"]
        cnt  = collections.Counter(acts)
        tot  = len(acts)
        h    = -sum((v / tot) * math.log2(v / tot) for v in cnt.values() if v > 0)
        entropy.append(round(h, 3))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor(DARK)

    ax1.bar(eps, n_unique, width=25, color=GREEN,  alpha=0.8, label="Unique actions")
    ax1.bar(eps, n_total,  width=25, color=ACCENT, alpha=0.35, label="Total actions")
    ax1.set_ylabel("Action count", fontsize=10)
    ax1.set_title("Action Diversity per Checkpoint Episode", fontsize=13,
                  fontweight="bold", color=WHITE, pad=12)
    ax1.legend(facecolor=PANEL, edgecolor=GREY)
    ax1.grid(True, axis="y")

    ax2.plot(eps, entropy, "o-", color=AMBER, linewidth=2, markersize=8)
    ax2.axhline(1.0, color=GREEN,  linestyle="--", linewidth=1, alpha=0.6, label="Healthy min (1.0)")
    ax2.axhline(0.3, color=RED,    linestyle="--", linewidth=1, alpha=0.6, label="Collapse threshold (0.3)")
    ax2.fill_between(eps, entropy, alpha=0.15, color=AMBER)
    ax2.set_xlabel("Episode", fontsize=10)
    ax2.set_ylabel("Shannon Entropy (bits)", fontsize=10)
    ax2.set_title("Action Entropy Over Training", fontsize=11,
                  fontweight="bold", color=WHITE, pad=8)
    ax2.legend(facecolor=PANEL, edgecolor=GREY, fontsize=8)
    ax2.grid(True, axis="y")

    plt.tight_layout()
    path = OUT_DIR / "5_action_diversity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Chart 6: Overall UMBRA scorecard radar ────────────────────────────────────
def chart_scorecard_radar():
    labels = [
        "Deception\nResistance", "Calibration", "Coalition\nDetection",
        "Low\nSycophancy", "Low Suboptimal\nActions", "Adversarial\nRobustness"
    ]
    # Average across easy/medium/hard where applicable
    def avg(key):
        return sum(results[d].get(key, 0) for d in ["easy","medium","hard"]) / 3

    values = [
        avg("Deception catch rate"),
        avg("Calibration score"),
        avg("Coalition detection rate"),
        1 - avg("Sycophancy rate"),          # invert: lower syco = better
        1 - avg("Suboptimal action rate"),   # invert: lower suboptimal = better
        results["adversarial_robustness_score"],
    ]

    N     = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(PANEL)

    ax.plot(angles, values, "o-", color=ACCENT, linewidth=2.5)
    ax.fill(angles, values, color=ACCENT, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=WHITE)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8, color=GREY)
    ax.grid(color=GREY, alpha=0.4)
    ax.spines["polar"].set_color(GREY)

    # Annotate each vertex
    for angle, val, label in zip(angles[:-1], values[:-1], labels):
        ax.annotate(f"{val:.2f}",
                    xy=(angle, val),
                    xytext=(angle, val + 0.08),
                    ha="center", fontsize=9, color=WHITE, fontweight="bold")

    ax.set_title("UMBRA Overall Scorecard", fontsize=14,
                 fontweight="bold", color=WHITE, pad=20)
    plt.tight_layout()
    path = OUT_DIR / "6_scorecard_radar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating UMBRA pictorial analysis…\n")
    chart_reward_progression()
    chart_metrics_by_difficulty()
    chart_reward_by_stage()
    chart_npc_frequency()
    chart_action_diversity()
    chart_scorecard_radar()
    print(f"\n✅ All 6 charts saved to {OUT_DIR}/")
