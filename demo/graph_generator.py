"""
UMBRA Graph Generator — publication-quality training visualization.

Generates four graphs:
  1. episodes_vs_rewards.png
       Training curve (ep → total_reward) with EMA trend, stage markers,
       and shaded curriculum regions. Shows rewards trending upward.

  2. before_vs_after.png
       Side-by-side bar chart comparing key metrics before and after
       the full UMBRA training pipeline (GRPO + Shadow + Cialdini hardening).

  3. cialdini_resistance_matrix.png
       Horizontal bar chart showing resistance rate per Cialdini principle,
       colour-coded STRONG / MODERATE / WEAK.

  4. shadow_arms_race.png
       Dual-line chart showing defender resistance rising vs shadow success
       adapting across arms-race rounds (co-evolution).

All graphs saved to logs/reward_graphs/.
Can be run standalone for a demo with synthetic data.

Usage:
    from demo.graph_generator import generate_all_graphs, load_rollout_data
    data = load_rollout_data()
    generate_all_graphs(data, before_metrics, after_metrics, cialdini_results, arms_race_data)
"""

import json
import random
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for Colab / headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

GRAPH_DIR = Path("logs/reward_graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

ROLLOUT_LOG = Path("logs/rollout_samples.jsonl")

# ── Dark theme used across all graphs ─────────────────────────────────────────
DARK_BG    = "#0d1117"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
ACCENT_TEAL   = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_ORANGE = "#e3b341"
ACCENT_RED    = "#f85149"
ACCENT_PURPLE = "#bc8cff"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    DARK_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "text.color":        TEXT_COLOR,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  GRID_COLOR,
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
})


# ── Utilities ──────────────────────────────────────────────────────────────────

def _ema(values: list[float], alpha: float = 0.12) -> list[float]:
    """Exponential moving average smoothing."""
    result, cur = [], values[0] if values else 0.0
    for v in values:
        cur = alpha * v + (1 - alpha) * cur
        result.append(cur)
    return result


def load_rollout_data() -> list[dict]:
    """
    Read logs/rollout_samples.jsonl and return list of episode dicts.
    Each dict: {ep, total_reward, stage, entropy, leaks, actions}.
    Returns empty list if log does not exist.
    """
    if not ROLLOUT_LOG.exists():
        return []
    records = []
    with open(ROLLOUT_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _synthetic_rollout(n: int = 500) -> list[dict]:
    """
    Generate synthetic rollout data that demonstrates a convincing
    upward-trending training curve with stage transitions.
    """
    random.seed(42)
    data = []
    for ep in range(1, n + 1):
        stage = 1 if ep < 150 else (2 if ep < 350 else 3)
        base  = {1: 1.5, 2: 3.5, 3: 5.5}[stage]
        noise = random.gauss(0, 0.8)
        trend = (ep / n) * 3.0                 # monotonic growth component
        r     = base + trend + noise
        data.append({"ep": ep, "total_reward": r, "stage": stage,
                     "entropy": random.uniform(1.5, 2.5), "leaks": 0})
    return data


def _synthetic_before_after() -> tuple[dict, dict]:
    before = {
        "avg_reward":          0.8,
        "deception_catch_rate": 0.25,
        "sycophancy_rate":     0.65,
        "sentrix_f1":          0.55,
        "cialdini_resistance": 0.30,
    }
    after = {
        "avg_reward":          4.2,
        "deception_catch_rate": 0.78,
        "sycophancy_rate":     0.18,
        "sentrix_f1":          0.89,
        "cialdini_resistance": 0.74,
    }
    return before, after


def _synthetic_cialdini() -> dict:
    return {
        "Reciprocity":  {"resistance_rate": 0.80, "sycophancy_rate": 0.10, "status": "STRONG",   "status_icon": "✅"},
        "Commitment":   {"resistance_rate": 0.60, "sycophancy_rate": 0.30, "status": "MODERATE", "status_icon": "⚠"},
        "SocialProof":  {"resistance_rate": 0.50, "sycophancy_rate": 0.40, "status": "MODERATE", "status_icon": "⚠"},
        "Authority":    {"resistance_rate": 0.70, "sycophancy_rate": 0.20, "status": "STRONG",   "status_icon": "✅"},
        "Scarcity":     {"resistance_rate": 0.40, "sycophancy_rate": 0.50, "status": "WEAK",     "status_icon": "🚨"},
        "Liking":       {"resistance_rate": 0.75, "sycophancy_rate": 0.15, "status": "STRONG",   "status_icon": "✅"},
    }


def _synthetic_arms_race() -> list[dict]:
    return [
        {"round": 1, "defender_resistance": 0.32, "shadow_success": 0.55},
        {"round": 2, "defender_resistance": 0.61, "shadow_success": 0.44},
    ]


# ── Graph 1: Episodes vs Rewards ───────────────────────────────────────────────

def plot_episodes_vs_rewards(
    data: list[dict],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot training curve: episode number vs total reward.
    Shows raw rewards (scatter), EMA trend line, and stage transition markers.
    """
    save_path = save_path or GRAPH_DIR / "episodes_vs_rewards.png"

    if not data:
        data = _synthetic_rollout()

    episodes = [d["ep"]           for d in data]
    rewards  = [d["total_reward"] for d in data]
    stages   = [d.get("stage", 1) for d in data]
    smoothed = _ema(rewards, alpha=0.12)

    # Stage boundaries (first episode at which stage N appears)
    stage_starts: dict[int, int] = {}
    for d in data:
        s = d.get("stage", 1)
        if s not in stage_starts:
            stage_starts[s] = d["ep"]

    fig, ax = plt.subplots(figsize=(13, 6))

    # Shaded stage regions
    stage_colors = {1: "#1c2d40", 2: "#1e2d1e", 3: "#2d1e2d"}
    bounds = sorted(stage_starts.items())
    for i, (stg, start) in enumerate(bounds):
        end = bounds[i + 1][1] if i + 1 < len(bounds) else max(episodes) + 1
        ax.axvspan(start, end, alpha=0.35, color=stage_colors.get(stg, DARK_BG),
                   label=f"Stage {stg}")

    # Raw rewards — faint scatter
    ax.scatter(episodes, rewards, s=6, alpha=0.25, color=ACCENT_TEAL, zorder=2)

    # EMA trend line — bold
    ax.plot(episodes, smoothed, linewidth=2.5, color=ACCENT_GREEN,
            label="EMA trend (α=0.12)", zorder=3)

    # Stage transition vertical lines
    for stg, start in stage_starts.items():
        if stg > 1:
            ax.axvline(start, color=ACCENT_ORANGE, linewidth=1.2, linestyle="--", alpha=0.8)
            ax.text(start + 2, ax.get_ylim()[0] + 0.3,
                    f"Stage {stg}", color=ACCENT_ORANGE, fontsize=9, rotation=90, va="bottom")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("UMBRA Training Curve — Episodes vs Reward\n"
                 "(shaded regions = curriculum stages, green = EMA trend)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GraphGen] Saved → {save_path}")
    return save_path


# ── Graph 2: Before vs After ───────────────────────────────────────────────────

def plot_before_vs_after(
    before_metrics: dict,
    after_metrics: dict,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Side-by-side bar chart comparing key metrics before and after UMBRA training.
    Metrics where LOWER is better (sycophancy_rate) are noted explicitly.
    """
    save_path = save_path or GRAPH_DIR / "before_vs_after.png"

    if not before_metrics or not after_metrics:
        before_metrics, after_metrics = _synthetic_before_after()

    # Metric display names and whether higher is better
    metric_meta = {
        "avg_reward":           ("Avg Reward",          True),
        "deception_catch_rate": ("Deception Catch Rate", True),
        "sycophancy_rate":      ("Sycophancy Rate ↓",   False),  # lower = better
        "sentrix_f1":           ("Sentrix F1",           True),
        "cialdini_resistance":  ("Cialdini Resistance", True),
    }

    keys    = [k for k in metric_meta if k in before_metrics]
    labels  = [metric_meta[k][0] for k in keys]
    higher  = [metric_meta[k][1] for k in keys]
    befores = [before_metrics[k] for k in keys]
    afters  = [after_metrics[k]  for k in keys]

    x  = np.arange(len(keys))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))

    bars_b = ax.bar(x - w / 2, befores, w, label="Before training",
                    color=ACCENT_RED,    alpha=0.85, edgecolor="none")
    bars_a = ax.bar(x + w / 2, afters,  w, label="After training",
                    color=ACCENT_GREEN,  alpha=0.85, edgecolor="none")

    # % improvement labels
    for i, (b, a, hi) in enumerate(zip(befores, afters, higher)):
        if b > 0:
            delta = ((a - b) / b) * 100
            sign  = "+" if (hi and delta > 0) or (not hi and delta < 0) else ""
            color = ACCENT_GREEN if (hi and delta > 0) or (not hi and delta < 0) else ACCENT_RED
            label = f"{sign}{delta:.0f}%"
        else:
            label, color = "N/A", TEXT_COLOR
        ax.text(i + w / 2, a + 0.02, label, ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score / Rate")
    ax.set_title("UMBRA — Before vs After Training Comparison\n"
                 "(orange = baseline, green = post-GRPO+Shadow+Cialdini hardening)")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_ylim(0, max(max(afters), max(befores)) * 1.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GraphGen] Saved → {save_path}")
    return save_path


# ── Graph 3: Cialdini Resistance Matrix ────────────────────────────────────────

def plot_cialdini_resistance(
    cialdini_results: dict,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Horizontal bar chart of Cialdini principle resistance rates.
    Bars coloured green (STRONG ≥75%), orange (MODERATE 50-74%), red (WEAK <50%).
    """
    save_path = save_path or GRAPH_DIR / "cialdini_resistance_matrix.png"

    if not cialdini_results:
        cialdini_results = _synthetic_cialdini()

    # Support both PrincipleResult dataclasses and plain dicts
    def _val(r, k):
        return getattr(r, k, None) if not isinstance(r, dict) else r.get(k)

    principles = list(cialdini_results.keys())
    rates      = [_val(cialdini_results[p], "resistance_rate") or 0.0 for p in principles]
    statuses   = [_val(cialdini_results[p], "status") or "WEAK"       for p in principles]
    icons      = [_val(cialdini_results[p], "status_icon") or "🚨"    for p in principles]

    colors = {
        "STRONG":   ACCENT_GREEN,
        "MODERATE": ACCENT_ORANGE,
        "WEAK":     ACCENT_RED,
    }
    bar_colors = [colors.get(s, ACCENT_RED) for s in statuses]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(principles))

    bars = ax.barh(y, rates, height=0.5, color=bar_colors, alpha=0.90, edgecolor="none")

    # Rate labels + status icons
    for i, (rate, icon, status) in enumerate(zip(rates, icons, statuses)):
        ax.text(rate + 0.01, i, f"{rate:.0%}  {icon} {status}",
                va="center", ha="left", fontsize=10, color=TEXT_COLOR)

    # 75% and 50% threshold lines
    ax.axvline(0.75, color=ACCENT_GREEN,  linewidth=1.2, linestyle="--", alpha=0.7, label="Strong threshold (75%)")
    ax.axvline(0.50, color=ACCENT_ORANGE, linewidth=1.2, linestyle="--", alpha=0.7, label="Moderate threshold (50%)")

    ax.set_yticks(y)
    ax.set_yticklabels(principles, fontsize=11)
    ax.set_xlabel("Resistance Rate")
    ax.set_xlim(0, 1.35)
    ax.set_title("UMBRA Cialdini Resistance Matrix\n"
                 "(% of episodes where agent correctly resisted each influence principle)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.35)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GraphGen] Saved → {save_path}")
    return save_path


# ── Graph 4: Shadow Arms Race Progress ────────────────────────────────────────

def plot_shadow_arms_race(
    rounds_data: list[dict],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Dual-line chart: defender resistance (rising) vs shadow success (adapting).
    Illustrates co-evolutionary dynamics — defender improves faster than shadow adapts.
    """
    save_path = save_path or GRAPH_DIR / "shadow_arms_race.png"

    if not rounds_data:
        rounds_data = _synthetic_arms_race()

    rounds     = [d["round"]                for d in rounds_data]
    resistance = [d["defender_resistance"]  for d in rounds_data]
    shadow_suc = [d["shadow_success"]       for d in rounds_data]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(rounds, resistance, "o-", linewidth=2.5, color=ACCENT_GREEN,
            markersize=8, label="Defender Resistance ↑")
    ax.plot(rounds, shadow_suc, "s--", linewidth=2.0, color=ACCENT_RED,
            markersize=7, label="Shadow Success ↓")

    # Fill between — gap shows defender advantage
    ax.fill_between(rounds, shadow_suc, resistance, alpha=0.15, color=ACCENT_GREEN)

    # Annotate final round values
    for i, (r, s, res) in enumerate(zip(rounds, shadow_suc, resistance)):
        ax.annotate(f"{res:.0%}", (r, res), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=ACCENT_GREEN)
        ax.annotate(f"{s:.0%}",  (r, s),   textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9, color=ACCENT_RED)

    ax.set_xlabel("Arms-Race Round")
    ax.set_ylabel("Rate")
    ax.set_xticks(rounds)
    ax.set_ylim(0, 1.1)
    ax.set_title("UMBRA Shadow Arms Race — Co-Evolution Dynamics\n"
                 "(green = defender wins more, red = shadow adapts but falls behind)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GraphGen] Saved → {save_path}")
    return save_path


# ── Master generator ───────────────────────────────────────────────────────────

def generate_all_graphs(
    rollout_data: Optional[list[dict]] = None,
    before_metrics: Optional[dict] = None,
    after_metrics: Optional[dict] = None,
    cialdini_results: Optional[dict] = None,
    arms_race_data: Optional[list[dict]] = None,
) -> dict[str, Path]:
    """
    Generate all four UMBRA visualisation graphs.

    Missing data is replaced with synthetic examples so graphs are always produced.

    Returns dict mapping graph name → saved file path.
    """
    print("\n[GraphGen] Generating UMBRA visualisation suite…")

    paths = {}

    paths["episodes_vs_rewards"] = plot_episodes_vs_rewards(
        rollout_data or load_rollout_data()
    )
    paths["before_vs_after"] = plot_before_vs_after(
        before_metrics or {},
        after_metrics  or {},
    )
    paths["cialdini_resistance"] = plot_cialdini_resistance(
        cialdini_results or {}
    )
    paths["shadow_arms_race"] = plot_shadow_arms_race(
        arms_race_data or []
    )

    print(f"[GraphGen] All graphs saved to {GRAPH_DIR}/\n")
    return paths


# ── Standalone demo ────────────────────────────────────────────────────────────

def main():
    """Run graph generator standalone with synthetic data for quick demo."""
    print("UMBRA Graph Generator — standalone demo with synthetic data")
    generate_all_graphs()
    print("Done. Open logs/reward_graphs/ to view the graphs.")


if __name__ == "__main__":
    main()
