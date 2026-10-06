"""Publication-grade 4-panel comparison plot for the 9-Run DP-DQN Hyperparameter Study.

Generates:
(a) 50-Episode Moving Average Return
(b) Cumulative Return
(c) Upright Balancing Steps (50-ep moving avg)
(d) Cumulative Upright Balancing Steps
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


# Palette for 9 distinct configurations
COLORS = [
    "#1f77b4",  # 1. Baseline (Blue)
    "#d62728",  # 2. No-LN (Red)
    "#2ca02c",  # 3. Pure Upright (Green)
    "#ff7f0e",  # 4. Stochastic Sparks (Orange)
    "#9467bd",  # 5. Energy Coupled (Purple)
    "#8c564b",  # 6. Low Alpha (Brown)
    "#e377c2",  # 7. High Alpha (Pink)
    "#17becf",  # 8. Target Warmstart (Cyan)
    "#bcbd22",  # 9. Fast Target (Olive)
]

STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]


def moving_average(arr, window=50):
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    ma = ret / np.minimum(np.arange(1, len(arr) + 1), window)
    return ma


def plot_study(json_path: str, output_image: str):
    with open(json_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.20)

    for i, (key, run_data) in enumerate(data.items()):
        color = COLORS[i % len(COLORS)]
        style = STYLES[i % len(STYLES)]
        label = run_data["short_name"]
        episodes = np.arange(1, len(run_data["returns"]) + 1)

        # 1. Moving average return
        ret_ma = moving_average(run_data["returns"], window=50)
        axes[0, 0].plot(episodes, ret_ma, label=label, color=color, linestyle=style, linewidth=1.8, alpha=0.9)

        # 2. Cumulative return
        cum_ret = np.cumsum(run_data["returns"])
        axes[0, 1].plot(episodes, cum_ret, label=label, color=color, linestyle=style, linewidth=1.8, alpha=0.9)

        # 3. Upright steps moving average
        upr_ma = moving_average(run_data["upright_steps"], window=50)
        axes[1, 0].plot(episodes, upr_ma, label=label, color=color, linestyle=style, linewidth=1.8, alpha=0.9)

        # 4. Cumulative upright steps
        cum_upr = np.cumsum(run_data["upright_steps"])
        axes[1, 1].plot(episodes, cum_upr, label=label, color=color, linestyle=style, linewidth=1.8, alpha=0.9)

    # Panel styling
    axes[0, 0].set_title("(a) 50-Episode Moving Average Return", fontsize=13, fontweight="bold", pad=8)
    axes[0, 0].set_xlabel("Episode", fontsize=11)
    axes[0, 0].set_ylabel("Episodic Return", fontsize=11)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].axhline(0, color="gray", linestyle=":", linewidth=1.0)

    axes[0, 1].set_title("(b) Cumulative Return", fontsize=13, fontweight="bold", pad=8)
    axes[0, 1].set_xlabel("Episode", fontsize=11)
    axes[0, 1].set_ylabel("Cumulative Return", fontsize=11)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    axes[1, 0].set_title("(c) Upright Balancing Steps per Episode (50-Ep MA)", fontsize=13, fontweight="bold", pad=8)
    axes[1, 0].set_xlabel("Episode", fontsize=11)
    axes[1, 0].set_ylabel("Upright Balancing Steps", fontsize=11)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    axes[1, 1].set_title("(d) Cumulative Upright Balancing Steps", fontsize=13, fontweight="bold", pad=8)
    axes[1, 1].set_xlabel("Episode", fontsize=11)
    axes[1, 1].set_ylabel("Cumulative Upright Steps", fontsize=11)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    # Put legend on panel (a) or outside
    axes[0, 0].legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    suptitle = fig.suptitle(
        "9-Run Systematic DP-DQN Hyperparameter & Ablation Study (2,500 Episodes, Identical Seed = 42)\nCart-Pole Swing-Up Benchmark (arXiv:1703.07608 Section 7.2.2)",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    os.makedirs(os.path.dirname(output_image) or ".", exist_ok=True)
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure to: {output_image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="results_rl/study_9runs_results.json")
    parser.add_argument("--out_image", type=str, default="results_rl/study_9runs_comparison.png")
    args = parser.parse_args()
    plot_study(args.results_file, args.out_image)
