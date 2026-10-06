"""Publication-grade 4-panel comparison plot for the 9-Run PSRL & Sparse SGD Study.

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

# Paired color scheme: solid = WarmStart, dashed = NoWarmStart
STYLE_MAP = {
    "PSRL-a15-WarmStart": {"color": "#17becf", "ls": "-", "lw": 2.0, "label": "PSRL a=15 + WarmStart"},
    "PSRL-a15-NoWarmStart": {"color": "#17becf", "ls": "--", "lw": 1.6, "label": "PSRL a=15 (No WarmStart)"},
    "PSRL-a50-WarmStart": {"color": "#1f77b4", "ls": "-", "lw": 2.0, "label": "PSRL a=50 + WarmStart"},
    "PSRL-a50-NoWarmStart": {"color": "#1f77b4", "ls": "--", "lw": 1.6, "label": "PSRL a=50 (No WarmStart)"},
    "SGD4-WarmStart": {"color": "#2ca02c", "ls": "-", "lw": 2.0, "label": "SGD-Period=4 + WarmStart"},
    "SGD4-NoWarmStart": {"color": "#2ca02c", "ls": "--", "lw": 1.6, "label": "SGD-Period=4 (No WarmStart)"},
    "SGD8-WarmStart": {"color": "#ff7f0e", "ls": "-", "lw": 2.0, "label": "SGD-Period=8 + WarmStart"},
    "SGD8-NoWarmStart": {"color": "#ff7f0e", "ls": "--", "lw": 1.6, "label": "SGD-Period=8 (No WarmStart)"},
    "Winner-UninformedGauss": {"color": "#d62728", "ls": "-.", "lw": 2.2, "label": "Winner + Uninformed F0 (Gaussian)"},
}


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

    for short_name, run_data in data.items():
        style_info = STYLE_MAP.get(short_name, {"color": "black", "ls": "-", "lw": 1.5, "label": short_name})
        color = style_info["color"]
        ls = style_info["ls"]
        lw = style_info["lw"]
        label = style_info["label"]

        episodes = np.arange(1, len(run_data["returns"]) + 1)

        # 1. Moving average return
        ret_ma = moving_average(run_data["returns"], window=50)
        axes[0, 0].plot(episodes, ret_ma, label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)

        # 2. Cumulative return
        cum_ret = np.cumsum(run_data["returns"])
        axes[0, 1].plot(episodes, cum_ret, label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)

        # 3. Upright steps moving average
        upr_ma = moving_average(run_data["upright_steps"], window=50)
        axes[1, 0].plot(episodes, upr_ma, label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)

        # 4. Cumulative upright steps
        cum_upr = np.cumsum(run_data["upright_steps"])
        axes[1, 1].plot(episodes, cum_upr, label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)

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

    # Put legend on panel (a)
    axes[0, 0].legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        "9-Run Systematic PSRL, Sparse SGD, & Base Measure Study (2,500 Episodes, Seed = 42)\nCart-Pole Swing-Up Benchmark (Layer Normalization Enabled)",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image, bbox_inches="tight")
    plt.close()
    print(f"Saved publication comparison plot to: {output_image}")


def main():
    parser = argparse.ArgumentParser(description="Plot 9-Run PSRL & Sparse SGD Study")
    parser.add_argument("--json", type=str, default="results_rl/psrl_sgd_study_results.json", help="Path to JSON results")
    parser.add_argument("--out", type=str, default="results_rl/psrl_sgd_study_comparison.png", help="Output PNG path")
    args = parser.parse_args()

    plot_study(args.json, args.out)


if __name__ == "__main__":
    main()
