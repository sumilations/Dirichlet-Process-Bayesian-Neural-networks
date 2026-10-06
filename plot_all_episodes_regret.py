"""Dedicated visualization of episode-by-episode regret across all algorithms and seeds on Deep Sea N=50."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_50 = "results_deepsea_50/deep_sea_50_results.json"
OUTPUT_PLOT = "results_deepsea_50/deep_sea_50_all_episodes_regret.png"


def main():
    with open(RESULTS_50, "r") as f:
        data_50 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {
        "DP-DQN (Structured BM)": "#1f77b4",       # Blue
        "DP-DQN (Uniform BM)": "#2ca02c",          # Green
        "BootDQN-RP (20 Heads)": "#ff7f0e",        # Orange
        "DQN-Dithering": "#d62728",                # Red
    }

    # ----------------------------------------------------
    # Panel 1: Full-Scale Regret across all 10,000 Episodes (with Threshold)
    # ----------------------------------------------------
    ax = axes[0]
    for algo, color in colors.items():
        runs = [r for r in data_50 if r["algo_name"] == algo]
        if not runs:
            continue

        # Plot individual seed traces lightly
        for r in runs:
            hist = r.get("regret_history", [])
            if hist:
                eps = [pt["ep"] for pt in hist]
                regs = [pt["avg_regret"] for pt in hist]
                ax.plot(eps, regs, color=color, alpha=0.25, lw=1.0)

        # Plot mean trace boldly
        # Find common x-axis
        max_eps = max(max(pt["ep"] for pt in r.get("regret_history", [])) for r in runs if r.get("regret_history"))
        common_x = [pt["ep"] for pt in runs[0].get("regret_history", []) if pt["ep"] <= max_eps]
        
        all_y = []
        for r in runs:
            hist = {pt["ep"]: pt["avg_regret"] for pt in r.get("regret_history", [])}
            all_y.append([hist[ep] for ep in common_x if ep in hist])
        
        if all_y and len(all_y[0]) == len(common_x):
            mean_y = np.mean(all_y, axis=0)
            ax.plot(common_x, mean_y, color=color, lw=2.5, label=f"{algo} (Mean of 4 Seeds)")

    ax.axhline(0.90, color="black", linestyle="--", lw=2, label="Target Learn Threshold (Avg Regret < 0.90)")
    ax.set_xlabel("Episode Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Average Regret", fontsize=12, fontweight="bold")
    ax.set_title("Deep Sea N=50: Cumulative Average Regret Across All Episodes\n(Full Scale: 0.0 to 1.1)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.set_xlim(0, 10000)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")

    # ----------------------------------------------------
    # Panel 2: High-Resolution Zoomed-in View (1.000 to 1.010)
    # ----------------------------------------------------
    ax2 = axes[1]
    for algo, color in colors.items():
        runs = [r for r in data_50 if r["algo_name"] == algo]
        if not runs:
            continue

        # Plot mean trace
        max_eps = max(max(pt["ep"] for pt in r.get("regret_history", [])) for r in runs if r.get("regret_history"))
        common_x = [pt["ep"] for pt in runs[0].get("regret_history", []) if pt["ep"] <= max_eps]
        all_y = []
        for r in runs:
            hist = {pt["ep"]: pt["avg_regret"] for pt in r.get("regret_history", [])}
            all_y.append([hist[ep] for ep in common_x if ep in hist])

        if all_y and len(all_y[0]) == len(common_x):
            mean_y = np.mean(all_y, axis=0)
            std_y = np.std(all_y, axis=0)
            ax2.plot(common_x, mean_y, color=color, lw=2.5, label=f"{algo}")
            ax2.fill_between(common_x, mean_y - std_y, mean_y + std_y, color=color, alpha=0.15)

    ax2.set_xlabel("Episode Number", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cumulative Average Regret (Zoomed)", fontsize=12, fontweight="bold")
    ax2.set_title("Zoomed View: Micro-Dynamics of Exploration Cost\n(Regret ~ 1.000 + Movement Penalty)", fontsize=13, fontweight="bold")
    ax2.set_ylim(1.001, 1.008)
    ax2.set_xlim(0, 10000)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="upper right")

    # Annotations explaining the mechanics
    ax2.text(
        0.03, 0.15,
        "Why Regret is ~1.005:\n"
        "• Each Right step costs -0.01/50 = -0.0002.\n"
        "• Exploratory agents taking right turns accumulate small movement penalties,\n"
        "  yielding episode reward ~ -0.005 => Regret = 1.0 - (-0.005) = 1.005.\n"
        "• Dithering decays epsilon to 0.01, retreating into inaction (Left action only,\n"
        "  0 penalty => Regret drifts down to exactly 1.000).",
        transform=ax2.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9db", edgecolor="#e67700", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Regret plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
