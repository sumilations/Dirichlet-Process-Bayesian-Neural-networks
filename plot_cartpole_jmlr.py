"""Plotting script for Cart-Pole Swing-Up Benchmark (arXiv:1703.07608 Section 7.2.2).

Generates a 4-panel publication-ready comparison figure:
- (a) Episodic Return (smoothed moving average + std-err band)
- (b) Cumulative Return across training
- (c) Upright Balancing Steps per episode (cos(theta) > 0.95)
- (d) Peak Swing-Up Height cos(theta)
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def moving_average(a, n=25):
    """Compute centered moving average with edge padding."""
    if len(a) < n:
        return np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    smoothed = ret[n - 1:] / n
    pad_left = np.full(n // 2, smoothed[0])
    pad_right = np.full(len(a) - len(smoothed) - len(pad_left), smoothed[-1])
    return np.concatenate([pad_left, smoothed, pad_right])


def plot_benchmark():
    res_path = "results_rl/cartpole_jmlr_results.json"
    if not os.path.exists(res_path):
        print(f"Results file {res_path} not found.")
        return

    with open(res_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

    colors = {
        "DP-DQN": "#0047AB",       # Deep Cobalt Blue
        "BootDQN-RP": "#9467bd",   # Purple
        "Standard-DQN": "#d62728"  # Red
    }
    linestyles = {
        "DP-DQN": "-",
        "BootDQN-RP": "--",
        "Standard-DQN": ":"
    }

    # Determine episode axis
    first_agent = next(iter(data.values()))
    episodes = np.arange(1, len(first_agent["mean_returns"]) + 1)

    # (a) Episodic Return
    ax1 = axes[0, 0]
    for name, d in data.items():
        raw_m = np.array(d["mean_returns"])
        raw_s = np.array(d.get("stderr_returns", np.zeros_like(raw_m)))
        m = moving_average(raw_m, n=25)
        s = moving_average(raw_s, n=25)
        c = colors.get(name, "black")
        ls = linestyles.get(name, "-")
        lw = 2.4 if name == "DP-DQN" else 1.8

        ax1.plot(episodes, m, label=name, color=c, linestyle=ls, linewidth=lw)
        ax1.fill_between(episodes, m - s, m + s, color=c, alpha=0.12)

    ax1.axhline(0.0, color="gray", linestyle=":", alpha=0.7, label="Idle Inaction Trap (0.0)")
    ax1.set_title("(a) Episodic Return (arXiv:1703.07608 Sec 7.2.2)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Return (Reward - Action Cost)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    # (b) Cumulative Return
    ax2 = axes[0, 1]
    for name, d in data.items():
        m = np.array(d["mean_cumulative_returns"])
        s = np.array(d.get("stderr_cumulative_returns", np.zeros_like(m)))
        c = colors.get(name, "black")
        ls = linestyles.get(name, "-")
        lw = 2.6 if name == "DP-DQN" else 2.0

        ax2.plot(episodes, m, label=name, color=c, linestyle=ls, linewidth=lw)
        if s.any():
            ax2.fill_between(episodes, m - s, m + s, color=c, alpha=0.12)

    ax2.set_title("(b) Cumulative Return Across Training", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Cumulative Return (∑ R)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True, fontsize=9)

    # (c) Upright Balancing Steps per Episode (cos(theta) > 0.95)
    ax3 = axes[1, 0]
    for name, d in data.items():
        raw_u = np.array(d.get("mean_upright_steps", np.zeros_like(episodes)))
        raw_s = np.array(d.get("stderr_upright_steps", np.zeros_like(raw_u)))
        m = moving_average(raw_u, n=25)
        s = moving_average(raw_s, n=25)
        c = colors.get(name, "black")
        ls = linestyles.get(name, "-")
        lw = 2.4 if name == "DP-DQN" else 1.8

        ax3.plot(episodes, m, label=name, color=c, linestyle=ls, linewidth=lw)
        ax3.fill_between(episodes, np.maximum(0, m - s), m + s, color=c, alpha=0.12)

    ax3.set_title("(c) Upright Balancing Steps (cos θ > 0.95)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Episode", fontsize=11)
    ax3.set_ylabel("Steps per Episode", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", frameon=True, fontsize=9)

    # (d) Peak Swing-Up Height cos(theta)
    ax4 = axes[1, 1]
    for name, d in data.items():
        raw_p = np.array(d.get("mean_peak_cos", np.zeros_like(episodes)))
        raw_s = np.array(d.get("stderr_peak_cos", np.zeros_like(raw_p)))
        m = moving_average(raw_p, n=25)
        s = moving_average(raw_s, n=25)
        c = colors.get(name, "black")
        ls = linestyles.get(name, "-")
        lw = 2.4 if name == "DP-DQN" else 1.8

        ax4.plot(episodes, m, label=name, color=c, linestyle=ls, linewidth=lw)
        ax4.fill_between(episodes, m - s, m + s, color=c, alpha=0.12)

    ax4.axhline(0.95, color="forestgreen", linestyle="--", alpha=0.8, label="Goal Upright Threshold (0.95)")
    ax4.axhline(0.0, color="gray", linestyle=":", alpha=0.5, label="Horizontal (0.0)")
    ax4.axhline(-1.0, color="darkred", linestyle=":", alpha=0.5, label="Hanging Down (-1.0)")
    ax4.set_title("(d) Peak Swing-Up Angle (cos θ)", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xlabel("Episode", fontsize=11)
    ax4.set_ylabel("Peak cos(θ)", fontsize=11)
    ax4.set_ylim(-1.05, 1.05)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="lower right", frameon=True, fontsize=9)

    plt.suptitle("Cart-Pole Swing-Up Benchmark: arXiv:1703.07608 (Osband, Van Roy et al.)\n"
                 "Comparing Single-Net DP-DQN vs. 20-Model Ensemble RLSVI vs. Standard DQN",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_img = "results_rl/figure3_cartpole_jmlr.png"
    out_pdf = "results_rl/figure3_cartpole_jmlr.pdf"
    manuscript_dir = "/Users/sumitvashishtha/Desktop/New Folder With Items/RLC_2026-3"
    brain_dir = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"

    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(os.path.join(manuscript_dir, "figure3_cartpole_jmlr.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(manuscript_dir, "figure3_cartpole_jmlr.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(brain_dir, "figure3_cartpole_jmlr.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(brain_dir, "figure3_cartpole_jmlr.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 3 comparison figure at {out_img} and {out_pdf}")


if __name__ == "__main__":
    plot_benchmark()
