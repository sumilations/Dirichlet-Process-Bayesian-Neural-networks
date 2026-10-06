"""Publication-grade 4-panel comparison plot for Deep Sea Scaling Benchmark.

Replicates and extends Figure 3 & Figure 8 from arXiv:1806.03335 (Osband et al., NeurIPS 2018):
(a) Linear Scale Learning Time T_learn vs Problem Scale N (with 2^N dithering bound)
(b) Log-Log Scale Empirical Scaling log(T_learn) vs log(N) with polynomial degree fit
(c) Cumulative Regret vs Problem Scale N
(d) Parameter and Compute Efficiency (Single-Network DP-DQN vs 20-Head Ensemble BootDQN)
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "DP-DQN": "#17becf",         # Cyan / Teal (Ours)
    "BootDQN-RP": "#9467bd",     # Purple (Osband et al. 2018)
    "DQN-Dithering": "#d62728",  # Red (Epsilon-greedy failure)
}

MARKERS = {
    "DP-DQN": "o",
    "BootDQN-RP": "s",
    "DQN-Dithering": "^",
}


def plot_scaling(json_path: str, output_image: str):
    with open(json_path) as f:
        records = json.load(f)

    # Group by (agent, size)
    data = {}
    for r in records:
        agent = r["agent"]
        size = r["size"]
        if agent not in data:
            data[agent] = {}
        if size not in data[agent]:
            data[agent][size] = {"learn_times": [], "regrets": [], "times": []}
        data[agent][size]["learn_times"].append(r["learn_time"])
        data[agent][size]["regrets"].append(r["cumulative_regret"])
        data[agent][size]["times"].append(r["elapsed_seconds"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    all_sizes = sorted(list(set(r["size"] for r in records)))

    # -------------------------------------------------------------
    # Panel (a): Linear Scale T_learn vs N
    # -------------------------------------------------------------
    for agent, color in COLORS.items():
        if agent not in data:
            continue
        sizes = sorted(data[agent].keys())
        means = [np.mean(data[agent][s]["learn_times"]) for s in sizes]
        stds = [np.std(data[agent][s]["learn_times"]) / np.sqrt(len(data[agent][s]["learn_times"])) for s in sizes]

        axes[0, 0].errorbar(
            sizes, means, yerr=stds, label=f"{agent}",
            color=color, marker=MARKERS[agent], linewidth=2.0, capsize=4, alpha=0.9
        )

    # Plot 2^N theoretical dithering lower bound (up to N=14)
    dither_n = np.arange(5, min(15, max(all_sizes) + 1))
    dither_curve = 2.0 ** dither_n
    axes[0, 0].plot(dither_n, dither_curve, "--", color="gray", alpha=0.7, label=r"Dithering lower bound $\Omega(2^N)$")

    axes[0, 0].set_title("(a) Learning Time vs. Problem Scale N (Linear)", fontsize=12, fontweight="bold", pad=8)
    axes[0, 0].set_xlabel("Problem Scale N", fontsize=11)
    axes[0, 0].set_ylabel("Episodes to Learn (T_learn)", fontsize=11)
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="upper left", fontsize=9, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (b): Log-Log Scale Empirical Scaling (Figure 8 in paper)
    # -------------------------------------------------------------
    for agent in ["DP-DQN", "BootDQN-RP"]:
        if agent not in data:
            continue
        sizes = np.array(sorted(data[agent].keys()))
        means = np.array([np.mean(data[agent][s]["learn_times"]) for s in sizes])

        # Filter positive values
        valid = means > 0
        s_val = sizes[valid]
        m_val = means[valid]

        log_s = np.log10(s_val)
        log_m = np.log10(m_val)

        # Fit polynomial exponent
        if len(s_val) >= 2:
            poly = np.polyfit(log_s, log_m, 1)
            slope = poly[0]
            label = f"{agent} (Fit: $\\mathcal{{O}}(N^{{{slope:.2f}}})$)"
        else:
            label = agent

        axes[0, 1].plot(s_val, m_val, marker=MARKERS[agent], color=COLORS[agent], linewidth=2.0, label=label)

    # Reference slopes O(N^2) and O(N^3)
    ref_s = np.array(all_sizes)
    if len(ref_s) > 0:
        c2 = 10.0 / (ref_s[0] ** 2)
        c3 = 2.0 / (ref_s[0] ** 3)
        axes[0, 1].plot(ref_s, c2 * (ref_s ** 2), ":", color="gray", alpha=0.8, label=r"Reference $\mathcal{O}(N^2)$")
        axes[0, 1].plot(ref_s, c3 * (ref_s ** 3), "-.", color="black", alpha=0.8, label=r"Reference $\mathcal{O}(N^3)$ (Paper BSP)")

    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("(b) Empirical Scaling (Log-Log Scale)", fontsize=12, fontweight="bold", pad=8)
    axes[0, 1].set_xlabel("Problem Scale N (log scale)", fontsize=11)
    axes[0, 1].set_ylabel("Episodes to Learn T_learn (log scale)", fontsize=11)
    axes[0, 1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="upper left", fontsize=9, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (c): Cumulative Regret vs Problem Scale N
    # -------------------------------------------------------------
    for agent, color in COLORS.items():
        if agent not in data:
            continue
        sizes = sorted(data[agent].keys())
        regret_means = [np.mean(data[agent][s]["regrets"]) for s in sizes]
        axes[1, 0].plot(sizes, regret_means, marker=MARKERS[agent], color=color, linewidth=2.0, label=agent)

    axes[1, 0].set_title("(c) Cumulative Regret vs. Problem Scale N", fontsize=12, fontweight="bold", pad=8)
    axes[1, 0].set_xlabel("Problem Scale N", fontsize=11)
    axes[1, 0].set_ylabel("Cumulative Regret", fontsize=11)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper left", fontsize=9, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (d): Computational & Parameter Efficiency
    # -------------------------------------------------------------
    bar_agents = ["DP-DQN", "BootDQN-RP"]
    # Single 20-unit MLP with input N^2: (N^2 * 20 + 20) + (20 * 2 + 2)
    # BootDQN-RP has 20 trainable + 20 prior + 20 target nets = 60 networks!
    ref_n = 20
    params_dp = (ref_n * ref_n * 20 + 20) + (20 * 2 + 2)  # ~8,062 parameters
    params_boot = 60 * ((ref_n * ref_n * 20 + 20) + (20 * 2 + 2))  # ~483,720 parameters

    categories = [f"Parameters\nat N={ref_n}\n(Log Scale)", "Active\nNetworks\nin Memory", "Optimizers\nin RAM"]
    dp_vals = [params_dp, 2, 1]
    boot_vals = [params_boot, 60, 20]

    x = np.arange(len(categories))
    width = 0.35

    ax_d = axes[1, 1]
    b1 = ax_d.bar(x - width/2, dp_vals, width, label="DP-DQN (Ours)", color=COLORS["DP-DQN"], alpha=0.9)
    b2 = ax_d.bar(x + width/2, boot_vals, width, label="BootDQN-RP (20 Heads)", color=COLORS["BootDQN-RP"], alpha=0.9)

    ax_d.set_yscale("log")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(categories, fontsize=10)
    ax_d.set_ylabel("Quantity (Log Scale)", fontsize=11)
    ax_d.set_title("(d) Computational & Architecture Efficiency", fontsize=12, fontweight="bold", pad=8)
    ax_d.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_d.legend(loc="upper left", fontsize=9, framealpha=0.9)

    for bar in b1:
        yval = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, f"{int(yval)}", ha='center', va='bottom', fontsize=8.5, fontweight="bold", color=COLORS["DP-DQN"])
    for bar in b2:
        yval = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, f"{int(yval)}", ha='center', va='bottom', fontsize=8.5, fontweight="bold", color=COLORS["BootDQN-RP"])

    fig.suptitle(
        "Deep Sea Scaling Benchmark: DP-DQN vs. BootDQN-RP (arXiv:1806.03335 Section 4.2.1)\nSingle-Network Dirichlet Process Value Learning vs. 20-Head Ensemble",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image, bbox_inches="tight")
    plt.close()
    print(f"Saved publication comparison plot to: {output_image}")


def main():
    parser = argparse.ArgumentParser(description="Plot Deep Sea Scaling Results")
    parser.add_argument("--json", type=str, default="results_deepsea/deep_sea_scaling_results.json", help="Path to JSON results")
    parser.add_argument("--out", type=str, default="results_deepsea/deep_sea_scaling_comparison.png", help="Output PNG path")
    args = parser.parse_args()

    plot_scaling(args.json, args.out)


if __name__ == "__main__":
    main()
