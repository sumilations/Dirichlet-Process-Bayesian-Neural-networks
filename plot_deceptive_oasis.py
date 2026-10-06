import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_deceptive_oasis():
    results_path = "results/deceptive_oasis_results.json"
    if not os.path.exists(results_path):
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    colors = {
        "DP-BNN": "#0047AB",          # Deep Blue (Winner)
        "BootDQN-RP": "#9467bd",      # Purple
        "Deep-Ensemble": "#ff7f0e",   # Orange
        "Neural-Linear": "#2ca02c",   # Green
        "Eps-Greedy": "#d62728"       # Red
    }
    linestyles = {
        "DP-BNN": "-",
        "BootDQN-RP": "--",
        "Deep-Ensemble": "-.",
        "Neural-Linear": ":",
        "Eps-Greedy": (0, (3, 1, 1, 1))
    }

    # Panel 1: Cumulative Regret over Time
    ax1 = axes[0]
    steps = np.arange(1500)
    for name, stats in data.items():
        m = np.array(stats["mean_regret"])
        s = np.array(stats["stderr_regret"])
        c = colors.get(name, "black")
        ls = linestyles.get(name, "-")
        lw = 2.8 if name == "DP-BNN" else 2.0
        ax1.plot(steps, m, label=name, color=c, linestyle=ls, linewidth=lw)
        ax1.fill_between(steps, m - s, m + s, color=c, alpha=0.12)

    ax1.set_title("Deceptive Oasis: Cumulative Regret", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Rounds / Steps", fontsize=11)
    ax1.set_ylabel("Expected Cumulative Regret", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)

    # Panel 2: Final Total Regret Bar Chart
    ax2 = axes[1]
    agents = list(data.keys())
    finals = [data[k]["mean_final_regret"] for k in agents]
    errs = [data[k]["stderr_final_regret"] for k in agents]
    bar_colors = [colors.get(k, "gray") for k in agents]

    bars = ax2.bar(range(len(agents)), finals, yerr=errs, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.1)
    ax2.set_title("Final Cumulative Regret (Lower is Better)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Final Regret at T=1500", fontsize=11)
    ax2.set_xticks(range(len(agents)))
    ax2.set_xticklabels(agents, rotation=20, ha="right", fontsize=9.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, finals):
        ax2.annotate(f"{val:.0f}",
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 3: Oasis Discovery Rate (%)
    ax3 = axes[2]
    disc_rates = [data[k]["mean_oasis_hits_pct"] for k in agents]
    bars_disc = ax3.bar(range(len(agents)), disc_rates, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.1)
    ax3.set_title("Oasis Jackpot Discovery Rate (%)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_ylabel("Jackpot Hit Rate inside Oasis (%)", fontsize=11)
    ax3.set_xticks(range(len(agents)))
    ax3.set_xticklabels(agents, rotation=20, ha="right", fontsize=9.5)
    ax3.set_ylim(0, 115)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars_disc, disc_rates):
        ax3.annotate(f"{val:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out_file = "results/deceptive_oasis_showdown.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Plot saved to {out_file}")


if __name__ == "__main__":
    plot_deceptive_oasis()
