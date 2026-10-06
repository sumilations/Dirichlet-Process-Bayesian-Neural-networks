import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_combined_showdown():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)

    colors = {
        "DP-BNN": "#1f77b4",         # Deep Blue
        "BootDQN-RP": "#9467bd",     # Purple
        "Neural-Linear": "#2ca02c",   # Green
        "Deep-Ensemble": "#ff7f0e",   # Orange
        "Eps-Greedy": "#d62728",      # Red
        "Uniform-Random": "#7f7f7f"   # Gray
    }
    linestyles = {
        "DP-BNN": "-",
        "BootDQN-RP": "-.",
        "Neural-Linear": "--",
        "Deep-Ensemble": ":",
        "Eps-Greedy": (0, (3, 1, 1, 1)),
        "Uniform-Random": ":"
    }

    datasets = [
        ("results/showdown_results.json", r"Wheel Bandit ($\delta = 0.7$, Standard)", axes[0, 0], axes[0, 1]),
        ("results_delta09/showdown_results.json", r"Wheel Bandit ($\delta = 0.9$, Hard Exploration)", axes[1, 0], axes[1, 1])
    ]

    for path, title_prefix, ax_curve, ax_bar in datasets:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)

        # 1. Regret curve
        for agent_name, stats in data.items():
            mean_regret = np.array(stats["mean_regret"])
            stderr_regret = np.array(stats["stderr_regret"])
            steps = np.arange(len(mean_regret))

            c = colors.get(agent_name, "black")
            ls = linestyles.get(agent_name, "-")

            ax_curve.plot(steps, mean_regret, label=agent_name, color=c, linestyle=ls, linewidth=2.2)
            ax_curve.fill_between(steps, mean_regret - stderr_regret, mean_regret + stderr_regret, color=c, alpha=0.15)

        ax_curve.set_title(f"{title_prefix}: Cumulative Regret", fontsize=12, fontweight="bold", pad=8)
        ax_curve.set_xlabel("Rounds / Steps", fontsize=10)
        ax_curve.set_ylabel("Expected Cumulative Regret", fontsize=10)
        ax_curve.grid(True, linestyle="--", alpha=0.5)
        ax_curve.legend(loc="upper left", frameon=True, fontsize=9)

        # 2. Bar chart
        agents = list(data.keys())
        final_means = [data[a]["mean_final_regret"] for a in agents]
        final_stderrs = [data[a]["stderr_final_regret"] for a in agents]
        bar_colors = [colors.get(a, "gray") for a in agents]

        bars = ax_bar.bar(range(len(agents)), final_means, yerr=final_stderrs, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.1)
        ax_bar.set_title(f"{title_prefix}: Total Regret (Lower is Better)", fontsize=12, fontweight="bold", pad=8)
        ax_bar.set_ylabel("Final Regret", fontsize=10)
        ax_bar.set_xticks(range(len(agents)))
        ax_bar.set_xticklabels(agents, rotation=15, ha="right", fontsize=9)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            h = bar.get_height()
            ax_bar.annotate(f"{h:.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out_file = "results/wheel_bandit_showdown_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Combined plot saved to {out_file}")


if __name__ == "__main__":
    plot_combined_showdown()
