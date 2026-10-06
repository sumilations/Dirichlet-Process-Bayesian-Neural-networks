import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_showdown_results(results_path="results/showdown_results.json", out_path="results/wheel_bandit_showdown.png"):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} does not exist.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Styling colors and markers
    colors = {
        "DP-BNN": "#1f77b4",         # Deep Blue
        "Neural-Linear": "#2ca02c",   # Green
        "Deep-Ensemble": "#ff7f0e",   # Orange
        "Eps-Greedy": "#d62728",      # Red
        "Uniform-Random": "#7f7f7f"   # Gray
    }
    linestyles = {
        "DP-BNN": "-",
        "Neural-Linear": "--",
        "Deep-Ensemble": "-.",
        "Eps-Greedy": ":",
        "Uniform-Random": ":"
    }

    # 1. Cumulative Regret Curves
    ax1 = axes[0]
    for agent_name, stats in data.items():
        mean_regret = np.array(stats["mean_regret"])
        stderr_regret = np.array(stats["stderr_regret"])
        steps = np.arange(len(mean_regret))

        color = colors.get(agent_name, "black")
        ls = linestyles.get(agent_name, "-")

        ax1.plot(steps, mean_regret, label=agent_name, color=color, linestyle=ls, linewidth=2.2)
        ax1.fill_between(steps, mean_regret - stderr_regret, mean_regret + stderr_regret, color=color, alpha=0.15)

    ax1.set_title("Wheel Bandit: Cumulative Regret over Time", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Rounds / Steps", fontsize=11)
    ax1.set_ylabel("Expected Cumulative Regret", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # 2. Final Cumulative Regret Bar Chart
    ax2 = axes[1]
    agents = list(data.keys())
    final_means = [data[a]["mean_final_regret"] for a in agents]
    final_stderrs = [data[a]["stderr_final_regret"] for a in agents]
    bar_colors = [colors.get(a, "gray") for a in agents]

    bars = ax2.bar(agents, final_means, yerr=final_stderrs, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    ax2.set_title("Total Cumulative Regret (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Final Cumulative Regret", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_xticklabels(agents, rotation=20, ha="right", fontsize=10)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{height:.0f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Showdown plot saved successfully to {out_path}")


if __name__ == "__main__":
    plot_showdown_results()
