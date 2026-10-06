import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_prior_comparison(results_path="results_delta09/showdown_results.json", out_path="results/dp_vs_randomized_priors.png"):
    if not os.path.exists(results_path):
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    selected = ["DP-BNN", "BootDQN-RP", "Deep-Ensemble", "Neural-Linear"]
    colors = {
        "DP-BNN": "#1f77b4",       # Blue
        "BootDQN-RP": "#9467bd",   # Purple
        "Deep-Ensemble": "#ff7f0e", # Orange
        "Neural-Linear": "#2ca02c"  # Green
    }
    linestyles = {
        "DP-BNN": "-",
        "BootDQN-RP": "-.",
        "Deep-Ensemble": ":",
        "Neural-Linear": "--"
    }

    # 1. Regret Curves
    ax1 = axes[0]
    for name in selected:
        if name not in data:
            continue
        stats = data[name]
        mean_regret = np.array(stats["mean_regret"])
        stderr_regret = np.array(stats["stderr_regret"])
        steps = np.arange(len(mean_regret))

        ax1.plot(steps, mean_regret, label=name, color=colors[name], linestyle=linestyles[name], linewidth=2.5)
        ax1.fill_between(steps, mean_regret - stderr_regret, mean_regret + stderr_regret, color=colors[name], alpha=0.15)

    ax1.set_title(r"Wheel Bandit ($\delta = 0.9$): Exploration Dynamics", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Rounds / Steps", fontsize=11)
    ax1.set_ylabel("Cumulative Expected Regret", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # 2. Resource vs Performance Tradeoff: Regret vs Network Footprint
    ax2 = axes[1]
    # Memory footprint: Number of networks maintained
    net_counts = {
        "DP-BNN": 1,          # Single network
        "Neural-Linear": 1,   # Feature net + linear stats
        "Deep-Ensemble": 5,   # 5 networks
        "BootDQN-RP": 10      # 5 trainable + 5 frozen prior nets
    }

    x_nets = [net_counts[k] for k in selected]
    y_regrets = [data[k]["mean_final_regret"] for k in selected]
    y_errs = [data[k]["stderr_final_regret"] for k in selected]
    point_colors = [colors[k] for k in selected]

    for i, name in enumerate(selected):
        ax2.errorbar(x_nets[i], y_regrets[i], yerr=y_errs[i], fmt='o', color=point_colors[i],
                     markersize=10, capsize=6, elinewidth=2, label=name)
        # Annotation offset
        offset_y = 350 if name != "BootDQN-RP" else -600
        ax2.annotate(f"{name}\n({data[name]['mean_final_regret']:.0f} regret)",
                     xy=(x_nets[i], y_regrets[i]),
                     xytext=(10, offset_y / 100), textcoords="offset points",
                     fontsize=9, fontweight="bold", color=point_colors[i])

    ax2.set_title("Exploration Performance vs Model Footprint", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Number of Neural Networks in Memory", fontsize=11)
    ax2.set_ylabel("Final Regret (Lower is Better)", fontsize=11)
    ax2.set_xlim(0, 12)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Focused prior comparison plot saved to {out_path}")


if __name__ == "__main__":
    plot_prior_comparison()
