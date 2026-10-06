import json
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_ids_benchmark(env_name="mismatch", out_dir="results_ids"):
    json_path = os.path.join(out_dir, f"ids_{env_name}_results.json")
    if not os.path.exists(json_path):
        print(f"File {json_path} not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    colors = {
        "DP-IDS (Mutual Info)": "#00897B",       # Teal
        "DP-IDS (Variance)": "#1E88E5",          # Royal Blue
        "DP-TS (Thompson Sampling)": "#D81B60",  # Magenta/Crimson
        "BootDQN-RP": "#FF8C00",                 # Amber/Orange
        "Epsilon-Greedy": "#757575",             # Gray
    }

    linestyles = {
        "DP-IDS (Mutual Info)": "-",
        "DP-IDS (Variance)": "--",
        "DP-TS (Thompson Sampling)": "-.",
        "BootDQN-RP": ":",
        "Epsilon-Greedy": ":",
    }

    # 1. Cumulative Regret Across Rounds
    ax1 = axes[0]
    rounds = None

    for name, d in data.items():
        m = np.array(d["mean_cum_regret"])
        s = np.array(d["stderr_cum_regret"])
        if rounds is None:
            rounds = np.arange(1, len(m) + 1)

        c = colors.get(name, "black")
        ls = linestyles.get(name, "-")
        lw = 2.4 if "IDS" in name else 1.8

        ax1.plot(rounds, m, label=name, color=c, linestyle=ls, linewidth=lw)
        ax1.fill_between(rounds, m - s, m + s, color=c, alpha=0.15)

    env_title = "Information-Action Mismatch" if env_name == "mismatch" else "Hard Wheel Bandit (δ=0.9)"
    ax1.set_title(f"(a) Cumulative Regret: {env_title}", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Round t", fontsize=11)
    ax1.set_ylabel("Cumulative Regret", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=9)

    # 2. Final Cumulative Regret Bar Chart
    ax2 = axes[1]
    agent_names = list(data.keys())
    final_means = [data[k]["final_regret_mean"] for k in agent_names]
    final_errs = [data[k]["final_regret_stderr"] for k in agent_names]
    bar_colors = [colors[k] for k in agent_names]

    y_pos = np.arange(len(agent_names))
    bars = ax2.barh(y_pos, final_means, xerr=final_errs, color=bar_colors, alpha=0.85, capsize=5, height=0.6)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        err = final_errs[i]
        ax2.text(w + err + (max(final_means) * 0.02), bar.get_y() + bar.get_height() / 2,
                 f"{w:.1f}", va="center", ha="left", fontsize=9, fontweight="bold", color=bar_colors[i])

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(agent_names, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_title("(b) Final Cumulative Regret at Horizon", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Final Cumulative Regret", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")

    # 3. Arm Selection Distribution
    ax3 = axes[2]
    num_arms = len(list(data.values())[0]["action_distribution"])
    x = np.arange(len(agent_names))
    width = 0.8 / num_arms

    if env_name == "mismatch":
        arm_labels = ["Arm 0 (Safe)", "Arm 1 (Risky)", "Arm 2 (Info)"]
        arm_palette = ["#90CAF9", "#EF5350", "#66BB6A"]
    else:
        arm_labels = [f"Arm {i}" for i in range(num_arms)]
        arm_palette = plt.cm.viridis(np.linspace(0.1, 0.9, num_arms))

    for a_idx in range(num_arms):
        counts = [data[k]["action_distribution"][a_idx] for k in agent_names]
        total_pulls = [sum(data[k]["action_distribution"]) for k in agent_names]
        fractions = [c / t for c, t in zip(counts, total_pulls)]
        offset = (a_idx - num_arms / 2 + 0.5) * width
        ax3.bar(x + offset, fractions, width=width, label=arm_labels[a_idx], color=arm_palette[a_idx], alpha=0.9)

    ax3.set_xticks(x)
    clean_labels = [k.replace(" (Thompson Sampling)", "").replace(" (Mutual Info)", "-MI").replace(" (Variance)", "-Var") for k in agent_names]
    ax3.set_xticklabels(clean_labels, fontsize=9, rotation=15)
    ax3.set_title("(c) Arm Pull Distribution", fontsize=12, fontweight="bold", pad=10)
    ax3.set_ylabel("Arm Pull Fraction", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax3.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    out_file = os.path.join(out_dir, f"ids_{env_name}_showdown.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"IDS plot saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="mismatch")
    parser.add_argument("--out_dir", type=str, default="results_ids")
    args = parser.parse_args()
    plot_ids_benchmark(args.env, args.out_dir)
