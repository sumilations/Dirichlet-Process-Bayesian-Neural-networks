import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def moving_average(a, n=15):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_cartpole():
    res_path = "results_rl/cartpole_swingup_results.json"
    if not os.path.exists(res_path):
        print(f"File {res_path} not found.")
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

    episodes = None

    # (a) Episodic Return
    ax1 = axes[0, 0]
    for name, d in data.items():
        m = np.array(d["mean_returns"])
        s = np.array(d.get("stderr_returns", np.zeros_like(m)))
        if episodes is None:
            episodes = np.arange(1, len(m) + 1)
        c = colors.get(name, "black")
        ls = linestyles.get(name, "-")
        lw = 2.4 if name == "DP-DQN" else 1.8

        ax1.plot(episodes, m, label=name, color=c, linestyle=ls, linewidth=lw)
        if s.any():
            ax1.fill_between(episodes, m - s, m + s, color=c, alpha=0.12)

    ax1.axhline(0.0, color="gray", linestyle=":", alpha=0.7, label="Trivial Coasting (0.0)")
    ax1.set_title("(a) Episodic Return (bsuite Swing-Up)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Episodic Return", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    # (b) Cumulative Return (Requested by user)
    ax2 = axes[0, 1]
    for name, d in data.items():
        if "mean_cumulative_returns" in d:
            m = np.array(d["mean_cumulative_returns"])
            s = np.array(d.get("stderr_cumulative_returns", np.zeros_like(m)))
        else:
            m = np.cumsum(np.array(d["mean_returns"]))
            s = np.zeros_like(m)

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

    # (c) Upright Steps per Episode (cos θ > 0)
    ax3 = axes[1, 0]
    for name, d in data.items():
        if "mean_upright_steps" in d:
            m = np.array(d["mean_upright_steps"])
            s = np.array(d.get("stderr_upright_steps", np.zeros_like(m)))
        else:
            m = np.zeros_like(episodes)
            s = np.zeros_like(m)

        c = colors.get(name, "black")
        ls = linestyles.get(name, "-")
        lw = 2.4 if name == "DP-DQN" else 1.8

        ax3.plot(episodes, m, label=name, color=c, linestyle=ls, linewidth=lw)
        if s.any():
            ax3.fill_between(episodes, m - s, m + s, color=c, alpha=0.12)

    ax3.set_title("(c) Deep Exploration: Upright Steps (cos θ > 0)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Episode", fontsize=11)
    ax3.set_ylabel("Steps Upright per Episode", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", frameon=True, fontsize=9)

    # (d) Cumulative Upright Steps (Total Balance Duration)
    ax4 = axes[1, 1]
    for name, d in data.items():
        if "mean_cumulative_upright_steps" in d:
            m = np.array(d["mean_cumulative_upright_steps"])
            s = np.array(d.get("stderr_cumulative_upright_steps", np.zeros_like(m)))
        elif "mean_upright_steps" in d:
            m = np.cumsum(np.array(d["mean_upright_steps"]))
            s = np.zeros_like(m)
        else:
            m = np.zeros_like(episodes)
            s = np.zeros_like(m)

        c = colors.get(name, "black")
        ls = linestyles.get(name, "-")
        lw = 2.6 if name == "DP-DQN" else 2.0

        ax4.plot(episodes, m, label=name, color=c, linestyle=ls, linewidth=lw)
        if s.any():
            ax4.fill_between(episodes, m - s, m + s, color=c, alpha=0.12)

    ax4.set_title("(d) Cumulative Upright Steps (Total Exploration)", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xlabel("Episode", fontsize=11)
    ax4.set_ylabel("Cumulative Upright Steps", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    out_file = "results_rl/cartpole_swingup_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Cartpole swing-up plot saved to {out_file}")


if __name__ == "__main__":
    plot_cartpole()
