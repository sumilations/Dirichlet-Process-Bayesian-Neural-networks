"""Live plotting tool for Cart-Pole benchmark progress."""

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_live_benchmark(data_dir: str, out_png: str, out_pdf: str):
    files = glob.glob(os.path.join(data_dir, "*_ckpt.json")) + glob.glob(os.path.join(data_dir, "*_s*.json"))
    # Filter to unique runs (prefer final json over ckpt if both exist)
    run_dict = {}
    for f in files:
        base = os.path.basename(f)
        if base.endswith("_ckpt.json"):
            key = base.replace("_ckpt.json", "")
            if key not in run_dict:
                run_dict[key] = f
        elif base.endswith(".json"):
            key = base.replace(".json", "")
            run_dict[key] = f

    if not run_dict:
        print("No checkpoint or result files found in", data_dir)
        return False

    runs_by_algo = {}
    for key, fpath in run_dict.items():
        try:
            with open(fpath) as f:
                data = json.load(f)
            algo = data.get("algo", "unknown")
            if algo not in runs_by_algo:
                runs_by_algo[algo] = []
            runs_by_algo[algo].append(data)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    # Plotting styles
    colors = {
        "dp_dqn_haar": "#1f77b4",       # Deep Blue
        "dp_dqn_non_haar": "#17becf",   # Cyan
        "boot_dqn": "#ff7f0e",          # Orange
        "bdqn": "#2ca02c",              # Green
        "dp_dqn_alpha_small": "#9467bd",# Purple
        "vanilla_dqn": "#7f7f7f"        # Gray
    }
    labels = {
        "dp_dqn_haar": r"DP-DQN (Haar, $\alpha=3$)",
        "dp_dqn_non_haar": r"DP-DQN (Uniform, $\alpha=3$)",
        "boot_dqn": "BootDQN (K=10)",
        "bdqn": "Bayesian DQN (BBB)",
        "dp_dqn_alpha_small": r"DP-DQN ($\alpha \to 0$)",
        "vanilla_dqn": r"Vanilla DQN ($\epsilon$-greedy)"
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # 1. Cumulative Regret
    ax = axes[0]
    max_ep = 0
    for algo, runs in runs_by_algo.items():
        all_cum_reg = []
        min_len = min(len(r["cumulative_regrets"]) for r in runs)
        max_ep = max(max_ep, min_len)
        for r in runs:
            all_cum_reg.append(r["cumulative_regrets"][:min_len])
        arr = np.array(all_cum_reg) / 1000.0  # in thousands
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        stderr = std / np.sqrt(len(runs))
        xs = np.arange(1, min_len + 1)
        c = colors.get(algo, "#333333")
        lbl = f"{labels.get(algo, algo)} (N={len(runs)})"
        ax.plot(xs, mean, label=lbl, color=c, lw=2.0)
        ax.fill_between(xs, mean - stderr, mean + stderr, color=c, alpha=0.15)

    ax.set_title(f"Cumulative Regret (Live: Ep 1–{max_ep})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(r"Cumulative Regret ($\times 10^3$)", fontsize=11)
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # 2. Episode Return (Moving average of 20)
    ax2 = axes[1]
    for algo, runs in runs_by_algo.items():
        all_ret = []
        min_len = min(len(r["returns"]) for r in runs)
        for r in runs:
            all_ret.append(r["returns"][:min_len])
        arr = np.array(all_ret)
        mean = np.mean(arr, axis=0)
        # 20-episode moving average
        w = min(20, min_len)
        if min_len >= w:
            smooth_mean = np.convolve(mean, np.ones(w)/w, mode="valid")
            xs = np.arange(w, min_len + 1)
            c = colors.get(algo, "#333333")
            lbl = f"{labels.get(algo, algo)}"
            ax2.plot(xs, smooth_mean, label=lbl, color=c, lw=2.0)

    ax2.set_title(f"Episode Return (20-Ep Moving Avg)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Undiscounted Return", fontsize=11)
    ax2.grid(True, alpha=0.3, ls="--")
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Successfully generated {out_png} and {out_pdf}")
    return True


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "./results_cartpole_30runs"
    png = sys.argv[2] if len(sys.argv) > 2 else "live_cartpole_benchmark.png"
    pdf = sys.argv[3] if len(sys.argv) > 3 else "live_cartpole_benchmark.pdf"
    plot_live_benchmark(d, png, pdf)
