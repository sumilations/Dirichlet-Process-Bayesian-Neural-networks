"""Plotting script for Neural-Linear DP-DQN Deep Sea N=20 Benchmark."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = "neural_linear_dp_dqn/deep_sea_20_results.json"
OUTPUT_PLOT = "neural_linear_dp_dqn/deep_sea_20_comparison.png"


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    algos = [
        "Neural-Linear DP-DQN",
        "Standard DP-DQN",
        "BootDQN-RP (20 Heads)",
        "DQN-Dithering",
    ]

    grouped = {a: [] for a in algos}
    for run in data:
        name = run["algo_name"]
        if name in grouped:
            grouped[name].append(run)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "Neural-Linear DP-DQN": "#2ca02c",     # Forest Green (Our New Model)
        "Standard DP-DQN": "#1f77b4",          # Deep Blue
        "BootDQN-RP (20 Heads)": "#ff7f0e",    # Orange
        "DQN-Dithering": "#d62728",            # Red
    }

    # ----------------------------------------------------
    # Panel (a): Cumulative Average Regret vs. Episodes
    # ----------------------------------------------------
    ax = axes[0, 0]
    for a in algos:
        runs = grouped[a]
        if not runs:
            continue

        all_curves = []
        ep_axis = None
        for r in runs:
            hist = r.get("regret_history", [])
            if not hist:
                continue
            eps = [pt["ep"] for pt in hist]
            regs = [pt["avg_regret"] for pt in hist]
            if ep_axis is None or len(eps) > len(ep_axis):
                ep_axis = eps
            all_curves.append((eps, regs))

        if ep_axis is not None and all_curves:
            interpolated = []
            for eps, regs in all_curves:
                interp = np.interp(ep_axis, eps, regs)
                interpolated.append(interp)

            arr = np.array(interpolated)
            mean_c = np.mean(arr, axis=0)
            std_c = np.std(arr, axis=0) if len(arr) > 1 else np.zeros_like(mean_c)

            ax.plot(ep_axis, mean_c, label=a, color=colors[a], lw=2.5)
            ax.fill_between(ep_axis, np.maximum(0, mean_c - std_c), mean_c + std_c, color=colors[a], alpha=0.15)

    ax.axhline(0.9, color="black", linestyle="--", lw=1.5, label="Threshold (Avg Regret = 0.9)")
    ax.set_xlabel("Training Episodes", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Average Regret", fontsize=11, fontweight="bold")
    ax.set_title("(a) Average Regret Trajectory on Deep Sea $N=20$\n($2^{20} = 1,048,576$ Policies)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.2, 1.05)
    ax.set_xlim(0, 6000)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

    # ----------------------------------------------------
    # Panel (b): Learning Time (T_learn)
    # ----------------------------------------------------
    ax = axes[0, 1]
    labels = []
    means = []
    stds = []
    bar_cols = []

    for a in algos:
        runs = grouped[a]
        if not runs:
            continue
        t_learns = [r["t_learn"] for r in runs if r.get("t_learn") is not None]
        labels.append(a.replace(" (20 Heads)", ""))
        bar_cols.append(colors[a])

        if len(t_learns) > 0:
            means.append(np.mean(t_learns))
            stds.append(np.std(t_learns) if len(t_learns) > 1 else 0.0)
        else:
            max_ep = max(r["total_episodes"] for r in runs)
            means.append(max_ep)
            stds.append(0.0)

    bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=5, color=bar_cols, alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Episodes to Learn (Avg Regret < 0.9)", fontsize=11, fontweight="bold")
    ax.set_title("(b) Learning Time ($T_{\\mathrm{learn}}$) on Deep Sea $N=20$", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    for i, a in enumerate(algos):
        runs = grouped[a]
        if not runs:
            continue
        t_learns = [r["t_learn"] for r in runs if r.get("t_learn") is not None]
        if len(t_learns) > 0:
            val = np.mean(t_learns)
            ax.text(i, val * 1.05, f"{val:.0f} ep\n({len(t_learns)}/{len(runs)} solved)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax.text(i, means[i] * 0.5, "TIMEOUT\n(0% Solved)", ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    # ----------------------------------------------------
    # Panel (c): First Treasure Discovery Time (T_first)
    # ----------------------------------------------------
    ax = axes[1, 0]
    disc_means = []
    disc_stds = []

    for a in algos:
        runs = grouped[a]
        if not runs:
            continue
        firsts = [r["t_first_treasure"] for r in runs if r.get("t_first_treasure") is not None]
        if len(firsts) > 0:
            disc_means.append(np.mean(firsts))
            disc_stds.append(np.std(firsts) if len(firsts) > 1 else 0.0)
        else:
            disc_means.append(max(r["total_episodes"] for r in runs))
            disc_stds.append(0.0)

    ax.bar(range(len(labels)), disc_means, yerr=disc_stds, capsize=5, color=bar_cols, alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Episodes to First Treasure ($T_{\\mathrm{first}}$)", fontsize=11, fontweight="bold")
    ax.set_title("(c) Needle in Haystack Discovery ($T_{\\mathrm{first}}$)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    for i, a in enumerate(algos):
        runs = grouped[a]
        if not runs:
            continue
        firsts = [r["t_first_treasure"] for r in runs if r.get("t_first_treasure") is not None]
        if len(firsts) > 0:
            val = np.mean(firsts)
            ax.text(i, val * 1.05, f"Ep {val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax.text(i, disc_means[i] * 0.5, "NOT FOUND\n($\\Omega(2^{20})$)", ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    # ----------------------------------------------------
    # Panel (d): Computational Speed (Episodes / Second)
    # ----------------------------------------------------
    ax = axes[1, 1]
    speed_means = []
    for a in algos:
        runs = grouped[a]
        if not runs:
            continue
        speeds = [r.get("ep_per_sec", r["total_episodes"]/r["elapsed_seconds"]) for r in runs]
        speed_means.append(np.mean(speeds))

    bars_s = ax.bar(range(len(labels)), speed_means, color=bar_cols, alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Throughput (Episodes / Second)", fontsize=11, fontweight="bold")
    ax.set_title("(d) Training Throughput (Episodes / Sec)\nClosed-Form Inverse vs. Gradient Warm-Start", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    for i, val in enumerate(speed_means):
        ax.text(i, val * 1.03, f"{val:.1f} ep/s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
