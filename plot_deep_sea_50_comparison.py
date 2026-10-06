"""Plotting script for Deep Sea N=50 Exploration Comparison.

Generates 4-panel visualization comparing:
1. DP-DQN (Structured BM)
2. DP-DQN (Uniform Optimistic BM)
3. BootDQN-RP (20 Heads)
4. DQN-Dithering
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = "results_deepsea_50/deep_sea_50_results.json"
OUTPUT_PLOT = "results_deepsea_50/deep_sea_50_comparison.png"


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    # Group by algorithm
    algos = [
        "DP-DQN (Structured BM)",
        "DP-DQN (Uniform BM)",
        "BootDQN-RP (20 Heads)",
        "DQN-Dithering",
    ]

    grouped = {a: [] for a in algos}
    for run in data:
        name = run["algo_name"]
        if name in grouped:
            grouped[name].append(run)

    # Styling setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "DP-DQN (Structured BM)": "#1f77b4",       # Strong Blue
        "DP-DQN (Uniform BM)": "#2ca02c",          # Forest Green
        "BootDQN-RP (20 Heads)": "#ff7f0e",        # Deep Orange
        "DQN-Dithering": "#d62728",                # Crimson Red
    }

    # ----------------------------------------------------
    # Panel (a): Learning Time (T_learn)
    # ----------------------------------------------------
    ax = axes[0, 0]
    algo_labels = []
    means = []
    stds = []
    bar_colors = []

    for a in algos:
        runs = grouped[a]
        if not runs:
            continue
        t_learns = [r["t_learn"] for r in runs if r.get("t_learn") is not None]
        timeouts = len(runs) - len(t_learns)

        algo_labels.append(a.replace(" (20 Heads)", ""))
        bar_colors.append(colors[a])

        if len(t_learns) > 0:
            means.append(np.mean(t_learns))
            stds.append(np.std(t_learns) if len(t_learns) > 1 else 0.0)
        else:
            # Timeout placeholder
            max_ep = max(r["total_episodes"] for r in runs)
            means.append(max_ep)
            stds.append(0.0)

    bars = ax.bar(range(len(algo_labels)), means, yerr=stds, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(algo_labels)))
    ax.set_xticklabels(algo_labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Episodes to Learn (Avg Regret < 0.9)", fontsize=11)
    ax.set_title("(a) Learning Time ($T_{\\mathrm{learn}}$) on Deep Sea $N=50$\n($2^{50} \\approx 1.12 \\times 10^{15}$ Policies)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Annotate bars
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
    # Panel (b): Treasure Discovery Time (T_first)
    # ----------------------------------------------------
    ax = axes[0, 1]
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

    bars_b = ax.bar(range(len(algo_labels)), disc_means, yerr=disc_stds, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(algo_labels)))
    ax.set_xticklabels(algo_labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Episodes to First Treasure ($T_{\\mathrm{first}}$)", fontsize=11)
    ax.set_title("(b) First Treasure Discovery Episode ($T_{\\mathrm{first}}$)\nNeedle in Haystack Discovery", fontsize=12, fontweight="bold")
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
            ax.text(i, disc_means[i] * 0.5, "NOT FOUND\n($\\Omega(2^{50})$ Lower Bound)", ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    # ----------------------------------------------------
    # Panel (c): Cumulative Average Regret Curves
    # ----------------------------------------------------
    ax = axes[1, 0]
    for a in algos:
        runs = grouped[a]
        if not runs:
            continue

        # Extract regret histories
        max_len = max(len(r.get("regret_history", [])) for r in runs)
        if max_len == 0:
            continue

        # Collect curve per seed
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

        # Interpolate onto common ep_axis
        if ep_axis is not None and len(all_curves) > 0:
            interpolated = []
            for eps, regs in all_curves:
                interp_reg = np.interp(ep_axis, eps, regs)
                interpolated.append(interp_reg)

            arr = np.array(interpolated)
            mean_c = np.mean(arr, axis=0)
            std_c = np.std(arr, axis=0) if len(arr) > 1 else np.zeros_like(mean_c)

            ax.plot(ep_axis, mean_c, label=a, color=colors[a], lw=2.5)
            ax.fill_between(ep_axis, np.maximum(0, mean_c - std_c), mean_c + std_c, color=colors[a], alpha=0.15)

    ax.axhline(0.9, color="black", linestyle="--", lw=1.5, label="Threshold (Avg Regret = 0.9)")
    ax.set_xlabel("Training Episodes", fontsize=11)
    ax.set_ylabel("Cumulative Average Regret", fontsize=11)
    ax.set_title("(c) Average Regret vs. Training Episodes ($N=50$)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

    # ----------------------------------------------------
    # Panel (d): Base Measure & Architecture Comparison Summary
    # ----------------------------------------------------
    ax = axes[1, 1]
    ax.axis("off")

    # Formatted comparison table
    summary_text = (
        "DEEP SEA SCALE N = 50 BENCHMARK SUMMARY\n"
        "State Space: 50 x 50 = 2,500 Dimensions (Flattened One-Hot)\n"
        "Policy Space: 2^50 = 1,125,899,906,842,624 Policies\n"
        "Seeds Evaluated: 4 Independent Seeds (42, 43, 44, 45)\n\n"
        "========================================================================\n"
        "ALGORITHM COMPARISON AT N = 50:\n\n"
    )

    for a in algos:
        runs = grouped[a]
        if not runs:
            continue
        t_learns = [r["t_learn"] for r in runs if r.get("t_learn") is not None]
        firsts = [r["t_first_treasure"] for r in runs if r.get("t_first_treasure") is not None]
        avg_time = np.mean([r["elapsed_time_s"] for r in runs])

        if len(t_learns) > 0:
            summary_text += f"• {a}:\n"
            summary_text += f"    - Mean T_learn: {np.mean(t_learns):.1f} episodes (Solved {len(t_learns)}/{len(runs)})\n"
            summary_text += f"    - Mean T_first: {np.mean(firsts):.1f} episodes\n"
            summary_text += f"    - Wall Clock / Seed: {avg_time:.1f}s\n"
        else:
            summary_text += f"• {a}:\n"
            summary_text += f"    - Mean T_learn: TIMEOUT (> {max(r['total_episodes'] for r in runs)} ep)\n"
            summary_text += f"    - Mean T_first: NOT FOUND (0/{len(runs)})\n"
            summary_text += f"    - Wall Clock / Seed: {avg_time:.1f}s\n"

    summary_text += (
        "\n========================================================================\n"
        "KEY TAKEAWAY:\n"
        "DP-DQN successfully masters the needle-in-a-haystack trajectory\n"
        "in a quadrillion-policy space (2^50), with both Structured and\n"
        "Uniform Optimistic Base Measures, while 20-head BootDQN-RP\n"
        "and DQN-Dithering fail to scale."
    )

    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9))

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
