"""
Generates publication-quality figures comparing Mutual Information estimators:
DP-BNN, DP-MINE, MINE, NWJ, InfoNCE, and KSG.
Produces:
(a) Estimated MI vs True MI (Correlation sweep)
(b) Mean Absolute Error (MAE) across correlation sweep
(c) High-Dimensional Scaling (d in {1, 2, 5, 10, 20})
(d) Non-linear & Circular Dependencies
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_mi_showdown(results_file: str = "results_mi/mi_results.json", out_file: str = "results_mi/mi_showdown.png"):
    if not os.path.exists(results_file):
        print(f"File {results_file} not found.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # Color and styling palette
    colors = {
        "DP-BNN (Predictive MI)": "#00897B",       # Teal
        "DP-MINE": "#43A047",                      # Green
        "MINE": "#1E88E5",                         # Blue
        "NWJ": "#FB8C00",                          # Amber
        "InfoNCE": "#8E24AA",                      # Purple
        "KSG": "#D81B60",                          # Crimson
        "Ground Truth": "#212121"                  # Charcoal
    }

    markers = {
        "DP-BNN (Predictive MI)": "o",
        "DP-MINE": "s",
        "MINE": "^",
        "NWJ": "v",
        "InfoNCE": "D",
        "KSG": "P",
        "Ground Truth": ""
    }

    # --------------------------------------------------------------------------
    # Panel (a): Estimated MI vs True MI (Correlation Sweep)
    # --------------------------------------------------------------------------
    ax_a = axes[0, 0]
    sweep_data = data["gaussian_correlation_sweep"]
    rhos = [v["rho"] for v in sweep_data.values()]
    true_mis = [v["true_mi"] for v in sweep_data.values()]

    # Diagonal reference line: perfect estimation
    max_mi = max(true_mis) * 1.05
    ax_a.plot([0, max_mi], [0, max_mi], "k--", label="Ground Truth (Ideal)", linewidth=2.0, alpha=0.7)

    estimators = list(list(sweep_data.values())[0]["estimators"].keys())

    for est in estimators:
        means = [v["estimators"][est]["mean"] for v in sweep_data.values()]
        stds = [v["estimators"][est]["stderr"] for v in sweep_data.values()]
        c = colors.get(est, "black")
        m = markers.get(est, "o")
        lw = 2.4 if "DP-BNN" in est else 1.6

        ax_a.errorbar(true_mis, means, yerr=stds, label=est, color=c, marker=m,
                      markersize=6, capsize=3, linewidth=lw, alpha=0.9)

    ax_a.set_title("(a) Estimated MI vs. True MI (Correlated Gaussian, d=2)", fontsize=11, fontweight="bold", pad=8)
    ax_a.set_xlabel("True Mutual Information (nats)", fontsize=10)
    ax_a.set_ylabel("Estimated Mutual Information (nats)", fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(loc="upper left", fontsize=8.5, frameon=True)

    # --------------------------------------------------------------------------
    # Panel (b): Mean Absolute Error (MAE) Across Correlation Sweep
    # --------------------------------------------------------------------------
    ax_b = axes[0, 1]
    mae_dict = {}
    for est in estimators:
        maes = [v["estimators"][est]["mae"] for v in sweep_data.values()]
        mae_dict[est] = np.mean(maes)

    sorted_ests = sorted(mae_dict.keys(), key=lambda k: mae_dict[k])
    y_pos = np.arange(len(sorted_ests))
    mae_vals = [mae_dict[k] for k in sorted_ests]
    bar_cols = [colors.get(k, "gray") for k in sorted_ests]

    bars = ax_b.barh(y_pos, mae_vals, color=bar_cols, alpha=0.85, height=0.55)
    for bar, val in zip(bars, mae_vals):
        ax_b.text(val + max(mae_vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                  f"{val:.3f} nats", va="center", ha="left", fontsize=9, fontweight="bold")

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(sorted_ests, fontsize=9.5)
    ax_b.invert_yaxis()
    ax_b.set_title("(b) Average Estimation Error (MAE Across Correlation Sweep)", fontsize=11, fontweight="bold", pad=8)
    ax_b.set_xlabel("Mean Absolute Error (nats, lower is better)", fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.5, axis="x")

    # --------------------------------------------------------------------------
    # Panel (c): High-Dimensional Scaling (dim in {1, 2, 5, 10, 20})
    # --------------------------------------------------------------------------
    ax_c = axes[1, 0]
    dim_data = data["dimension_scaling_sweep"]
    dims = [v["dim"] for v in dim_data.values()]
    dim_true_mis = [v["true_mi"] for v in dim_data.values()]

    ax_c.plot(dims, dim_true_mis, "k--", label="Ground Truth", linewidth=2.2, alpha=0.8)

    for est in estimators:
        means = [v["estimators"][est]["mean"] for v in dim_data.values()]
        stds = [v["estimators"][est]["stderr"] for v in dim_data.values()]
        c = colors.get(est, "black")
        m = markers.get(est, "o")
        lw = 2.4 if "DP-BNN" in est else 1.6

        ax_c.errorbar(dims, means, yerr=stds, label=est, color=c, marker=m,
                      markersize=6, capsize=3, linewidth=lw, alpha=0.9)

    ax_c.set_title("(c) Dimensionality Scaling: d=1 to d=20 (rho=0.6)", fontsize=11, fontweight="bold", pad=8)
    ax_c.set_xlabel("Dimension d", fontsize=10)
    ax_c.set_ylabel("Estimated Mutual Information (nats)", fontsize=10)
    ax_c.set_xticks(dims)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(loc="upper left", fontsize=8.5, frameon=True)

    # --------------------------------------------------------------------------
    # Panel (d): Non-linear & Circular Benchmarks
    # --------------------------------------------------------------------------
    ax_d = axes[1, 1]
    nl_data = data["nonlinear_benchmarks"]
    task_names = list(nl_data.keys())
    x_pos = np.arange(len(task_names))
    width = 0.8 / (len(estimators) + 1)

    # Plot Ground Truth bars first
    gt_vals = [nl_data[t]["true_mi"] for t in task_names]
    ax_d.bar(x_pos - 0.4 + width / 2, gt_vals, width=width, label="Ground Truth",
             color=colors["Ground Truth"], alpha=0.85, hatch="//")

    for i, est in enumerate(estimators):
        means = [nl_data[t]["estimators"][est]["mean"] for t in task_names]
        errs = [nl_data[t]["estimators"][est]["stderr"] for t in task_names]
        offset = -0.4 + (i + 1.5) * width
        ax_d.bar(x_pos + offset, means, yerr=errs, width=width, label=est,
                 color=colors.get(est, "gray"), alpha=0.85, capsize=2)

    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels([t.split(" ")[0] for t in task_names], fontsize=10, fontweight="bold")
    ax_d.set_title("(d) Non-Linear Benchmarks (Cubic, Sinusoid, Circular)", fontsize=11, fontweight="bold", pad=8)
    ax_d.set_ylabel("Estimated Mutual Information (nats)", fontsize=10)
    ax_d.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax_d.legend(loc="upper right", fontsize=8.0, frameon=True)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Mutual Information figure saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results_mi/mi_results.json")
    parser.add_argument("--out", type=str, default="results_mi/mi_showdown.png")
    args = parser.parse_args()
    plot_mi_showdown(args.results, args.out)
