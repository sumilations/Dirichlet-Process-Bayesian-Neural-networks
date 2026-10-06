"""Plot scaling comparison between Pure TS (SampleOnce=True) and Multi-Sample (SampleOnce=False).
Publication-grade styling for TMLR submission.
"""

import glob
import json
import os
import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
})

BASE_DIR = "/Users/sumitvashishtha/Desktop/DP-BNNs"
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"


def plot_scaling(data_dir="./results_scaling_sampler_comparison", out_prefix="scaling_sampler_comparison"):
    files = glob.glob(os.path.join(data_dir, "*.json"))
    records = []
    for f in files:
        if "summary" in f:
            continue
        try:
            with open(f, "r") as fp:
                d = json.load(fp)
            if d.get("completed", False):
                records.append(d)
        except Exception:
            pass

    if not records:
        print("No valid records found in", data_dir)
        return

    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        mode = r.get("mode", "pure_ts" if r.get("sample_once", False) else "multi_sample")
        size = r["size"]
        solved = r.get("solved_episode") or r.get("total_episodes")
        disc = r.get("first_discovery") or r.get("total_episodes")
        time_s = r.get("elapsed_seconds", 0.0)
        regret = r.get("final_cum_regret", 0.0)
        grouped[mode][size].append({
            "solved": solved,
            "disc": disc,
            "time": time_s,
            "regret": regret,
        })

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    fig.suptitle(
        r"$\mathbf{Figure\ 6:}$ Deep Sea Scaling Suite from $N=10$ to $N=50$ (180 Runs, 30-Point Grid)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    modes_meta = {
        "pure_ts": {
            "label": r"Pure TS (1 DP draw/ep, Ours)",
            "color": "#00897B",     # Deep Teal
            "marker": "o",
        },
        "multi_sample": {
            "label": r"Multi-Sample BNP-DQN",
            "color": "#E65100",     # Deep Amber
            "marker": "s",
        },
    }

    # --------------------------------------------------------------------------
    # Panel 1: Solved Episode Scaling (Log-Log)
    # --------------------------------------------------------------------------
    ax1 = axes[0]
    for mode, meta in modes_meta.items():
        if mode not in grouped:
            continue
        sizes = sorted(grouped[mode].keys())
        means = np.array([np.mean([x["solved"] for x in grouped[mode][s]]) for s in sizes])
        stds = np.array([np.std([x["solved"] for x in grouped[mode][s]]) for s in sizes])

        ax1.errorbar(
            sizes, means, yerr=stds,
            fmt=f"{meta['marker']}-",
            color=meta["color"],
            label=meta["label"],
            capsize=4,
            alpha=0.95,
            linewidth=2.2,
            zorder=4,
        )
        ax1.fill_between(
            sizes,
            np.maximum(1, means - stds),
            means + stds,
            color=meta["color"],
            alpha=0.12,
            zorder=3,
        )

        if len(sizes) >= 3:
            log_n = np.log(sizes)
            log_y = np.log(means)
            slope, intercept = np.polyfit(log_n, log_y, 1)
            fit_x = np.linspace(min(sizes), max(sizes), 50)
            fit_y = np.exp(intercept) * (fit_x ** slope)
            ax1.plot(
                fit_x, fit_y, "--",
                color=meta["color"],
                alpha=0.8,
                linewidth=1.8,
                label=f"Fit: $\\mathcal{{O}}(N^{{{slope:.2f}}})$",
                zorder=3,
            )

    # Theoretical \Omega(N^2) Lower Bound
    if "pure_ts" in grouped:
        ref_sizes = sorted(grouped["pure_ts"].keys())
        ref_means = [np.mean([x["solved"] for x in grouped["pure_ts"][s]]) for s in ref_sizes]
        c_bound = ref_means[len(ref_means)//2] / (ref_sizes[len(ref_sizes)//2] ** 2)
        bound_x = np.linspace(min(ref_sizes), max(ref_sizes), 50)
        ax1.plot(
            bound_x, c_bound * (bound_x ** 2), ":",
            color="#37474F",
            linewidth=2.0,
            label=r"$\Omega(N^2)$ Lower Bound",
            zorder=2,
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Deep Sea Horizon $N$ ($|\mathcal{S}| = 2^N$)")
    ax1.set_ylabel(r"Episodes to Solve (Return $\geq 0.85$)")
    ax1.set_title(r"$\mathbf{(a)}$ Sample Complexity Scaling vs. Bound", fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.92, edgecolor="#CCCCCC")
    ax1.grid(True, which="both", ls="--", alpha=0.35)

    # --------------------------------------------------------------------------
    # Panel 2: First Goal Discovery Episode
    # --------------------------------------------------------------------------
    ax2 = axes[1]
    for mode, meta in modes_meta.items():
        if mode not in grouped:
            continue
        sizes = sorted(grouped[mode].keys())
        means = np.array([np.mean([x["disc"] for x in grouped[mode][s]]) for s in sizes])
        stds = np.array([np.std([x["disc"] for x in grouped[mode][s]]) for s in sizes])

        ax2.errorbar(
            sizes, means, yerr=stds,
            fmt=f"{meta['marker']}-",
            color=meta["color"],
            label=meta["label"],
            capsize=4,
            alpha=0.95,
            linewidth=2.2,
            zorder=4,
        )
        ax2.fill_between(
            sizes,
            np.maximum(1, means - stds),
            means + stds,
            color=meta["color"],
            alpha=0.12,
            zorder=3,
        )

    # Add Dithered Random Walk Failure Line \Omega(2^N)
    rw_x = np.linspace(10, 18, 50)
    ax2.plot(
        rw_x, 2.0 ** (rw_x - 1), ":",
        color="#D32F2F",
        linewidth=2.0,
        label=r"Dithering / $\epsilon$-greedy: $\Omega(2^N)$",
        zorder=2,
    )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Deep Sea Horizon $N$")
    ax2.set_ylabel("First Goal Discovery Episode")
    ax2.set_title(r"$\mathbf{(b)}$ Goal Discovery Horizon ($T_{\mathrm{disc}}$)", fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.92, edgecolor="#CCCCCC")
    ax2.grid(True, which="both", ls="--", alpha=0.35)

    # --------------------------------------------------------------------------
    # Panel 3: Total Wallclock Time
    # --------------------------------------------------------------------------
    ax3 = axes[2]
    for mode, meta in modes_meta.items():
        if mode not in grouped:
            continue
        sizes = sorted(grouped[mode].keys())
        means = np.array([np.mean([x["time"] for x in grouped[mode][s]]) for s in sizes])
        stds = np.array([np.std([x["time"] for x in grouped[mode][s]]) for s in sizes])

        ax3.plot(
            sizes, means,
            f"{meta['marker']}-",
            color=meta["color"],
            label=meta["label"],
            linewidth=2.2,
            zorder=4,
        )
        ax3.fill_between(
            sizes,
            np.maximum(0, means - stds),
            means + stds,
            color=meta["color"],
            alpha=0.12,
            zorder=3,
        )

    # Highlight N=50 runtime
    if "pure_ts" in grouped and 50 in grouped["pure_ts"]:
        t50 = np.mean([x["time"] for x in grouped["pure_ts"][50]])
        ax3.annotate(
            f"$N=50$: {t50/60:.1f} min\n(1 CPU core)",
            xy=(50, t50),
            xytext=(34, t50 * 0.75),
            arrowprops=dict(arrowstyle="->", color="#00897B", lw=1.5),
            fontsize=9.5,
            fontweight="bold",
            color="#004D40",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E0F2F1", ec="#00897B", alpha=0.9),
        )

    ax3.set_xlabel(r"Deep Sea Horizon $N$")
    ax3.set_ylabel("Wallclock Runtime (seconds)")
    ax3.set_title(r"$\mathbf{(c)}$ Single-Core CPU Runtime Scaling", fontweight="bold")
    ax3.legend(loc="upper left", framealpha=0.92, edgecolor="#CCCCCC")
    ax3.grid(True, ls="--", alpha=0.35)

    plt.tight_layout()
    pdf_path = f"{out_prefix}.pdf"
    png_path = f"{out_prefix}.png"
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    print(f"[Plot Export] Saved scaling figure to {pdf_path} and {png_path}")

    # Copy to brain artifact directory as well
    shutil.copy(png_path, os.path.join(ARTIFACT_DIR, "scaling_sampler_comparison.png"))
    shutil.copy(pdf_path, os.path.join(ARTIFACT_DIR, "scaling_sampler_comparison.pdf"))


if __name__ == "__main__":
    plot_scaling()
