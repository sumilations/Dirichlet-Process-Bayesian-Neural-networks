"""Generate a clear, high-resolution dedicated Regret Plot for Deep Sea 20x20 (10,000 episodes)."""

import argparse
import json
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default=None)
    args = parser.parse_args()

    if args.results_file is not None:
        json_path = args.results_file
    elif os.path.exists("results_deepsea/presampling_4seeds_N10_5000ep.json"):
        json_path = "results_deepsea/presampling_4seeds_N10_5000ep.json"
    elif os.path.exists("results_deepsea/presampling_4seeds_10000ep.json"):
        json_path = "results_deepsea/presampling_4seeds_10000ep.json"
    else:
        json_path = "results_deepsea/presampling_4seeds_4000ep.json"

    if not os.path.exists(json_path):
        print(f"Results file {json_path} not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    without_runs = [d for d in data if d["mode"] == "Without Pre-sampling"]
    with_runs = [d for d in data if d["mode"] == "With Pre-sampling"]
    seeds = [d["seed"] for d in without_runs]
    size = without_runs[0].get("size", 20)
    prior_r = without_runs[0].get("prior_reward_mean", 1.0)
    r_tag = f"_r{int(prior_r)}" if prior_r != 1.0 else ""
    num_ep = without_runs[0]["num_episodes"]

    n_points = len(without_runs[0]["subsampled_regrets"])
    step_size = max(1, num_ep // n_points)
    ep_axis = np.arange(1, n_points + 1) * step_size

    cum_without = np.array([np.cumsum(d["subsampled_regrets"]) * step_size for d in without_runs])
    cum_with = np.array([np.cumsum(d["subsampled_regrets"]) * step_size for d in with_runs])

    per_without = np.array([d["subsampled_regrets"] for d in without_runs])
    per_with = np.array([d["subsampled_regrets"] for d in with_runs])

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    plt.subplots_adjust(wspace=0.22)

    c_without = "#D84315"  # Deep Vermilion
    c_with = "#1565C0"     # Royal Blue

    # -------------------------------------------------------------
    # Plot 1: Cumulative Regret
    # -------------------------------------------------------------
    mean_cum_wo = np.mean(cum_without, axis=0)
    std_cum_wo = np.std(cum_without, axis=0)

    mean_cum_w = np.mean(cum_with, axis=0)
    std_cum_w = np.std(cum_with, axis=0)

    # Individual seed lines
    for i, s in enumerate(seeds):
        ax1.plot(ep_axis, cum_without[i], color=c_without, alpha=0.3, linewidth=1.2, linestyle="--",
                 label=f"Without Pre-sampling (Seed {s})" if i == 0 else None)
        ax1.plot(ep_axis, cum_with[i], color=c_with, alpha=0.3, linewidth=1.2, linestyle=":",
                 label=f"With Pre-sampling (Seed {s})" if i == 0 else None)

    # Shaded error bands
    ax1.fill_between(ep_axis, mean_cum_wo - std_cum_wo, mean_cum_wo + std_cum_wo, color=c_without, alpha=0.15)
    ax1.fill_between(ep_axis, mean_cum_w - std_cum_w, mean_cum_w + std_cum_w, color=c_with, alpha=0.15)

    # Mean lines
    ax1.plot(ep_axis, mean_cum_wo, color=c_without, linewidth=2.8,
             label=f"Mean Without Pre-sampling (Final: {mean_cum_wo[-1]:.1f})")
    ax1.plot(ep_axis, mean_cum_w, color=c_with, linewidth=2.8,
             label=f"Mean With Pre-sampling (Final: {mean_cum_w[-1]:.1f})")

    # Annotate treasure discoveries
    discoveries = []
    for d in without_runs:
        if d.get("first_discovery") is not None:
            discoveries.append((f"Seed {d['seed']} (Seq)", d["first_discovery"], 180))
    for d in with_runs:
        if d.get("first_discovery") is not None:
            discoveries.append((f"Seed {d['seed']} (Pre)", d["first_discovery"], -250))

    for seed_lbl, ep, offset_y in discoveries[:4]:
        idx = min(len(ep_axis) - 1, ep // step_size)
        y_val = mean_cum_wo[idx]
        ax1.scatter([ep], [y_val], color="#C62828", s=110, zorder=6, marker="*")
        x_off = -0.06 * num_ep if ep > num_ep * 0.6 else 0.02 * num_ep
        ax1.annotate(f"{seed_lbl} Discovery\n(Ep {ep:,})",
                     xy=(ep, y_val),
                     xytext=(ep + x_off, y_val + offset_y),
                     arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5),
                     fontsize=9.5, fontweight="bold", color="#B71C1C",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#C62828", alpha=0.9))

    ax1.set_title(f"Cumulative Regret vs. Episode (4 Seeds, {num_ep:,} Episodes)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Cumulative Regret $\sum_{t=1}^E (V^* - R_t)$", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, num_ep)
    ax1.set_ylim(0, max(np.max(cum_without), np.max(cum_with)) * 1.08)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", framealpha=0.92, fontsize=9.5)

    # -------------------------------------------------------------
    # Plot 2: Instantaneous Per-Episode Regret
    # -------------------------------------------------------------
    def smooth(arr, w=5):
        return np.convolve(arr, np.ones(w)/w, mode='same')

    mean_per_wo = np.mean(per_without, axis=0)
    mean_per_w = np.mean(per_with, axis=0)

    ax2.plot(ep_axis, smooth(mean_per_wo), color=c_without, linewidth=2.2, label="Without Pre-sampling (Rolling Avg)")
    ax2.plot(ep_axis, smooth(mean_per_w), color=c_with, linewidth=2.2, linestyle="-.", label="With Pre-sampling (Rolling Avg)")

    # Discovery markers on instantaneous regret
    for seed_lbl, ep, _ in discoveries[:4]:
        idx = min(len(ep_axis) - 1, ep // step_size)
        y_val = smooth(mean_per_wo)[idx]
        ax2.scatter([ep], [y_val], color="#C62828", s=100, zorder=6, marker="v")
        ax2.annotate(f"{seed_lbl}\n(Ep {ep:,})",
                     xy=(ep, y_val),
                     xytext=(ep - 0.04 * num_ep, y_val - 0.12),
                     arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2),
                     fontsize=8.5, fontweight="bold", color="#B71C1C")

    ax2.axhline(0.99, color="#757575", linestyle=":", linewidth=1.5, label=r"Failed Exploration ($V^* - R_{fail} \approx 0.99$)")
    ax2.axhline(0.0, color="#2E7D32", linestyle="--", linewidth=1.5, label=r"Optimal Policy ($V^* - R^* = 0.0$)")

    ax2.set_title("Instantaneous Regret per Episode (Rolling Average)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Per-Episode Regret $(V^* - R_t)$", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, num_ep)
    ax2.set_ylim(-0.05, 1.10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", framealpha=0.92, fontsize=9.5)

    plt.suptitle(f"Deep Sea {size}x{size} Regret Evaluation (Prior Reward r=+{prior_r:.0f}): DP-DQN With vs Without Pre-sampling ({num_ep:,} Episodes)",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = f"results_deepsea/deep_sea_{size}{r_tag}_regret_plot_{num_ep}ep.png"
    plt.savefig(out_file, dpi=300)
    plt.savefig(f"results_deepsea/deep_sea_{size}{r_tag}_regret_plot.png", dpi=300)
    print(f"Saved regret plot to: {out_file}")

    artifact_dir = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
    dest_path = os.path.join(artifact_dir, f"deep_sea_{size}{r_tag}_regret_plot.png")
    shutil.copy(out_file, dest_path)
    shutil.copy(out_file, os.path.join(artifact_dir, f"deep_sea_{size}_regret_plot.png"))
    print(f"Copied to artifact directory: {dest_path}")

if __name__ == "__main__":
    main()
