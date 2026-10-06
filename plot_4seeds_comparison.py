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
    num_ep = without_runs[0]["num_episodes"]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    c_without = "#E64A19"  # Deep Orange
    c_with = "#1976D2"     # Royal Blue
    seeds = [d["seed"] for d in without_runs]

    # -------------------------------------------------------------
    # Panel 1: Throughput (Episodes / Second) per Seed & Mean
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    bar_width = 0.35
    x = np.arange(len(seeds))

    speeds_without = [d["episodes_per_sec"] for d in without_runs]
    speeds_with = [d["episodes_per_sec"] for d in with_runs]

    b1 = ax1.bar(x - bar_width/2, speeds_without, bar_width, label="Without Pre-sampling (Sequential)", color=c_without, alpha=0.85)
    b2 = ax1.bar(x + bar_width/2, speeds_with, bar_width, label="With Pre-sampling (Episodic)", color=c_with, alpha=0.85)

    # Annotate values
    for rect in b1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight="bold")

    mean_spd_without = np.mean(speeds_without)
    mean_spd_with = np.mean(speeds_with)
    ax1.axhline(mean_spd_without, color=c_without, linestyle="--", alpha=0.7, label=f"Mean Without: {mean_spd_without:.1f} eps/s")
    ax1.axhline(mean_spd_with, color=c_with, linestyle="--", alpha=0.7, label=f"Mean With: {mean_spd_with:.1f} eps/s")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Throughput (Episodes / Second)", fontsize=11)
    ax1.set_title("Training Throughput Across 4 Seeds (4,000 Episodes Each)", fontsize=13, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

    # -------------------------------------------------------------
    # Panel 2: Total Wall-Clock Execution Time (Seconds)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    times_without = [d["elapsed_seconds"] for d in without_runs]
    times_with = [d["elapsed_seconds"] for d in with_runs]

    t1 = ax2.bar(x - bar_width/2, times_without, bar_width, label="Without Pre-sampling (Sequential)", color=c_without, alpha=0.85)
    t2 = ax2.bar(x + bar_width/2, times_with, bar_width, label="With Pre-sampling (Episodic)", color=c_with, alpha=0.85)

    for rect in t1:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}s", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight="bold")
    for rect in t2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}s", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight="bold")

    mean_time_without = np.mean(times_without)
    mean_time_with = np.mean(times_with)
    ax2.axhline(mean_time_without, color=c_without, linestyle="--", alpha=0.7, label=f"Mean Without: {mean_time_without:.1f}s")
    ax2.axhline(mean_time_with, color=c_with, linestyle="--", alpha=0.7, label=f"Mean With: {mean_time_with:.1f}s")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=11, fontweight="bold")
    ax2.set_ylabel("Wall-Clock Time (Seconds)", fontsize=11)
    ax2.set_title("Total Execution Time for 4,000 Episodes", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    # -------------------------------------------------------------
    # Panel 3: Mean Cumulative Regret Trajectories
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    for d in without_runs:
        ax3.plot(np.arange(len(d["subsampled_regrets"])) * 10, d["subsampled_regrets"], color=c_without, alpha=0.25, linewidth=1.2)
    for d in with_runs:
        ax3.plot(np.arange(len(d["subsampled_regrets"])) * 10, d["subsampled_regrets"], color=c_with, alpha=0.25, linewidth=1.2)

    mean_reg_without = np.mean([d["subsampled_regrets"] for d in without_runs], axis=0)
    mean_reg_with = np.mean([d["subsampled_regrets"] for d in with_runs], axis=0)
    x_reg = np.arange(len(mean_reg_without)) * 10

    ax3.plot(x_reg, mean_reg_without, color=c_without, linewidth=2.5, label=f"Without Pre-sampling (Mean Final: {np.mean([d['cumulative_regret'] for d in without_runs]):.0f})")
    ax3.plot(x_reg, mean_reg_with, color=c_with, linewidth=2.5, label=f"With Pre-sampling (Mean Final: {np.mean([d['cumulative_regret'] for d in with_runs]):.0f})")

    ax3.set_title("Mean Regret Trajectories Across All 4 Seeds", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Episode", fontsize=11)
    ax3.set_ylabel("Regret per Episode", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=10)

    # -------------------------------------------------------------
    # Panel 4: Statistical Summary Table
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.axis("off")

    speed_delta = (mean_spd_with - mean_spd_without) / mean_spd_without * 100
    time_delta = (mean_time_with - mean_time_without) / mean_time_without * 100
    reg_without = np.mean([d["cumulative_regret"] for d in without_runs])
    reg_with = np.mean([d["cumulative_regret"] for d in with_runs])
    reg_delta = (reg_with - reg_without) / reg_without * 100

    table_data = [
        ["Statistical Metric (N=4 seeds)", "Without Pre-sampling", "With Pre-sampling", "Impact / Benefit"],
        ["Average Speed (eps/s)", f"{mean_spd_without:.1f} ± {np.std(speeds_without):.1f}", f"{mean_spd_with:.1f} ± {np.std(speeds_with):.1f}", f"{speed_delta:+.1f}% faster"],
        ["Average Time per Seed", f"{mean_time_without:.1f}s ± {np.std(times_without):.1f}s", f"{mean_time_with:.1f}s ± {np.std(times_with):.1f}s", f"{time_delta:+.1f}% time saved"],
        ["Total Time (All 4 Seeds)", f"{np.sum(times_without):.1f}s", f"{np.sum(times_with):.1f}s", f"{np.sum(times_with)-np.sum(times_without):.1f}s faster"],
        ["Mean Cumulative Regret", f"{reg_without:.1f}", f"{reg_with:.1f}", f"{reg_delta:+.1f}% regret"],
        ["Intra-Episode Latency", "10 SGD steps mid-episode", "0 steps (Pure C++ inference)", "Eliminated"],
        ["PSRL Theoretical Faithfulness", "Policy jitter mid-trajectory", "Frozen hypothesis during ep", "Strictly faithful"]
    ]

    table = ax4.table(cellText=table_data, colWidths=[0.31, 0.22, 0.22, 0.25], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.85)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#1A237E")
        table[(0, i)].get_text().set_color("white")
        table[(0, i)].get_text().set_fontweight("bold")

    for row_idx in range(1, len(table_data)):
        bg = "#F5F5F5" if row_idx % 2 == 1 else "#FFFFFF"
        for col_idx in range(len(table_data[0])):
            table[(row_idx, col_idx)].set_facecolor(bg)
            if col_idx == 3:
                table[(row_idx, col_idx)].get_text().set_fontweight("bold")
                table[(row_idx, col_idx)].get_text().set_color("#1B5E20")

    plt.suptitle(f"Deep Sea {size}x{size} Benchmark: DP-DQN With vs Without Episodic Pre-sampling (4 Seeds, {num_ep:,} Episodes Each)", 
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = f"results_deepsea/deep_sea_{size}_presampling_4seeds_{num_ep}ep.png"
    plt.savefig(out_file, dpi=300)
    plt.savefig(f"results_deepsea/deep_sea_{size}_presampling_4seeds.png", dpi=300)
    print(f"Saved figure to: {out_file}")

    artifact_dir = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
    dest_path = os.path.join(artifact_dir, f"deep_sea_{size}_presampling_4seeds_{num_ep}ep.png")
    shutil.copy(out_file, dest_path)
    shutil.copy(out_file, os.path.join(artifact_dir, f"deep_sea_{size}_presampling_4seeds.png"))
    print(f"Copied figure to artifact directory: {dest_path}")

if __name__ == "__main__":
    main()
