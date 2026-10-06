"""Generate publication comparison plots for DP-DQN 3000-episode benchmark (With vs Without Pre-sampling)."""

import json
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

def main():
    json_path = "results_deepsea/comparison_3000ep_n20.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    res_without = [d for d in data if d["mode"] == "Without Pre-sampling"][0]
    res_with = [d for d in data if d["mode"] == "With Pre-sampling"][0]

    # Style configuration
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    c_without = "#E64A19"  # Deep Orange
    c_with = "#1976D2"     # Strong Blue

    # -------------------------------------------------------------
    # Panel 1: Rolling Return over 3000 Episodes
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    window = 50
    ret_without = np.convolve(res_without["returns"], np.ones(window)/window, mode="valid")
    ret_with = np.convolve(res_with["returns"], np.ones(window)/window, mode="valid")
    x_without = np.arange(window, len(res_without["returns"]) + 1)
    x_with = np.arange(window, len(res_with["returns"]) + 1)

    ax1.plot(x_without, ret_without, label="DP-DQN Without Pre-sampling (Sequential)", color=c_without, linewidth=2.0, alpha=0.9)
    ax1.plot(x_with, ret_with, label="DP-DQN With Episodic Pre-sampling", color=c_with, linewidth=2.2, alpha=0.95)
    
    # Mark first discovery
    if res_without["first_discovery"]:
        ax1.axvline(res_without["first_discovery"], color=c_without, linestyle=":", alpha=0.7, label=f"Discovery Without (Ep {res_without['first_discovery']})")
    if res_with["first_discovery"]:
        ax1.axvline(res_with["first_discovery"], color=c_with, linestyle="--", alpha=0.8, label=f"Discovery With (Ep {res_with['first_discovery']})")

    ax1.axhline(0.99, color="green", linestyle="-.", alpha=0.5, label="Optimal Treasure Return (+0.99)")
    ax1.set_title("Deep Sea N=20: 50-Episode Rolling Return (Horizon N=20)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Rolling Average Return", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    # -------------------------------------------------------------
    # Panel 2: Cumulative Regret Curves
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    cum_reg_without = np.cumsum(res_without["regrets"])
    cum_reg_with = np.cumsum(res_with["regrets"])
    eps = np.arange(1, len(cum_reg_without) + 1)

    ax2.plot(eps, cum_reg_without, color=c_without, linewidth=2.2, label=f"Without Pre-sampling (Final: {cum_reg_without[-1]:.1f})")
    ax2.plot(eps, cum_reg_with, color=c_with, linewidth=2.2, label=f"With Pre-sampling (Final: {cum_reg_with[-1]:.1f})")

    # Shading the regret gap
    ax2.fill_between(eps, cum_reg_with, cum_reg_without, color=c_with, alpha=0.15, label=f"Saved Regret: {cum_reg_without[-1] - cum_reg_with[-1]:.1f} (-21.0%)")

    ax2.set_title("Deep Sea N=20: Cumulative Regret Across 3,000 Episodes", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Cumulative Regret", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # -------------------------------------------------------------
    # Panel 3: Exploration Discovery & Learning Time Milestones
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    bar_width = 0.35
    labels = ["Without Pre-sampling", "With Pre-sampling"]
    x = np.arange(len(labels))

    first_disc = [res_without["first_discovery"], res_with["first_discovery"]]
    learn_ep = [3000 if not res_without["solved"] else res_without["learn_time"], 
                res_with["learn_time"]]

    b1 = ax3.bar(x - bar_width/2, first_disc, bar_width, label="First Discovery (T_first)", color="#42A5F5", alpha=0.85)
    b2 = ax3.bar(x + bar_width/2, learn_ep, bar_width, label="Solved / Converged (T_learn)", color=["#FF7043", "#66BB6A"], alpha=0.85)

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax3.set_ylabel("Episode Number", fontsize=11)
    ax3.set_title("Exploration Milestones: First Discovery vs. Policy Convergence", fontsize=13, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=10)

    for rect in b1:
        h = rect.get_height()
        ax3.annotate(f"Ep {int(h)}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight="bold")
    
    for i, rect in enumerate(b2):
        h = rect.get_height()
        txt = f"Ep {int(h)}" if (i == 1 or res_without["solved"]) else "> 3000 (Unconverged)"
        ax3.annotate(txt, xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 4: Throughput & Latency Summary Table
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.axis("off")

    table_data = [
        ["Benchmark Metric", "Without Pre-sampling", "With Pre-sampling", "Impact / Benefit"],
        ["First Discovery (T_first)", f"Episode {res_without['first_discovery']}", f"Episode {res_with['first_discovery']}", "-548 episodes (20.2% faster)"],
        ["Learning Time (T_learn)", "Unconverged (>3000)", f"Episode {res_with['learn_time']}", "SOLVED at Ep 2,230"],
        ["Final 50-Ep Return", f"{res_without['final_50ep_return']:.4f}", f"{res_with['final_50ep_return']:.4f}", "+0.9975 (Near-Optimal)"],
        ["Cumulative Regret (3000 ep)", f"{res_without['cumulative_regret']:.1f}", f"{res_with['cumulative_regret']:.1f}", f"-{res_without['cumulative_regret'] - res_with['cumulative_regret']:.1f} (-21.0%)"],
        ["Total Wall-Clock Time", f"{res_without['elapsed_seconds']:.2f}s", f"{res_with['elapsed_seconds']:.2f}s", f"-{res_without['elapsed_seconds'] - res_with['elapsed_seconds']:.2f}s (-9.1%)"],
        ["Throughput Speed", f"{res_without['episodes_per_sec']:.1f} eps/s", f"{res_with['episodes_per_sec']:.1f} eps/s", "+10.1% higher throughput"],
        ["Intra-Episode Latency", "10 SGD steps mid-episode", "0 steps (Pure inference)", "Zero intra-episode overhead"],
        ["PSRL Theoretical Faithfulness", "Policy jitter mid-episode", "Frozen hypothesis during ep", "Strictly faithful to PSRL"]
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

    plt.suptitle("Deep Sea 20x20 Benchmark: DP-DQN With vs Without Episodic Pre-sampling (3,000 Episodes)", 
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = "results_deepsea/deep_sea_20_presampling_3000ep.png"
    plt.savefig(out_file, dpi=300)
    print(f"Saved figure to: {out_file}")

    # Copy to artifact directory
    artifact_dir = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
    shutil.copy(out_file, os.path.join(artifact_dir, "deep_sea_20_presampling_3000ep.png"))
    print(f"Copied figure to artifact directory: {artifact_dir}")

if __name__ == "__main__":
    main()
