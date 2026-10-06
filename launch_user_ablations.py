"""Parallel Launcher & Plotter for User Requested Deep Sea Ablations on 4 Separate CPUs."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior_reward_mean", type=float, default=0.1)
    parser.add_argument("--prior_reward_std", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs("results_deepsea", exist_ok=True)

    experiments = [
        {
            "exp_id": 0,
            "name": "Baseline (5 SGD/ep, Warm-Start, Polyak τ=0.05, Thompson)",
            "sgd_per_episode": 5,
            "warm_start_target": True,
            "target_update_freq": 1,
            "target_tau": 1.0,
            "use_thompson_bias": True,
        },
        {
            "exp_id": 1,
            "name": "Exp 1: SGD 1x/ep (Warm-Start, Polyak τ=0.05, Thompson)",
            "sgd_per_episode": 1,
            "warm_start_target": True,
            "target_update_freq": 1,
            "target_tau": 1.0,
            "use_thompson_bias": True,
        },
        {
            "exp_id": 2,
            "name": "Exp 2: SGD 1x/ep + Target Sync 100ep (No Warm-Start, Thompson)",
            "sgd_per_episode": 1,
            "warm_start_target": False,
            "target_update_freq": 100,
            "target_tau": 1.0,
            "use_thompson_bias": True,
        },
        {
            "exp_id": 3,
            "name": "Exp 3: SGD 1x/ep + Target Sync 100ep + Value Net Greedy",
            "sgd_per_episode": 1,
            "warm_start_target": False,
            "target_update_freq": 100,
            "target_tau": 1.0,
            "use_thompson_bias": False,
        },
    ]

    print("=" * 85)
    print(f"LAUNCHING 4 ABLATION EXPERIMENTS IN PARALLEL ACROSS 4 SEPARATE CPU PROCESSES")
    print(f"Environment: Deep Sea {args.size}x{args.size} | Seed: {args.seed} | Episodes: {args.episodes:,}")
    print("=" * 85)

    processes = []
    t0 = time.time()

    for exp in experiments:
        out_file = f"results_deepsea/ablation_exp{exp['exp_id']}.json"
        cmd = [
            sys.executable, "run_ablation_worker.py",
            "--exp_id", str(exp["exp_id"]),
            "--name", exp["name"],
            "--seed", str(args.seed),
            "--size", str(args.size),
            "--episodes", str(args.episodes),
            "--sgd_per_episode", str(exp["sgd_per_episode"]),
            "--warm_start_target", str(exp["warm_start_target"]),
            "--target_update_freq", str(exp["target_update_freq"]),
            "--target_tau", str(exp["target_tau"]),
            "--use_thompson_bias", str(exp["use_thompson_bias"]),
            "--prior_reward_mean", str(args.prior_reward_mean),
            "--prior_reward_std", str(args.prior_reward_std),
            "--out_file", out_file
        ]
        print(f"  --> Launching CPU Worker {exp['exp_id']}: {exp['name']}...")
        p = subprocess.Popen(cmd)
        processes.append((exp, p, out_file))

    print(f"\nAll 4 CPU worker processes running simultaneously! Waiting for completion...", flush=True)

    for exp, p, _ in processes:
        p.wait()
        if p.returncode != 0:
            print(f"Warning: Worker for Exp {exp['exp_id']} exited with code {p.returncode}")

    total_time = time.time() - t0
    print("\n" + "=" * 85)
    print(f"ALL 4 PARALLEL CPU EXPERIMENTS COMPLETED IN {total_time:.2f}s!")
    print("=" * 85)

    # Collect results
    all_data = []
    for exp, _, out_file in processes:
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                all_data.append(json.load(f))

    # Print summary table
    print(f"\nABLATION BENCHMARK RESULTS (Deep Sea {args.size}x{args.size}, {args.episodes:,} Episodes, Seed {args.seed}):")
    print(f"{'Exp ID':<7} | {'Condition Name':<55} | {'Solved':<7} | {'T_first':<8} | {'T_learn':<8} | {'Cum Regret':<10} | {'Speed':<10}")
    print("-" * 115)
    for d in all_data:
        t_first = str(d["first_discovery"]) if d["first_discovery"] is not None else "N/A"
        t_learn = str(d["learn_time"]) if d["learn_time"] is not None else "N/A"
        print(f"Exp {d['exp_id']:<3} | {d['name'][:55]:<55} | {str(d['solved']):<7} | {t_first:<8} | {t_learn:<8} | {d['cumulative_regret']:<10.1f} | {d['episodes_per_sec']:<10.1f}")
    print("=" * 115)

    # Generate Publication Figure
    print("\nPlotting comprehensive 4-panel ablation figure...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    colors = ["#757575", "#1E88E5", "#FB8C00", "#43A047"]
    styles = [":", "--", "-.", "-"]

    n_points = len(all_data[0]["subsampled_regrets"])
    step_size = args.episodes // n_points
    ep_axis = np.arange(1, n_points + 1) * step_size

    # Panel 1: Cumulative Regret
    ax1 = axes[0, 0]
    for i, d in enumerate(all_data):
        cum_reg = np.cumsum(d["subsampled_regrets"]) * step_size
        ax1.plot(ep_axis, cum_reg, color=colors[i], linestyle=styles[i], linewidth=2.4,
                 label=f"Exp {d['exp_id']}: {d['name'].split('(')[0].strip()} (Final: {cum_reg[-1]:.0f})")
        if d["first_discovery"] is not None:
            disc_idx = min(len(ep_axis) - 1, d["first_discovery"] // step_size)
            ax1.scatter([d["first_discovery"]], [cum_reg[disc_idx]], color=colors[i], s=90, marker="*", zorder=6)

    ax1.set_title("Cumulative Regret Comparison vs. Episode", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Cumulative Regret $\sum (V^* - R_t)$", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, args.episodes)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", framealpha=0.92, fontsize=9.5)

    # Panel 2: Instantaneous Regret (50-ep moving avg)
    ax2 = axes[0, 1]
    def smooth(arr, w=7):
        return np.convolve(arr, np.ones(w)/w, mode='same')

    for i, d in enumerate(all_data):
        per_reg = np.array(d["subsampled_regrets"])
        ax2.plot(ep_axis, smooth(per_reg), color=colors[i], linestyle=styles[i], linewidth=2.2,
                 label=f"Exp {d['exp_id']}: {d['name'].split('(')[0].strip()}")

    ax2.axhline(0.99, color="#B0BEC5", linestyle=":", linewidth=1.5, label=r"Failed Exploration ($V^* - R_{fail} \approx 0.99$)")
    ax2.axhline(0.0, color="#2E7D32", linestyle="--", linewidth=1.5, label=r"Optimal Policy ($V^* - R^* = 0.0$)")
    ax2.set_title("Instantaneous Regret per Episode (Rolling Average)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Per-Episode Regret $(V^* - R_t)$", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, args.episodes)
    ax2.set_ylim(-0.05, 1.10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", framealpha=0.92, fontsize=9.5)

    # Panel 3: Treasure Discovery & Solve Episode
    ax3 = axes[1, 0]
    bar_w = 0.35
    x_pos = np.arange(len(all_data))
    t_firsts = [d["first_discovery"] if d["first_discovery"] is not None else 0 for d in all_data]
    t_learns = [d["learn_time"] if d["learn_time"] is not None else 0 for d in all_data]

    b1 = ax3.bar(x_pos - bar_w/2, t_firsts, bar_w, label="First Discovery (Ep)", color="#42A5F5", alpha=0.85)
    b2 = ax3.bar(x_pos + bar_w/2, t_learns, bar_w, label="Solved / Converged (Ep)", color="#66BB6A", alpha=0.85)

    for rect in b1:
        h = rect.get_height()
        if h > 0:
            ax3.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        if h > 0:
            ax3.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight="bold")

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"Exp {d['exp_id']}" for d in all_data], fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Episode Number", fontsize=11, fontweight="bold")
    ax3.set_title("Exploration Timeline: Discovery vs. Solution Episode", fontsize=13, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    # Panel 4: Metrics Summary Table
    ax4 = axes[1, 1]
    ax4.axis("off")

    table_rows = [
        ["Exp Condition", "SGD Freq", "Target Update", "Decision Network", "Solved?", "Cum Regret"]
    ]
    for d in all_data:
        sgd_str = "1x / ep" if d["sgd_per_episode"] == 1 else "5x / ep (step=2)"
        tgt_str = "100 ep sync (No warm)" if not d["warm_start_target"] else "Warm-start + Polyak"
        dec_str = "Pure Value Net" if not d["use_thompson_bias"] else "Thompson Bias"
        table_rows.append([
            f"Exp {d['exp_id']}",
            sgd_str,
            tgt_str,
            dec_str,
            "YES" if d["solved"] else "NO",
            f"{d['cumulative_regret']:.1f}"
        ])

    table = ax4.table(cellText=table_rows, colWidths=[0.14, 0.18, 0.26, 0.20, 0.10, 0.14], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 2.0)

    for col_idx in range(6):
        table[(0, col_idx)].set_facecolor("#1A237E")
        table[(0, col_idx)].get_text().set_color("white")
        table[(0, col_idx)].get_text().set_fontweight("bold")

    for row_idx in range(1, len(table_rows)):
        bg = "#F5F5F5" if row_idx % 2 == 1 else "#FFFFFF"
        for col_idx in range(6):
            table[(row_idx, col_idx)].set_facecolor(bg)
            if col_idx == 4:
                table[(row_idx, col_idx)].get_text().set_fontweight("bold")
                table[(row_idx, col_idx)].get_text().set_color("#1B5E20" if "YES" in table_rows[row_idx][4] else "#B71C1C")

    plt.suptitle(f"Deep Sea {args.size}x{args.size} Ablations on 4 CPUs | Base Prior Reward: $r \\sim \\mathcal{{N}}({args.prior_reward_mean}, {args.prior_reward_std}^2)$",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_plot = "results_deepsea/deep_sea_user_ablations_comparison.png"
    plt.savefig(out_plot, dpi=300)
    print(f"Saved figure to: {out_plot}")

    artifact_dir = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
    dest_path = os.path.join(artifact_dir, "deep_sea_user_ablations_comparison.png")
    shutil.copy(out_plot, dest_path)
    print(f"Copied figure to artifact directory: {dest_path}")

if __name__ == "__main__":
    main()
