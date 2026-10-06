import json
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.envs.wheel_bandit import WheelBandit
from src.models.dp_bnn import DPBNNAgent


def run_and_plot_final_comparison():
    delta = 0.9
    num_steps = 1500
    num_seeds = 5

    # 1. Load existing baselines from delta=0.9
    with open("results_delta09/showdown_results.json", "r") as f:
        existing_data = json.load(f)

    # 2. Run DP-BNN (Optimized: Kt=200, mu_0=8.0, alpha=10.0)
    print("Running DP-BNN (Optimized, Kt=200) across 5 seeds...")
    opt_cumulative_regrets = []
    opt_final_regrets = []
    opt_outside_rates = []

    for seed_idx in range(num_seeds):
        seed = 100 + seed_idx * 42
        env = WheelBandit(delta=delta, seed=seed)
        agent = DPBNNAgent(
            context_dim=2,
            num_arms=5,
            hidden_dim=64,
            alpha=10.0,
            truncation_K=200,
            prior_reward_mean=8.0,
            prior_reward_std=1.0,
            lr=0.01,
            steps_per_decision=2,
            seed=seed
        )

        total_regret = 0.0
        reg_curve = np.zeros(num_steps, dtype=np.float32)
        outside_opt = []

        for t in range(num_steps):
            ctx = env.sample_context()
            act = agent.select_action(ctx)
            rew, exp_rew, opt_exp_rew, reg = env.step(ctx, act)
            agent.update(ctx, act, rew)

            total_regret += reg
            reg_curve[t] = total_regret

            if np.linalg.norm(ctx) > delta:
                outside_opt.append(1.0 if (exp_rew == opt_exp_rew) else 0.0)

        opt_cumulative_regrets.append(reg_curve.tolist())
        opt_final_regrets.append(float(total_regret))
        opt_outside_rates.append(float(np.mean(outside_opt)) if outside_opt else 0.0)
        print(f"  [Seed {seed_idx+1}/{num_seeds}] Final Regret: {total_regret:.1f}")

    opt_matrix = np.array(opt_cumulative_regrets)

    # 3. Assemble all agents for comparison
    all_agents = {
        "DP-BNN (Optimized, Kt=200)": {
            "mean_regret": np.mean(opt_matrix, axis=0).tolist(),
            "stderr_regret": (np.std(opt_matrix, axis=0) / np.sqrt(num_seeds)).tolist(),
            "mean_final_regret": float(np.mean(opt_final_regrets)),
            "stderr_final_regret": float(np.std(opt_final_regrets) / np.sqrt(num_seeds)),
            "mean_outside_opt_rate": float(np.mean(opt_outside_rates)),
            "color": "#0047AB",       # Cobalt Blue (Primary highlight)
            "ls": "-",
            "lw": 3.0,
            "order": 1
        },
        "BootDQN-RP (Osband et al.)": {
            "mean_regret": existing_data["BootDQN-RP"]["mean_regret"],
            "stderr_regret": existing_data["BootDQN-RP"]["stderr_regret"],
            "mean_final_regret": existing_data["BootDQN-RP"]["mean_final_regret"],
            "stderr_final_regret": existing_data["BootDQN-RP"]["stderr_final_regret"],
            "mean_outside_opt_rate": existing_data["BootDQN-RP"]["mean_outside_opt_rate"],
            "color": "#9467bd",       # Purple
            "ls": "--",
            "lw": 2.2,
            "order": 2
        },
        "DP-BNN (Baseline, Kt=50)": {
            "mean_regret": existing_data["DP-BNN"]["mean_regret"],
            "stderr_regret": existing_data["DP-BNN"]["stderr_regret"],
            "mean_final_regret": existing_data["DP-BNN"]["mean_final_regret"],
            "stderr_final_regret": existing_data["DP-BNN"]["stderr_final_regret"],
            "mean_outside_opt_rate": existing_data["DP-BNN"]["mean_outside_opt_rate"],
            "color": "#17becf",       # Cyan
            "ls": ":",
            "lw": 2.2,
            "order": 3
        },
        "Eps-Greedy (eps=0.05)": {
            "mean_regret": existing_data["Eps-Greedy"]["mean_regret"],
            "stderr_regret": existing_data["Eps-Greedy"]["stderr_regret"],
            "mean_final_regret": existing_data["Eps-Greedy"]["mean_final_regret"],
            "stderr_final_regret": existing_data["Eps-Greedy"]["stderr_final_regret"],
            "mean_outside_opt_rate": existing_data["Eps-Greedy"]["mean_outside_opt_rate"],
            "color": "#d62728",       # Red
            "ls": (0, (3, 1, 1, 1)),
            "lw": 2.0,
            "order": 4
        },
        "Deep-Ensemble (M=5)": {
            "mean_regret": existing_data["Deep-Ensemble"]["mean_regret"],
            "stderr_regret": existing_data["Deep-Ensemble"]["stderr_regret"],
            "mean_final_regret": existing_data["Deep-Ensemble"]["mean_final_regret"],
            "stderr_final_regret": existing_data["Deep-Ensemble"]["stderr_final_regret"],
            "mean_outside_opt_rate": existing_data["Deep-Ensemble"]["mean_outside_opt_rate"],
            "color": "#ff7f0e",       # Orange
            "ls": "-.",
            "lw": 2.0,
            "order": 5
        },
        "Neural-Linear": {
            "mean_regret": existing_data["Neural-Linear"]["mean_regret"],
            "stderr_regret": existing_data["Neural-Linear"]["stderr_regret"],
            "mean_final_regret": existing_data["Neural-Linear"]["mean_final_regret"],
            "stderr_final_regret": existing_data["Neural-Linear"]["stderr_final_regret"],
            "mean_outside_opt_rate": existing_data["Neural-Linear"]["mean_outside_opt_rate"],
            "color": "#2ca02c",       # Green
            "ls": "--",
            "lw": 2.0,
            "order": 6
        },
        "Uniform-Random": {
            "mean_regret": existing_data["Uniform-Random"]["mean_regret"],
            "stderr_regret": existing_data["Uniform-Random"]["stderr_regret"],
            "mean_final_regret": existing_data["Uniform-Random"]["mean_final_regret"],
            "stderr_final_regret": existing_data["Uniform-Random"]["stderr_final_regret"],
            "mean_outside_opt_rate": existing_data["Uniform-Random"]["mean_outside_opt_rate"],
            "color": "#7f7f7f",       # Gray
            "ls": ":",
            "lw": 1.8,
            "order": 7
        }
    }

    # Save to JSON
    with open("results/final_regret_comparison.json", "w") as f:
        json.dump(all_agents, f, indent=2)

    # 4. Generate Publication-Quality Figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Panel A: Cumulative Regret over Time
    ax1 = axes[0]
    steps = np.arange(num_steps)

    for name, d in all_agents.items():
        m = np.array(d["mean_regret"])
        s = np.array(d["stderr_regret"])
        ax1.plot(steps, m, label=name, color=d["color"], linestyle=d["ls"], linewidth=d["lw"])
        ax1.fill_between(steps, m - s, m + s, color=d["color"], alpha=0.12)

    ax1.set_title(r"Wheel Bandit ($\delta = 0.9$): Cumulative Regret Curves", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Rounds / Steps", fontsize=11)
    ax1.set_ylabel("Expected Cumulative Regret", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)

    # Panel B: Final Regret Bar Comparison
    ax2 = axes[1]
    names = list(all_agents.keys())
    finals = [all_agents[k]["mean_final_regret"] for k in names]
    errs = [all_agents[k]["stderr_final_regret"] for k in names]
    bar_colors = [all_agents[k]["color"] for k in names]

    bars = ax2.bar(range(len(names)), finals, yerr=errs, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.1)
    ax2.set_title("Total Cumulative Regret (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Final Regret at T=1500", fontsize=11)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=9.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, finals):
        ax2.annotate(f"{val:.0f}",
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out_file = "results/final_regret_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Final regret comparison plot saved to {out_file}")


if __name__ == "__main__":
    run_and_plot_final_comparison()
