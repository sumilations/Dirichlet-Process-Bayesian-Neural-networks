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


def evaluate_truncation_horizon(K_trunc, num_steps=1500, num_seeds=5, delta=0.9):
    all_regrets = []
    all_outside_opt = []
    start_time = time.time()

    for seed_idx in range(num_seeds):
        seed = 100 + seed_idx * 42
        env = WheelBandit(delta=delta, seed=seed)
        agent = DPBNNAgent(
            context_dim=2,
            num_arms=5,
            hidden_dim=64,
            alpha=10.0,
            truncation_K=K_trunc,
            prior_reward_mean=8.0,
            prior_reward_std=1.0,
            lr=0.01,
            steps_per_decision=2,
            seed=seed
        )

        total_regret = 0.0
        outside_opt = []

        for t in range(num_steps):
            ctx = env.sample_context()
            act = agent.select_action(ctx)
            rew, exp_rew, opt_exp_rew, reg = env.step(ctx, act)
            agent.update(ctx, act, rew)

            total_regret += reg
            if np.linalg.norm(ctx) > delta:
                outside_opt.append(1.0 if (exp_rew == opt_exp_rew) else 0.0)

        all_regrets.append(total_regret)
        all_outside_opt.append(np.mean(outside_opt) if outside_opt else 0.0)

    elapsed = (time.time() - start_time) / num_seeds
    mean_reg = float(np.mean(all_regrets))
    std_err = float(np.std(all_regrets) / np.sqrt(num_seeds))
    mean_opt = float(np.mean(all_outside_opt) * 100)

    return {
        "K_trunc": K_trunc,
        "mean_regret": mean_reg,
        "std_err": std_err,
        "mean_outside_opt": mean_opt,
        "avg_time_sec": float(elapsed)
    }


def main():
    K_values = [10, 25, 50, 100, 150, 200]
    results = []

    print("Sweeping Truncation Horizon K_t on Wheel Bandit (delta=0.9):")
    print("=" * 70)

    for K in K_values:
        res = evaluate_truncation_horizon(K)
        results.append(res)
        print(f"K_t = {K:3d} | Regret: {res['mean_regret']:6.1f} +/- {res['std_err']:5.1f} | "
              f"Outside Opt: {res['mean_outside_opt']:4.1f}% | Time: {res['avg_time_sec']:.2f}s")

    os.makedirs("results", exist_ok=True)
    with open("results/truncation_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    ks = [r["K_trunc"] for r in results]
    regrets = [r["mean_regret"] for r in results]
    errs = [r["std_err"] for r in results]
    opts = [r["mean_outside_opt"] for r in results]
    times = [r["avg_time_sec"] for r in results]

    # Regret & Discovery vs K_t
    ax1 = axes[0]
    c1 = "#1f77b4"
    ax1.errorbar(ks, regrets, yerr=errs, fmt='-o', color=c1, linewidth=2.2, markersize=7, capsize=5, label="Cumulative Regret")
    ax1.set_xlabel(r"Stick-Breaking Truncation Horizon ($K_t$)", fontsize=11)
    ax1.set_ylabel("Final Cumulative Regret", fontsize=11, color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title(r"Performance vs Truncation Horizon $K_t$", fontsize=12, fontweight="bold")

    ax1_twin = ax1.twinx()
    c2 = "#2ca02c"
    ax1_twin.plot(ks, opts, '--s', color=c2, linewidth=2.0, markersize=6, label="Outer Opt Discovery %")
    ax1_twin.set_ylabel("Outer Quadrant Discovery Rate (%)", fontsize=11, color=c2)
    ax1_twin.tick_params(axis='y', labelcolor=c2)

    best_idx = regrets.index(min(regrets))
    ax1.annotate(f"Optimal $K_t = {ks[best_idx]}$\n({regrets[best_idx]:.0f} regret)",
                 xy=(ks[best_idx], regrets[best_idx]),
                 xytext=(15, 20), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                 fontsize=9, fontweight="bold")

    # Runtime scaling
    ax2 = axes[1]
    c3 = "#d62728"
    ax2.plot(ks, times, '-^', color=c3, linewidth=2.2, markersize=7)
    ax2.set_xlabel(r"Stick-Breaking Truncation Horizon ($K_t$)", fontsize=11)
    ax2.set_ylabel("Execution Time per Run (seconds)", fontsize=11, color=c3)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title(r"Computational Cost vs $K_t$", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_plot = "results/truncation_sweep.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Plot saved to {out_plot}")


if __name__ == "__main__":
    main()
