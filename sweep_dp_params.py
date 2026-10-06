import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.envs.wheel_bandit import WheelBandit
from src.models.dp_bnn import DPBNNAgent


def evaluate_dp_config(alpha, prior_mean, prior_std, truncation_K, steps_per_decision, num_steps=1500, num_seeds=5, delta=0.9):
    all_regrets = []
    all_outside_opt = []

    for seed_idx in range(num_seeds):
        seed = 100 + seed_idx * 42
        env = WheelBandit(delta=delta, seed=seed)
        agent = DPBNNAgent(
            context_dim=2,
            num_arms=5,
            hidden_dim=64,
            alpha=alpha,
            truncation_K=truncation_K,
            prior_reward_mean=prior_mean,
            prior_reward_std=prior_std,
            lr=0.01,
            steps_per_decision=steps_per_decision,
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

    mean_reg = float(np.mean(all_regrets))
    std_err = float(np.std(all_regrets) / np.sqrt(num_seeds))
    mean_opt = float(np.mean(all_outside_opt) * 100)

    return {
        "alpha": alpha,
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "truncation_K": truncation_K,
        "steps_per_decision": steps_per_decision,
        "mean_regret": mean_reg,
        "std_err": std_err,
        "mean_outside_opt": mean_opt
    }


def main():
    print("Sweeping DP-BNN Hyperparameters on Wheel Bandit (delta=0.9):")
    print("=" * 75)

    # 1. Sweep Base Measure Prior Reward Mean: [2.0, 4.0, 8.0, 12.0, 20.0]
    prior_means = [2.0, 4.0, 8.0, 12.0, 20.0]
    results_mean = []
    print("\n--- Sweeping Base Measure Prior Reward Mean (alpha=10, K=50) ---")
    for pm in prior_means:
        res = evaluate_dp_config(alpha=10.0, prior_mean=pm, prior_std=1.0, truncation_K=50, steps_per_decision=2)
        results_mean.append(res)
        print(f"Prior Mean: {pm:4.1f} | Regret: {res['mean_regret']:6.1f} +/- {res['std_err']:5.1f} | Outside Opt: {res['mean_outside_opt']:4.1f}%")

    # 2. Sweep Concentration Parameter Alpha: [3.0, 10.0, 25.0, 50.0]
    best_pm = min(results_mean, key=lambda x: x["mean_regret"])["prior_mean"]
    alphas = [3.0, 10.0, 20.0, 35.0]
    results_alpha = []
    print(f"\n--- Sweeping Concentration Alpha (Prior Mean={best_pm}, K=50) ---")
    for a in alphas:
        res = evaluate_dp_config(alpha=a, prior_mean=best_pm, prior_std=1.0, truncation_K=50, steps_per_decision=2)
        results_alpha.append(res)
        print(f"Alpha: {a:4.1f} | Regret: {res['mean_regret']:6.1f} +/- {res['std_err']:5.1f} | Outside Opt: {res['mean_outside_opt']:4.1f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/dp_sweep_results.json", "w") as f:
        json.dump({"sweep_prior_mean": results_mean, "sweep_alpha": results_alpha}, f, indent=2)

    print("\n" + "=" * 75)
    print("Sweep complete! Saved to results/dp_sweep_results.json")


if __name__ == "__main__":
    main()
