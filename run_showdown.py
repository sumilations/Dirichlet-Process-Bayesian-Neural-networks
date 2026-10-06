import argparse
import json
import os
import sys
import time
import numpy as np

# Ensure root directory in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.envs.wheel_bandit import WheelBandit
from src.models.dp_bnn import DPBNNAgent
from src.models.baselines import (
    EpsilonGreedyAgent,
    DeepEnsembleAgent,
    NeuralLinearAgent,
    RandomizedPriorEnsembleAgent
)


class UniformRandomAgent:
    """Uniform Random Exploration baseline."""
    def __init__(self, num_arms=5, seed=None):
        self.num_arms = num_arms
        self.rng = np.random.RandomState(seed)

    def select_action(self, context):
        return self.rng.randint(0, self.num_arms)

    def update(self, context, action, reward):
        pass


def get_agent(agent_name, seed, delta, truncation_K=50, alpha=10.0):
    if agent_name == "DP-BNN":
        # DP-BNN with domain-informed optimistic prior in F_0
        return DPBNNAgent(
            context_dim=2,
            num_arms=5,
            hidden_dim=64,
            alpha=alpha,
            truncation_K=truncation_K,
            prior_reward_mean=4.0,  # Optimistic prior reward
            prior_reward_std=1.0,
            lr=0.01,
            steps_per_decision=2,
            seed=seed
        )
    elif agent_name == "BootDQN-RP":
        return RandomizedPriorEnsembleAgent(
            context_dim=2,
            num_arms=5,
            num_models=5,
            hidden_dim=64,
            prior_scale=3.0,
            lr=0.01,
            steps_per_decision=2,
            seed=seed
        )
    elif agent_name == "Deep-Ensemble":
        return DeepEnsembleAgent(
            context_dim=2,
            num_arms=5,
            num_models=5,
            hidden_dim=64,
            lr=0.01,
            steps_per_decision=2,
            seed=seed
        )
    elif agent_name == "Neural-Linear":
        return NeuralLinearAgent(
            context_dim=2,
            num_arms=5,
            hidden_dim=64,
            feature_dim=32,
            prior_variance=1.0,
            noise_variance=0.01,
            lr=0.01,
            steps_per_decision=2,
            seed=seed
        )
    elif agent_name == "Eps-Greedy":
        return EpsilonGreedyAgent(
            context_dim=2,
            num_arms=5,
            hidden_dim=64,
            epsilon=0.05,
            lr=0.01,
            steps_per_decision=2,
            seed=seed
        )
    elif agent_name == "Uniform-Random":
        return UniformRandomAgent(num_arms=5, seed=seed)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def run_experiment(agent_name, num_steps, delta, seed):
    env = WheelBandit(delta=delta, seed=seed)
    agent = get_agent(agent_name, seed, delta)

    cumulative_regret = np.zeros(num_steps, dtype=np.float32)
    optimal_action_chosen = np.zeros(num_steps, dtype=np.float32)
    outside_delta_optimal_chosen = []

    total_reg = 0.0
    start_time = time.time()

    for t in range(num_steps):
        context = env.sample_context()
        action = agent.select_action(context)
        reward, exp_reward, opt_exp_reward, regret = env.step(context, action)
        agent.update(context, action, reward)

        total_reg += regret
        cumulative_regret[t] = total_reg

        is_opt = (exp_reward == opt_exp_reward)
        optimal_action_chosen[t] = 1.0 if is_opt else 0.0

        if np.linalg.norm(context) > delta:
            outside_delta_optimal_chosen.append(1.0 if is_opt else 0.0)

    elapsed = time.time() - start_time
    outside_opt_rate = float(np.mean(outside_delta_optimal_chosen)) if outside_delta_optimal_chosen else 0.0

    return {
        "cumulative_regret": cumulative_regret.tolist(),
        "final_regret": float(total_reg),
        "overall_opt_rate": float(np.mean(optimal_action_chosen)),
        "outside_delta_opt_rate": outside_opt_rate,
        "elapsed_sec": elapsed
    }


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian Bandit Showdown on Wheel Bandit")
    parser.add_argument("--num_steps", type=int, default=1500, help="Number of bandit steps per run")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--delta", type=float, default=0.7, help="Radius threshold for Wheel Bandit")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save output results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    agent_names = ["DP-BNN", "BootDQN-RP", "Neural-Linear", "Deep-Ensemble", "Eps-Greedy", "Uniform-Random"]
    all_results = {}

    print(f"Starting Wheel Bandit Showdown: Steps={args.num_steps}, Seeds={args.num_seeds}, Delta={args.delta}")
    print("=" * 70)

    for agent_name in agent_names:
        print(f"\nEvaluating {agent_name}:")
        all_results[agent_name] = {
            "seeds": [],
            "final_regrets": [],
            "overall_opt_rates": [],
            "outside_opt_rates": [],
            "all_cumulative_regrets": []
        }

        for seed_idx in range(args.num_seeds):
            seed = 100 + seed_idx * 42
            res = run_experiment(agent_name, args.num_steps, args.delta, seed)
            all_results[agent_name]["seeds"].append(seed)
            all_results[agent_name]["final_regrets"].append(res["final_regret"])
            all_results[agent_name]["overall_opt_rates"].append(res["overall_opt_rate"])
            all_results[agent_name]["outside_opt_rates"].append(res["outside_delta_opt_rate"])
            all_results[agent_name]["all_cumulative_regrets"].append(res["cumulative_regret"])

            print(f"  [Seed {seed_idx+1}/{args.num_seeds}] Final Regret: {res['final_regret']:.1f} | "
                  f"Outside Delta Opt Rate: {res['outside_delta_opt_rate']*100:.1f}% | "
                  f"Time: {res['elapsed_sec']:.1f}s")

        mean_final_reg = np.mean(all_results[agent_name]["final_regrets"])
        std_final_reg = np.std(all_results[agent_name]["final_regrets"]) / np.sqrt(args.num_seeds)
        mean_outside_opt = np.mean(all_results[agent_name]["outside_opt_rates"]) * 100
        print(f"  --> {agent_name} Mean Final Regret: {mean_final_reg:.1f} +/- {std_final_reg:.1f} | "
              f"Outside Opt Rate: {mean_outside_opt:.1f}%")

    # Compute summary statistics
    summary = {}
    for agent_name in agent_names:
        reg_matrix = np.array(all_results[agent_name]["all_cumulative_regrets"])  # [seeds, steps]
        summary[agent_name] = {
            "mean_regret": np.mean(reg_matrix, axis=0).tolist(),
            "stderr_regret": (np.std(reg_matrix, axis=0) / np.sqrt(args.num_seeds)).tolist(),
            "mean_final_regret": float(np.mean(all_results[agent_name]["final_regrets"])),
            "stderr_final_regret": float(np.std(all_results[agent_name]["final_regrets"]) / np.sqrt(args.num_seeds)),
            "mean_outside_opt_rate": float(np.mean(all_results[agent_name]["outside_opt_rates"]))
        }

    out_file = os.path.join(args.out_dir, "showdown_results.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Results successfully saved to {out_file}")


if __name__ == "__main__":
    main()
