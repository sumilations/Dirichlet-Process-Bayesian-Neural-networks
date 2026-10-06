import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.envs.information_mismatch_bandit import InformationMismatchBandit
from src.envs.wheel_bandit import WheelBandit
from src.models.dp_bnn_ids import DPBNN_IDSAgent
from src.models.dp_bnn import DPBNNAgent
from src.models.baselines import RandomizedPriorEnsembleAgent, EpsilonGreedyAgent


def create_agent(agent_name, env_name, context_dim, num_arms, seed):
    if env_name == "mismatch":
        prior_mean = 0.0
        prior_std = 0.5
        noise_std = 0.1
    else:
        prior_mean = 3.0
        prior_std = 1.0
        noise_std = 0.5

    if agent_name == "DP-IDS (Mutual Info)":
        return DPBNN_IDSAgent(
            context_dim=context_dim,
            num_arms=num_arms,
            num_models=20,
            hidden_dim=64,
            alpha=10.0,
            truncation_K=50,
            prior_reward_mean=prior_mean,
            prior_reward_std=prior_std,
            noise_std=noise_std,
            info_type="mutual_information",
            action_selection="randomized",
            seed=seed,
        )
    elif agent_name == "DP-IDS (Variance)":
        return DPBNN_IDSAgent(
            context_dim=context_dim,
            num_arms=num_arms,
            num_models=20,
            hidden_dim=64,
            alpha=10.0,
            truncation_K=50,
            prior_reward_mean=prior_mean,
            prior_reward_std=prior_std,
            noise_std=noise_std,
            info_type="variance",
            action_selection="randomized",
            seed=seed,
        )
    elif agent_name == "DP-TS (Thompson Sampling)":
        return DPBNNAgent(
            context_dim=context_dim,
            num_arms=num_arms,
            hidden_dim=64,
            alpha=10.0,
            truncation_K=50,
            prior_reward_mean=prior_mean,
            prior_reward_std=prior_std,
            seed=seed,
        )
    elif agent_name == "BootDQN-RP":
        return RandomizedPriorEnsembleAgent(
            context_dim=context_dim,
            num_arms=num_arms,
            num_models=5,
            hidden_dim=64,
            prior_scale=3.0,
            seed=seed,
        )
    elif agent_name == "Epsilon-Greedy":
        return EpsilonGreedyAgent(
            context_dim=context_dim,
            num_arms=num_arms,
            hidden_dim=64,
            epsilon=0.05,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown agent {agent_name}")


def run_experiment(env_name, agent_names, num_rounds, seeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    print(f"Starting IDS Showdown on {env_name.upper()}:")
    print(f"Rounds: {num_rounds} | Seeds: {len(seeds)}")
    print("=" * 75)

    for agent_name in agent_names:
        print(f"\nEvaluating {agent_name}...")
        seed_regrets = []
        seed_cum_regrets = []
        seed_actions = []
        total_time = 0.0

        for s_idx, seed in enumerate(seeds):
            t0 = time.time()
            if env_name == "mismatch":
                env = InformationMismatchBandit(delta=1.0, epsilon=0.05, risky_noise=1.0, info_noise=0.05, safe_noise=0.1, context_dim=2, seed=seed)
                context_dim = 2
                num_arms = 3
            elif env_name == "wheel":
                env = WheelBandit(delta=0.9, seed=seed)
                context_dim = 2
                num_arms = 5
            else:
                raise ValueError(f"Unknown environment {env_name}")

            agent = create_agent(agent_name, env_name, context_dim, num_arms, seed)

            regrets = []
            actions_taken = []
            context = env.reset()

            for step in range(num_rounds):
                action = agent.select_action(context)
                next_context, reward, opt_reward, info = env.step(action)
                agent.update(context, action, reward)
                context = next_context

                regret = info["regret"]
                regrets.append(float(regret))
                actions_taken.append(int(action))

            elapsed = time.time() - t0
            total_time += elapsed
            cum_regret = np.cumsum(regrets)

            seed_regrets.append(regrets)
            seed_cum_regrets.append(cum_regret.tolist())
            seed_actions.append(actions_taken)

            print(f"  [Seed {s_idx+1}/{len(seeds)}] Final Cumulative Regret: {cum_regret[-1]:6.2f} | Time: {elapsed:4.1f}s")

        cum_mat = np.array(seed_cum_regrets)
        results[agent_name] = {
            "mean_cum_regret": np.mean(cum_mat, axis=0).tolist(),
            "stderr_cum_regret": (np.std(cum_mat, axis=0) / np.sqrt(len(seeds))).tolist(),
            "final_regret_mean": float(np.mean(cum_mat[:, -1])),
            "final_regret_stderr": float(np.std(cum_mat[:, -1]) / np.sqrt(len(seeds))),
            "total_time_sec": float(total_time),
            "action_distribution": np.bincount(np.array(seed_actions).flatten(), minlength=num_arms).tolist(),
        }

        print(f"--> {agent_name} Final Regret: {results[agent_name]['final_regret_mean']:.2f} +/- {results[agent_name]['final_regret_stderr']:.2f}")

    out_file = os.path.join(out_dir, f"ids_{env_name}_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print(f"Saved benchmark results to {out_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run DP-BNN Information Directed Sampling Benchmark")
    parser.add_argument("--env", type=str, default="mismatch", choices=["mismatch", "wheel"], help="Environment to test")
    parser.add_argument("--rounds", type=int, default=1000, help="Rounds per seed")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--out_dir", type=str, default="results_ids", help="Output directory")
    args = parser.parse_args()

    agent_names = [
        "DP-IDS (Mutual Info)",
        "DP-IDS (Variance)",
        "DP-TS (Thompson Sampling)",
        "BootDQN-RP",
        "Epsilon-Greedy",
    ]

    seeds = [100 + i * 42 for i in range(args.seeds)]
    run_experiment(args.env, agent_names, args.rounds, seeds, args.out_dir)


if __name__ == "__main__":
    main()
