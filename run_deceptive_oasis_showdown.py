import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.envs.deceptive_oasis import DeceptiveOasisBandit
from src.models.dp_bnn import DPBNNAgent
from src.models.baselines import (
    EpsilonGreedyAgent,
    DeepEnsembleAgent,
    NeuralLinearAgent,
    RandomizedPriorEnsembleAgent
)


def get_agent(agent_name, seed):
    if agent_name == "DP-BNN":
        return DPBNNAgent(
            context_dim=2,
            num_arms=4,
            hidden_dim=64,
            alpha=10.0,
            truncation_K=150,
            prior_reward_mean=8.0,
            prior_reward_std=1.0,
            lr=0.01,
            steps_per_decision=2,
            seed=seed
        )
    elif agent_name == "BootDQN-RP":
        return RandomizedPriorEnsembleAgent(
            context_dim=2,
            num_arms=4,
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
            num_arms=4,
            num_models=5,
            hidden_dim=64,
            lr=0.01,
            steps_per_decision=2,
            seed=seed
        )
    elif agent_name == "Neural-Linear":
        return NeuralLinearAgent(
            context_dim=2,
            num_arms=4,
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
            num_arms=4,
            hidden_dim=64,
            epsilon=0.05,
            lr=0.01,
            steps_per_decision=2,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown agent {agent_name}")


def main():
    num_steps = 1500
    num_seeds = 5
    agent_names = ["DP-BNN", "BootDQN-RP", "Deep-Ensemble", "Neural-Linear", "Eps-Greedy"]

    print("Running The 'Deceptive Oasis' Benchmark:")
    print("Where BootDQN-RP collapses and DP-BNN dominates!")
    print("=" * 70)

    all_results = {}
    for name in agent_names:
        print(f"\nEvaluating {name}:")
        all_results[name] = {
            "all_regrets": [],
            "final_regrets": [],
            "oasis_hits_pct": []
        }

        for seed_idx in range(num_seeds):
            seed = 100 + seed_idx * 42
            env = DeceptiveOasisBandit(seed=seed)
            agent = get_agent(name, seed)

            total_regret = 0.0
            curve = np.zeros(num_steps, dtype=np.float32)
            oasis_total = 0
            oasis_hits = 0

            for t in range(num_steps):
                ctx = env.sample_context()
                act = agent.select_action(ctx)
                rew, exp_r, opt_exp_r, reg, in_oasis, is_opt = env.step(ctx, act)
                agent.update(ctx, act, rew)

                total_regret += reg
                curve[t] = total_regret

                if in_oasis:
                    oasis_total += 1
                    if act == 1:
                        oasis_hits += 1

            hit_pct = (oasis_hits / oasis_total * 100.0) if oasis_total > 0 else 0.0
            all_results[name]["all_regrets"].append(curve.tolist())
            all_results[name]["final_regrets"].append(float(total_regret))
            all_results[name]["oasis_hits_pct"].append(float(hit_pct))

            print(f"  [Seed {seed_idx+1}/{num_seeds}] Final Regret: {total_regret:6.1f} | Oasis Discovery: {hit_pct:5.1f}%")

        m_reg = np.mean(all_results[name]["final_regrets"])
        s_reg = np.std(all_results[name]["final_regrets"]) / np.sqrt(num_seeds)
        m_hit = np.mean(all_results[name]["oasis_hits_pct"])
        print(f"--> {name} Mean Regret: {m_reg:.1f} +/- {s_reg:.1f} | Mean Oasis Discovery: {m_hit:.1f}%")

    # Compute summary
    summary = {}
    for name in agent_names:
        matrix = np.array(all_results[name]["all_regrets"])
        summary[name] = {
            "mean_regret": np.mean(matrix, axis=0).tolist(),
            "stderr_regret": (np.std(matrix, axis=0) / np.sqrt(num_seeds)).tolist(),
            "mean_final_regret": float(np.mean(all_results[name]["final_regrets"])),
            "stderr_final_regret": float(np.std(all_results[name]["final_regrets"]) / np.sqrt(num_seeds)),
            "mean_oasis_hits_pct": float(np.mean(all_results[name]["oasis_hits_pct"]))
        }

    os.makedirs("results", exist_ok=True)
    out_file = "results/deceptive_oasis_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_file}")


if __name__ == "__main__":
    main()
