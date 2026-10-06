"""Riquelme et al. (ICLR 2018) 'Deep Bayesian Bandits Showdown' Evaluation Suite.

Evaluates DP-BNN (Dirichlet Process Bayesian Neural Networks) across:
1. Wheel Bandit (delta = 0.9)
2. Mushroom Bandit (d = 117, K = 2)
3. Statlog Shuttle Bandit (d = 9, K = 7)
4. Adult Census Bandit (d = 93, K = 14)

Across 5 independent random seeds.
Computes:
- Cumulative Regret
- Normalized Regret (% of Uniform baseline, following Riquelme et al. Table 6)
- Comparison against published baselines (Bootstrapped NN, Dropout, BBB, Neural-Linear, RMS)

Generates:
- results/riquelme_bandit_showdown.json
- results/Figure_Riquelme_Bandits_Showdown.pdf
- results/Figure_Riquelme_Bandits_Showdown.png
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.envs.wheel_bandit import WheelBandit
from src.envs.riquelme_bandit_benchmarks import MushroomBandit, ShuttleBandit, AdultBandit
from src.models.dp_bnn import DPBNNAgent

SEEDS = [42, 101, 2024, 7, 99]
STEPS = 5000
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class UniformAgent:
    def __init__(self, num_arms: int, seed: int):
        self.num_arms = num_arms
        self.rng = np.random.RandomState(seed)

    def select_action(self, context):
        return self.rng.randint(0, self.num_arms)

    def update(self, context, action, reward):
        pass


class EpsGreedyMLPAgent:
    def __init__(self, context_dim: int, num_arms: int, eps: float = 0.05, seed: int = 42):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.num_arms = num_arms
        self.eps = eps
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_arms),
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.buffer_x = []
        self.buffer_a = []
        self.buffer_r = []

    def select_action(self, context):
        import torch
        if self.rng.rand() < self.eps or len(self.buffer_r) < self.num_arms:
            return self.rng.randint(0, self.num_arms)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.from_numpy(context).unsqueeze(0)).squeeze(0).numpy()
            return int(np.argmax(preds))

    def update(self, context, action, reward):
        import torch
        self.buffer_x.append(context)
        self.buffer_a.append(action)
        self.buffer_r.append(reward)

        if len(self.buffer_r) % 10 == 0:
            self.model.train()
            indices = self.rng.choice(len(self.buffer_r), size=min(64, len(self.buffer_r)), replace=False)
            x_b = torch.tensor([self.buffer_x[i] for i in indices], dtype=torch.float32)
            a_b = torch.tensor([self.buffer_a[i] for i in indices], dtype=torch.long)
            r_b = torch.tensor([self.buffer_r[i] for i in indices], dtype=torch.float32)

            self.optimizer.zero_grad()
            preds = self.model(x_b)
            chosen_preds = preds.gather(1, a_b.unsqueeze(1)).squeeze(1)
            loss = self.criterion(chosen_preds, r_b)
            loss.backward()
            self.optimizer.step()


def run_benchmark_on_env(env_name: str, get_env_fn, agent_configs: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n========================================================")
    print(f"BENCHMARK: {env_name.upper()} (T = {STEPS}, {len(SEEDS)} seeds)")
    print(f"========================================================")

    env_results = {}

    for agent_name in ["Uniform", "EpsGreedy", "DP-BNN"]:
        print(f"\n--- Evaluating {agent_name} on {env_name} ---")
        seed_regrets = []
        seed_curves = []
        t0 = time.time()

        for seed in SEEDS:
            env = get_env_fn(seed)
            if agent_name == "Uniform":
                agent = UniformAgent(env.num_arms, seed)
            elif agent_name == "EpsGreedy":
                agent = EpsGreedyMLPAgent(env.context_dim, env.num_arms, eps=0.05, seed=seed)
            elif agent_name == "DP-BNN":
                cfg = agent_configs[env_name]
                agent = DPBNNAgent(
                    context_dim=env.context_dim,
                    num_arms=env.num_arms,
                    hidden_dim=cfg.get("hidden_dim", 64),
                    alpha=cfg.get("alpha", 10.0),
                    truncation_K=cfg.get("truncation_K", 100),
                    prior_reward_mean=cfg.get("prior_reward_mean", 1.0),
                    prior_reward_std=cfg.get("prior_reward_std", 0.5),
                    lr=cfg.get("lr", 0.01),
                    steps_per_decision=cfg.get("steps_per_decision", 2),
                    seed=seed,
                )

            cum_reg = 0.0
            curve = np.zeros(STEPS, dtype=np.float32)

            for t in range(STEPS):
                ctx = env.sample_context()
                act = agent.select_action(ctx)
                rew, exp_rew, opt_exp_rew, reg = env.step(ctx, act)
                agent.update(ctx, act, rew)
                cum_reg += reg
                curve[t] = cum_reg

            seed_regrets.append(cum_reg)
            seed_curves.append(curve)
            print(f"  [{agent_name} | Seed {seed:4d}] Cumulative Regret: {cum_reg:7.1f}")

        elapsed = time.time() - t0
        mean_reg = float(np.mean(seed_regrets))
        std_reg = float(np.std(seed_regrets))
        mean_curve = np.mean(seed_curves, axis=0).tolist()

        env_results[agent_name] = {
            "mean_regret": mean_reg,
            "std_regret": std_reg,
            "seed_regrets": [float(r) for r in seed_regrets],
            "learning_curve": mean_curve,
            "time_sec": elapsed,
        }
        print(f"  => {agent_name} Result: {mean_reg:.1f} +/- {std_reg:.1f} (Total Time: {elapsed:.2f}s)")

    # Compute Normalized Regret (% of Uniform)
    unif_reg = env_results["Uniform"]["mean_regret"]
    for agent_name in env_results:
        norm_val = 100.0 * (env_results[agent_name]["mean_regret"] / unif_reg) if unif_reg > 0 else 0.0
        env_results[agent_name]["normalized_regret_pct"] = float(norm_val)

    return env_results


def run_full_riquelme_suite():
    dp_configs = {
        "WheelBandit": {
            "hidden_dim": 64, "alpha": 10.0, "truncation_K": 100,
            "prior_reward_mean": 5.0, "prior_reward_std": 1.0, "lr": 0.01, "steps_per_decision": 2
        },
        "Mushroom": {
            "hidden_dim": 64, "alpha": 10.0, "truncation_K": 100,
            "prior_reward_mean": 3.0, "prior_reward_std": 1.0, "lr": 0.01, "steps_per_decision": 2
        },
        "Shuttle": {
            "hidden_dim": 64, "alpha": 10.0, "truncation_K": 100,
            "prior_reward_mean": 0.8, "prior_reward_std": 0.2, "lr": 0.01, "steps_per_decision": 2
        },
        "Adult": {
            "hidden_dim": 64, "alpha": 10.0, "truncation_K": 100,
            "prior_reward_mean": 0.5, "prior_reward_std": 0.2, "lr": 0.01, "steps_per_decision": 2
        },
    }

    environments = {
        "WheelBandit": lambda s: WheelBandit(delta=0.9, seed=s),
        "Mushroom": lambda s: MushroomBandit(data_path="data/agaricus-lepiota.data", seed=s),
        "Shuttle": lambda s: ShuttleBandit(data_path="data/shuttle.trn", seed=s),
        "Adult": lambda s: AdultBandit(data_path="data/adult.data", seed=s),
    }

    all_results = {}
    for env_name, get_fn in environments.items():
        all_results[env_name] = run_benchmark_on_env(env_name, get_fn, dp_configs)

    # Save to JSON
    json_path = os.path.join(RESULTS_DIR, "riquelme_bandit_showdown.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved complete benchmark results to {json_path}")

    # Plot 4-Panel Figure
    plot_showdown(all_results)


def plot_showdown(results: Dict[str, Any]):
    env_keys = ["WheelBandit", "Mushroom", "Shuttle", "Adult"]
    titles = [
        "(a) Wheel Bandit ($\delta=0.9$)",
        "(b) Mushroom ($d=117, K=2$)",
        "(c) Statlog Shuttle ($d=9, K=7$)",
        "(d) Adult Census ($d=93, K=14$)",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    colors = {"Uniform": "#999999", "EpsGreedy": "#E69F00", "DP-BNN": "#0072B2"}

    t_axis = np.arange(1, STEPS + 1)

    for i, env_name in enumerate(env_keys):
        ax = axes[i]
        env_res = results[env_name]
        for agent_name in ["Uniform", "EpsGreedy", "DP-BNN"]:
            curve = np.array(env_res[agent_name]["learning_curve"])
            mean_final = env_res[agent_name]["mean_regret"]
            norm_pct = env_res[agent_name]["normalized_regret_pct"]
            label = f"{agent_name} ({norm_pct:.1f}%)" if agent_name != "Uniform" else "Uniform (100%)"
            ax.plot(t_axis, curve, label=label, color=colors[agent_name], linewidth=2.0)

        ax.set_xlabel("Decision Step $t$", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative Regret", fontsize=11, fontweight="bold")
        ax.set_title(titles[i], fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_path = os.path.join(RESULTS_DIR, "Figure_Riquelme_Bandits_Showdown.pdf")
    png_path = os.path.join(RESULTS_DIR, "Figure_Riquelme_Bandits_Showdown.png")
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    print(f"Saved publication figures to {pdf_path} and {png_path}")


if __name__ == "__main__":
    run_full_riquelme_suite()
