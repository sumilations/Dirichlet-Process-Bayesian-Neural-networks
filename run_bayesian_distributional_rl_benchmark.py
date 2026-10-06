"""
Benchmark Runner: Bayesian Distributional RL with Dirichlet Processes (Algorithm 5)
vs. Standard Q-Learning and Baselines on RiverSwim
Reference: Vashishtha PhD Thesis, Section 4.5 (Eq. 4.15 - 4.16 & Algorithm 5)
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.bayesian_distributional_rl.dp_posterior import BaseMeasure, sample_dp_posterior, evaluate_functional
from src.bayesian_distributional_rl.agent import BayesianDistributionalRL_DP
from src.bayesian_distributional_rl.environments import RiverSwimEnv


class StandardQLearning:
    """Standard Q-learning with fixed geometric discount gamma and epsilon-greedy."""
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        init_val: float = 0.0,
        seed: int = 42
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.full((num_states, num_actions), init_val, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def select_action(self, s: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(self.num_actions))
        max_q = np.max(self.q_table[s])
        best = np.where(np.isclose(self.q_table[s], max_q))[0]
        return int(self.rng.choice(best))

    def update_step(self, s: int, a: int, r: float, s_next: int, done: bool):
        target = r if done or s_next is None else r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s, a] += self.lr * (target - self.q_table[s, a])


def run_single_experiment(agent_type: str, env_seed: int, num_episodes: int = 300, n_states: int = 6):
    env = RiverSwimEnv(n_states=n_states, max_steps=40, seed=env_seed)
    
    if agent_type == "DP_RL_Thompson":
        # Algorithm 5: phi_1 = mean (Thompson sampling over DP random measure)
        agent = BayesianDistributionalRL_DP(
            num_states=env.num_states,
            num_actions=env.num_actions,
            alpha_0=5.0,
            base_measure=BaseMeasure("optimistic", loc=1.0, scale=0.3),
            phi_1="mean",
            phi_2="mean",
            synthesis_mode="direct_union",
            seed=env_seed
        )
    elif agent_type == "DP_RL_Quantile":
        # Algorithm 5: phi_1 = Upper Quantile (tau=0.85) for optimistic exploration
        agent = BayesianDistributionalRL_DP(
            num_states=env.num_states,
            num_actions=env.num_actions,
            alpha_0=5.0,
            base_measure=BaseMeasure("optimistic", loc=1.0, scale=0.3),
            phi_1="quantile",
            phi_1_kwargs={"tau": 0.85},
            phi_2="mean",
            synthesis_mode="direct_union",
            seed=env_seed
        )
    elif agent_type == "DP_RL_Bellman":
        # Algorithm 5 with explicit Eq. 4.15 distributional Bellman return synthesis
        agent = BayesianDistributionalRL_DP(
            num_states=env.num_states,
            num_actions=env.num_actions,
            alpha_0=5.0,
            base_measure=BaseMeasure("optimistic", loc=1.0, scale=0.3),
            phi_1="mean",
            phi_2="mean",
            synthesis_mode="distributional_bellman",
            seed=env_seed
        )
    elif agent_type == "QLearning_EpsGreedy":
        agent = StandardQLearning(
            num_states=env.num_states,
            num_actions=env.num_actions,
            lr=0.1,
            gamma=0.95,
            epsilon=0.1,
            init_val=0.0,
            seed=env_seed
        )
    elif agent_type == "QLearning_Optimistic":
        agent = StandardQLearning(
            num_states=env.num_states,
            num_actions=env.num_actions,
            lr=0.1,
            gamma=0.95,
            epsilon=0.02,
            init_val=2.0,  # Optimistic initialization
            seed=env_seed
        )
    elif agent_type == "Random":
        agent = None
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    ep_returns = []
    cum_returns = []
    goal_visits = []
    variances = []
    discounts = []
    
    cum_r = 0.0
    total_goals = 0

    for ep in range(num_episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        
        while not done:
            if agent is None:
                a = int(env.rng.choice(env.num_actions))
            else:
                a = agent.select_action(s)
                
            s_next, r, done, _ = env.step(a)
            ep_r += r
            
            if s == n_states - 1 and a == 1 and r > 0:
                total_goals += 1
                
            if agent is not None:
                if isinstance(agent, BayesianDistributionalRL_DP):
                    agent.update_step(s, a, r, s_next if not done else None, done)
                else:
                    agent.update_step(s, a, r, s_next, done)
                    
            s = s_next

        cum_r += ep_r
        ep_returns.append(ep_r)
        cum_returns.append(cum_r)
        goal_visits.append(total_goals)
        
        if isinstance(agent, BayesianDistributionalRL_DP):
            agent.episodes += 1
            diag = agent.record_diagnostics()
            variances.append(diag["mean_variance"])
            discounts.append(diag["mean_emergent_discount"])
        else:
            variances.append(0.0)
            discounts.append(0.95)

    return {
        "ep_returns": ep_returns,
        "cum_returns": cum_returns,
        "goal_visits": goal_visits,
        "variances": variances,
        "discounts": discounts,
        "final_cum_return": cum_r,
        "total_goal_visits": total_goals,
        "agent": agent
    }


def main():
    print("=" * 80)
    print("Starting Bayesian Distributional RL (Algorithm 5) Benchmark")
    print("Vashishtha PhD Thesis Section 4.5: Natural Discounting & DP Posteriors")
    print("=" * 80)

    seeds = [42, 43, 44, 45, 46]
    num_episodes = 250
    n_states = 6
    
    models = [
        ("DP_RL_Thompson", "Bayesian Dist-RL (Alg 5, Thompson)", "#1f77b4", "-"),
        ("DP_RL_Quantile", "Bayesian Dist-RL (Alg 5, Quantile 0.85)", "#2ca02c", "--"),
        ("DP_RL_Bellman", "Bayesian Dist-RL (Eq 4.15 Bellman)", "#9467bd", "-."),
        ("QLearning_Optimistic", "Q-Learning (Optimistic Init, gamma=0.95)", "#ff7f0e", ":"),
        ("QLearning_EpsGreedy", "Q-Learning (eps=0.10, gamma=0.95)", "#d62728", "-"),
        ("Random", "Random Exploration", "#7f7f7f", ":")
    ]
    
    results = {}
    last_agents = {}
    
    t0 = time.time()
    for model_key, label, color, ls in models:
        print(f"Running benchmark for {label} across {len(seeds)} seeds...")
        seed_runs = []
        for s in seeds:
            res = run_single_experiment(model_key, env_seed=s, num_episodes=num_episodes, n_states=n_states)
            last_agents[model_key] = res.pop("agent")
            seed_runs.append(res)
            
        # Aggregate across seeds
        all_cum_ret = np.array([run["cum_returns"] for run in seed_runs])
        all_ep_ret = np.array([run["ep_returns"] for run in seed_runs])
        all_goals = np.array([run["goal_visits"] for run in seed_runs])
        all_vars = np.array([run["variances"] for run in seed_runs])
        all_discounts = np.array([run["discounts"] for run in seed_runs])
        
        results[model_key] = {
            "label": label,
            "color": color,
            "ls": ls,
            "mean_cum_returns": np.mean(all_cum_ret, axis=0).tolist(),
            "std_cum_returns": np.std(all_cum_ret, axis=0).tolist(),
            "mean_ep_returns": np.mean(all_ep_ret, axis=0).tolist(),
            "std_ep_returns": np.std(all_ep_ret, axis=0).tolist(),
            "mean_goals": np.mean(all_goals, axis=0).tolist(),
            "std_goals": np.std(all_goals, axis=0).tolist(),
            "mean_vars": np.mean(all_vars, axis=0).tolist(),
            "mean_discounts": np.mean(all_discounts, axis=0).tolist(),
            "final_cum_return_mean": float(np.mean(all_cum_ret[:, -1])),
            "final_cum_return_std": float(np.std(all_cum_ret[:, -1])),
            "total_goals_mean": float(np.mean(all_goals[:, -1])),
            "total_goals_std": float(np.std(all_goals[:, -1]))
        }
        print(f"  --> Final Cum Return: {results[model_key]['final_cum_return_mean']:.2f} +/- {results[model_key]['final_cum_return_std']:.2f}")
        print(f"  --> Total State-5 Goals: {results[model_key]['total_goals_mean']:.1f} +/- {results[model_key]['total_goals_std']:.1f}")

    total_time = time.time() - t0
    print(f"\nAll benchmark runs completed in {total_time:.2f} seconds.")

    # ---------------------------------------------------------
    # Generate 4-Panel Publication Figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)
    episodes = np.arange(1, num_episodes + 1)
    
    # Panel A: Cumulative Return Curves
    ax = axes[0, 0]
    for m_key, label, color, ls in models:
        m = np.array(results[m_key]["mean_cum_returns"])
        s = np.array(results[m_key]["std_cum_returns"])
        ax.plot(episodes, m, label=label, color=color, linestyle=ls, linewidth=2.5)
        ax.fill_between(episodes, m - s, m + s, color=color, alpha=0.15)
    ax.set_title(r"(a) Cumulative Return on 6-State RiverSwim ($5$ Seeds)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Cumulative Return", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel B: State 5 Optimal Goal Discoveries
    ax = axes[0, 1]
    for m_key, label, color, ls in models:
        if m_key == "Random":
            continue
        m = np.array(results[m_key]["mean_goals"])
        s = np.array(results[m_key]["std_goals"])
        ax.plot(episodes, m, label=label, color=color, linestyle=ls, linewidth=2.5)
        ax.fill_between(episodes, m - s, m + s, color=color, alpha=0.15)
    ax.set_title(r"(b) Cumulative Far-End Optimal Goal Visits ($s = 5$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Cumulative Optimal Chest Visits", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel C: Dirichlet Process Posterior Contraction (Page 66)
    ax = axes[1, 0]
    dp_models = ["DP_RL_Thompson", "DP_RL_Quantile", "DP_RL_Bellman"]
    for m_key in dp_models:
        v = np.array(results[m_key]["mean_vars"])
        label = results[m_key]["label"]
        color = results[m_key]["color"]
        ax.plot(episodes, v, label=f"{label} Variance", color=color, linewidth=2.5)
    ax.set_title(r"(c) Dirichlet Process Posterior Contraction: $\mathbb{V}[Z(s, a)] \to 0$", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(r"Mean Epistemic Variance $\mathbb{V}[Z(s, a)]$", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Panel D: Emergent Natural Discounting by State (Page 64-65)
    ax = axes[1, 1]
    dp_agent = last_agents["DP_RL_Thompson"]
    if dp_agent is not None:
        state_discounts_swim_left = []
        state_discounts_swim_right = []
        for s in range(n_states):
            state_discounts_swim_left.append(dp_agent.get_emergent_discount(s, 0))
            state_discounts_swim_right.append(dp_agent.get_emergent_discount(s, 1))
            
        x_indices = np.arange(n_states)
        width = 0.35
        ax.bar(x_indices - width/2, state_discounts_swim_left, width, label=r"Action $0$ (Swim Left)", color="#4c72b0")
        ax.bar(x_indices + width/2, state_discounts_swim_right, width, label=r"Action $1$ (Swim Right)", color="#55a868")
        ax.axhline(0.95, color="red", linestyle="--", linewidth=1.8, label=r"Fixed Bellman $\gamma = 0.95$")
        ax.set_title(r"(d) Emergent Natural Discounting: $\mathbb{E}[1 - V] = \frac{\alpha(s, a)}{1 + \alpha(s, a)}$", fontsize=13, fontweight="bold")
        ax.set_xlabel("State Index $s \in \{0, \dots, 5\}$", fontsize=11)
        ax.set_ylabel(r"Emergent Discount Factor $\mathbb{E}[1 - V]$", fontsize=11)
        ax.set_ylim(0.75, 1.01)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"s={i}" for i in range(n_states)])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    fig_suptitle = "Bayesian Distributional RL with Dirichlet Processes (Vashishtha PhD Thesis, Algorithm 5)\nCanonical Benchmark on RiverSwim Exploration MDP"
    fig.suptitle(fig_suptitle, fontsize=15, fontweight="bold", y=0.98)

    # Output paths
    artifact_dir = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
    workspace_results_dir = "/Users/sumitvashishtha/Desktop/DP-BNNs/results"
    os.makedirs(workspace_results_dir, exist_ok=True)
    
    fig_art = os.path.join(artifact_dir, "bayesian_dist_rl_showdown.png")
    fig_work = os.path.join(workspace_results_dir, "bayesian_dist_rl_showdown.png")
    plt.savefig(fig_art, dpi=300, bbox_inches="tight")
    plt.savefig(fig_work, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to:\n  - {fig_art}\n  - {fig_work}")

    # Save summary JSON
    summary_data = {
        "benchmark": "RiverSwim-6",
        "num_episodes": num_episodes,
        "seeds": seeds,
        "runtime_seconds": total_time,
        "models": {
            k: {
                "label": v["label"],
                "final_cum_return_mean": v["final_cum_return_mean"],
                "final_cum_return_std": v["final_cum_return_std"],
                "total_goals_mean": v["total_goals_mean"],
                "total_goals_std": v["total_goals_std"]
            }
            for k, v in results.items()
        }
    }
    
    json_art = os.path.join(artifact_dir, "bayesian_dist_rl_summary.json")
    json_work = os.path.join(workspace_results_dir, "bayesian_dist_rl_summary.json")
    with open(json_art, "w") as f:
        json.dump(summary_data, f, indent=2)
    with open(json_work, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON saved to:\n  - {json_art}\n  - {json_work}")


if __name__ == "__main__":
    main()
