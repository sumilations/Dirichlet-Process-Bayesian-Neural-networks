"""
Comprehensive Benchmark: Bayesian Distributional RL with Dirichlet Processes (Algorithm 5)
on the Deep Sea Exponential Exploration Task (Osband et al., 2018 / 2019)
Reference: Vashishtha PhD Thesis, Section 4.5 & Algorithm 5
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.bayesian_distributional_rl.dp_posterior import BaseMeasure
from src.bayesian_distributional_rl.agent import BayesianDistributionalRL_DP
from src.bayesian_distributional_rl.environments import DeepSeaTabularEnv


class StandardTabularQLearning:
    """Tabular Q-learning with epsilon-greedy dithering or optimistic initialization."""
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        lr: float = 0.1,
        gamma: float = 0.99,
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

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        target = r if done else r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s, a] += self.lr * (target - self.q_table[s, a])


def run_deepsea_run(
    algo_type: str,
    n_scale: int,
    seed: int,
    max_episodes: int = 1500,
    action_mapping: bool = True
):
    env = DeepSeaTabularEnv(n=n_scale, action_mapping=action_mapping, seed=seed)
    optimal_return = 1.0 - 0.01  # +0.99 optimal return
    
    agent = None
    if algo_type == "Bayesian_Dist_RL_Alg5":
        agent = BayesianDistributionalRL_DP(
            num_states=env.num_states,
            num_actions=2,
            alpha_0=1.0,
            base_measure=BaseMeasure("gaussian", loc=1.0, scale=0.1),
            phi_1="mean",
            phi_2="mean",
            synthesis_mode="direct_union",
            max_history_len=100,
            seed=seed
        )
    elif algo_type == "Bayesian_Dist_RL_Trajectory":
        agent = BayesianDistributionalRL_DP(
            num_states=env.num_states,
            num_actions=2,
            alpha_0=2.0,
            base_measure=BaseMeasure("gaussian", loc=1.0, scale=0.1),
            phi_1="mean",
            phi_2="mean",
            synthesis_mode="distributional_bellman",
            max_history_len=100,
            seed=seed
        )
    elif algo_type == "QLearning_Optimistic":
        agent = StandardTabularQLearning(
            num_states=env.num_states,
            num_actions=2,
            lr=0.1,
            gamma=0.99,
            epsilon=0.01,
            init_val=1.5,
            seed=seed
        )
    elif algo_type == "QLearning_EpsGreedy":
        agent = StandardTabularQLearning(
            num_states=env.num_states,
            num_actions=2,
            lr=0.1,
            gamma=0.99,
            epsilon=0.10,
            init_val=0.0,
            seed=seed
        )
    elif algo_type == "Random":
        agent = None
    else:
        raise ValueError(f"Unknown algo: {algo_type}")

    cum_regret = 0.0
    cum_returns = []
    regrets = []
    ep_returns = []
    variances = []
    first_discovery = None
    solved_ep = None
    
    recent_window = []

    for ep in range(1, max_episodes + 1):
        s = env.reset()
        done = False
        traj = []
        ep_r = 0.0
        
        while not done:
            if agent is None:
                a = int(env.rng.choice(2))
            else:
                a = agent.select_action(s)
                
            s_next, r, done, info = env.step(a)
            ep_r += r
            traj.append((s, a, r, s_next, done))
            s = s_next

        # Regret for this episode: optimal_return - ep_r
        regret_t = max(0.0, optimal_return - ep_r)
        cum_regret += regret_t
        regrets.append(cum_regret)
        ep_returns.append(ep_r)
        cum_returns.append(ep_r if len(cum_returns) == 0 else cum_returns[-1] + ep_r)
        
        recent_window.append(ep_r)
        if len(recent_window) > 20:
            recent_window.pop(0)

        if ep_r > 0.5 and first_discovery is None:
            first_discovery = ep

        if len(recent_window) >= 20 and np.mean(recent_window) > 0.8 and solved_ep is None:
            solved_ep = ep

        # Agent updates
        if algo_type == "Bayesian_Dist_RL_Alg5":
            for step_data in traj:
                agent.update_step(step_data[0], step_data[1], step_data[2], step_data[3] if not step_data[4] else None, step_data[4])
            agent.episodes += 1
            diag = agent.record_diagnostics()
            variances.append(diag["mean_variance"])
            
        elif algo_type == "Bayesian_Dist_RL_Trajectory":
            agent.update_trajectory([(t[0], t[1], t[2]) for t in traj])
            agent.episodes += 1
            diag = agent.record_diagnostics()
            variances.append(diag["mean_variance"])
            
        elif algo_type in ["QLearning_Optimistic", "QLearning_EpsGreedy"]:
            for step_data in traj:
                agent.update(step_data[0], step_data[1], step_data[2], step_data[3], step_data[4])
            variances.append(0.0)
        else:
            variances.append(0.0)

    return {
        "regrets": regrets,
        "cum_returns": cum_returns,
        "ep_returns": ep_returns,
        "variances": variances,
        "first_discovery": first_discovery,
        "solved_ep": solved_ep,
        "final_regret": cum_regret,
        "agent": agent
    }


def main():
    print("=" * 80)
    print("Benchmark: Bayesian Distributional RL with Dirichlet Processes on Deep Sea")
    print("Reference: Vashishtha PhD Thesis, Section 4.5 & Algorithm 5")
    print("=" * 80)

    n_scale = 10
    max_episodes = 1200
    seeds = [42, 43, 44, 45, 46]
    
    models = [
        ("Bayesian_Dist_RL_Alg5", "Bayesian Dist-RL (Alg 5, Step-Union)", "#1f77b4", "-"),
        ("Bayesian_Dist_RL_Trajectory", "Bayesian Dist-RL (Eq 4.15 Trajectory)", "#2ca02c", "--"),
        ("QLearning_Optimistic", "Q-Learning (Optimistic Init Q=1.5)", "#ff7f0e", "-."),
        ("QLearning_EpsGreedy", "Q-Learning (eps=0.10 Dithering)", "#d62728", ":"),
        ("Random", "Random Walk (2^-N Policy Hurdle)", "#7f7f7f", ":")
    ]
    
    results = {}
    last_agents = {}
    
    t0 = time.time()
    for m_key, label, color, ls in models:
        print(f"\nRunning {label} on Deep Sea N={n_scale} across {len(seeds)} seeds...")
        seed_runs = []
        disc_eps = []
        solve_eps = []
        
        for s in seeds:
            t_s = time.time()
            res = run_deepsea_run(m_key, n_scale=n_scale, seed=s, max_episodes=max_episodes)
            elapsed_s = time.time() - t_s
            last_agents[m_key] = res.pop("agent")
            seed_runs.append(res)
            
            d_str = str(res["first_discovery"]) if res["first_discovery"] else "Timeout"
            s_str = str(res["solved_ep"]) if res["solved_ep"] else "Timeout"
            print(f"  Seed {s}: Discovery={d_str:>7s}, Solved={s_str:>7s}, Final Regret={res['final_regret']:6.1f} ({elapsed_s:.1f}s)")
            
            if res["first_discovery"]:
                disc_eps.append(res["first_discovery"])
            if res["solved_ep"]:
                solve_eps.append(res["solved_ep"])
                
        all_regrets = np.array([r["regrets"] for r in seed_runs])
        all_cum_ret = np.array([r["cum_returns"] for r in seed_runs])
        all_vars = np.array([r["variances"] for r in seed_runs])
        
        solved_count = len(solve_eps)
        solved_rate = solved_count / len(seeds)
        mean_disc = float(np.mean(disc_eps)) if disc_eps else None
        mean_solve = float(np.mean(solve_eps)) if solve_eps else None
        
        results[m_key] = {
            "label": label,
            "color": color,
            "ls": ls,
            "mean_regrets": np.mean(all_regrets, axis=0).tolist(),
            "std_regrets": np.std(all_regrets, axis=0).tolist(),
            "mean_cum_returns": np.mean(all_cum_ret, axis=0).tolist(),
            "std_cum_returns": np.std(all_cum_ret, axis=0).tolist(),
            "mean_vars": np.mean(all_vars, axis=0).tolist(),
            "final_regret_mean": float(np.mean(all_regrets[:, -1])),
            "final_regret_std": float(np.std(all_regrets[:, -1])),
            "solved_rate": solved_rate,
            "mean_discovery": mean_disc,
            "mean_solved": mean_solve
        }
        
        print(f"  --> Solved Rate: {solved_count}/{len(seeds)} ({solved_rate*100:.0f}%) | Mean Disc: {mean_disc} | Mean Solved: {mean_solve}")
        print(f"  --> Mean Final Regret: {results[m_key]['final_regret_mean']:.1f} +/- {results[m_key]['final_regret_std']:.1f}")

    total_time = time.time() - t0
    print(f"\nAll Deep Sea runs completed in {total_time:.2f} seconds.")

    # ---------------------------------------------------------
    # Generate 4-Panel Publication Figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)
    episodes = np.arange(1, max_episodes + 1)
    
    # Panel A: Cumulative Regret Curves
    ax = axes[0, 0]
    for m_key, label, color, ls in models:
        m = np.array(results[m_key]["mean_regrets"])
        s = np.array(results[m_key]["std_regrets"])
        ax.plot(episodes, m, label=label, color=color, linestyle=ls, linewidth=2.5)
        ax.fill_between(episodes, np.maximum(0, m - s), m + s, color=color, alpha=0.15)
    ax.set_title(r"(a) Cumulative Regret on Deep Sea $N=10$ ($5$ Seeds)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Cumulative Regret", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel B: Cumulative Returns
    ax = axes[0, 1]
    for m_key, label, color, ls in models:
        m = np.array(results[m_key]["mean_cum_returns"])
        s = np.array(results[m_key]["std_cum_returns"])
        ax.plot(episodes, m, label=label, color=color, linestyle=ls, linewidth=2.5)
        ax.fill_between(episodes, m - s, m + s, color=color, alpha=0.15)
    ax.set_title(r"(b) Cumulative Return on Deep Sea $N=10$", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Cumulative Return", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel C: Dirichlet Process Posterior Contraction (Epistemic Uncertainty)
    ax = axes[1, 0]
    for m_key in ["Bayesian_Dist_RL_Alg5", "Bayesian_Dist_RL_Trajectory"]:
        v = np.array(results[m_key]["mean_vars"])
        label = results[m_key]["label"]
        color = results[m_key]["color"]
        ax.plot(episodes, v, label=f"{label} Variance", color=color, linewidth=2.5)
    ax.set_title(r"(c) Dirichlet Process Epistemic Contraction: $\mathbb{V}[Z(s, a)] \to 0$", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(r"Mean Posterior Variance $\mathbb{V}[Z(s, a)]$", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Panel D: Emergent Natural Discounting Profile Across Deep Sea Grid
    ax = axes[1, 1]
    dp_agent = last_agents["Bayesian_Dist_RL_Alg5"]
    if dp_agent is not None:
        row_indices = np.arange(n_scale)
        diag_discounts = []
        offdiag_discounts = []
        for r in range(n_scale):
            # Diagonal state (r, r)
            s_diag = r * n_scale + r
            d_diag = 0.5 * (dp_agent.get_emergent_discount(s_diag, 0) + dp_agent.get_emergent_discount(s_diag, 1))
            diag_discounts.append(d_diag)
            
            # Off-diagonal state (r, 0)
            s_off = r * n_scale + 0
            d_off = 0.5 * (dp_agent.get_emergent_discount(s_off, 0) + dp_agent.get_emergent_discount(s_off, 1))
            offdiag_discounts.append(d_off)
            
        width = 0.35
        ax.bar(row_indices - width/2, diag_discounts, width, label="Diagonal Path (Optimal)", color="#2ca02c")
        ax.bar(row_indices + width/2, offdiag_discounts, width, label="Off-Diagonal (Dead Ends)", color="#7f7f7f")
        ax.axhline(0.99, color="red", linestyle="--", linewidth=1.8, label=r"Standard Fixed $\gamma = 0.99$")
        ax.set_title(r"(d) Emergent Discount Factor Profile: $\mathbb{E}[1 - V] = \frac{\alpha(s, a)}{1 + \alpha(s, a)}$", fontsize=13, fontweight="bold")
        ax.set_xlabel("Depth Row $r \in \{0, \dots, N-1\}$", fontsize=11)
        ax.set_ylabel(r"Emergent Discount Factor $\mathbb{E}[1 - V]$", fontsize=11)
        ax.set_ylim(0.45, 1.02)
        ax.set_xticks(row_indices)
        ax.set_xticklabels([f"r={i}" for i in range(n_scale)])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    fig_suptitle = f"Bayesian Distributional RL with Dirichlet Processes (Algorithm 5)\nDeep Sea Benchmark ($N={n_scale}$, Policy Space = $2^{{{n_scale}}} = {2**n_scale}$)"
    fig.suptitle(fig_suptitle, fontsize=15, fontweight="bold", y=0.98)

    # Save artifacts
    artifact_dir = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
    workspace_results_dir = "/Users/sumitvashishtha/Desktop/DP-BNNs/results"
    os.makedirs(workspace_results_dir, exist_ok=True)
    
    fig_art = os.path.join(artifact_dir, "deepsea_bayesian_dist_rl_showdown.png")
    fig_work = os.path.join(workspace_results_dir, "deepsea_bayesian_dist_rl_showdown.png")
    plt.savefig(fig_art, dpi=300, bbox_inches="tight")
    plt.savefig(fig_work, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to:\n  - {fig_art}\n  - {fig_work}")

    summary_data = {
        "benchmark": f"DeepSea-{n_scale}",
        "policy_space": 2 ** n_scale,
        "max_episodes": max_episodes,
        "seeds": seeds,
        "runtime_seconds": total_time,
        "models": {
            k: {
                "label": v["label"],
                "final_regret_mean": v["final_regret_mean"],
                "final_regret_std": v["final_regret_std"],
                "solved_rate": v["solved_rate"],
                "mean_discovery": v["mean_discovery"],
                "mean_solved": v["mean_solved"]
            }
            for k, v in results.items()
        }
    }
    
    json_art = os.path.join(artifact_dir, "deepsea_bayesian_dist_rl_summary.json")
    json_work = os.path.join(workspace_results_dir, "deepsea_bayesian_dist_rl_summary.json")
    with open(json_art, "w") as f:
        json.dump(summary_data, f, indent=2)
    with open(json_work, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON saved to:\n  - {json_art}\n  - {json_work}")


if __name__ == "__main__":
    main()
