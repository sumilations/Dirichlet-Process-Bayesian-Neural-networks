"""
Benchmark Runner: DP-BNN Based REINFORCE vs. Standard Policy Gradient Baselines on CartPole-v1
Reference: Vashishtha PhD Thesis Chapter 4 (Algorithm 3 & Section 4.2)
"""

import os
import sys
import json
import time
import gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dp_reinforce.dp_reinforce_agent import DP_BNN_REINFORCE


def run_single_agent(
    agent_type: str,
    seed: int,
    num_episodes: int = 200,
    env_name: str = "CartPole-v1"
):
    env = gym.make(env_name)
    
    # Configure agent type
    if agent_type == "DP_BNN_REINFORCE":
        # Our flagship DP-BNN Policy Gradient (Algorithm 3 stick-breaking weighting + LayerNorm baseline)
        agent = DP_BNN_REINFORCE(
            state_dim=4, action_dim=2, is_continuous=False,
            hidden_dim=64, use_layer_norm=True, use_baseline=True,
            use_dp_weights=True, use_policy_ts=False,
            alpha_dp=2.0, k_trunc_prior=0,
            lr_policy=1e-3, lr_baseline=2e-3, seed=seed
        )
    elif agent_type == "DP_BNN_REINFORCE_LN_Only":
        # Standard REINFORCE with LayerNorm baseline, but uniform weights (no DP)
        agent = DP_BNN_REINFORCE(
            state_dim=4, action_dim=2, is_continuous=False,
            hidden_dim=64, use_layer_norm=True, use_baseline=True,
            use_dp_weights=False, use_policy_ts=False,
            lr_policy=1e-3, lr_baseline=2e-3, seed=seed
        )
    elif agent_type == "Standard_REINFORCE_Baseline":
        # Standard REINFORCE with Baseline (no LayerNorm, uniform weights)
        agent = DP_BNN_REINFORCE(
            state_dim=4, action_dim=2, is_continuous=False,
            hidden_dim=64, use_layer_norm=False, use_baseline=True,
            use_dp_weights=False, use_policy_ts=False,
            lr_policy=1e-3, lr_baseline=2e-3, seed=seed
        )
    elif agent_type == "Vanilla_REINFORCE":
        # Vanilla REINFORCE (Williams 1992: No Baseline, no LayerNorm, uniform weights)
        agent = DP_BNN_REINFORCE(
            state_dim=4, action_dim=2, is_continuous=False,
            hidden_dim=64, use_layer_norm=False, use_baseline=False,
            use_dp_weights=False, use_policy_ts=False,
            lr_policy=1e-3, seed=seed
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    ep_returns = []
    cum_returns = []
    grad_norms = []
    entropies = []
    
    cum_r = 0.0
    solved_ep = None

    for ep in range(1, num_episodes + 1):
        agent.start_episode()
        reset_res = env.reset(seed=seed + ep)
        s = reset_res[0] if isinstance(reset_res, tuple) else reset_res
        
        done = False
        states = []
        actions = []
        rewards = []
        
        while not done:
            a, _ = agent.select_action(s)
            step_res = env.step(a)
            s_next, r = step_res[0], step_res[1]
            done = step_res[2] or (step_res[3] if len(step_res) > 4 else False)
            
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s_next

        metrics = agent.update_with_episode(states, actions, rewards)
        total_ep_r = float(np.sum(rewards))
        cum_r += total_ep_r
        
        ep_returns.append(total_ep_r)
        cum_returns.append(cum_r)
        grad_norms.append(metrics["grad_norm"])
        entropies.append(metrics["mean_entropy"])
        
        if len(ep_returns) >= 20 and np.mean(ep_returns[-20:]) >= 475.0 and solved_ep is None:
            solved_ep = ep

    env.close()
    return {
        "ep_returns": ep_returns,
        "cum_returns": cum_returns,
        "grad_norms": grad_norms,
        "entropies": entropies,
        "solved_ep": solved_ep,
        "final_mean_return": float(np.mean(ep_returns[-20:])),
        "final_cum_return": cum_r
    }


def main():
    print("=" * 80)
    print("Benchmark: DP-BNN REINFORCE vs. Standard Policy Gradient Baselines")
    print("Environment: CartPole-v1 (5 Independent Seeds)")
    print("=" * 80)

    seeds = [42, 43, 44, 45, 46]
    num_episodes = 200

    models = [
        ("DP_BNN_REINFORCE_Full", "DP-BNN REINFORCE (Full: TS + DP-PG + LN)", "#1f77b4", "-"),
        ("DP_BNN_REINFORCE_NoTS", "DP-BNN REINFORCE (DP-PG + LN, No TS)", "#2ca02c", "--"),
        ("Standard_REINFORCE_Baseline", "Standard REINFORCE + Baseline (Williams 1992)", "#ff7f0e", "-."),
        ("Vanilla_REINFORCE", "Vanilla REINFORCE (No Baseline, No DP)", "#d62728", ":")
    ]

    results = {}
    t0 = time.time()

    for m_key, label, color, ls in models:
        print(f"\nEvaluating {label} across {len(seeds)} seeds...")
        seed_runs = []
        solve_eps = []
        for s in seeds:
            t_s = time.time()
            res = run_single_agent(m_key, seed=s, num_episodes=num_episodes)
            elapsed = time.time() - t_s
            seed_runs.append(res)
            s_str = str(res["solved_ep"]) if res["solved_ep"] else "Not Solved"
            print(f"  Seed {s}: Final 20-Ep Return = {res['final_mean_return']:5.1f} | Solved Ep = {s_str:>10s} ({elapsed:.1f}s)")
            if res["solved_ep"]:
                solve_eps.append(res["solved_ep"])

        all_ep_returns = np.array([r["ep_returns"] for r in seed_runs])
        all_cum_returns = np.array([r["cum_returns"] for r in seed_runs])
        all_grad_norms = np.array([r["grad_norms"] for r in seed_runs])
        all_entropies = np.array([r["entropies"] for r in seed_runs])

        results[m_key] = {
            "label": label,
            "color": color,
            "ls": ls,
            "mean_ep_returns": np.mean(all_ep_returns, axis=0).tolist(),
            "std_ep_returns": np.std(all_ep_returns, axis=0).tolist(),
            "mean_cum_returns": np.mean(all_cum_returns, axis=0).tolist(),
            "std_cum_returns": np.std(all_cum_returns, axis=0).tolist(),
            "mean_grad_norms": np.mean(all_grad_norms, axis=0).tolist(),
            "mean_entropies": np.mean(all_entropies, axis=0).tolist(),
            "final_return_mean": float(np.mean([r["final_mean_return"] for r in seed_runs])),
            "final_return_std": float(np.std([r["final_mean_return"] for r in seed_runs])),
            "solved_rate": len(solve_eps) / len(seeds),
            "mean_solved_ep": float(np.mean(solve_eps)) if solve_eps else None
        }

        print(f"  --> Final 20-Ep Return: {results[m_key]['final_return_mean']:.1f} +/- {results[m_key]['final_return_std']:.1f}")
        print(f"  --> Solved Rate: {len(solve_eps)}/{len(seeds)} ({results[m_key]['solved_rate']*100:.0f}%) | Mean Solved Ep: {results[m_key]['mean_solved_ep']}")

    total_time = time.time() - t0
    print(f"\nAll benchmark runs finished in {total_time:.2f} seconds.")

    # ---------------------------------------------------------
    # Generate 4-Panel Publication Figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)
    episodes = np.arange(1, num_episodes + 1)

    # Panel A: Episodic Return Learning Curves (Smoothed)
    ax = axes[0, 0]
    for m_key, label, color, ls in models:
        m = np.array(results[m_key]["mean_ep_returns"])
        s = np.array(results[m_key]["std_ep_returns"])
        # 10-episode moving average for smooth display
        window = 10
        m_smooth = np.convolve(m, np.ones(window)/window, mode='valid')
        s_smooth = np.convolve(s, np.ones(window)/window, mode='valid')
        ep_range = episodes[window-1:]
        ax.plot(ep_range, m_smooth, label=label, color=color, linestyle=ls, linewidth=2.5)
        ax.fill_between(ep_range, np.maximum(0, m_smooth - s_smooth), m_smooth + s_smooth, color=color, alpha=0.15)
    ax.axhline(475.0, color="green", linestyle=":", linewidth=1.5, label="Solved Threshold (475)")
    ax.set_title(r"(a) Episodic Return on CartPole-v1 ($5$ Seeds)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episodic Return (10-Ep MA)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel B: Cumulative Return Curves
    ax = axes[0, 1]
    for m_key, label, color, ls in models:
        m = np.array(results[m_key]["mean_cum_returns"])
        s = np.array(results[m_key]["std_cum_returns"])
        ax.plot(episodes, m, label=label, color=color, linestyle=ls, linewidth=2.5)
        ax.fill_between(episodes, m - s, m + s, color=color, alpha=0.15)
    ax.set_title(r"(b) Cumulative Return on CartPole-v1", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Cumulative Return", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel C: Policy Gradient Norm & Stability
    ax = axes[1, 0]
    for m_key, label, color, ls in models:
        g = np.array(results[m_key]["mean_grad_norms"])
        window = 10
        g_smooth = np.convolve(g, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], g_smooth, label=label, color=color, linestyle=ls, linewidth=2.0)
    ax.set_title(r"(c) Policy Gradient Norm Stability $\|\nabla_\theta J(\theta)\|_2$", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Gradient Norm (10-Ep MA)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Panel D: Policy Entropy Dynamics
    ax = axes[1, 1]
    for m_key, label, color, ls in models:
        h = np.array(results[m_key]["mean_entropies"])
        window = 10
        h_smooth = np.convolve(h, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], h_smooth, label=label, color=color, linestyle=ls, linewidth=2.0)
    ax.set_title(r"(d) Policy Entropy Dynamics $\mathcal{H}(\pi_\theta)$", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Mean Policy Entropy", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("DP-BNN REINFORCE: Dirichlet Process Bayesian Neural Networks for Policy Gradients\nBenchmark Comparison on CartPole-v1 (Williams 1992 vs. Ours)", fontsize=15, fontweight="bold", y=0.98)

    # Save artifacts
    artifact_dir = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
    workspace_results_dir = "/Users/sumitvashishtha/Desktop/DP-BNNs/results"
    os.makedirs(workspace_results_dir, exist_ok=True)

    fig_art = os.path.join(artifact_dir, "dp_reinforce_showdown.png")
    fig_work = os.path.join(workspace_results_dir, "dp_reinforce_showdown.png")
    plt.savefig(fig_art, dpi=300, bbox_inches="tight")
    plt.savefig(fig_work, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to:\n  - {fig_art}\n  - {fig_work}")

    summary_data = {
        "benchmark": "CartPole-v1",
        "num_episodes": num_episodes,
        "seeds": seeds,
        "runtime_seconds": total_time,
        "models": {
            k: {
                "label": v["label"],
                "final_return_mean": v["final_return_mean"],
                "final_return_std": v["final_return_std"],
                "solved_rate": v["solved_rate"],
                "mean_solved_ep": v["mean_solved_ep"]
            }
            for k, v in results.items()
        }
    }

    json_art = os.path.join(artifact_dir, "dp_reinforce_summary.json")
    json_work = os.path.join(workspace_results_dir, "dp_reinforce_summary.json")
    with open(json_art, "w") as f:
        json.dump(summary_data, f, indent=2)
    with open(json_work, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON saved to:\n  - {json_art}\n  - {json_work}")


if __name__ == "__main__":
    main()
