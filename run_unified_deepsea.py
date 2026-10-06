"""Run Deep Sea 10 benchmark using the Unified DP-DQN Framework (src.dp_dqn).

Evaluates the exact winning parameters:
- Base Measure: Uniformly optimistic with r ~ N(0.1, 0.05^2)
- Replay: Candidate sampling with replacement
- LayerNorm: Enabled
- 5 SGD updates per episode (sgd_period=2)
- Target warm-start with 5 steps
"""

import argparse
import json
import os
import shutil
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dp_dqn import DPDQNConfig, DPDQNAgent, train, make_env

# Single CPU execution
torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser(description="Unified DP-DQN Runner for Deep Sea 10")
    parser.add_argument("--size", type=int, default=10, help="Deep Sea grid size N")
    parser.add_argument("--episodes", type=int, default=10000, help="Total episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prior_reward_mean", type=float, default=0.1, help="Prior reward mean")
    parser.add_argument("--prior_reward_std", type=float, default=0.05, help="Prior reward std")
    parser.add_argument("--alpha", type=float, default=5.0, help="DP concentration parameter")
    parser.add_argument("--batch_size", type=int, default=100, help="Stick-breaking truncation parameter K")
    parser.add_argument("--candidate_batch_size", type=int, default=256, help="Candidate pool size B")
    parser.add_argument("--contraction_C", type=float, default=None, help="Scale factor C for posterior contraction (None = no contraction)")
    parser.add_argument("--trajectory_C", type=float, default=None, help="Scale factor C for trajectory denominator: alpha / (alpha + total_count / C)")
    parser.add_argument("--bayesian_alpha", action="store_true", help="Enable Bayesian Beta prior on alpha")
    parser.add_argument("--alpha_prior_a", type=float, default=1.0, help="Prior shape a_0 for Beta prior on theta")
    parser.add_argument("--alpha_evidence_scale", type=float, default=1.0, help="Evidence scale for Bayesian alpha")
    parser.add_argument("--hidden_dim", type=int, default=20, help="Hidden dimension")
    parser.add_argument("--out_dir", type=str, default="results_deepsea", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print(f"UNIFIED DP-DQN FRAMEWORK: DEEP SEA {args.size} BENCHMARK")
    print(f"Grid Size: {args.size}x{args.size} | Seed: {args.seed} | Episodes: {args.episodes:,}")
    print(f"Base Measure: r ~ N({args.prior_reward_mean}, {args.prior_reward_std}^2) uniformly everywhere")
    print(f"Parameters: Alpha = {args.alpha} | Stick Truncation K = {args.batch_size} | Cand Batch B = {args.candidate_batch_size}")
    if args.bayesian_alpha:
        print(f"Bayesian Alpha Prior: theta ~ Beta({args.alpha_prior_a}, b_0) | Evidence Scale = {args.alpha_evidence_scale}")
    print(f"Architecture: Hidden Dim {args.hidden_dim}, Layer Normalization Enabled")
    print("=" * 80)

    # 1. Unified Configuration
    config = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=args.size,
        state_dim=args.size * args.size,
        action_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        alpha=args.alpha,
        batch_size=args.batch_size,
        candidate_batch_size=args.candidate_batch_size,
        base_measure="deep_sea",
        prior_reward_mean=args.prior_reward_mean,
        prior_reward_std=args.prior_reward_std,
        contraction_C=args.contraction_C,
        trajectory_C=args.trajectory_C,
        use_bayesian_alpha=args.bayesian_alpha,
        alpha_prior_a=args.alpha_prior_a,
        alpha_evidence_scale=args.alpha_evidence_scale,
        lr=1e-3,
        gamma=0.99,
        tau=0.05,
        sgd_period=2,
        buffer_capacity=50000,
        target_warmstart=True,
        warmstart_steps=5,
        warmstart_lr_scale=0.5,
        num_episodes=args.episodes,
        max_episode_steps=args.size,
        seed=args.seed,
        eval_frequency=1000,
        eval_episodes=1,
        verbose=False,
    )

    env = make_env(config.env_name, deep_sea_size=args.size, seed=args.seed)
    agent = DPDQNAgent(config=config)

    # Optimal return for Deep Sea N: 1.0 - (N * 0.01 / N) = 0.99
    optimal_return = 1.0 - 0.01

    returns = []
    regrets = []
    first_discovery = None
    solved_ep = None
    t0 = time.time()

    print("\nStarting Training Loop via Unified DPDQNAgent...")

    for ep in range(1, args.episodes + 1):
        agent.reset_episode()
        obs = env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done and step_count < config.max_episode_steps:
            action = agent.act(obs, eval_mode=False)
            next_obs, reward, done, info = env.step(action)
            agent.step(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward
            step_count += 1

        returns.append(ep_reward)
        regret = float(optimal_return - ep_reward)
        regrets.append(regret)

        if ep_reward > 0.5 and first_discovery is None:
            first_discovery = ep
            print(f"  >>> [Discovery] First Treasure at Episode {first_discovery:5d} ({time.time() - t0:.2f}s)!", flush=True)

        if len(returns) >= 20 and np.mean(returns[-20:]) > 0.5 and solved_ep is None:
            solved_ep = ep - 10
            print(f"  >>> [Solved] Policy Converged at Episode {solved_ep:5d} ({time.time() - t0:.2f}s)!", flush=True)

        if ep % 1000 == 0:
            elapsed = time.time() - t0
            eps_sec = ep / elapsed
            rolling_50 = np.mean(returns[-50:])
            cum_reg = np.sum(regrets)
            alpha_str = f" | Alpha: {agent.current_alpha:4.2f} (Psyn: {agent.current_prob_syn*100:4.2f}%)" if args.bayesian_alpha else ""
            print(f"  Ep {ep:5d}/{args.episodes:,} | Ret(50): {rolling_50:+.4f} | Cum Regret: {cum_reg:7.1f}{alpha_str} | {elapsed:4.1f}s ({eps_sec:4.1f} eps/s)", flush=True)

    total_time = time.time() - t0
    episodes_per_sec = args.episodes / total_time
    cumulative_regret = float(np.sum(regrets))
    final_50_return = float(np.mean(returns[-50:]))

    print("\n" + "=" * 80)
    print("TRAINING FINISHED")
    print(f"Total Time: {total_time:.2f}s | Speed: {episodes_per_sec:.1f} eps/s")
    print(f"First Discovery Episode: {first_discovery}")
    print(f"Solved / Converged Episode: {solved_ep}")
    print(f"Final 50-Episode Mean Return: {final_50_return:+.4f}")
    print(f"Total Cumulative Regret: {cumulative_regret:.1f}")
    print("=" * 80)

    # Save JSON results
    res_data = {
        "framework": "Unified DP-DQN (src.dp_dqn)",
        "environment": f"Deep Sea {args.size}x{args.size}",
        "seed": args.seed,
        "episodes": args.episodes,
        "prior_reward_mean": args.prior_reward_mean,
        "prior_reward_std": args.prior_reward_std,
        "first_discovery": first_discovery,
        "solved_episode": solved_ep,
        "cumulative_regret": cumulative_regret,
        "final_50_return": final_50_return,
        "total_time_s": total_time,
        "episodes_per_sec": episodes_per_sec,
    }
    json_path = os.path.join(args.out_dir, f"unified_deepsea{args.size}_results.json")
    with open(json_path, "w") as f:
        json.dump(res_data, f, indent=2)
    print(f"\nSaved metrics JSON to: {json_path}")

    # Plot Figure
    print("Generating comprehensive evaluation figure...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    step_sub = max(1, args.episodes // 1000)
    sub_eps = np.arange(1, len(regrets) + 1, step_sub)
    sub_regrets = [np.mean(regrets[i:i + step_sub]) for i in range(0, len(regrets), step_sub)]
    sub_returns = [np.mean(returns[i:i + step_sub]) for i in range(0, len(returns), step_sub)]
    cum_regrets_curve = np.cumsum(sub_regrets) * step_sub

    # Panel 1: Cumulative Regret
    ax1 = axes[0, 0]
    ax1.plot(sub_eps, cum_regrets_curve, color="#1565C0", linewidth=2.5, label=f"Unified DP-DQN (Final: {cumulative_regret:.0f})")
    if first_discovery is not None:
        disc_idx = min(len(sub_eps) - 1, first_discovery // step_sub)
        ax1.scatter([first_discovery], [cum_regrets_curve[disc_idx]], color="#FF6F00", s=120, marker="*", zorder=5, label=f"First Discovery (Ep {first_discovery})")
    if solved_ep is not None:
        solv_idx = min(len(sub_eps) - 1, solved_ep // step_sub)
        ax1.scatter([solved_ep], [cum_regrets_curve[solv_idx]], color="#2E7D32", s=90, marker="o", zorder=5, label=f"Solved Policy (Ep {solved_ep})")

    ax1.set_title("Cumulative Regret vs. Episode", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Cumulative Regret $\sum (V^* - R_t)$", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, args.episodes)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", framealpha=0.92, fontsize=10)

    # Panel 2: Per-Episode Regret (Rolling Average)
    ax2 = axes[0, 1]
    def smooth(arr, w=9):
        return np.convolve(arr, np.ones(w)/w, mode="same")
    
    ax2.plot(sub_eps, smooth(sub_regrets), color="#E65100", linewidth=2.2, label="Instantaneous Regret (Smoothed)")
    ax2.axhline(optimal_return, color="#9E9E9E", linestyle=":", linewidth=1.5, label=r"Failed Exploration ($V^* - R_{fail} \approx 0.99$)")
    ax2.axhline(0.0, color="#2E7D32", linestyle="--", linewidth=1.8, label=r"Optimal Policy ($V^* - R^* = 0.0$)")
    ax2.set_title("Per-Episode Regret $(V^* - R_t)$", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Regret per Episode", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, args.episodes)
    ax2.set_ylim(-0.05, 1.10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", framealpha=0.92, fontsize=10)

    # Panel 3: Episode Returns
    ax3 = axes[1, 0]
    ax3.plot(sub_eps, smooth(sub_returns), color="#2E7D32", linewidth=2.2, label="Episode Return (Smoothed)")
    ax3.axhline(optimal_return, color="#1565C0", linestyle="--", linewidth=1.5, label=f"Optimal Return (+{optimal_return:.2f})")
    ax3.set_title("Episode Return Progression", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Undiscounted Return", fontsize=11, fontweight="bold")
    ax3.set_xlim(0, args.episodes)
    ax3.set_ylim(-0.02, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right", framealpha=0.92, fontsize=10)

    # Panel 4: Benchmark Summary Card
    ax4 = axes[1, 1]
    ax4.axis("off")

    summary_text = (
        "UNIFIED DP-DQN BENCHMARK METRICS\n"
        "--------------------------------------------------\n"
        f"• Framework:           Unified codebase (src.dp_dqn)\n"
        f"• Environment:         Deep Sea {args.size}x{args.size}\n"
        f"• Base Measure:        Uniform r ~ N({args.prior_reward_mean}, {args.prior_reward_std}^2)\n"
        f"• Concentration α:     {args.alpha}\n"
        f"• Stick Truncation K:  {args.batch_size}\n"
        f"• Candidate Pool B:    {args.candidate_batch_size}\n"
        f"• Layer Normalization: ENABLED (LayerNorm({args.hidden_dim}))\n"
        f"• Optimization:        5 SGD/ep (sgd_period=2)\n"
        f"• Target Warm-Start:   ENABLED (5 steps @ episode start)\n"
        f"• Candidate Sampling:  Strictly WITH REPLACEMENT\n"
        "--------------------------------------------------\n"
        f"• First Discovery:     Episode {first_discovery if first_discovery else 'N/A'}\n"
        f"• Solved Episode:      Episode {solved_ep if solved_ep else 'N/A'}\n"
        f"• Total Cum. Regret:   {cumulative_regret:.1f}\n"
        f"• Final 50-Ep Return:  {final_50_return:+.4f} (Max: +{optimal_return:.2f})\n"
        f"• Total Wall Time:     {total_time:.1f} s ({episodes_per_sec:.1f} eps/s)\n"
        f"• Solved Status:       {'SOLVED (Optimal Policy Locked)' if solved_ep else 'NOT SOLVED'}"
    )

    ax4.text(
        0.05, 0.5, summary_text,
        fontsize=11, fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="#ECEFF1", edgecolor="#37474F", linewidth=1.5)
    )

    plt.suptitle(
        f"Unified DP-DQN Deep Sea {args.size}x{args.size} | $\\alpha={args.alpha}$, Stick Truncation $K={args.batch_size}$ | Prior: $r \\sim \\mathcal{{N}}({args.prior_reward_mean}, {args.prior_reward_std}^2)$",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_plot = os.path.join(args.out_dir, f"unified_deepsea{args.size}_regret.png")
    plt.savefig(out_plot, dpi=300)
    print(f"Saved figure to: {out_plot}")

    artifact_dir = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
    dest_path = os.path.join(artifact_dir, f"unified_deepsea{args.size}_regret.png")
    shutil.copy(out_plot, dest_path)
    print(f"Copied figure to artifact directory: {dest_path}")


if __name__ == "__main__":
    main()
