#!/usr/bin/env python3
"""Unified DP-DQN Runner Script.

Run Dirichlet Process Deep Q-Network on any environment by simply specifying
hyperparameters and base measures via CLI or Python configuration.

Examples:
  python3 run_dp_dqn.py --env deep_sea --N 10 --episodes 500
  python3 run_dp_dqn.py --env cartpole_swingup --alpha 15.0 --episodes 1000
  python3 run_dp_dqn.py --env gym:CartPole-v1 --episodes 200
"""

import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure root directory is on PATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dp_dqn import DPDQNConfig, train


def parse_args():
    parser = argparse.ArgumentParser(description="Unified DP-DQN Runner")
    # Environment
    parser.add_argument("--env", type=str, default="deep_sea", help="Environment name: 'deep_sea', 'cartpole_swingup', or 'gym:<id>'")
    parser.add_argument("--N", type=int, default=10, help="Deep Sea problem scale N (if env is deep_sea)")

    # DP-BNN Hyperparameters
    parser.add_argument("--alpha", type=float, default=10.0, help="Dirichlet Process concentration parameter")
    parser.add_argument("--base_measure", type=str, default="default", help="Base measure: 'default', 'cartpole', 'deep_sea', 'gaussian', 'zero'")
    parser.add_argument("--prior_reward_mean", type=float, default=1.0, help="Optimistic prior reward scale in F_0")

    # Neural Network Architecture
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of MLP")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--no_layer_norm", action="store_true", help="Disable Layer Normalization")

    # Training & Optimization
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--sgd_period", type=int, default=2, help="Steps between online TD updates")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size for stick-breaking")
    parser.add_argument("--no_warmstart", action="store_true", help="Disable Target Warm-Start Thompson Sampling")
    parser.add_argument("--warmstart_steps", type=int, default=10, help="Number of gradient steps for target warm-start")

    # Execution
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_freq", type=int, default=50, help="Evaluation frequency in episodes")
    parser.add_argument("--save_json", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--plot", type=str, default=None, help="Path to save training plot PNG")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create config from args
    config = DPDQNConfig(
        env_name=args.env,
        deep_sea_size=args.N,
        alpha=args.alpha,
        base_measure=args.base_measure,
        prior_reward_mean=args.prior_reward_mean,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_layer_norm=not args.no_layer_norm,
        lr=args.lr,
        gamma=args.gamma,
        sgd_period=args.sgd_period,
        batch_size=args.batch_size,
        target_warmstart=not args.no_warmstart,
        warmstart_steps=args.warmstart_steps,
        num_episodes=args.episodes,
        max_episode_steps=args.max_steps,
        seed=args.seed,
        eval_frequency=args.eval_freq,
        verbose=True,
    )

    results = train(config)

    # Save JSON if requested
    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.save_json}")

    # Plot if requested
    if args.plot:
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Returns
        axes[0].plot(results["episode_returns"], alpha=0.3, color="gray", label="Raw Episode Return")
        # Moving average
        window = min(20, len(results["episode_returns"]))
        if window > 1:
            ma = np.convolve(results["episode_returns"], np.ones(window)/window, mode="valid")
            axes[0].plot(range(window - 1, len(results["episode_returns"])), ma, color="#1f77b4", lw=2, label=f"Moving Avg ({window})")
        if results["eval_episodes"]:
            axes[0].plot(results["eval_episodes"], results["eval_returns"], "ro-", lw=2, label="Evaluation Return")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Return")
        axes[0].set_title(f"DP-DQN Return ({config.env_name})")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Average Regret
        axes[1].plot(results["avg_regret_history"], color="#d62728", lw=2, label="Cumulative Avg Regret")
        axes[1].axhline(0.9, color="black", linestyle="--", label="T_learn Threshold (0.9)")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Average Regret")
        axes[1].set_title(f"Regret vs. Threshold (T_learn = {results['t_learn']})")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(args.plot, dpi=200)
        plt.close()
        print(f"Plot saved to: {args.plot}")


if __name__ == "__main__":
    main()
