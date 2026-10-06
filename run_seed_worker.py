"""Worker script running 1 seed on Deep Sea N=20 for 4000 episodes (With and Without Pre-sampling)."""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from bsuite.environments.deep_sea import DeepSea
from src.rl.batched_dp_dqn import BatchedDPDQNDeepSeaAgent

# Lock PyTorch to 1 thread per worker process so each worker gets its own dedicated physical CPU core
torch.set_num_threads(1)

def run_experiment(mode: bool, name: str, seed: int, num_episodes: int = 4000, size: int = 20, prior_reward_mean: float = 1.0) -> Dict[str, Any]:
    print(f"[Worker CPU | Seed {seed}] Starting {name} (Ep: {num_episodes}, Size: {size}, PriorReward: {prior_reward_mean})", flush=True)
    env = DeepSea(size=size, deterministic=True, randomize_actions=True, seed=seed, mapping_seed=seed)
    agent = BatchedDPDQNDeepSeaAgent(
        size=size,
        hidden_dim=max(20, size),
        alpha=5.0,
        prior_reward_mean=prior_reward_mean,
        candidate_batch_size=128,
        batch_size=32,
        gamma=0.99,
        tau=0.05,
        lr=1e-3,
        sgd_period=2,
        capacity=30000,
        use_layer_norm=True,
        presample_and_batch=mode,
        seed=seed,
    )

    t0 = time.time()
    returns = []
    regrets = []
    opt_return = env._optimal_return
    first_discovery = None
    solved_ep = None

    for ep in range(num_episodes):
        agent.start_episode()
        ts = env.reset()
        obs = ts.observation
        ep_ret = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_ts = env.step(action)
            reward = float(next_ts.reward or 0.0)
            done = next_ts.last()
            agent.step_update(obs, action, reward, next_ts.observation, done)
            obs = next_ts.observation
            ep_ret += reward

        agent.end_episode()
        returns.append(ep_ret)
        regret = float(opt_return - ep_ret)
        regrets.append(regret)

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep + 1
            print(f"  >>> [Seed {seed} | {name}] First Treasure at Ep {first_discovery:4d} ({time.time()-t0:.1f}s)", flush=True)

        if len(returns) >= 20 and np.mean(returns[-20:]) > 0.5 and solved_ep is None:
            solved_ep = ep + 1 - 10
            print(f"  >>> [Seed {seed} | {name}] SOLVED at Ep {solved_ep:4d} ({time.time()-t0:.1f}s)", flush=True)

        if (ep + 1) % 1000 == 0:
            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / elapsed
            print(f"  [Seed {seed} | {name[:12]}] Ep {ep+1:4d} | Ret: {np.mean(returns[-50:]):.4f} | {elapsed:4.1f}s ({eps_per_sec:4.1f} eps/s)", flush=True)

    elapsed_total = time.time() - t0
    episodes_per_sec = num_episodes / elapsed_total

    subsampled_regrets = [float(np.mean(regrets[i:i+10])) for i in range(0, len(regrets), 10)]

    print(f"[Worker CPU | Seed {seed}] FINISHED {name}: {elapsed_total:.1f}s | {episodes_per_sec:.1f} eps/s", flush=True)

    return {
        "agent": name,
        "size": size,
        "prior_reward_mean": prior_reward_mean,
        "mode": "With Pre-sampling" if mode else "Without Pre-sampling",
        "seed": seed,
        "num_episodes": num_episodes,
        "solved": solved_ep is not None,
        "learn_time": solved_ep,
        "first_discovery": first_discovery,
        "elapsed_seconds": elapsed_total,
        "episodes_per_sec": episodes_per_sec,
        "cumulative_regret": float(np.sum(regrets)),
        "final_50ep_return": float(np.mean(returns[-50:])),
        "subsampled_regrets": subsampled_regrets,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--episodes", type=int, default=4000, help="Number of episodes")
    parser.add_argument("--size", type=int, default=20, help="Grid size N (e.g. 10 or 20)")
    parser.add_argument("--prior_reward", type=float, default=1.0, help="Prior optimistic reward mean")
    args = parser.parse_args()

    os.makedirs("results_deepsea", exist_ok=True)
    r_tag = f"_r{int(args.prior_reward)}" if args.prior_reward != 1.0 else ""
    out_file = f"results_deepsea/seed_{args.seed}_N{args.size}{r_tag}.json"

    # 1. Run Without Pre-sampling
    res_without = run_experiment(
        mode=False,
        name="DP-DQN (Without Pre-sampling)",
        seed=args.seed,
        num_episodes=args.episodes,
        size=args.size,
        prior_reward_mean=args.prior_reward,
    )

    # 2. Run With Pre-sampling
    res_with = run_experiment(
        mode=True,
        name="DP-DQN (With Pre-sampling)",
        seed=args.seed,
        num_episodes=args.episodes,
        size=args.size,
        prior_reward_mean=args.prior_reward,
    )

    with open(out_file, "w") as f:
        json.dump([res_without, res_with], f, indent=2)
    print(f"[Worker CPU | Seed {args.seed}] All runs saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
