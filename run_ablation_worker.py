"""Worker script running a single experimental condition for user ablations on Deep Sea."""

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

# Pin to 1 physical CPU core
torch.set_num_threads(1)

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--sgd_per_episode", type=int, default=None)
    parser.add_argument("--warm_start_target", type=str2bool, default=True)
    parser.add_argument("--target_update_freq", type=int, default=1)
    parser.add_argument("--target_tau", type=float, default=1.0)
    parser.add_argument("--prior_reward_mean", type=float, default=0.1)
    parser.add_argument("--prior_reward_std", type=float, default=0.05)
    parser.add_argument("--use_thompson_bias", type=str2bool, default=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()

    print(f"[Worker CPU | Exp {args.exp_id}] Starting {args.name} on Seed {args.seed} (Ep: {args.episodes})", flush=True)

    env = DeepSea(size=args.size, deterministic=True, randomize_actions=True, seed=args.seed, mapping_seed=args.seed)
    agent = BatchedDPDQNDeepSeaAgent(
        size=args.size,
        hidden_dim=max(20, args.size),
        alpha=5.0,
        prior_reward_mean=args.prior_reward_mean,
        prior_reward_std=args.prior_reward_std,
        candidate_batch_size=128,
        batch_size=32,
        gamma=0.99,
        tau=0.05,
        lr=1e-3,
        sgd_period=2,
        capacity=30000,
        use_layer_norm=True,
        presample_and_batch=True,
        sgd_per_episode=args.sgd_per_episode,
        warm_start_target=args.warm_start_target,
        target_update_freq=args.target_update_freq,
        target_tau=args.target_tau if args.target_update_freq > 1 else None,
        use_thompson_bias=args.use_thompson_bias,
        seed=args.seed,
    )

    t0 = time.time()
    returns = []
    regrets = []
    opt_return = env._optimal_return
    first_discovery = None
    solved_ep = None

    for ep in range(args.episodes):
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
            print(f"  >>> [Exp {args.exp_id} | {args.name[:18]}] First Treasure at Ep {first_discovery:4d} ({time.time()-t0:.1f}s)", flush=True)

        if len(returns) >= 20 and np.mean(returns[-20:]) > 0.5 and solved_ep is None:
            solved_ep = ep + 1 - 10
            print(f"  >>> [Exp {args.exp_id} | {args.name[:18]}] SOLVED at Ep {solved_ep:4d} ({time.time()-t0:.1f}s)", flush=True)

        if (ep + 1) % 1000 == 0:
            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / elapsed
            print(f"  [Exp {args.exp_id} | {args.name[:18]}] Ep {ep+1:5d} | Ret(50): {np.mean(returns[-50:]):.4f} | {elapsed:4.1f}s ({eps_per_sec:4.1f} eps/s)", flush=True)

    elapsed_total = time.time() - t0
    episodes_per_sec = args.episodes / elapsed_total

    step_sub = max(1, args.episodes // 1000)
    subsampled_regrets = [float(np.mean(regrets[i:i+step_sub])) for i in range(0, len(regrets), step_sub)]

    print(f"[Worker CPU | Exp {args.exp_id}] FINISHED {args.name}: {elapsed_total:.1f}s | {episodes_per_sec:.1f} eps/s", flush=True)

    res = {
        "exp_id": args.exp_id,
        "name": args.name,
        "seed": args.seed,
        "size": args.size,
        "num_episodes": args.episodes,
        "sgd_per_episode": args.sgd_per_episode,
        "warm_start_target": args.warm_start_target,
        "target_update_freq": args.target_update_freq,
        "target_tau": args.target_tau,
        "use_thompson_bias": args.use_thompson_bias,
        "solved": solved_ep is not None,
        "learn_time": solved_ep,
        "first_discovery": first_discovery,
        "elapsed_seconds": elapsed_total,
        "episodes_per_sec": episodes_per_sec,
        "cumulative_regret": float(np.sum(regrets)),
        "final_50ep_return": float(np.mean(returns[-50:])),
        "subsampled_regrets": subsampled_regrets,
    }

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[Worker CPU | Exp {args.exp_id}] Results saved to {args.out_file}", flush=True)

if __name__ == "__main__":
    main()
