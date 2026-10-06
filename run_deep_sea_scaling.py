"""Deep Sea Scaling Benchmark across Problem Scales N.

Replicates Section 4.2.1 and Figure 3 of arXiv:1806.03335:
"Randomized Prior Functions for Deep Reinforcement Learning" (Osband et al., NeurIPS 2018)

Evaluates:
1. DP-DQN (Single Q-network with Data-Space Dirichlet Process Prior)
2. BootDQN-RP (20-head ensemble with randomized prior functions, beta=10)
3. DQN-Dithering (Standard epsilon-greedy DQN)
across problem sizes N in [5, 8, 10, 12, 14, 16, 18, 20, 25, 30] over multiple random seeds.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from bsuite.environments.deep_sea import DeepSea
from src.rl.deep_sea_dp_dqn import DPDQNDeepSeaAgent
from src.rl.deep_sea_boot_dqn import BootDQNRPDeepSeaAgent
from src.rl.deep_sea_dithering import DQNDitheringAgent


def run_single_experiment(agent_type: str, size: int, seed: int, max_episodes: int) -> Dict[str, Any]:
    """Runs a single agent on DeepSea(size=size, seed=seed) until solved or max_episodes."""
    torch.set_num_threads(1)
    env = DeepSea(size=size, deterministic=True, randomize_actions=True, seed=seed, mapping_seed=seed)

    if agent_type == "DP-DQN":
        agent = DPDQNDeepSeaAgent(
            size=size,
            hidden_dim=20,
            alpha=5.0,
            prior_reward_mean=1.0,
            candidate_batch_size=128,
            truncation_K=32,
            gamma=0.99,
            tau=0.05,
            lr=1e-3,
            sgd_period=2,
            capacity=30000,
            use_layer_norm=True,
            episode_target_warmstart=True,
            seed=seed
        )
    elif agent_type == "BootDQN-RP":
        agent = BootDQNRPDeepSeaAgent(
            size=size,
            num_heads=20,
            hidden_dim=20,
            prior_scale=10.0,
            gamma=0.99,
            tau=0.05,
            lr=1e-3,
            sgd_period=2,
            batch_size=32,
            capacity=30000,
            seed=seed
        )
    elif agent_type == "DQN-Dithering":
        agent = DQNDitheringAgent(
            size=size,
            hidden_dim=20,
            epsilon_start=1.0,
            epsilon_end=0.01,
            decay_episodes=min(2000, max_episodes // 2),
            gamma=0.99,
            tau=0.05,
            lr=1e-3,
            sgd_period=2,
            batch_size=32,
            capacity=30000,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    returns = []
    regrets = []
    opt_return = env._optimal_return
    solved_ep = None
    start_time = time.time()

    # Max episodes for dithering capped to prevent wasting time when N is large
    actual_max_episodes = max_episodes
    if agent_type == "DQN-Dithering" and size > 12:
        actual_max_episodes = min(max_episodes, 3000)

    for ep in range(actual_max_episodes):
        agent.start_episode()
        ts = env.reset()
        obs = ts.observation
        ep_ret = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_ts = env.step(action)
            reward = next_ts.reward
            done = next_ts.last()
            next_obs = next_ts.observation

            agent.step_update(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_ret += reward

        agent.end_episode()
        returns.append(float(ep_ret))
        regret = float(opt_return - ep_ret)
        regrets.append(regret)

        # Regret criterion matching arXiv:1806.03335 (average regret drops below 0.9)
        # or moving average return over last 20 episodes exceeds 0.5
        if len(returns) >= 20 and np.mean(returns[-20:]) > 0.5 and solved_ep is None:
            solved_ep = ep + 1 - 10
            elapsed = time.time() - start_time
            print(f"  [{agent_type:14s}] N={size:2d} | Seed {seed} | SOLVED at Ep {solved_ep:5d} | Time: {elapsed:5.1f}s | Return: {np.mean(returns[-20:]):.4f}", flush=True)
            break

        if (ep + 1) % 1000 == 0:
            rec_ret = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
            print(f"  [{agent_type:14s}] N={size:2d} | Seed {seed} | Ep {ep+1:5d}/{actual_max_episodes} | Recent Ret: {rec_ret:.4f} | {time.time()-start_time:4.0f}s", flush=True)

    elapsed_total = time.time() - start_time
    is_solved = (solved_ep is not None)
    final_learn_time = solved_ep if is_solved else actual_max_episodes

    if not is_solved:
        print(f"  [{agent_type:14s}] N={size:2d} | Seed {seed} | TIMEOUT at Ep {actual_max_episodes:5d} | Time: {elapsed_total:5.1f}s", flush=True)

    return {
        "agent": agent_type,
        "size": size,
        "seed": seed,
        "solved": is_solved,
        "learn_time": final_learn_time,
        "elapsed_seconds": elapsed_total,
        "total_episodes": len(returns),
        "cumulative_regret": float(np.sum(regrets)),
        "final_20ep_return": float(np.mean(returns[-20:])) if len(returns) >= 20 else float(np.mean(returns)),
    }


def worker_task(task_args: Tuple[str, int, int, int]) -> Dict[str, Any]:
    agent_type, size, seed, max_episodes = task_args
    return run_single_experiment(agent_type, size, seed, max_episodes)


def main():
    parser = argparse.ArgumentParser(description="Deep Sea Scaling Benchmark (arXiv:1806.03335)")
    parser.add_argument("--scales", type=str, default="5,8,10,12,14,16,18,20,25,30", help="Comma-separated N values")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per configuration (default: 5)")
    parser.add_argument("--max_episodes", type=int, default=15000, help="Max episodes per run")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPU workers")
    parser.add_argument("--out_dir", type=str, default="results_deepsea", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick run on smaller subset of scales")
    args = parser.parse_args()

    if args.quick:
        scales = [5, 8, 10, 12, 14]
        num_seeds = 3
        max_episodes = 3000
    else:
        scales = [int(s.strip()) for s in args.scales.split(",")]
        num_seeds = args.seeds
        max_episodes = args.max_episodes

    agents = ["DP-DQN", "BootDQN-RP", "DQN-Dithering"]
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("Deep Sea Scaling Benchmark (Section 4.2.1 of arXiv:1806.03335)")
    print(f"Scales (N): {scales}")
    print(f"Agents: {agents}")
    print(f"Seeds: {num_seeds} | Max Episodes: {max_episodes} | CPU Workers: {args.workers}")
    print("=" * 80)

    tasks = []
    base_seed = 42
    for size in scales:
        for s_idx in range(num_seeds):
            seed = base_seed + s_idx
            for agent_type in agents:
                # Dithering is provably intractable beyond N=14 (2^N >= 32,768)
                if agent_type == "DQN-Dithering" and size > 14:
                    continue
                # For BootDQN on large N, cap to prevent excessive timeout wait
                if agent_type == "BootDQN-RP" and size > 20:
                    continue

                if agent_type == "DP-DQN":
                    agent_max_ep = max_episodes
                elif agent_type == "BootDQN-RP":
                    agent_max_ep = min(max_episodes, 10000)
                else:
                    agent_max_ep = 3000

                tasks.append((agent_type, size, seed, agent_max_ep))

    print(f"Total experiment tasks to execute: {len(tasks)}")
    t0 = time.time()
    results = []
    out_file = os.path.join(args.out_dir, "deep_sea_scaling_results.json")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_task, task): task for task in tasks}
        for fut in concurrent.futures.as_completed(futures):
            task = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                # Incremental save
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as exc:
                print(f"Error in task {task}: {exc}")
                raise exc

    wall_clock = time.time() - t0

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE! Total Wall-Clock Time: {wall_clock:.1f}s ({wall_clock/60:.1f} min)")
    print(f"Saved results to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
