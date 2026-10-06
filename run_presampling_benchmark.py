"""Benchmark comparing DP-DQN with and without episodic pre-sampling on Deep Sea N=20."""

import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from bsuite.environments.deep_sea import DeepSea
from src.rl.batched_dp_dqn import BatchedDPDQNDeepSeaAgent

torch.set_num_threads(1)

def run_single(mode: bool, name: str, seed: int, max_episodes: int = 5500) -> Dict[str, Any]:
    print(f"\n---> Starting {name} | Seed {seed} | Max Ep: {max_episodes}", flush=True)
    env = DeepSea(size=20, deterministic=True, randomize_actions=True, seed=seed, mapping_seed=seed)
    agent = BatchedDPDQNDeepSeaAgent(
        size=20,
        hidden_dim=20,
        alpha=5.0,
        prior_reward_mean=1.0,
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

    for ep in range(max_episodes):
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
            print(f"  [{name:28s}] Seed {seed} | First Treasure Discovered at Ep {first_discovery:4d} | Time: {time.time()-t0:4.1f}s", flush=True)

        if len(returns) >= 20 and np.mean(returns[-20:]) > 0.5 and solved_ep is None:
            solved_ep = ep + 1 - 10
            elapsed = time.time() - t0
            print(f"  [{name:28s}] Seed {seed} | SOLVED at Ep {solved_ep:4d} | Time: {elapsed:4.1f}s | eps/s: {len(returns)/elapsed:4.1f}", flush=True)
            break

        if (ep + 1) % 1000 == 0:
            rec_ret = np.mean(returns[-50:])
            elapsed = time.time() - t0
            print(f"  [{name:28s}] Seed {seed} | Ep {ep+1:4d} | Ret: {rec_ret:.4f} | Time: {elapsed:4.1f}s | eps/s: {len(returns)/elapsed:4.1f}", flush=True)

    elapsed_total = time.time() - t0
    is_solved = (solved_ep is not None)
    final_learn_time = solved_ep if is_solved else max_episodes

    # Subsample regret curve every 10 episodes for clean plotting
    subsampled_regrets = [float(np.mean(regrets[i:i+10])) for i in range(0, len(regrets), 10)]

    return {
        "agent": name,
        "mode": "With Pre-sampling" if mode else "Without Pre-sampling",
        "seed": seed,
        "solved": is_solved,
        "learn_time": final_learn_time,
        "first_discovery": first_discovery,
        "elapsed_seconds": elapsed_total,
        "total_episodes": len(returns),
        "episodes_per_sec": len(returns) / elapsed_total,
        "cumulative_regret": float(np.sum(regrets)),
        "final_20ep_return": float(np.mean(returns[-20:])),
        "subsampled_regrets": subsampled_regrets,
    }


def main():
    os.makedirs("results_deepsea", exist_ok=True)
    out_file = "results_deepsea/presampling_comparison_n20.json"
    seeds = [42, 43, 44]
    modes = [
        (False, "DP-DQN (Without Pre-sampling)"),
        (True, "DP-DQN (With Pre-sampling)"),
    ]

    all_results = []
    print("=" * 80)
    print("Deep Sea 20x20 Benchmark: With vs Without Episodic Pre-sampling")
    print(f"Seeds: {seeds} | Total Runs: {len(seeds) * len(modes)}")
    print("=" * 80, flush=True)

    for mode, name in modes:
        for seed in seeds:
            res = run_single(mode, name, seed, max_episodes=5500)
            all_results.append(res)
            with open(out_file, "w") as f:
                json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE! Summary Table:")
    print(f"{'Method':<32} | {'Seed':<5} | {'Solved':<7} | {'T_first':<8} | {'T_learn':<8} | {'Time (s)':<9} | {'Eps/s':<7}")
    print("-" * 80)
    for r in all_results:
        t_first_str = str(r['first_discovery']) if r['first_discovery'] is not None else "N/A"
        print(f"{r['agent']:<32} | {r['seed']:<5} | {str(r['solved']):<7} | {t_first_str:<8} | {r['learn_time']:<8} | {r['elapsed_seconds']:<9.1f} | {r['episodes_per_sec']:<7.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
