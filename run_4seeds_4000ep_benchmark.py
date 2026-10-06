"""4-Seed 4000-Episode Benchmark: DP-DQN With vs Without Pre-sampling on Deep Sea N=20."""

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

torch.set_num_threads(1)

def run_single_seed(mode: bool, name: str, seed: int, num_episodes: int = 4000) -> Dict[str, Any]:
    print(f"\n---> Starting {name} | Seed {seed} | Episodes: {num_episodes}", flush=True)
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
            print(f"  [{name:28s}] Seed {seed} | First Discovery at Ep {first_discovery:4d} | Time: {time.time()-t0:4.1f}s", flush=True)

        if len(returns) >= 20 and np.mean(returns[-20:]) > 0.5 and solved_ep is None:
            solved_ep = ep + 1 - 10
            print(f"  [{name:28s}] Seed {seed} | SOLVED at Ep {solved_ep:4d} | Time: {time.time()-t0:4.1f}s", flush=True)

        if (ep + 1) % 1000 == 0:
            rec_ret = np.mean(returns[-50:])
            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / elapsed
            print(f"  [{name:28s}] Seed {seed} | Ep {ep+1:4d}/{num_episodes} | Ret: {rec_ret:.4f} | Time: {elapsed:5.1f}s | Speed: {eps_per_sec:5.1f} eps/s", flush=True)

    elapsed_total = time.time() - t0
    episodes_per_sec = num_episodes / elapsed_total

    # Subsample regret every 10 episodes
    subsampled_regrets = [float(np.mean(regrets[i:i+10])) for i in range(0, len(regrets), 10)]

    return {
        "agent": name,
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
    os.makedirs("results_deepsea", exist_ok=True)
    out_file = "results_deepsea/presampling_4seeds_4000ep.json"
    seeds = [42, 43, 44, 45]
    num_episodes = 4000

    print("=" * 80)
    print(f"Deep Sea 20x20 Benchmark: 4 Seeds x 4000 Episodes")
    print(f"Seeds: {seeds} | Max Episodes: {num_episodes}")
    print("=" * 80, flush=True)

    all_results = []
    # Interleave modes across seeds for fair comparison
    for seed in seeds:
        # Run Without Pre-sampling
        res_without = run_single_seed(
            mode=False,
            name="DP-DQN (Without Pre-sampling)",
            seed=seed,
            num_episodes=num_episodes
        )
        all_results.append(res_without)
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)

        # Run With Pre-sampling
        res_with = run_single_seed(
            mode=True,
            name="DP-DQN (With Pre-sampling)",
            seed=seed,
            num_episodes=num_episodes
        )
        all_results.append(res_with)
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL 8 RUNS COMPLETE! Final Summary:")
    print(f"{'Method':<32} | {'Seed':<5} | {'Solved':<7} | {'T_first':<8} | {'T_learn':<8} | {'Time (s)':<9} | {'Speed (eps/s)':<13}")
    print("-" * 80)
    for r in all_results:
        t_first = str(r['first_discovery']) if r['first_discovery'] is not None else "N/A"
        t_learn = str(r['learn_time']) if r['learn_time'] is not None else "N/A"
        print(f"{r['agent']:<32} | {r['seed']:<5} | {str(r['solved']):<7} | {t_first:<8} | {t_learn:<8} | {r['elapsed_seconds']:<9.2f} | {r['episodes_per_sec']:<13.1f}")
    print("=" * 80)

    # Aggregate Speed Statistics
    without_speeds = [r['episodes_per_sec'] for r in all_results if r['mode'] == 'Without Pre-sampling']
    with_speeds = [r['episodes_per_sec'] for r in all_results if r['mode'] == 'With Pre-sampling']
    without_times = [r['elapsed_seconds'] for r in all_results if r['mode'] == 'Without Pre-sampling']
    with_times = [r['elapsed_seconds'] for r in all_results if r['mode'] == 'With Pre-sampling']

    print(f"\nAVERAGE SPEED COMPARISON (N=4 seeds, 4000 episodes each):")
    print(f"  Without Pre-sampling: {np.mean(without_speeds):.2f} +/- {np.std(without_speeds):.2f} eps/s (Mean Time: {np.mean(without_times):.2f}s)")
    print(f"  With Pre-sampling:    {np.mean(with_speeds):.2f} +/- {np.std(with_speeds):.2f} eps/s (Mean Time: {np.mean(with_times):.2f}s)")
    delta = (np.mean(with_speeds) - np.mean(without_speeds)) / np.mean(without_speeds) * 100
    print(f"  Throughput Advantage: {delta:+.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
