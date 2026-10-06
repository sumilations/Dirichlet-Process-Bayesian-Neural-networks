"""Deep Sea N=50 Exploration Benchmark across 4 Random Seeds.

Compares:
1. DP-DQN (Structured Base Measure - Reachable Grid Prior)
2. DP-DQN (Uniform Optimistic Base Measure - Random Uniform Sampling over State & Action Space)
3. BootDQN-RP (20 Heads, Prior Scale beta=10.0, Osband et al. NeurIPS 2018)
4. DQN-Dithering (Standard epsilon-greedy)

State space: N x N = 50 x 50 = 2,500 dimensions.
Policy space: 2^50 = 1,125,899,906,842,624 (> 1.12 x 10^15 policies).
"""

import concurrent.futures
import json
import os
import sys
import time
from typing import Dict, Any, Optional
import numpy as np

# Ensure root is on PATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from bsuite.environments.deep_sea import DeepSea
from src.rl.deep_sea_dp_dqn import DPDQNDeepSeaAgent
from src.rl.deep_sea_boot_dqn import BootDQNRPDeepSeaAgent
from src.rl.deep_sea_dithering import DQNDitheringAgent

OUTPUT_DIR = "results_deepsea_50"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "deep_sea_50_results.json")
SEEDS = [42, 43, 44, 45]
N = 50


def run_single_experiment(algo_name: str, seed: int, max_episodes: int) -> Dict[str, Any]:
    """Execute a single training run on DeepSea N=50."""
    t_start = time.time()
    env = DeepSea(size=N, deterministic=True, randomize_actions=True, seed=seed)

    # Agent setup
    if algo_name == "DP-DQN (Structured BM)":
        agent = DPDQNDeepSeaAgent(
            size=N,
            hidden_dim=32,
            alpha=5.0,
            truncation_K=16,
            use_layer_norm=True,
            base_measure_type="structured",
            seed=seed,
        )
    elif algo_name == "DP-DQN (Uniform BM)":
        agent = DPDQNDeepSeaAgent(
            size=N,
            hidden_dim=32,
            alpha=5.0,
            truncation_K=16,
            use_layer_norm=True,
            base_measure_type="uniform_optimistic",
            seed=seed,
        )
    elif algo_name == "BootDQN-RP (20 Heads)":
        agent = BootDQNRPDeepSeaAgent(
            size=N,
            num_heads=20,
            hidden_dim=20,
            prior_scale=10.0,
            batch_size=64,
            seed=seed,
        )
    elif algo_name == "DQN-Dithering":
        agent = DQNDitheringAgent(
            size=N,
            hidden_dim=20,
            batch_size=64,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    cum_regret = 0.0
    t_learn: Optional[int] = None
    t_first_treasure: Optional[int] = None
    regret_history = []
    return_history = []

    print(f"[{algo_name} | Seed {seed}] Starting {max_episodes} episodes on DeepSea N={N}...")

    for ep in range(1, max_episodes + 1):
        agent.start_episode()
        ts = env.reset()
        obs = ts.observation
        done = False
        ep_return = 0.0

        while not done:
            action = agent.select_action(obs)
            next_ts = env.step(action)
            reward = float(next_ts.reward or 0.0)
            done = bool(next_ts.last())
            next_obs = next_ts.observation

            agent.step_update(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward

        agent.end_episode()
        return_history.append(ep_return)

        # Track treasure discovery
        if ep_return > 0.5 and t_first_treasure is None:
            t_first_treasure = ep
            print(f"[{algo_name} | Seed {seed}] *** TREASURE FOUND at Episode {ep}! (Return: {ep_return:.3f}) ***")

        # Regret tracking
        regret = 1.0 - ep_return
        cum_regret += regret
        avg_regret = cum_regret / ep

        # Subsample regret history every 25 episodes to save space
        if ep % 25 == 0 or ep == max_episodes:
            regret_history.append({"ep": ep, "avg_regret": float(avg_regret)})

        if t_learn is None and ep >= 50 and avg_regret < 0.9:
            t_learn = ep
            print(f"[{algo_name} | Seed {seed}] *** LEARNED (Avg Regret < 0.9) at Episode {ep}! ***")
            # Once learned, continue for 100 verification episodes then terminate
            if ep + 100 < max_episodes:
                max_episodes = ep + 100

        if ep % 500 == 0:
            elapsed = time.time() - t_start
            print(f"[{algo_name} | Seed {seed}] Ep {ep}/{max_episodes} | Avg Regret: {avg_regret:.3f} | Elapsed: {elapsed:.1f}s")

    elapsed_total = time.time() - t_start
    print(f"[{algo_name} | Seed {seed}] Finished in {elapsed_total:.1f}s. T_first: {t_first_treasure}, T_learn: {t_learn}")

    return {
        "algo_name": algo_name,
        "seed": seed,
        "N": N,
        "t_first_treasure": t_first_treasure,
        "t_learn": t_learn,
        "final_avg_regret": float(cum_regret / max(ep, 1)),
        "cum_regret": float(cum_regret),
        "total_episodes": ep,
        "elapsed_time_s": float(elapsed_total),
        "regret_history": regret_history,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define experimental tasks
    tasks = []
    # 1. DP-DQN (Structured BM): max 10,000 episodes
    for s in SEEDS:
        tasks.append(("DP-DQN (Structured BM)", s, 10000))
    # 2. DP-DQN (Uniform BM): max 10,000 episodes
    for s in SEEDS:
        tasks.append(("DP-DQN (Uniform BM)", s, 10000))
    # 3. BootDQN-RP (20 Heads): max 4,000 episodes (sufficient to test scaling or timeout)
    for s in SEEDS:
        tasks.append(("BootDQN-RP (20 Heads)", s, 4000))
    # 4. DQN-Dithering: max 2,500 episodes (dithering lower bound proves 2^50 failure)
    for s in SEEDS:
        tasks.append(("DQN-Dithering", s, 2500))

    print(f"=== LAUNCHING DEEP SEA N={N} BENCHMARK ===")
    print(f"Total Runs: {len(tasks)} across {len(SEEDS)} seeds")
    print(f"State Dimension: {N*N} | Policy Space: 2^{N} = {2**N:,}")
    print("-" * 60)

    results = []
    # Execute across 8 CPU worker processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        future_to_task = {
            executor.submit(run_single_experiment, algo, seed, max_ep): (algo, seed)
            for algo, seed, max_ep in tasks
        }

        for future in concurrent.futures.as_completed(future_to_task):
            algo, seed = future_to_task[future]
            try:
                res = future.result()
                results.append(res)
                # Incremental write
                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"[PROGRESS] Completed {len(results)}/{len(tasks)} runs. Saved to {RESULTS_FILE}")
            except Exception as exc:
                print(f"[ERROR] Run ({algo}, Seed {seed}) generated exception: {exc}")

    print("\n=== ALL DEEP SEA N=50 EXPERIMENTS COMPLETED ===")
    print(f"Final results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
