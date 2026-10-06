"""Benchmark script: Neural-Linear DP-DQN on Deep Sea N=20.

Runs 4 seeds of Neural-Linear DP-DQN with multiprocessing 'spawn' context.
"""

import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, Any, Optional
import numpy as np

# Add project root to PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bsuite.environments.deep_sea import DeepSea
from neural_linear_dp_dqn import NeuralLinearDPDQNAgent

OUTPUT_DIR = "neural_linear_dp_dqn"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "deep_sea_20_results.json")
N = 20
SEEDS = [42, 43, 44, 45]


def run_single_seed(seed: int, max_episodes: int = 6000) -> Dict[str, Any]:
    """Execute Neural-Linear DP-DQN on Deep Sea N=20."""
    t_start = time.time()
    env = DeepSea(size=N, deterministic=True, randomize_actions=True, seed=seed)

    agent = NeuralLinearDPDQNAgent(
        size=N,
        feature_dim=20,
        alpha=5.0,
        prior_precision=1.0,
        noise_variance=0.1,
        candidate_batch_size=128,
        sample_batch_size=64,
        use_layer_norm=True,
        seed=seed,
    )

    cum_regret = 0.0
    t_first_treasure: Optional[int] = None
    t_learn: Optional[int] = None
    regret_history = []
    episode_returns = []

    print(f"[Seed {seed}] Starting {max_episodes} episodes on Deep Sea N={N}...", flush=True)

    for ep in range(1, max_episodes + 1):
        agent.start_episode()
        ts = env.reset()
        obs = ts.observation
        done = False
        ep_ret = 0.0

        while not done:
            action = agent.select_action(obs)
            next_ts = env.step(action)
            reward = float(next_ts.reward or 0.0)
            done = bool(next_ts.last())
            next_obs = next_ts.observation

            agent.step_update(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_ret += reward

        agent.end_episode()
        episode_returns.append(ep_ret)

        if ep_ret > 0.5 and t_first_treasure is None:
            t_first_treasure = ep
            print(f"  --> [Seed {seed}] *** FIRST TREASURE at Episode {ep}! (Return: {ep_ret:.3f}) ***", flush=True)

        regret = 1.0 - ep_ret
        cum_regret += regret
        avg_regret = cum_regret / ep

        if ep % 20 == 0 or ep == max_episodes:
            regret_history.append({"ep": ep, "avg_regret": float(avg_regret), "ep_return": float(ep_ret)})

        if t_learn is None and ep >= 50 and avg_regret < 0.9:
            t_learn = ep
            print(f"  --> [Seed {seed}] *** LEARNED (Avg Regret < 0.9) at Episode {ep}! ***", flush=True)
            if ep + 100 < max_episodes:
                max_episodes = ep + 100

        if ep % 1000 == 0:
            elapsed = time.time() - t_start
            print(f"  [Seed {seed}] Ep {ep:4d}/{max_episodes} | Avg Regret: {avg_regret:.4f} | Time: {elapsed:5.1f}s", flush=True)

    elapsed_total = time.time() - t_start
    ep_per_sec = ep / max(elapsed_total, 1e-3)
    print(f"[Seed {seed} DONE] Time: {elapsed_total:.1f}s ({ep_per_sec:.1f} ep/s) | T_first: {t_first_treasure} | T_learn: {t_learn}", flush=True)

    return {
        "algo_name": "Neural-Linear DP-DQN",
        "seed": seed,
        "N": N,
        "t_first_treasure": t_first_treasure,
        "t_learn": t_learn,
        "final_avg_regret": float(cum_regret / max(ep, 1)),
        "total_episodes": ep,
        "elapsed_seconds": float(elapsed_total),
        "ep_per_sec": float(ep_per_sec),
        "regret_history": regret_history,
    }


def _worker_wrapper(arg):
    seed, max_ep = arg
    return run_single_seed(seed, max_ep)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== LAUNCHING NEURAL-LINEAR DP-DQN BENCHMARK (N={N}) ===", flush=True)
    print(f"Seeds: {SEEDS} | State Dim: {N*N} | Policy Space: 2^{N} = {2**N:,}", flush=True)
    print("-" * 65, flush=True)

    args_list = [(s, 6000) for s in SEEDS]
    results = []

    # Use multiprocessing 'spawn' context to avoid macOS fork issues
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(SEEDS)) as pool:
        for res in pool.imap_unordered(_worker_wrapper, args_list):
            results.append(res)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[PROGRESS] Completed Seed {res['seed']} ({len(results)}/{len(SEEDS)} seeds finished).", flush=True)

    print("\nAll 4 seeds completed!", flush=True)
    print(f"Results written to: {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
