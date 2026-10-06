"""Deep Sea Scaling Benchmark: DAG vs Non-DAG Base Measures (N=5 to N=40).

Runs 20 sizes x 3 seeds x 2 base measures = 120 runs.
Algorithmic setup:
- Living network across episodes (NO re-initialization)
- Episodic posterior sampling: warmstart 2 steps on q_net with alpha=3.0 synthetic transitions
- Standard replay SGD during episode (batch 64, lr 1e-3)
- Polyak target tracking tau=0.05
- Replay capacity 1,000,000 (zero FIFO eviction)
- Jaynes' Uniform Max-Ent reward optimism: r0 ~ Uniform(0, 1.0)
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import get_base_measure


def run_single_experiment(
    size: int,
    seed: int,
    base_measure_type: str,
    max_episodes: Optional[int] = None,
    out_dir: str = "./results_deepsea_scaling",
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    fname = f"deepsea_N{size}_{base_measure_type}_s{seed}.json"
    fpath = os.path.join(out_dir, fname)

    # Check if already completed
    if os.path.exists(fpath):
        try:
            with open(fpath, "r") as fp:
                data = json.load(fp)
            if data.get("completed", False):
                return data
        except Exception:
            pass

    if max_episodes is None:
        if size >= 50:
            max_episodes = 10000
        elif size >= 30:
            max_episodes = 4000
        else:
            max_episodes = min(3000, max(600, 60 * size))

    t0 = time.time()
    state_dim = size * size
    action_dim = 2

    # Build agent configuration matching winning Deep Sea 20 construct
    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=seed,
        alpha=3.0,
        batch_size=64,
        candidate_batch_size=96,
        base_measure=base_measure_type,
        prior_reward_mean=0.5,
        prior_reward_std=0.288,
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.05,
        sgd_period=2,
        buffer_capacity=1000000,
        target_warmstart=True,
        warmstart_steps=2,
        warmstart_lr_scale=0.8,
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        dp_sampled_target=False,
        use_td_info_gain_decay=True,
        td_info_scale=1.0,
        verbose=False,
    )

    env = make_env("deep_sea", size=size, seed=seed)
    base_measure = get_base_measure(base_measure_type, state_dim, action_dim, deep_sea_size=size)
    agent = DPDQNAgent(cfg, base_measure=base_measure)

    returns: List[float] = []
    regrets: List[float] = []
    first_discovery: Optional[int] = None
    solved_episode: Optional[int] = None
    cum_regret = 0.0

    optimal_return = 1.0 - 0.01 * (size - 1)

    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        s = env.reset()
        ep_ret = 0.0
        done = False

        while not done:
            a = agent.act(s)
            sn, r, done, _ = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn
            ep_ret += float(r)

        returns.append(ep_ret)
        regret = optimal_return - ep_ret
        cum_regret += regret
        regrets.append(cum_regret)

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep

        # Solved criteria: 50-episode moving average return >= 0.85
        if ep >= 50:
            recent_ret = np.mean(returns[-50:])
            if recent_ret >= 0.85:
                if solved_episode is None:
                    solved_episode = ep
                elif ep >= solved_episode + 20:
                    break

    elapsed = time.time() - t0
    result = {
        "size": size,
        "seed": seed,
        "base_measure": base_measure_type,
        "max_episodes": max_episodes,
        "total_episodes": len(returns),
        "first_discovery": first_discovery,
        "solved_episode": solved_episode,
        "final_cum_regret": cum_regret,
        "recent_return_50": float(np.mean(returns[-50:])) if len(returns) >= 50 else float(np.mean(returns)),
        "elapsed_sec": round(elapsed, 2),
        "completed": True,
    }

    with open(fpath, "w") as fp:
        json.dump(result, fp, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="./results_deepsea_scaling")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--priors", type=str, nargs="+", default=["dag_maxent", "nondag_maxent"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    for sz in args.sizes:
        for bm in args.priors:
            for s in args.seeds:
                tasks.append((sz, s, bm))

    print(f"Total benchmark tasks: {len(tasks)} across {args.workers} workers.")
    sys.stdout.flush()

    results = []
    completed_count = 0
    t_start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_single_experiment, sz, s, bm, None, args.out_dir): (sz, s, bm)
            for sz, s, bm in tasks
        }

        for fut in concurrent.futures.as_completed(futures):
            sz, s, bm = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                completed_count += 1
                disc = res.get("first_discovery")
                solv = res.get("solved_episode")
                el = res.get("elapsed_sec")
                print(f"[{completed_count:3d}/{len(tasks):3d}] N={sz:2d} | Seed {s} | {bm:14s} -> Disc: Ep {str(disc):4s} | Solved: Ep {str(solv):4s} | {el:5.1f}s")
                sys.stdout.flush()
            except Exception as e:
                print(f"ERROR task N={sz} s={s} bm={bm}: {e}")
                sys.stdout.flush()

    summary_path = os.path.join(args.out_dir, "deepsea_scaling_results.json")
    with open(summary_path, "w") as fp:
        json.dump(results, fp, indent=2)

    total_el = time.time() - t_start
    print(f"\nALL {len(results)} TASKS COMPLETED in {total_el/60.0:.2f} minutes!")
    print(f"Saved aggregated results to {summary_path}")


if __name__ == "__main__":
    main()
