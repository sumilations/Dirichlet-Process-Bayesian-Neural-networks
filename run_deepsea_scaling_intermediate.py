#!/usr/bin/env python3
"""Run DeepSea Intermediate Sizes with FIXED K_prior (vm_prior_multiplier = 10.0, 50 atoms).
Sizes: 43, 47 (alongside existing 40 and 50) to give 4 data points in [40, 50].
Pure Thompson Sampling (sample_once_per_episode = True).
Seeds: 42, 43, 44.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import concurrent.futures
import json
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
torch.set_num_threads(1)

from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import get_base_measure


def run_single(
    size: int,
    seed: int,
    base_measure_type: str = "deep_sea_dag_maxent",
    max_episodes: int = 10000,
    out_dir: str = "./results_deepsea_intermediate",
) -> Dict[str, Any]:
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"deepsea_N{size}_fixed_kprior_{base_measure_type}_s{seed}.json"
    fpath = os.path.join(out_dir, fname)

    if os.path.exists(fpath):
        try:
            with open(fpath, "r") as fp:
                data = json.load(fp)
            if data.get("completed", False):
                print(f"[CACHED] N={size} Seed={seed} -> Solved={data.get('solved_episode')}")
                return data
        except Exception:
            pass

    t0 = time.time()
    state_dim = size * size
    action_dim = 2
    alpha = 5.0

    # STRICTLY FIXED K_prior = 10 * alpha = 50 atoms
    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=seed,
        alpha=alpha,
        batch_size=64,
        candidate_batch_size=256,
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
        episodic_sgd=True,
        buffer_capacity=1000000,
        target_warmstart=True,
        warmstart_steps=10,
        warmstart_lr_scale=1.0,
        sample_once_per_episode=True,  # Pure TS
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,  # FIXED K_prior (50 atoms)
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
    regret_curve: List[Dict[str, Any]] = []
    first_discovery: Optional[int] = None
    solved_episode: Optional[int] = None
    cum_regret = 0.0
    optimal_return = 1.0 - 0.01 * (size - 1)

    print(f"--> [START] DeepSea N={size} Seed={seed} FIXED K_prior ({base_measure_type})", flush=True)

    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        s = env.reset()
        ep_ret = 0.0
        done = False
        steps = 0

        while not done:
            a = agent.act(s)
            sn, r, done, _ = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn
            ep_ret += float(r)
            steps += 1

        agent.end_episode(steps)
        returns.append(ep_ret)
        cum_regret += max(0.0, optimal_return - ep_ret)

        if ep % 10 == 0:
            recent_ret = float(np.mean(returns[-min(ep, 50):]))
            regret_curve.append({
                "episode": ep,
                "cum_regret": float(cum_regret),
                "recent_ret": recent_ret,
            })

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep
            print(f"*** [DISCOVERY] N={size} Seed={seed} at Ep {ep}! ({time.time()-t0:.1f}s)", flush=True)

        if ep >= 50:
            rec_ret = np.mean(returns[-50:])
            if rec_ret >= 0.85:
                if solved_episode is None:
                    solved_episode = ep
                    print(f"*** [SOLVED] N={size} Seed={seed} at Ep {ep}! ({time.time()-t0:.1f}s, Regret: {cum_regret:.1f})", flush=True)
                elif ep >= solved_episode + 50:
                    break

        if ep % 500 == 0:
            print(f"[PROGRESS] N={size} Seed={seed} Ep={ep}/{max_episodes} | Disc={first_discovery} | Solv={solved_episode} | Reg={cum_regret:.1f} ({time.time()-t0:.1f}s)", flush=True)

    elapsed = time.time() - t0
    res = {
        "size": size,
        "seed": seed,
        "fixed_kprior": True,
        "k_prior_atoms": 50,
        "base_measure": base_measure_type,
        "first_discovery": first_discovery,
        "solved_episode": solved_episode,
        "total_episodes": len(returns),
        "final_cum_regret": float(cum_regret),
        "regret_curve": regret_curve,
        "elapsed_seconds": elapsed,
        "completed": True,
    }

    with open(fpath, "w") as fp:
        json.dump(res, fp, indent=2)

    print(f"--> [FINISH] N={size} Seed={seed} -> Solved={solved_episode} in {elapsed:.1f}s (Final Regret: {cum_regret:.1f})", flush=True)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[43, 47])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out_dir", type=str, default="./results_deepsea_intermediate")
    args = parser.parse_args()

    tasks = [(sz, s) for sz in args.sizes for s in args.seeds]
    print(f"Running {len(tasks)} tasks across {args.workers} workers on MacBook:")
    for sz, s in tasks:
        print(f"  - Size N={sz}, Seed {s} (FIXED K_prior = 50 atoms)")

    t_start = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single, sz, s, "deep_sea_dag_maxent", 10000, args.out_dir): (sz, s) for sz, s in tasks}
        for fut in concurrent.futures.as_completed(futures):
            sz, s = futures[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                print(f"[ERROR] Task N={sz} s={s} failed: {e}", flush=True)

    summary_file = os.path.join(args.out_dir, "intermediate_summary.json")
    with open(summary_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nAll tasks finished in {time.time()-t_start:.1f}s! Summary written to {summary_file}")


if __name__ == '__main__':
    main()
