"""Targeted Benchmark for Pure TS on Deep Sea 40 and 50:
Comparing Standard K_prior vs Scaled K_prior (K_prior = 4*N) with Extended Episode Budget (up to 15,000 eps).

Runs:
1. Standard Pure TS (K_prior = 50):
   - N=40 (Seed 44)
   - N=50 (Seeds 42, 43, 44)
2. Scaled K_prior Pure TS (K_prior = 4*N):
   - N=40 (Seeds 42, 43, 44)
   - N=50 (Seeds 42, 43, 44)
Total: 10 runs.
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


def run_single_experiment(
    size: int,
    seed: int,
    scaled_kprior: bool,
    max_episodes: int = 15000,
    out_dir: str = "./results_pure_ts_deepsea",
) -> Dict[str, Any]:
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    mode_str = "pure_ts_scaled_kprior" if scaled_kprior else "pure_ts_standard"
    fname = f"deepsea_N{size}_{mode_str}_s{seed}.json"
    fpath = os.path.join(out_dir, fname)

    # Check cache
    if os.path.exists(fpath):
        try:
            with open(fpath, "r") as fp:
                data = json.load(fp)
            if data.get("completed", False):
                print(f"[CACHED] N={size} Mode={mode_str} Seed={seed} -> Solved={data.get('solved_episode')}", flush=True)
                return data
        except Exception:
            pass

    t0 = time.time()
    state_dim = size * size
    action_dim = 2
    alpha = 5.0

    # Determine K_prior multiplier
    if scaled_kprior:
        # Scale K_prior to 4 * N (e.g. 160 atoms for N=40, 200 atoms for N=50)
        vm_mult = float(4.0 * size / alpha)
        batch_sz = 128
        cand_sz = 256
    else:
        vm_mult = 10.0  # Standard K_prior = 50
        batch_sz = 64
        cand_sz = 256

    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=seed,
        alpha=alpha,
        batch_size=batch_sz,
        candidate_batch_size=cand_sz,
        base_measure="deep_sea_dag_maxent",
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
        sample_once_per_episode=True,  # 100% Pure TS
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=vm_mult,
        one_living_network=True,
        dp_sampled_target=False,
        use_td_info_gain_decay=True,
        td_info_scale=1.0,
        verbose=False,
    )

    env = make_env("deep_sea", size=size, seed=seed)
    base_measure = get_base_measure("deep_sea_dag_maxent", state_dim, action_dim, deep_sea_size=size)
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
        regret = optimal_return - ep_ret
        cum_regret += regret
        regrets.append(cum_regret)

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep
            print(f"*** [DISCOVERY] N={size} {mode_str.upper()} Seed={seed} at Episode {ep}! ({time.time()-t0:.1f}s)", flush=True)

        # Solved criteria: 50-episode moving average return >= 0.85
        if ep >= 50:
            recent_ret = np.mean(returns[-50:])
            if recent_ret >= 0.85:
                if solved_episode is None:
                    solved_episode = ep
                elif ep >= solved_episode + 20:
                    break

        if ep % 200 == 0:
            print(f"[{mode_str.upper()}] N={size} Seed={seed} Ep={ep}/{max_episodes} Disc={first_discovery} Solved={solved_episode} ({time.time()-t0:.1f}s)", flush=True)

    elapsed = time.time() - t0
    result = {
        "size": size,
        "seed": seed,
        "scaled_kprior": scaled_kprior,
        "mode": mode_str,
        "base_measure": "deep_sea_dag_maxent",
        "max_episodes": max_episodes,
        "total_episodes": len(returns),
        "first_discovery": first_discovery,
        "solved_episode": solved_episode,
        "final_cum_regret": cum_regret,
        "recent_return_50": float(np.mean(returns[-50:])) if len(returns) >= 50 else float(np.mean(returns)),
        "elapsed_seconds": elapsed,
        "completed": True,
        "timestamp": time.time(),
    }

    with open(fpath, "w") as fp:
        json.dump(result, fp, indent=2)

    status = f"SOLVED at Ep {solved_episode}" if solved_episode else (f"Discovered at Ep {first_discovery}" if first_discovery else "FAILED")
    print(f"--> [FINISH] N={size} {mode_str.upper()} Seed={seed} -> {status} in {elapsed:.1f}s ({len(returns)} eps)", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--out_dir", type=str, default="./results_pure_ts_deepsea")
    args = parser.parse_args()

    tasks = [
        # 1. Standard Pure TS (K_prior = 50) with extended episodes
        (40, 44, False),
        (50, 42, False),
        (50, 43, False),
        (50, 44, False),
        # 2. Scaled K_prior Pure TS (K_prior = 4*N)
        (40, 42, True),
        (40, 43, True),
        (40, 44, True),
        (50, 42, True),
        (50, 43, True),
        (50, 44, True),
    ]

    print(f"Total tasks: {len(tasks)} across {args.workers} workers")
    t_start = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_single_experiment,
                size=t[0],
                seed=t[1],
                scaled_kprior=t[2],
                out_dir=args.out_dir,
            ): t
            for t in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"[ERROR] Task {t} crashed: {e}")

    total_time = time.time() - t_start
    summary_path = os.path.join(args.out_dir, "pure_ts_scaling_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"\n==========================================")
    print(f"All {len(results)}/{len(tasks)} experiments finished in {total_time:.1f}s ({total_time/60:.2f} mins)!")
    print(f"Summary written to {summary_path}")
    print(f"==========================================\n")


if __name__ == "__main__":
    main()
