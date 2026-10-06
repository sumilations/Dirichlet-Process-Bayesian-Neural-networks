#!/usr/bin/env python3
"""DeepSea 20x20 Benchmark: DAG vs Non-DAG Max-Entropy Uniform Optimistic Priors across 10 Seeds.
Strictly with LayerNorm (use_layer_norm = True).
Matching the proven scaling benchmark setup from run_deepsea_scaling.py.
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
from typing import Any, Dict, List
import numpy as np
import torch
torch.set_num_threads(1)

from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import get_base_measure

SIZE = 20
MAX_EPISODES = 2500
SEEDS = list(range(42, 52))  # 10 seeds: 42 to 51
BASE_MEASURES = ["deep_sea_dag_maxent", "deep_sea_nondag_maxent"]
OUT_DIR = "./results_deepsea20_10seeds_ln"


def run_single(size: int, bm_name: str, seed: int, max_episodes: int = MAX_EPISODES, out_dir: str = OUT_DIR) -> Dict[str, Any]:
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    clean_bm = "dag_maxent" if "dag" in bm_name and "nondag" not in bm_name else "nondag_maxent"
    fname = f"deepsea_N{size}_{clean_bm}_s{seed}.json"
    fpath = os.path.join(out_dir, fname)

    if os.path.exists(fpath):
        try:
            with open(fpath, "r") as fp:
                data = json.load(fp)
            if data.get("completed", False):
                print(f"[CACHED] N={size} {clean_bm} Seed={seed} -> Disc={data.get('first_discovery')} Solved={data.get('solved_episode')}", flush=True)
                return data
        except Exception:
            pass

    t0 = time.time()
    state_dim = size * size
    action_dim = 2

    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=seed,
        alpha=3.0,
        batch_size=64,
        candidate_batch_size=96,
        base_measure=bm_name,
        prior_reward_mean=0.5,
        prior_reward_std=0.288,
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,  # STRICTLY LayerNorm enabled
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
    base_measure = get_base_measure(bm_name, state_dim, action_dim, deep_sea_size=size)
    agent = DPDQNAgent(cfg, base_measure=base_measure)

    returns: List[float] = []
    regrets: List[float] = []
    cumulative_regrets: List[float] = []
    cum_regret = 0.0
    first_discovery = None
    solved_episode = None
    recent_returns = []

    r_star = 1.0 - (1.0 / size)

    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        state = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            ep_ret += reward
            state = next_state

        inst_regret = r_star - ep_ret
        cum_regret += inst_regret
        returns.append(float(ep_ret))
        regrets.append(float(inst_regret))
        cumulative_regrets.append(float(cum_regret))
        recent_returns.append(float(ep_ret))
        if len(recent_returns) > 50:
            recent_returns.pop(0)

        if ep_ret > 0.5:
            if first_discovery is None:
                first_discovery = ep
                print(f"[DISCOVERY] N={size} {clean_bm} Seed={seed} at Episode {ep}! Time: {time.time()-t0:.1f}s", flush=True)

        ma50 = np.mean(recent_returns) if len(recent_returns) >= 20 else 0.0
        if ma50 >= 0.85 and solved_episode is None:
            solved_episode = ep
            print(f"[SOLVED] N={size} {clean_bm} Seed={seed} SOLVED at Episode {ep}! Time: {time.time()-t0:.1f}s", flush=True)

        if solved_episode is not None and (ep - solved_episode >= 50):
            break

        if ep % 200 == 0:
            print(f"[{clean_bm} s{seed}] Ep {ep}/{max_episodes} | Ret(50): {ma50:.3f} | CumReg: {cum_regret:.1f} | Disc: {first_discovery}", flush=True)

    elapsed = time.time() - t0
    result = {
        "size": size,
        "base_measure": clean_bm,
        "seed": seed,
        "first_discovery": first_discovery,
        "solved_episode": solved_episode,
        "total_episodes": len(returns),
        "final_cum_regret": cum_regret,
        "regret_curve": [
            {"episode": i + 1, "cum_regret": cumulative_regrets[i], "return": returns[i]}
            for i in range(0, len(returns), max(1, len(returns) // 100))
        ],
        "elapsed_seconds": elapsed,
        "completed": True,
    }

    with open(fpath, "w") as fp:
        json.dump(result, fp, indent=2)

    status_str = f"SOLVED at ep {solved_episode}" if solved_episode else (f"FOUND at ep {first_discovery}" if first_discovery else "NOT FOUND")
    print(f"[FINISHED] N={size} {clean_bm} Seed={seed} -> {status_str} in {elapsed:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    args = parser.parse_args()

    tasks = []
    for bm in BASE_MEASURES:
        for seed in SEEDS:
            tasks.append((SIZE, bm, seed))

    print(f"Launching DeepSea {SIZE}x{SIZE} Benchmark: {len(tasks)} tasks across {args.workers} workers...")
    print(f"Base measures: {BASE_MEASURES}")
    print(f"Seeds: {SEEDS}")
    print(f"LayerNorm: True | Output: {OUT_DIR}", flush=True)

    t_start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single, sz, bm, sd): (sz, bm, sd) for (sz, bm, sd) in tasks}
        for fut in concurrent.futures.as_completed(futures):
            sz, bm, sd = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[ERROR] N={sz} {bm} Seed={sd} raised exception: {e}", flush=True)

    total_time = time.time() - t_start
    print(f"All DeepSea 20x20 tasks finished in {total_time:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
