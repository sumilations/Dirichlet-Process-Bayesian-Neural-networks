#!/usr/bin/env python3
"""Cart-Pole Swing-Up Pure Thompson Sampling Benchmark across 5 Seeds.
Strictly with LayerNorm (use_layer_norm = True) and Pure TS (sample_once_per_episode = True).
Runs until 1000 episodes.
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

from unified_dp_dqn import (
    DPDQNAgent,
    DPDQNConfig,
    make_env,
)

SEEDS = [42, 43, 44, 45, 46]
ALPHA = 3.0
EPISODES = 1000
OUT_DIR = "./results_cartpole_pure_ts_1000_ln"


def run_single_seed(seed: int, alpha: float = ALPHA, episodes: int = EPISODES, out_dir: str = OUT_DIR) -> Dict[str, Any]:
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"cartpole_pure_ts_ln_a{alpha}_s{seed}.json")
    ckpt_path = os.path.join(out_dir, f"cartpole_pure_ts_ln_a{alpha}_s{seed}_ckpt.json")

    if os.path.exists(fpath):
        try:
            with open(fpath, "r") as fp:
                data = json.load(fp)
            if data.get("completed", False):
                print(f"[CACHED] Cart-Pole Seed={seed} -> Disc={data.get('first_discovery')} Solved={data.get('solved_episode')}", flush=True)
                return data
        except Exception:
            pass

    t0 = time.time()
    env = make_env("cartpole_swingup", seed=seed)
    state_dim = env.state_dim
    action_dim = env.action_dim

    cfg = DPDQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,  # STRICTLY LayerNorm enabled
        activation="relu",
        alpha=alpha,
        base_measure_type="gaussian",
        haar_angle=False,
        sampler_type="vashishtha_maillard",
        w_min=0.05,
        seed=seed,
        buffer_capacity=1000000,
        warmstart_steps=2,
        sample_once_per_episode=True,  # Pure Thompson Sampling
        dp_online_sgd=False,
        tau=0.05,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
    )

    agent = DPDQNAgent(cfg)

    returns: List[float] = []
    uprights: List[int] = []
    regrets: List[float] = []
    cumulative_regrets: List[float] = []
    cum_regret = 0.0
    first_discovery: Optional[int] = None
    solved_episode: Optional[int] = None
    recent_solved_counter = 0

    r_star = 850.0

    for ep in range(1, episodes + 1):
        agent.reset_episode()
        s = env.reset()
        done = False
        ep_ret = 0.0
        upr = 0

        while not done:
            a = agent.act(s)
            sn, r, done, info = env.step(a)
            agent.step(s, a, r, sn, done)
            ep_ret += r
            if info.get("is_upright", False):
                upr += 1
            s = sn

        inst_regret = r_star - ep_ret
        cum_regret += inst_regret
        returns.append(float(ep_ret))
        uprights.append(int(upr))
        regrets.append(float(inst_regret))
        cumulative_regrets.append(float(cum_regret))

        if ep_ret > 0.5:
            if first_discovery is None:
                first_discovery = ep
                print(f"[DISCOVERY] CartPole Pure TS Seed {seed} at Ep {ep}! Time: {time.time()-t0:.1f}s", flush=True)
            recent_solved_counter += 1
            if recent_solved_counter >= 10 and solved_episode is None:
                solved_episode = ep - 9
                print(f"[SOLVED] CartPole Pure TS Seed {seed} SOLVED at Ep {solved_episode}! Time: {time.time()-t0:.1f}s", flush=True)
        else:
            recent_solved_counter = 0

        if ep % 50 == 0 or ep == episodes:
            elapsed = time.time() - t0
            ret_50 = np.mean(returns[-50:])
            upr_50 = np.mean(uprights[-50:])
            print(f"[CartPole s{seed}] Ep {ep:4d}/{episodes} | Ret(50): {ret_50:6.2f} | Upr(50): {upr_50:5.1f} | CumReg: {cum_regret/1e3:5.1f}k | Time: {elapsed:5.1f}s", flush=True)

            ckpt = {
                "env": "cartpole_swingup",
                "algo": "dp_dqn_pure_ts_ln",
                "seed": seed,
                "alpha": alpha,
                "episodes": ep,
                "first_discovery": first_discovery,
                "solved_episode": solved_episode,
                "returns": returns,
                "upright_steps": uprights,
                "regrets": regrets,
                "cumulative_regrets": cumulative_regrets,
                "elapsed_sec": elapsed,
                "completed": (ep == episodes),
            }
            with open(ckpt_path, "w") as fp:
                json.dump(ckpt, fp)

    with open(fpath, "w") as fp:
        json.dump(ckpt, fp, indent=2)

    status_str = f"SOLVED at ep {solved_episode}" if solved_episode else (f"FOUND at ep {first_discovery}" if first_discovery else "NOT FOUND")
    print(f"[FINISHED] CartPole Pure TS Seed {seed} -> {status_str} in {time.time()-t0:.1f}s", flush=True)
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers")
    args = parser.parse_args()

    print(f"Launching CartPole Swing-Up Pure TS Benchmark: 5 seeds across {args.workers} workers...")
    print(f"Seeds: {SEEDS}")
    print(f"Episodes: {EPISODES} | Alpha: {ALPHA} | LayerNorm: True | Pure TS: True", flush=True)

    t_start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_seed, sd): sd for sd in SEEDS}
        for fut in concurrent.futures.as_completed(futures):
            sd = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[ERROR] CartPole Seed {sd} raised exception: {e}", flush=True)

    total_time = time.time() - t_start
    print(f"All CartPole tasks finished in {total_time:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
