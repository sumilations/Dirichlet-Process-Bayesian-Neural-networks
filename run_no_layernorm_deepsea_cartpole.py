#!/usr/bin/env python3
"""Targeted Benchmark on MacBook:
Evaluating DP-DQN without LayerNorm (use_layer_norm=False) on:
1. DeepSea N=40 (Pure TS, Scaled K_prior = 160 atoms) across seeds [42, 43, 44]
2. CartPole Swing-Up (Pure TS, Gaussian base measure, alpha=15.0) across seeds [42, 43, 44]
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


def run_deepsea40_no_ln(seed: int, max_episodes: int = 6000, out_dir: str = "./results_no_ln") -> Dict[str, Any]:
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"deepsea40_no_ln_s{seed}.json")
    
    t0 = time.time()
    size = 40
    state_dim = size * size
    action_dim = 2
    alpha = 5.0
    vm_mult = float(4.0 * size / alpha)  # 32.0 (160 atoms)

    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=seed,
        alpha=alpha,
        batch_size=128,
        candidate_batch_size=256,
        base_measure="deep_sea_dag_maxent",
        prior_reward_mean=0.5,
        prior_reward_std=0.288,
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=False,  # STRICTLY NO LAYERNORM
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
    first_discovery: Optional[int] = None
    solved_episode: Optional[int] = None
    cum_regret = 0.0
    optimal_return = 0.99

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

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep
            print(f"*** [DEEPSEA-40 NO-LN] DISCOVERY Seed={seed} at Ep {ep}! ({time.time()-t0:.1f}s)", flush=True)

        if ep >= 50:
            recent_ret = np.mean(returns[-50:])
            if recent_ret >= 0.85:
                if solved_episode is None:
                    solved_episode = ep
                    print(f"*** [DEEPSEA-40 NO-LN] SOLVED Seed={seed} at Ep {ep}! ({time.time()-t0:.1f}s)", flush=True)
                elif ep >= solved_episode + 50:
                    break

        if ep % 500 == 0:
            print(f"[DEEPSEA-40 NO-LN] Seed={seed} Ep={ep}/{max_episodes} Disc={first_discovery} Solv={solved_episode} ({time.time()-t0:.1f}s)", flush=True)

    elapsed = time.time() - t0
    res = {
        "benchmark": "deepsea40_no_layernorm",
        "seed": seed,
        "use_layer_norm": False,
        "first_discovery": first_discovery,
        "solved_episode": solved_episode,
        "total_episodes": len(returns),
        "final_cum_regret": float(cum_regret),
        "elapsed_seconds": elapsed,
        "completed": True,
    }
    with open(fpath, "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"--> [FINISH DEEPSEA-40 NO-LN] Seed={seed} -> Solved={solved_episode} in {elapsed:.1f}s", flush=True)
    return res


def run_cartpole_no_ln(seed: int, max_episodes: int = 1000, out_dir: str = "./results_no_ln") -> Dict[str, Any]:
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"cartpole_no_ln_s{seed}.json")

    t0 = time.time()
    env = make_env("cartpole_swingup")
    state_dim = env.state_dim
    action_dim = env.action_dim
    alpha = 15.0

    cfg = DPDQNConfig(
        env_name="cartpole_swingup",
        state_dim=state_dim,
        action_dim=action_dim,
        seed=seed,
        alpha=alpha,
        batch_size=64,
        candidate_batch_size=256,
        base_measure="cartpole_deepsea_construct_gaussian",
        hidden_dim=128,
        num_layers=2,
        use_layer_norm=False,  # STRICTLY NO LAYERNORM
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
        max_episode_steps=1000,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        dp_sampled_target=False,
        use_td_info_gain_decay=True,
        td_info_scale=1.0,
        verbose=False,
    )

    base_measure = get_base_measure("cartpole_deepsea_construct_gaussian", state_dim, action_dim)
    agent = DPDQNAgent(cfg, base_measure=base_measure)

    returns: List[float] = []
    uprights: List[int] = []
    first_swingup: Optional[int] = None
    solved_episode: Optional[int] = None
    cum_upright = 0

    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        s = env.reset()
        ep_ret = 0.0
        ep_upr = 0
        done = False
        steps = 0

        while not done:
            a = agent.act(s)
            sn, r, done, info = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn
            ep_ret += float(r)
            if info.get("is_upright", False):
                ep_upr += 1
            steps += 1

        agent.end_episode(steps)
        returns.append(ep_ret)
        uprights.append(ep_upr)
        cum_upright += ep_upr

        if ep_upr > 100 and first_swingup is None:
            first_swingup = ep
            print(f"*** [CARTPOLE NO-LN] FIRST SWING-UP Seed={seed} at Ep {ep}! (Upr={ep_upr}, Ret={ep_ret:.1f}, {time.time()-t0:.1f}s)", flush=True)

        if ep >= 50:
            rec_ret = np.mean(returns[-50:])
            rec_upr = np.mean(uprights[-50:])
            if rec_ret >= 500.0 or rec_upr >= 500.0:
                if solved_episode is None:
                    solved_episode = ep
                    print(f"*** [CARTPOLE NO-LN] SOLVED Seed={seed} at Ep {ep}! (50-MA Ret={rec_ret:.1f}, {time.time()-t0:.1f}s)", flush=True)
                elif ep >= solved_episode + 50:
                    break

        if ep % 100 == 0:
            rec_ret = np.mean(returns[-min(ep, 50):])
            rec_upr = np.mean(uprights[-min(ep, 50):])
            print(f"[CARTPOLE NO-LN] Seed={seed} Ep={ep}/{max_episodes} 50-MA Ret={rec_ret:.1f} Upr={rec_upr:.1f} CumUpr={cum_upright} ({time.time()-t0:.1f}s)", flush=True)

    elapsed = time.time() - t0
    res = {
        "benchmark": "cartpole_no_layernorm",
        "seed": seed,
        "use_layer_norm": False,
        "first_swingup": first_swingup,
        "solved_episode": solved_episode,
        "peak_return": float(np.max(returns)),
        "final_50_return": float(np.mean(returns[-min(len(returns), 50):])),
        "total_cum_upright": cum_upright,
        "total_episodes": len(returns),
        "elapsed_seconds": elapsed,
        "completed": True,
    }
    with open(fpath, "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"--> [FINISH CARTPOLE NO-LN] Seed={seed} -> Solved={solved_episode} PeakRet={res['peak_return']:.1f} in {elapsed:.1f}s", flush=True)
    return res


def main():
    seeds = [42, 43, 44]
    print("==========================================================")
    print(" Running No-LayerNorm Ablation on MacBook (Pure TS)")
    print(" 1. DeepSea N=40 (Seeds 42, 43, 44)")
    print(" 2. CartPole Swing-Up (Seeds 42, 43, 44)")
    print("==========================================================")

    # Run tasks with ProcessPoolExecutor (6 workers on local machine)
    t_start = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = {}
        for s in seeds:
            futures[executor.submit(run_deepsea40_no_ln, s)] = f"deepsea40_s{s}"
            futures[executor.submit(run_cartpole_no_ln, s)] = f"cartpole_s{s}"

        for future in concurrent.futures.as_completed(futures):
            tag = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"[ERROR] Task {tag} failed: {e}", flush=True)

    print(f"\n==========================================================")
    print(f" All No-LayerNorm runs finished in {time.time()-t_start:.1f}s!")
    print("==========================================================")
    with open("./results_no_ln/no_ln_ablation_summary.json", "w") as fp:
        json.dump(results, fp, indent=2)


if __name__ == '__main__':
    main()
