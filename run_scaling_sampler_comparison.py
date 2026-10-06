"""Scaling Benchmark: Pure TS (SampleOnce=True) vs Multi-Sample (SampleOnce=False).

Sizes: N in {10, 15, 20, 25, 30, 35, 40, 45, 50}
Seeds: 3 seeds {42, 43, 44}
Modes: sample_once in {True, False}
Total: 9 sizes x 2 modes x 3 seeds = 54 runs.
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
    sample_once: bool,
    base_measure_type: str = "deep_sea_dag_maxent",
    max_episodes: Optional[int] = None,
    warmstart_steps: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    num_layers: int = 1,
    out_dir: str = "./results_scaling_sampler_comparison",
) -> Dict[str, Any]:
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    mode_str = "pure_ts" if sample_once else "multi_sample"
    ws = warmstart_steps if warmstart_steps is not None else max(10, size // 2)
    if warmstart_steps is not None:
        fname = f"deepsea_N{size}_{mode_str}_ws{ws}_s{seed}.json"
    else:
        fname = f"deepsea_N{size}_{mode_str}_s{seed}.json"
    fpath = os.path.join(out_dir, fname)

    # Check if already completed
    if os.path.exists(fpath):
        try:
            with open(fpath, "r") as fp:
                data = json.load(fp)
            if data.get("completed", False):
                print(f"[CACHED] N={size} Mode={mode_str} Seed={seed} -> Solved={data.get('solved_episode')}")
                return data
        except Exception:
            pass

    if max_episodes is None:
        if size >= 50:
            max_episodes = 8000
        elif size >= 40:
            max_episodes = 6000
        elif size >= 30:
            max_episodes = 4000
        else:
            max_episodes = 2500

    t0 = time.time()
    state_dim = size * size
    action_dim = 2

    env = make_env("deep_sea", size=size, seed=seed)
    base_measure = get_base_measure(base_measure_type, state_dim, action_dim, deep_sea_size=size)

    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=seed,
        alpha=5.0,
        batch_size=64,
        candidate_batch_size=256,
        base_measure=base_measure_type,
        prior_reward_mean=float(base_measure.prior_reward_mean),
        prior_reward_std=float(base_measure.prior_reward_std),
        hidden_dim=hidden_dim if hidden_dim is not None else 20,
        num_layers=num_layers,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.05,
        sgd_period=2,
        episodic_sgd=True,
        buffer_capacity=1000000,
        target_warmstart=True,
        warmstart_steps=ws,
        warmstart_lr_scale=1.0,
        sample_once_per_episode=sample_once,
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=float(4.0 * size / 5.0),
        one_living_network=True,
        dp_sampled_target=False,
        use_td_info_gain_decay=True,
        td_info_scale=1.0,
        verbose=False,
    )

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

        if ep % 100 == 0:
            print(f"[{mode_str.upper()}] N={size} Seed={seed} Ep={ep}/{max_episodes} Disc={first_discovery} Solved={solved_episode} ({time.time()-t0:.1f}s)", flush=True)

    elapsed = time.time() - t0
    result = {
        "size": size,
        "seed": seed,
        "sample_once": sample_once,
        "mode": mode_str,
        "warmstart_steps": ws,
        "base_measure": base_measure_type,
        "max_episodes": max_episodes,
        "total_episodes": len(returns),
        "first_discovery": first_discovery,
        "solved_episode": solved_episode,
        "final_cum_regret": cum_regret,
        "recent_return_50": float(np.mean(returns[-50:])) if len(returns) >= 50 else float(np.mean(returns)),
        "returns": returns,
        "cum_regrets": regrets,
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
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base_measure", type=str, default="deep_sea_nondag_maxent", help="Base measure type: deep_sea_nondag_maxent (model-free uniform) or deep_sea_dag_maxent")
    parser.add_argument("--warmstart_steps", type=int, default=None, help="Fixed warmstart steps (default: max(10, N//2))")
    parser.add_argument("--hidden_dim", type=int, default=20, help="Hidden dimension (default: 20)")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of hidden layers (default: 1)")
    parser.add_argument("--out_dir", type=str, default="./results_scaling_sampler_comparison")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    modes = [True, False]  # sample_once: True (Pure TS) vs False (Multi-Sample)

    tasks = []
    for size in args.sizes:
        for mode in modes:
            for seed in args.seeds:
                tasks.append((size, seed, mode))

    print(f"Total tasks to execute: {len(tasks)} across {args.workers} workers")
    print(f"Base Measure: {args.base_measure}")
    print(f"Sizes: {args.sizes}")
    print(f"Network: {args.num_layers} hidden layer(s) with {args.hidden_dim} units (Single MLP-{args.hidden_dim})")
    print(f"Warmstart Steps: {args.warmstart_steps if args.warmstart_steps is not None else 'Adaptive max(10, N//2)'}")
    print(f"Modes: Pure TS (sample_once=True) vs Multi-Sample (sample_once=False)")
    print(f"Seeds: {args.seeds}")

    # Prioritize smaller sizes first so we get immediate feedback across the board
    tasks.sort(key=lambda x: (x[0], x[2], x[1]))

    results = []
    t_start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_single_experiment,
                size=t[0],
                seed=t[1],
                sample_once=t[2],
                base_measure_type=args.base_measure,
                warmstart_steps=args.warmstart_steps,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
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
    summary_path = os.path.join(args.out_dir, "scaling_comparison_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"\n==========================================")
    print(f"All {len(results)}/{len(tasks)} experiments finished in {total_time:.1f}s ({total_time/60:.2f} mins)!")
    print(f"Summary written to {summary_path}")
    print(f"==========================================\n")


if __name__ == "__main__":
    main()
