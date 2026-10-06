"""DeepSea N=50 Pure TS 10-Seed Benchmark:
Comparing Scaled K_prior (K_prior = 4*N = 200) vs Standard K_prior (K_prior = 50)
Runs up to 20,000 episodes across 10 distinct seeds.
Records high-resolution cumulative regret trajectories.
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
    max_episodes: int = 20000,
    out_dir: str = "./results_deepsea50_pure_ts_10seeds",
) -> Dict[str, Any]:
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    mode_str = "scaled_kprior" if scaled_kprior else "standard_kprior"
    fname = f"deepsea_N{size}_{mode_str}_s{seed}.json"
    fpath = os.path.join(out_dir, fname)

    # Check cache if already completed
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

    if scaled_kprior:
        # 4 atoms per level -> 4 * 50 = 200 atoms
        vm_mult = float(4.0 * size / alpha)  # 40.0
        batch_sz = 128
        cand_sz = 256
    else:
        # Standard K_prior = 50 atoms
        vm_mult = 10.0
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
    first_discovery: Optional[int] = None
    solved_episode: Optional[int] = None
    cum_regret = 0.0

    # In DeepSea bsuite, reaching treasure gives 1.0 - 0.01 = 0.99 return
    optimal_return = 0.99

    # Record regret trajectory at intervals
    regret_curve: List[Dict[str, float]] = []

    # Checkpoint every 500 episodes
    ckpt_path = fpath + ".ckpt"

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
        regret = max(0.0, optimal_return - ep_ret)
        cum_regret += regret

        # Record every 10 episodes or at key events
        if ep % 10 == 0 or ep == 1:
            regret_curve.append({
                "episode": ep,
                "cum_regret": float(cum_regret),
                "recent_ret": float(np.mean(returns[-min(ep, 50):]))
            })

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep
            print(f"*** [DISCOVERY] N={size} {mode_str.upper()} Seed={seed} at Ep {ep}! ({time.time()-t0:.1f}s)", flush=True)

        # Solved criteria: 50-episode moving average return >= 0.85
        if ep >= 50:
            recent_ret = np.mean(returns[-50:])
            if recent_ret >= 0.85:
                if solved_episode is None:
                    solved_episode = ep
                    print(f"*** [SOLVED] N={size} {mode_str.upper()} Seed={seed} at Ep {ep}! ({time.time()-t0:.1f}s)", flush=True)
                # Run for 200 post-solve episodes to demonstrate flatline regret curve
                elif ep >= solved_episode + 200:
                    print(f"--> [CONVERGED] N={size} {mode_str.upper()} Seed={seed} stabilized after solve. Halting early at Ep {ep}.", flush=True)
                    break

        if ep % 500 == 0:
            print(f"[{mode_str.upper()}] N={size} Seed={seed} Ep={ep}/{max_episodes} Disc={first_discovery} Solv={solved_episode} CumReg={cum_regret:.1f} ({time.time()-t0:.1f}s)", flush=True)
            # Intermediate checkpoint
            ckpt_data = {
                "size": size,
                "seed": seed,
                "scaled_kprior": scaled_kprior,
                "mode": mode_str,
                "current_episode": ep,
                "first_discovery": first_discovery,
                "solved_episode": solved_episode,
                "cum_regret": cum_regret,
                "regret_curve": regret_curve,
                "elapsed_seconds": time.time() - t0,
            }
            with open(ckpt_path, "w") as fp:
                json.dump(ckpt_data, fp)

    elapsed = time.time() - t0
    # Final regret point
    if regret_curve[-1]["episode"] != len(returns):
        regret_curve.append({
            "episode": len(returns),
            "cum_regret": float(cum_regret),
            "recent_ret": float(np.mean(returns[-min(len(returns), 50):]))
        })

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
        "final_cum_regret": float(cum_regret),
        "recent_return_50": float(np.mean(returns[-50:])) if len(returns) >= 50 else float(np.mean(returns)),
        "regret_curve": regret_curve,
        "elapsed_seconds": elapsed,
        "completed": True,
        "timestamp": time.time(),
    }

    with open(fpath, "w") as fp:
        json.dump(result, fp, indent=2)

    if os.path.exists(ckpt_path):
        try:
            os.remove(ckpt_path)
        except OSError:
            pass

    status = f"SOLVED at Ep {solved_episode}" if solved_episode else (f"Discovered at Ep {first_discovery}" if first_discovery else "FAILED/UNSOLVED")
    print(f"--> [FINISH] N={size} {mode_str.upper()} Seed={seed} -> {status} in {elapsed:.1f}s ({len(returns)} eps, Final Regret: {cum_regret:.1f})", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--max_episodes", type=int, default=20000)
    parser.add_argument("--out_dir", type=str, default="./results_deepsea50_pure_ts_10seeds")
    args = parser.parse_args()

    size = 50
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    tasks = []
    # 1. Scaled K_prior (4*N = 200 atoms) across 10 seeds
    for s in seeds:
        tasks.append((size, s, True))
    # 2. Standard K_prior (50 atoms) across 10 seeds
    for s in seeds:
        tasks.append((size, s, False))

    print(f"============================================================")
    print(f" DeepSea N=50 Pure TS 10-Seed Benchmark")
    print(f" Total Tasks : {len(tasks)} (10 Scaled K_prior + 10 Standard K_prior)")
    print(f" Workers     : {args.workers}")
    print(f" Max Episodes: {args.max_episodes}")
    print(f" Output Dir  : {args.out_dir}")
    print(f"============================================================")

    t_start = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_single_experiment,
                size=t[0],
                seed=t[1],
                scaled_kprior=t[2],
                max_episodes=args.max_episodes,
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
                print(f"[ERROR] Task {t} crashed: {e}", flush=True)

    total_time = time.time() - t_start
    summary_path = os.path.join(args.out_dir, "deepsea50_10seeds_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"\n============================================================")
    print(f"All {len(results)}/{len(tasks)} experiments finished in {total_time:.1f}s ({total_time/60:.2f} mins)!")
    print(f"Summary written to {summary_path}")
    print(f"============================================================\n")


if __name__ == "__main__":
    main()
