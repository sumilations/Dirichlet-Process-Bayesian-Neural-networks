"""Cart-Pole Swing-Up Pure Thompson Sampling Benchmark.

Evaluates Pure TS (sample_once_per_episode=True) vs Multi-Sample (morning baseline)
with Gaussian State Base Measure (s ~ N(0, I)) and Optimistic Reward Prior (r ~ N(0.5, 0.5)).

Key Features:
1. Pure Model-Free State Base Measure: isotropic Gaussian, zero angle/physics priors.
2. Pure TS: Draws one single DP posterior sample at episode boundary and freezes the hypothesis.
3. Multi-process parallel execution with torch.set_num_threads(1).
4. Live JSON checkpointing and metrics logging.
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
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

from unified_dp_dqn import (
    DPDQNAgent,
    DPDQNConfig,
    make_env,
)


def run_single_cartpole_experiment(
    algo: str,
    alpha: float,
    seed: int,
    pure_ts: bool,
    episodes: int = 2500,
    warmstart_steps: int = 2,
    out_dir: str = "./results_cartpole_pure_ts",
) -> Dict[str, Any]:
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)

    mode_tag = "pure_ts" if pure_ts else "multi_sample"
    fname = f"cartpole_gaussian_a{alpha}_{mode_tag}_s{seed}.json"
    ckpt_name = f"cartpole_gaussian_a{alpha}_{mode_tag}_s{seed}_ckpt.json"
    out_path = os.path.join(out_dir, fname)
    ckpt_path = os.path.join(out_dir, ckpt_name)

    # Check cache
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as fp:
                data = json.load(fp)
            if data.get("completed", False):
                print(f"[CACHED] CartPole {mode_tag} a={alpha} s={seed} -> Solved={data.get('solved_episode')}", flush=True)
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
        use_layer_norm=True,
        activation="relu",
        alpha=alpha,
        base_measure_type="gaussian",
        haar_angle=False,
        sampler_type="vashishtha_maillard",
        w_min=0.05,
        seed=seed,
        buffer_capacity=1000000,
        warmstart_steps=warmstart_steps,
        sample_once_per_episode=pure_ts,
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

    r_star = 850.0  # Optimal return upper bound for regret tracking

    for ep in range(1, episodes + 1):
        agent.reset_episode()
        s = env.reset()
        done = False
        ep_ret = 0.0
        upr = 0

        while not done:
            a = agent.act(s)
            sn, r, done, info = env.step(a)
            agent.step(s, a, float(r), sn, done)

            ep_ret += float(r)
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
                print(f"*** [{mode_tag.upper()} a={alpha} s={seed}] FIRST DISCOVERY at Episode {ep}! ({time.time()-t0:.1f}s)", flush=True)
            recent_solved_counter += 1
            if recent_solved_counter >= 10 and solved_episode is None:
                solved_episode = ep - 9
                print(f"*** [{mode_tag.upper()} a={alpha} s={seed}] SOLVED at Episode {solved_episode}! (Max Upright: {max(uprights)}) ({time.time()-t0:.1f}s)", flush=True)
        else:
            recent_solved_counter = 0

        if ep % 50 == 0 or ep == episodes:
            elapsed = time.time() - t0
            ret_50 = np.mean(returns[-50:])
            upr_50 = np.mean(uprights[-50:])
            print(f"[{mode_tag.upper()} a={alpha} s={seed}] Ep {ep:4d}/{episodes} | "
                  f"Ret(50): {ret_50:6.2f} | Upr(50): {upr_50:5.1f} | "
                  f"Disc: {first_discovery} | Solved: {solved_episode} | "
                  f"CumReg: {cum_regret/1e3:6.1f}k | Time: {elapsed:5.1f}s", flush=True)

            ckpt = {
                "env": "cartpole_swingup",
                "mode": mode_tag,
                "pure_ts": pure_ts,
                "alpha": alpha,
                "seed": seed,
                "base_measure": "gaussian",
                "episodes": ep,
                "first_discovery": first_discovery,
                "solved_episode": solved_episode,
                "returns": returns,
                "upright_steps": uprights,
                "regrets": regrets,
                "cumulative_regrets": cumulative_regrets,
                "elapsed_sec": elapsed,
                "completed": (ep == episodes)
            }
            with open(ckpt_path, "w") as fp:
                json.dump(ckpt, fp)

    ckpt["completed"] = True
    with open(out_path, "w") as fp:
        json.dump(ckpt, fp, indent=2)

    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Cart-Pole Swing-Up Pure TS Benchmark")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--episodes", type=int, default=2500, help="Episodes per run")
    parser.add_argument("--warmstart_steps", type=int, default=2, help="Warmstart steps per episode")
    parser.add_argument("--out_dir", type=str, default="./results_cartpole_pure_ts", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    # 1. Pure TS (sample_once_per_episode = True)
    for a in [3.0, 15.0]:
        for s in [42, 43, 44, 45, 46]:
            tasks.append({
                "algo": "dp_dqn_construct",
                "alpha": a,
                "seed": s,
                "pure_ts": True,
                "episodes": args.episodes,
                "warmstart_steps": args.warmstart_steps,
                "out_dir": args.out_dir,
            })

    # 2. Multi-Sample baseline (sample_once_per_episode = False) for comparison
    for a in [3.0]:
        for s in [42, 43, 44]:
            tasks.append({
                "algo": "dp_dqn_construct",
                "alpha": a,
                "seed": s,
                "pure_ts": False,
                "episodes": args.episodes,
                "warmstart_steps": args.warmstart_steps,
                "out_dir": args.out_dir,
            })

    print(f"Launching Cart-Pole Benchmark: {len(tasks)} tasks across {args.workers} workers", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_single_cartpole_experiment,
                t["algo"],
                t["alpha"],
                t["seed"],
                t["pure_ts"],
                t["episodes"],
                t["warmstart_steps"],
                t["out_dir"],
            ): t for t in tasks
        }

        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                res = fut.result()
                print(f"[DONE] CartPole Mode={res['mode']} a={res['alpha']} s={res['seed']} -> Disc={res.get('first_discovery')} Solved={res.get('solved_episode')}", flush=True)
            except Exception as e:
                print(f"[ERROR] Task failed: {t} -> {e}", flush=True)


if __name__ == "__main__":
    main()
