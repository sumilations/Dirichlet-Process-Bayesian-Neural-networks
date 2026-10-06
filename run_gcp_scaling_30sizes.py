#!/usr/bin/env python3
"""Option C Scaling Benchmark across 30 sizes in [10, 50] on Google Cloud.

Protocol (Option C):
- True-Support Uniform Base Measure: U(-0.01/N, 1.0)
- TD Information Gain decay enabled
- Dynamic fresh sampling of prior atoms and empirical batch at every warmstart step
- Single living MLP network (64 hidden, 2 layers, LayerNorm)
- Hard Snapshot target network (K=5)
- Horizon-scaled episode cap (up to 8,000 for N=50)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import argparse
import concurrent.futures
from typing import Dict, Any, List, Optional
import numpy as np
import torch
torch.set_num_threads(1)

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import BaseMeasure
from test_td_info_deepsea20 import HardSnapshotTDInfoAgent

# 30 sizes between 10 and 50
DEFAULT_30_SIZES = [
    10, 11, 13, 14, 16, 17, 18, 20, 21, 22,
    24, 25, 27, 28, 29, 31, 32, 33, 35, 36,
    38, 39, 40, 42, 43, 44, 46, 47, 49, 50
]


class TrueSupportDAGBaseMeasure(BaseMeasure):
    """Option C Base Measure with DAG support constraint (col <= row)."""
    def __init__(self, size: int):
        self.size = size
        state_dim = size * size
        self.r_min = -0.01 / size
        self.r_max = 1.0
        mean = (self.r_min + self.r_max) / 2.0
        std = (self.r_max - self.r_min) / np.sqrt(12)
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=mean, prior_reward_std=std)

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        r = rng.uniform(self.r_min, self.r_max, size=count).astype(np.float32)

        for i in range(count):
            row = rng.randint(0, self.size)
            col = rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                act = a[i]
                next_col = min(self.size - 1, col + 1) if act == 1 else max(0, col - 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


class TrueSupportNonDAGBaseMeasure(BaseMeasure):
    """Option C Base Measure without DAG constraint (uniform over entire N x N grid)."""
    def __init__(self, size: int):
        self.size = size
        state_dim = size * size
        self.r_min = -0.01 / size
        self.r_max = 1.0
        mean = (self.r_min + self.r_max) / 2.0
        std = (self.r_max - self.r_min) / np.sqrt(12)
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=mean, prior_reward_std=std)

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        r = rng.uniform(self.r_min, self.r_max, size=count).astype(np.float32)

        for i in range(count):
            row = rng.randint(0, self.size)
            col = rng.randint(0, self.size)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                act = a[i]
                next_col = min(self.size - 1, col + 1) if act == 1 else max(0, col - 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


def run_scaling_trial(
    size: int,
    seed: int,
    variant: str = "dag",
    out_dir: str = "./results_gcp_scaling_option_c",
    max_episodes: Optional[int] = None
) -> Dict[str, Any]:
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"deepsea_N{size}_{variant}_s{seed}.json"
    fpath = os.path.join(out_dir, fname)

    # Check cache
    if os.path.exists(fpath):
        try:
            with open(fpath, "r") as fp:
                cached = json.load(fp)
            if cached.get("completed", False):
                print(f"[CACHED] N={size:2d} Seed={seed} ({variant}) -> Solved: {cached.get('solved_episode')}")
                return cached
        except Exception:
            pass

    if max_episodes is None:
        if size >= 45:
            max_episodes = 8000
        elif size >= 35:
            max_episodes = 6000
        elif size >= 25:
            max_episodes = 4500
        elif size >= 15:
            max_episodes = 3000
        else:
            max_episodes = 2000

    state_dim = size * size
    r_min = -0.01 / size
    mean = (r_min + 1.0) / 2.0
    std = (1.0 - r_min) / np.sqrt(12)

    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=state_dim,
        action_dim=2,
        seed=seed,
        alpha=5.0,
        batch_size=64,
        candidate_batch_size=128,
        base_measure="deep_sea",
        prior_reward_mean=float(mean),
        prior_reward_std=float(std),
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.0,
        target_warmstart=False,
        warmstart_steps=max(5, size // 2),
        sample_once_per_episode=False,
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        episodic_sgd=False,
        use_td_info_gain_decay=True,
        priority_positive_slot=True,
    )

    env = make_env("deep_sea", seed=seed, size=size)
    agent = HardSnapshotTDInfoAgent(cfg, period_k=5)

    if variant.lower() == "dag":
        agent.base_measure = TrueSupportDAGBaseMeasure(size)
    else:
        agent.base_measure = TrueSupportNonDAGBaseMeasure(size)
    agent.sampler.base_measure = agent.base_measure

    returns = []
    t0 = time.time()
    first_discovery = None
    solved_ep = None
    cum_regret = 0.0
    cum_regrets_subsampled = []

    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        s = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            a = agent.act(s)
            sn, r, done, _ = env.step(a)
            agent.replay.push(s, a, float(r), sn, done)
            s = sn
            ep_ret += float(r)

        returns.append(ep_ret)
        regret = max(0.0, 0.99 - ep_ret)
        cum_regret += regret

        if ep % 5 == 0 or ep == 1 or ep == max_episodes:
            cum_regrets_subsampled.append((ep, float(cum_regret)))

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep
            print(f"*** [DISCOVERY] N={size:2d} Seed={seed} ({variant}) at Ep {ep}! ({time.time() - t0:.1f}s)")

        if ep >= 20 and solved_ep is None:
            recent_ret = np.mean(returns[-20:])
            if recent_ret >= 0.8:
                solved_ep = ep
                print(f"--> [SOLVED]    N={size:2d} Seed={seed} ({variant}) at Ep {ep}! Regret={cum_regret:.1f} ({time.time() - t0:.1f}s)")
                break

    elapsed = time.time() - t0
    res = {
        "completed": True,
        "size": size,
        "seed": seed,
        "variant": variant,
        "first_discovery": first_discovery,
        "solved_episode": solved_ep or max_episodes,
        "final_cum_regret": float(cum_regret),
        "elapsed_time": float(elapsed),
        "max_episodes": max_episodes,
        "cum_regrets": cum_regrets_subsampled,
    }

    with open(fpath, "w") as fp:
        json.dump(res, fp, indent=2)

    status_str = f"Solved: {solved_ep}" if solved_ep else f"Timeout ({max_episodes})"
    print(f"Done: N={size:2d} Seed={seed} ({variant}) -> Disc: {first_discovery} | {status_str} | Time: {elapsed:.1f}s")
    return res


def main():
    parser = argparse.ArgumentParser(description="GCP DeepSea 30-Size Scaling Suite")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_30_SIZES, help="Grid sizes")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--variant", type=str, default="dag", choices=["dag", "nondag"], help="Prior support variant")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--out_dir", type=str, default="./results_gcp_scaling_option_c", help="Output directory")
    args = parser.parse_args()

    sizes = args.sizes
    seeds = args.seeds
    variant = args.variant
    max_workers = min(args.workers, os.cpu_count() or 4)

    tasks = [(n, s, variant, args.out_dir) for n in sizes for s in seeds]
    print("=" * 80)
    print(f"Option C Scaling Suite on GCP ({variant.upper()} Prior)")
    print(f"Sizes ({len(sizes)}): {sizes}")
    print(f"Seeds: {seeds}")
    print(f"Total tasks: {len(tasks)} across {max_workers} CPU workers")
    print(f"Output directory: {args.out_dir}")
    print("=" * 80)

    t_start = time.time()
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_scaling_trial, n, s, v, out): (n, s) for (n, s, v, out) in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Error in task: {e}")

    print(f"\nAll {len(results)} tasks completed in {time.time() - t_start:.2f}s!")

    results.sort(key=lambda r: (r["size"], r["seed"]))
    summary_path = os.path.join(args.out_dir, "scaling_comparison_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Saved complete summary to {summary_path}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"SCALING SUMMARY ({variant.upper()} Prior, Option C):")
    print("=" * 80)
    for n in sorted(list(set(r["size"] for r in results))):
        n_res = [r for r in results if r["size"] == n]
        solvs = [r["solved_episode"] for r in n_res]
        discs = [r["first_discovery"] for r in n_res if r["first_discovery"] is not None]
        print(f"N={n:2d} | Solved: {np.mean(solvs):6.1f} ± {np.std(solvs):5.1f} | Disc: {np.mean(discs) if discs else float('nan'):6.1f} (rate: {len(discs)}/{len(n_res)})")


if __name__ == "__main__":
    main()
