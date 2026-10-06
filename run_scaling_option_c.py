import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import concurrent.futures
from typing import Dict, Any, List, Optional
import numpy as np
import torch
torch.set_num_threads(1)

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import BaseMeasure
from test_td_info_deepsea20 import HardSnapshotTDInfoAgent

class TrueSupportDAGBaseMeasure(BaseMeasure):
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


def run_scaling_trial(size: int, seed: int, max_episodes: int = 3000) -> Dict[str, Any]:
    torch.set_num_threads(1)
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
        warmstart_steps=size // 2,
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
    agent.base_measure = TrueSupportDAGBaseMeasure(size)
    agent.sampler.base_measure = agent.base_measure

    returns = []
    t0 = time.time()
    first_discovery = None
    solved_ep = None
    cum_regret = 0.0

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

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep

        if ep >= 20 and solved_ep is None:
            recent_ret = np.mean(returns[-20:])
            if recent_ret >= 0.8:
                solved_ep = ep
                break

    elapsed = time.time() - t0
    res = {
        "size": size,
        "seed": seed,
        "first_discovery": first_discovery,
        "solved_episode": solved_ep or max_episodes,
        "final_cum_regret": cum_regret,
        "elapsed_seconds": elapsed,
        "completed": True,
    }
    print(f"Done: N={size:2d} Seed={seed} -> Disc: {str(first_discovery):5s} | Solved: {str(solved_ep):5s} | Time: {elapsed:.1f}s", flush=True)
    return res


def main():
    sizes = [10, 15, 20, 25, 30, 40, 50]
    seeds = [42, 43, 44]
    max_workers = min(8, os.cpu_count() or 4)

    tasks = [(n, s) for n in sizes for s in seeds]
    print(f"Launching Option C Scaling Suite: {len(tasks)} tasks across sizes {sizes} on {max_workers} CPU workers...")
    t_start = time.time()

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_scaling_trial, n, s): (n, s) for (n, s) in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)

    print(f"\nAll {len(results)} tasks completed in {time.time() - t_start:.2f}s!")

    # Sort results
    results.sort(key=lambda r: (r["size"], r["seed"]))
    with open("results_scaling_option_c.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("Saved results to results_scaling_option_c.json")

    # Print summary by size
    print("\n" + "=" * 60)
    print("SCALING SUMMARY (Option C True-Support Uniform):")
    print("=" * 60)
    for n in sizes:
        n_res = [r for r in results if r["size"] == n]
        solvs = [r["solved_episode"] for r in n_res]
        discs = [r["first_discovery"] for r in n_res if r["first_discovery"] is not None]
        print(f"N={n:2d} | Solved: {np.mean(solvs):.1f} ± {np.std(solvs):.1f} | Disc: {np.mean(discs):.1f} ± {np.std(discs):.1f} (rate: {len(discs)}/{len(n_res)})")

if __name__ == "__main__":
    main()
