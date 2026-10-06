import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import BaseMeasure
from test_td_info_deepsea20 import HardSnapshotTDInfoAgent

class TrueSupportDAGBaseMeasure(BaseMeasure):
    """Causal DAG Prior with True Physical Support Uniform Rewards [r_min, R_max]."""
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
    """Model-Free Uniform Base Measure (Zero DAG Constraint) with True Physical Support Rewards."""
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
                next_col = rng.randint(0, next_row + 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


def run_single(variant="dag", size=20, seed=42, max_episodes=800):
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

    if variant == "dag":
        agent.base_measure = TrueSupportDAGBaseMeasure(size)
    else:
        agent.base_measure = TrueSupportNonDAGBaseMeasure(size)
    agent.sampler.base_measure = agent.base_measure

    returns = []
    cum_regrets = []
    cum_reg = 0.0
    t0 = time.time()
    first_discovery = None
    solved_ep = None

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
        cum_reg += regret
        cum_regrets.append(cum_reg)

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep

        if ep >= 20 and solved_ep is None:
            recent_ret = np.mean(returns[-20:])
            if recent_ret >= 0.8:
                solved_ep = ep

    elapsed = time.time() - t0
    return {
        "variant": variant,
        "seed": seed,
        "first_discovery": first_discovery,
        "solved_episode": solved_ep,
        "final_cum_regret": cum_reg,
        "cum_regrets": cum_regrets,
        "elapsed_sec": elapsed,
    }


def main():
    seeds = [42, 43, 44, 45, 46]
    size = 20
    print("=" * 80)
    print(f"HEAD-TO-HEAD BENCHMARK: DAG vs Non-DAG on DeepSea N={size}")
    print("Option C: True-Support Uniform U(-0.01/N, 1.0) + TD Info Gain")
    print(f"Fresh Sampling at EACH Warmstart Step (W={size//2}), Hard Snapshot (K=5)")
    print(f"Seeds: {seeds}")
    print("=" * 80)

    results_dag = []
    print("\n--- 1. RUNNING DAG PRIOR (True Support Uniform) ---")
    for s in seeds:
        res = run_single(variant="dag", size=size, seed=s)
        results_dag.append(res)
        print(f"DAG Seed {s} | Discovery: Ep {str(res['first_discovery']):5s} | Solved: Ep {str(res['solved_episode']):5s} | Regret: {res['final_cum_regret']:.1f} | Time: {res['elapsed_sec']:.2f}s", flush=True)

    results_nondag = []
    print("\n--- 2. RUNNING Non-DAG PRIOR (True Support Uniform) ---")
    for s in seeds:
        res = run_single(variant="nondag", size=size, seed=s)
        results_nondag.append(res)
        print(f"Non-DAG Seed {s} | Discovery: Ep {str(res['first_discovery']):5s} | Solved: Ep {str(res['solved_episode']):5s} | Regret: {res['final_cum_regret']:.1f} | Time: {res['elapsed_sec']:.2f}s", flush=True)

    # Compute stats
    dag_sol = [r["solved_episode"] for r in results_dag if r["solved_episode"] is not None]
    dag_disc = [r["first_discovery"] for r in results_dag if r["first_discovery"] is not None]
    dag_reg = [r["final_cum_regret"] for r in results_dag]

    nondag_sol = [r["solved_episode"] for r in results_nondag if r["solved_episode"] is not None]
    nondag_disc = [r["first_discovery"] for r in results_nondag if r["first_discovery"] is not None]
    nondag_reg = [r["final_cum_regret"] for r in results_nondag]

    print("\n" + "=" * 80)
    print("FINAL SUMMARY: DAG vs Non-DAG (Option C: True-Support Uniform)")
    print("=" * 80)
    print(f"DAG Prior:     Solved: {len(dag_sol)}/{len(seeds)} | Avg Solve: {np.mean(dag_sol):.1f} ± {np.std(dag_sol):.1f} | Avg Disc: {np.mean(dag_disc):.1f} ± {np.std(dag_disc):.1f} | Avg Regret: {np.mean(dag_reg):.1f}")
    print(f"Non-DAG Prior: Solved: {len(nondag_sol)}/{len(seeds)} | Avg Solve: {np.mean(nondag_sol):.1f} ± {np.std(nondag_sol):.1f} | Avg Disc: {np.mean(nondag_disc):.1f} ± {np.std(nondag_disc):.1f} | Avg Regret: {np.mean(nondag_reg):.1f}")

    out_data = {
        "seeds": seeds,
        "dag": results_dag,
        "nondag": results_nondag,
    }
    with open("results_option_c_dag_vs_nondag.json", "w") as fp:
        json.dump(out_data, fp, indent=2)
    print("\nSaved benchmark data to results_option_c_dag_vs_nondag.json")

if __name__ == "__main__":
    main()
