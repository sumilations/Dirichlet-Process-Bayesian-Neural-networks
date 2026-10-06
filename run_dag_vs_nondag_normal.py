import os
import sys
import time
import json
import numpy as np
import torch

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import DeepSeaDownwardDAGBaseMeasure, DeepSeaBaseMeasure
from test_td_info_deepsea20 import HardSnapshotTDInfoAgent

def run_single_experiment(variant="dag", seed=42, size=20, max_episodes=800):
    state_dim = size * size
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
        prior_reward_mean=0.0,          # PURE ZERO MEAN
        prior_reward_std=1.0,           # STANDARD NORMAL (SIGMA = 1.0)
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.0,
        target_warmstart=False,
        warmstart_steps=size // 2,      # W = 10 steps
        sample_once_per_episode=False,  # FRESH PER STEP!
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

    # Set exact base measure: DAG vs Non-DAG with Standard Normal rewards N(0, 1)
    if variant == "dag":
        agent.base_measure = DeepSeaDownwardDAGBaseMeasure(
            size, prior_reward_mean=0.0, prior_reward_std=1.0, goal_bonus=False
        )
    else:
        agent.base_measure = DeepSeaBaseMeasure(
            size, prior_reward_mean=0.0, prior_reward_std=1.0
        )
    agent.sampler.base_measure = agent.base_measure

    returns = []
    cum_regrets = []
    optimal_ret = 0.99
    running_regret = 0.0

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
        regret = max(0.0, optimal_ret - ep_ret)
        running_regret += regret
        cum_regrets.append(running_regret)

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
        "final_cum_regret": running_regret,
        "cum_regrets": cum_regrets,
        "elapsed_sec": elapsed,
    }


def main():
    seeds = [42, 43, 44, 45, 46]
    size = 20
    print("=" * 80)
    print(f"HEAD-TO-HEAD BENCHMARK: DAG vs Non-DAG on DeepSea N={size} with STANDARD NORMAL REWARDS")
    print(f"Fresh Sampling at EACH Warmstart Step (W={size//2}), Hard Snapshot (K=5), TD Info Gain")
    print(f"Seeds: {seeds}")
    print("=" * 80)

    results_dag = []
    print("\n--- 1. RUNNING DAG PRIOR (Standard Normal N(0, 1)) ---")
    for s in seeds:
        res = run_single_experiment(variant="dag", seed=s, size=size)
        results_dag.append(res)
        print(f"DAG Seed {s} | Discovery: Ep {str(res['first_discovery']):5s} | Solved: Ep {str(res['solved_episode']):5s} | Regret: {res['final_cum_regret']:.1f} | Time: {res['elapsed_sec']:.2f}s", flush=True)

    results_nondag = []
    print("\n--- 2. RUNNING Non-DAG PRIOR (Standard Normal N(0, 1)) ---")
    for s in seeds:
        res = run_single_experiment(variant="nondag", seed=s, size=size)
        results_nondag.append(res)
        print(f"Non-DAG Seed {s} | Discovery: Ep {str(res['first_discovery']):5s} | Solved: Ep {str(res['solved_episode']):5s} | Regret: {res['final_cum_regret']:.1f} | Time: {res['elapsed_sec']:.2f}s", flush=True)

    # Compute Summary Stats
    dag_sol = [r["solved_episode"] for r in results_dag if r["solved_episode"] is not None]
    dag_disc = [r["first_discovery"] for r in results_dag if r["first_discovery"] is not None]
    dag_reg = [r["final_cum_regret"] for r in results_dag]

    nondag_sol = [r["solved_episode"] for r in results_nondag if r["solved_episode"] is not None]
    nondag_disc = [r["first_discovery"] for r in results_nondag if r["first_discovery"] is not None]
    nondag_reg = [r["final_cum_regret"] for r in results_nondag]

    print("\n" + "=" * 80)
    print("FINAL SUMMARY: DAG vs Non-DAG (STANDARD NORMAL REWARDS)")
    print("=" * 80)
    print(f"DAG Prior:     Solved: {len(dag_sol)}/{len(seeds)} | Avg Solve: {np.mean(dag_sol):.1f} ± {np.std(dag_sol):.1f} | Avg Disc: {np.mean(dag_disc):.1f} ± {np.std(dag_disc):.1f} | Avg Regret: {np.mean(dag_reg):.1f}")
    if nondag_sol:
        print(f"Non-DAG Prior: Solved: {len(nondag_sol)}/{len(seeds)} | Avg Solve: {np.mean(nondag_sol):.1f} ± {np.std(nondag_sol):.1f} | Avg Disc: {np.mean(nondag_disc):.1f} ± {np.std(nondag_disc):.1f} | Avg Regret: {np.mean(nondag_reg):.1f}")
    else:
        print(f"Non-DAG Prior: Solved: {len(nondag_sol)}/{len(seeds)} | Avg Disc: {np.mean(nondag_disc):.1f} if nondag_disc else 'None'")

    # Save to JSON for plotting
    out_data = {
        "seeds": seeds,
        "dag": results_dag,
        "nondag": results_nondag,
    }
    with open("results_dag_vs_nondag_stdnormal.json", "w") as fp:
        json.dump(out_data, fp, indent=2)
    print("\nSaved benchmark data to results_dag_vs_nondag_stdnormal.json")

if __name__ == "__main__":
    main()
