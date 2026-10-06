"""9-Run Systematic PSRL, Sparse SGD, & Base Measure Study for DP-DQN on Cart-Pole Swing-Up.

Executes 9 configurations in parallel across 9 CPU workers on identical seed:
1. PSRL-Alpha15-WarmStart: Pure episodic burst (100 steps at boundary), alpha=15, Warm-Start Target
2. PSRL-Alpha15-NoWarmStart: Pure episodic burst (100 steps at boundary), alpha=15, Standard Target
3. PSRL-Alpha50-WarmStart: Pure episodic burst (100 steps at boundary), alpha=50, Warm-Start Target
4. PSRL-Alpha50-NoWarmStart: Pure episodic burst (100 steps at boundary), alpha=50, Standard Target
5. SGD4-WarmStart: Online sparse updating (sgd_period=4), alpha=50, Warm-Start Target
6. SGD4-NoWarmStart: Online sparse updating (sgd_period=4), alpha=50, Standard Target
7. SGD8-WarmStart: Online sparse updating (sgd_period=8), alpha=50, Warm-Start Target
8. SGD8-NoWarmStart: Online sparse updating (sgd_period=8), alpha=50, Standard Target
9. Winner-UninformedGaussian: Previous winning architecture (sgd_period=2, alpha=50, WarmStart)
   tested with completely uninformed Uniform-State + Gaussian-Reward Base Measure
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.envs.cartpole_jmlr import CartpoleSwingupJMLR
from src.rl.dp_dqn import DPDQNAgent


CONFIGURATIONS = [
    {
        "name": "1. PSRL-Alpha15-WarmStart",
        "short_name": "PSRL-a15-WarmStart",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 15.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episodic_burst_steps": 100,
            "episode_target_warmstart": True,
        }
    },
    {
        "name": "2. PSRL-Alpha15-NoWarmStart",
        "short_name": "PSRL-a15-NoWarmStart",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 15.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episodic_burst_steps": 100,
            "episode_target_warmstart": False,
        }
    },
    {
        "name": "3. PSRL-Alpha50-WarmStart",
        "short_name": "PSRL-a50-WarmStart",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episodic_burst_steps": 100,
            "episode_target_warmstart": True,
        }
    },
    {
        "name": "4. PSRL-Alpha50-NoWarmStart",
        "short_name": "PSRL-a50-NoWarmStart",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episodic_burst_steps": 100,
            "episode_target_warmstart": False,
        }
    },
    {
        "name": "5. SGD4-WarmStart",
        "short_name": "SGD4-WarmStart",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 4,
            "episodic_burst_steps": 0,
            "episode_target_warmstart": True,
        }
    },
    {
        "name": "6. SGD4-NoWarmStart",
        "short_name": "SGD4-NoWarmStart",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 4,
            "episodic_burst_steps": 0,
            "episode_target_warmstart": False,
        }
    },
    {
        "name": "7. SGD8-WarmStart",
        "short_name": "SGD8-WarmStart",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 8,
            "episodic_burst_steps": 0,
            "episode_target_warmstart": True,
        }
    },
    {
        "name": "8. SGD8-NoWarmStart",
        "short_name": "SGD8-NoWarmStart",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 8,
            "episodic_burst_steps": 0,
            "episode_target_warmstart": False,
        }
    },
    {
        "name": "9. Winner-UninformedGaussian",
        "short_name": "Winner-UninformedGauss",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "uniform_gaussian_optimistic",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episodic_burst_steps": 0,
            "episode_target_warmstart": True,
        }
    },
]


def run_single_config(config: Dict[str, Any], num_episodes: int, seed: int) -> Dict[str, Any]:
    """Execute episodes for a single DP-DQN configuration."""
    torch.set_num_threads(1)
    env = CartpoleSwingupJMLR(seed=seed)
    params = config["params"]

    agent = DPDQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=50,
        candidate_batch_size=256,
        truncation_K=128,
        prior_reward_mean=1.0,
        gamma=0.99,
        lr=1e-3,
        capacity=100000,
        seed=seed,
        **params
    )

    returns = []
    upright_steps = []
    peak_cos_thetas = []

    start_time = time.time()
    tag = config["short_name"]

    for ep in range(num_episodes):
        agent.start_episode()
        state = env.reset()
        done = False
        ep_reward = 0.0
        upr_count = 0
        peak_cos = -1.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.step_update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if info.get("is_upright", False):
                upr_count += 1
            if info.get("cos_theta", -1.0) > peak_cos:
                peak_cos = info["cos_theta"]

        agent.end_episode()

        returns.append(float(ep_reward))
        upright_steps.append(int(upr_count))
        peak_cos_thetas.append(float(peak_cos))

        if (ep + 1) % 100 == 0 or (ep + 1) == num_episodes:
            recent_ret = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
            recent_upr = np.mean(upright_steps[-50:]) if len(upright_steps) >= 50 else np.mean(upright_steps)
            cum_upr = int(np.sum(upright_steps))
            elapsed = time.time() - start_time
            print(f"  [{tag:22s}] Ep {ep+1:4d}/{num_episodes} | Ret(50): {recent_ret:6.2f} | Upr(50): {recent_upr:4.1f} | CumUpr: {cum_upr:6d} | PeakCos: {peak_cos:5.2f} | {elapsed:4.0f}s", flush=True)

    elapsed_total = time.time() - start_time
    return {
        "name": config["name"],
        "short_name": config["short_name"],
        "params": params,
        "returns": returns,
        "upright_steps": upright_steps,
        "peak_cos_thetas": peak_cos_thetas,
        "final_50ep_mean_return": float(np.mean(returns[-50:])),
        "final_50ep_mean_upright": float(np.mean(upright_steps[-50:])),
        "final_cumulative_return": float(np.sum(returns)),
        "final_cumulative_upright": int(np.sum(upright_steps)),
        "max_cos_theta_reached": float(np.max(peak_cos_thetas)),
        "elapsed": elapsed_total
    }


def worker_task(args: Tuple[Dict[str, Any], int, int]) -> Dict[str, Any]:
    config, num_episodes, seed = args
    return run_single_config(config, num_episodes, seed)


def main():
    parser = argparse.ArgumentParser(description="9-Run Systematic PSRL, Sparse SGD, & Base Measure Study for DP-DQN")
    parser.add_argument("--episodes", type=int, default=2500, help="Episodes per run (default: 2500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for all runs (default: 42)")
    parser.add_argument("--workers", type=int, default=9, help="Parallel CPU workers (default: 9)")
    parser.add_argument("--out_dir", type=str, default="results_rl", help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Run 2 episodes only to test pipeline")
    args = parser.parse_args()

    num_episodes = 2 if args.dry_run else args.episodes
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print(f"9-Run Systematic PSRL, Sparse SGD, & Base Measure Study")
    print(f"Episodes per run: {num_episodes} | Identical Seed: {args.seed} | CPU Workers: {args.workers}")
    print("=" * 80)

    task_args = [(cfg, num_episodes, args.seed) for cfg in CONFIGURATIONS]
    t0 = time.time()
    results = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_task, arg): arg[0]["short_name"] for arg in task_args}
        for fut in concurrent.futures.as_completed(futures):
            short_name = futures[fut]
            try:
                res = fut.result()
                results[short_name] = res
                print(f"--> Finished {short_name} in {res['elapsed']:.1f}s | Ret(50): {res['final_50ep_mean_return']:.2f} | CumUpr: {res['final_cumulative_upright']}")
            except Exception as exc:
                print(f"Error in {short_name}: {exc}")
                raise exc

    wall_clock = time.time() - t0
    out_file = os.path.join(args.out_dir, "psrl_sgd_study_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"STUDY COMPLETE! Total Wall-Clock Time: {wall_clock:.1f}s ({wall_clock/60:.1f} min)")
    print(f"Saved results to: {out_file}")
    print("=" * 80)

    # Leaderboard summary
    print("\nFinal Performance Leaderboard:")
    print(f"{'Run Configuration':<26} | {'Final 50-Ep Return':<18} | {'Cum. Upright Steps':<18} | {'Peak cos(th)':<12} | {'Runtime':<10}")
    print("-" * 94)
    sorted_runs = sorted(results.values(), key=lambda x: x["final_cumulative_upright"], reverse=True)
    for r in sorted_runs:
        print(f"{r['short_name']:<26} | {r['final_50ep_mean_return']:>18.2f} | {r['final_cumulative_upright']:>18d} | {r['max_cos_theta_reached']:>12.3f} | {r['elapsed']:>9.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
