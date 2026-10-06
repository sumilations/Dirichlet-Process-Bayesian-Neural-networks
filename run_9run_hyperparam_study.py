"""9-Run Systematic Hyperparameter & Ablation Study for DP-DQN on Cart-Pole Swing-Up.

Executes 9 configurations in parallel across 9 CPU workers on identical seed:
1. Baseline-LayerNorm: Reference 50/50 base measure, alpha=50, tau=0.01, with LayerNorm
2. No-LayerNorm: Same as baseline, but without LayerNorm (ablation)
3. Pure-Upright-F0: 100% Upright anchors, 0% synthetic swing
4. Stochastic-Sparks: High-variance mixture with optimistic jackpot reward sparks
5. Energy-Coupled-F0: Directional swing atoms coupled to angular velocity sign(theta_dot)
6. Low-Alpha-15: Low concentration alpha=15.0 (fast transition to empirical data)
7. High-Alpha-150: High concentration alpha=150.0 (persistent heavy prior anchor)
8. Target-WarmStart: Target network adapts 10 fast steps on fresh DP draw per episode
9. Fast-Target-Tau05: 5x faster target tracking tau=0.05
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
        "name": "1. Baseline-LayerNorm",
        "short_name": "Baseline-LayerNorm",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episode_target_warmstart": False,
        }
    },
    {
        "name": "2. No-LayerNorm (Ablation)",
        "short_name": "No-LayerNorm",
        "params": {
            "use_layer_norm": False,
            "base_measure_type": "default",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episode_target_warmstart": False,
        }
    },
    {
        "name": "3. Pure-Upright-F0",
        "short_name": "Pure-Upright-F0",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "pure_upright",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episode_target_warmstart": False,
        }
    },
    {
        "name": "4. Stochastic-Sparks-F0",
        "short_name": "Stochastic-Sparks",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "stochastic_sparks",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episode_target_warmstart": False,
        }
    },
    {
        "name": "5. Energy-Coupled-F0",
        "short_name": "Energy-Coupled-F0",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "energy_coupled",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episode_target_warmstart": False,
        }
    },
    {
        "name": "6. Low-Alpha (alpha=15)",
        "short_name": "Low-Alpha-15",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 15.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episode_target_warmstart": False,
        }
    },
    {
        "name": "7. High-Alpha (alpha=150)",
        "short_name": "High-Alpha-150",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 150.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episode_target_warmstart": False,
        }
    },
    {
        "name": "8. Target-WarmStart (User Idea)",
        "short_name": "Target-WarmStart",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 50.0,
            "tau": 0.01,
            "sgd_period": 2,
            "episode_target_warmstart": True,
        }
    },
    {
        "name": "9. Fast-Target-Tau05 (tau=0.05)",
        "short_name": "Fast-Target-Tau05",
        "params": {
            "use_layer_norm": True,
            "base_measure_type": "default",
            "alpha": 50.0,
            "tau": 0.05,
            "sgd_period": 2,
            "episode_target_warmstart": False,
        }
    },
]


def run_single_config(config: Dict[str, Any], num_episodes: int, seed: int) -> Dict[str, Any]:
    """Execute 2500 episodes for a single DP-DQN configuration."""
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
            print(f"  [{tag:18s}] Ep {ep+1:4d}/{num_episodes} | Ret(50): {recent_ret:6.2f} | Upr(50): {recent_upr:4.1f} | CumUpr: {cum_upr:5d} | PeakCos: {peak_cos:5.2f} | {elapsed:4.0f}s", flush=True)

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
    parser = argparse.ArgumentParser(description="9-Run Systematic Hyperparameter & Ablation Study for DP-DQN")
    parser.add_argument("--episodes", type=int, default=2500, help="Episodes per run (default: 2500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for all runs (default: 42)")
    parser.add_argument("--workers", type=int, default=9, help="Parallel CPU workers (default: 9)")
    parser.add_argument("--out_dir", type=str, default="results_rl", help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Run 2 episodes only to test pipeline")
    args = parser.parse_args()

    num_episodes = 2 if args.dry_run else args.episodes
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print(f"9-Run Systematic DP-DQN Hyperparameter & Ablation Study")
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
    out_file = os.path.join(args.out_dir, "study_9runs_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"STUDY COMPLETE! Total Wall-Clock Time: {wall_clock:.1f}s ({wall_clock/60:.1f} min)")
    print(f"Saved results to: {out_file}")
    print("=" * 80)

    # Leaderboard summary
    print("\nFinal Performance Leaderboard:")
    print(f"{'Run Configuration':<24} | {'Final 50-Ep Return':<18} | {'Cum. Upright Steps':<18} | {'Peak cos(th)':<12} | {'Runtime':<10}")
    print("-" * 92)
    sorted_runs = sorted(results.values(), key=lambda x: x["final_cumulative_upright"], reverse=True)
    for r in sorted_runs:
        print(f"{r['short_name']:<24} | {r['final_50ep_mean_return']:>18.2f} | {r['final_cumulative_upright']:>18d} | {r['max_cos_theta_reached']:>12.3f} | {r['elapsed']:>9.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
