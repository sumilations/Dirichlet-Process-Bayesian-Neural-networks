"""Cart-Pole Swing-Up Benchmark Runner from arXiv:1703.07608 Section 7.2.2.

Evaluates:
1. DP-DQN (Vashishtha Data-Space Dirichlet Process Prior)
2. BootDQN-RP (Ensemble RLSVI with K=20 heads and prior networks, Osband & Van Roy JMLR)
3. Standard DQN (Linear epsilon-greedy annealing, Mnih et al. / Figure 16)
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from typing import Dict, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.envs.cartpole_jmlr import CartpoleSwingupJMLR
from src.rl.dp_dqn import DPDQNAgent
from src.rl.baselines_rl import BootDQNRPAgent, StandardDQNAgent


def run_agent_trial(
    agent_name: str,
    num_episodes: int,
    seed: int,
    base_measure_type: str = "uniform_max_entropy"
) -> Dict:
    """Run a single random seed trial for the specified agent on CartpoleSwingupJMLR."""
    env = CartpoleSwingupJMLR(seed=seed)

    if agent_name == "DP-DQN":
        agent = DPDQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=50,
            alpha=50.0,
            candidate_batch_size=256,
            truncation_K=128,
            base_measure_type=base_measure_type,
            prior_reward_mean=1.0,
            gamma=0.99,
            tau=0.01,
            lr=1e-3,
            sgd_period=2,
            capacity=100000,
            seed=seed
        )
    elif agent_name == "BootDQN-RP":
        agent = BootDQNRPAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            num_models=20,
            hidden_dim=50,
            prior_scale=3.0,
            gamma=0.99,
            tau=0.01,
            lr=1e-3,
            sgd_period=2,
            batch_size=128,
            capacity=100000,
            seed=seed
        )
    elif agent_name == "Standard-DQN":
        agent = StandardDQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=50,
            epsilon_start=1.0,
            epsilon_end=0.0,
            anneal_episodes=int(num_episodes * 0.8),
            gamma=0.99,
            tau=0.01,
            lr=1e-3,
            sgd_period=2,
            batch_size=128,
            capacity=100000,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")

    returns = []
    upright_steps = []
    peak_cos_thetas = []

    start_time = time.time()

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
            print(f"  [{agent_name} S{seed}] Ep {ep+1:4d}/{num_episodes} | 50-Ep Return: {recent_ret:6.2f} | Upright Steps: {recent_upr:4.1f} | Peak cos: {peak_cos:5.2f} | Time: {time.time() - start_time:4.0f}s", flush=True)

    elapsed = time.time() - start_time
    return {
        "returns": returns,
        "upright_steps": upright_steps,
        "peak_cos_thetas": peak_cos_thetas,
        "elapsed": elapsed
    }


def worker_task(task_args: Tuple[str, int, int, str]) -> Tuple[str, int, Dict]:
    agent_name, num_episodes, seed, base_measure_type = task_args
    torch.set_num_threads(1)
    trial_result = run_agent_trial(agent_name, num_episodes, seed, base_measure_type=base_measure_type)
    return agent_name, seed, trial_result


def main():
    parser = argparse.ArgumentParser(description="Run Cartpole Swing-Up Benchmark (arXiv:1703.07608 Section 7.2.2)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes per seed (default: 1000)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds (default: 3)")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run seeds in parallel across CPU cores")
    parser.add_argument("--base_measure", type=str, default="uniform_max_entropy", help="Base measure for DP-DQN (default: uniform_max_entropy)")
    parser.add_argument("--out_dir", type=str, default="results_rl", help="Output directory (default: results_rl)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    agent_names = ["DP-DQN", "BootDQN-RP", "Standard-DQN"]

    num_workers = min(os.cpu_count() or 4, args.seeds * len(agent_names))
    print("=" * 75)
    print("Cart-Pole Swing-Up Showdown (arXiv:1703.07608 Section 7.2.2)")
    print(f"Episodes: {args.episodes} | Seeds: {args.seeds} | Parallel CPU Workers: {num_workers}")
    print(f"DP-DQN Base Measure: {args.base_measure} (Jaynes Maximum Entropy)")
    print("=" * 75)

    tasks = []
    for name in agent_names:
        for s_idx in range(args.seeds):
            seed = 100 + s_idx * 42
            tasks.append((name, args.episodes, seed, args.base_measure))

    agent_seed_results = {name: [] for name in agent_names}
    start_all = time.time()

    if args.parallel and num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                name, seed, data = fut.result()
                agent_seed_results[name].append((seed, data))
                final_ret = np.mean(data["returns"][-50:])
                cum_ret = np.sum(data["returns"])
                cum_upr = np.sum(data["upright_steps"])
                peak_cos = np.max(data["peak_cos_thetas"])
                print(f"  [{name} Seed {seed}] Final 50-Ep Return: {final_ret:6.2f} | "
                      f"Cum Return: {cum_ret:7.1f} | Cum Upright Steps: {cum_upr:4d} | Max cos(th): {peak_cos:5.2f} | Time: {data['elapsed']:5.1f}s")
    else:
        for t in tasks:
            name, seed, data = worker_task(t)
            agent_seed_results[name].append((seed, data))
            final_ret = np.mean(data["returns"][-50:])
            cum_ret = np.sum(data["returns"])
            cum_upr = np.sum(data["upright_steps"])
            peak_cos = np.max(data["peak_cos_thetas"])
            print(f"  [{name} Seed {seed}] Final 50-Ep Return: {final_ret:6.2f} | "
                  f"Cum Return: {cum_ret:7.1f} | Cum Upright Steps: {cum_upr:4d} | Max cos(th): {peak_cos:5.2f} | Time: {data['elapsed']:5.1f}s")

    # Aggregate statistics
    results = {}
    print("\n" + "-" * 75)
    print("Summary of Final Benchmark Results:")
    print("-" * 75)

    for name in agent_names:
        runs = [d for _, d in sorted(agent_seed_results[name], key=lambda x: x[0])]
        ret_matrix = np.array([r["returns"] for r in runs])
        upr_matrix = np.array([r["upright_steps"] for r in runs])
        peak_matrix = np.array([r["peak_cos_thetas"] for r in runs])
        cum_ret_matrix = np.cumsum(ret_matrix, axis=1)
        cum_upr_matrix = np.cumsum(upr_matrix, axis=1)
        total_runtime = sum(r["elapsed"] for r in runs)

        results[name] = {
            "mean_returns": np.mean(ret_matrix, axis=0).tolist(),
            "stderr_returns": (np.std(ret_matrix, axis=0) / np.sqrt(args.seeds)).tolist(),
            "mean_cumulative_returns": np.mean(cum_ret_matrix, axis=0).tolist(),
            "stderr_cumulative_returns": (np.std(cum_ret_matrix, axis=0) / np.sqrt(args.seeds)).tolist(),
            "mean_upright_steps": np.mean(upr_matrix, axis=0).tolist(),
            "stderr_upright_steps": (np.std(upr_matrix, axis=0) / np.sqrt(args.seeds)).tolist(),
            "mean_cumulative_upright_steps": np.mean(cum_upr_matrix, axis=0).tolist(),
            "stderr_cumulative_upright_steps": (np.std(cum_upr_matrix, axis=0) / np.sqrt(args.seeds)).tolist(),
            "mean_peak_cos": np.mean(peak_matrix, axis=0).tolist(),
            "stderr_peak_cos": (np.std(peak_matrix, axis=0) / np.sqrt(args.seeds)).tolist(),
            "final_50ep_mean_return": float(np.mean(ret_matrix[:, -50:])),
            "final_50ep_stderr_return": float(np.std(np.mean(ret_matrix[:, -50:], axis=1)) / np.sqrt(args.seeds)),
            "final_cumulative_return": float(np.mean(cum_ret_matrix[:, -1])),
            "final_cumulative_upright": float(np.mean(cum_upr_matrix[:, -1])),
            "max_cos_theta_reached": float(np.max(peak_matrix)),
            "total_runtime_sec": float(total_runtime),
            "wall_clock_time_sec": float(time.time() - start_all)
        }

        print(f"--> {name:15s} | Final 50-Ep Return: {results[name]['final_50ep_mean_return']:6.2f} +/- {results[name]['final_50ep_stderr_return']:4.2f} | "
              f"Cum Return: {results[name]['final_cumulative_return']:7.1f} | "
              f"Cum Upright: {results[name]['final_cumulative_upright']:5.0f} | "
              f"Max cos(th): {results[name]['max_cos_theta_reached']:5.2f} | Runtime: {results[name]['total_runtime_sec']:5.1f}s")

    out_file = os.path.join(args.out_dir, "cartpole_jmlr_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 75)
    print(f"Saved benchmark results to {out_file}")
    print(f"Total Wall-Clock Time: {time.time() - start_all:.1f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
