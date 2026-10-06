import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.rl.bsuite_wrapper import BSuiteCartpoleWrapper
from src.rl.dp_dqn import DPDQNAgent
from src.rl.baselines_rl import BootDQNRPAgent, StandardDQNAgent


def run_agent(agent_name, num_episodes, seed, bsuite_id="cartpole_swingup/0"):
    env = BSuiteCartpoleWrapper(bsuite_id)

    if agent_name == "DP-DQN":
        agent = DPDQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=64,
            alpha=25.0,
            candidate_batch_size=256,
            truncation_K=150,
            prior_reward_mean=1.5,
            gamma=0.99,
            tau=0.1,
            lr=0.002,
            warmstart_steps=15,
            seed=seed
        )
    elif agent_name == "BootDQN-RP":
        agent = BootDQNRPAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            num_models=10,
            hidden_dim=64,
            prior_scale=3.0,
            gamma=0.99,
            tau=0.1,
            lr=0.002,
            train_steps_per_episode=15,
            seed=seed
        )
    elif agent_name == "Standard-DQN":
        agent = StandardDQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=64,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.99,
            gamma=0.99,
            tau=0.1,
            lr=0.002,
            train_steps_per_episode=15,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    returns = []
    upright_steps_history = []
    max_heights = []
    start_time = time.time()

    for ep in range(num_episodes):
        agent.start_episode()
        state = env.reset()
        done = False
        ep_reward = 0.0
        upright_count = 0
        peak_cos = -1.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.step_update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if info.get("is_upright", False):
                upright_count += 1
            cos_th = float(state[3])
            if cos_th > peak_cos:
                peak_cos = cos_th

        agent.end_episode()
        returns.append(float(ep_reward))
        upright_steps_history.append(int(upright_count))
        max_heights.append(float(peak_cos))

    elapsed = time.time() - start_time
    return {
        "returns": returns,
        "upright_steps": upright_steps_history,
        "max_heights": max_heights,
        "elapsed": elapsed
    }


import concurrent.futures
import torch


def run_worker(task):
    agent_name, num_episodes, seed, bsuite_id = task
    torch.set_num_threads(1)
    res = run_agent(agent_name, num_episodes, seed, bsuite_id)
    return agent_name, seed, res


def main():
    parser = argparse.ArgumentParser(description="Run Cartpole Swing-Up Deep RL Benchmark")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes per seed")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run seeds in parallel across CPU cores")
    parser.add_argument("--out_dir", type=str, default="results_rl", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    agent_names = ["DP-DQN", "BootDQN-RP", "Standard-DQN"]
    results = {}

    num_workers = min(os.cpu_count() or 4, args.seeds * len(agent_names))
    print(f"Starting bsuite Cartpole Swing-Up Benchmark:")
    print(f"Episodes per run: {args.episodes} | Seeds: {args.seeds} | Parallel Workers: {num_workers}")
    print("=" * 70)

    tasks = []
    for name in agent_names:
        for s_idx in range(args.seeds):
            seed = 100 + s_idx * 42
            tasks.append((name, args.episodes, seed, "cartpole_swingup/0"))

    agent_seed_results = {name: [] for name in agent_names}
    start_total = time.time()

    if args.parallel and num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_worker, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                name, seed, run_data = fut.result()
                agent_seed_results[name].append((seed, run_data))
                avg_ret = np.mean(run_data["returns"][-50:])
                avg_upr = np.mean(run_data["upright_steps"][-50:])
                cum_ret = np.sum(run_data["returns"])
                cum_upr = np.sum(run_data["upright_steps"])
                print(f"  [{name} Seed {seed}] Final 50-Ep Return: {avg_ret:6.2f} | "
                      f"Upright Steps: {avg_upr:4.1f} | Cum Return: {cum_ret:7.1f} | Cum Upright: {cum_upr:5.0f} | Time: {run_data['elapsed']:5.1f}s")
    else:
        for t in tasks:
            name, seed, run_data = run_worker(t)
            agent_seed_results[name].append((seed, run_data))
            avg_ret = np.mean(run_data["returns"][-50:])
            avg_upr = np.mean(run_data["upright_steps"][-50:])
            cum_ret = np.sum(run_data["returns"])
            cum_upr = np.sum(run_data["upright_steps"])
            print(f"  [{name} Seed {seed}] Final 50-Ep Return: {avg_ret:6.2f} | "
                  f"Upright Steps: {avg_upr:4.1f} | Cum Return: {cum_ret:7.1f} | Cum Upright: {cum_upr:5.0f} | Time: {run_data['elapsed']:5.1f}s")

    for name in agent_names:
        runs = [data for _, data in sorted(agent_seed_results[name], key=lambda x: x[0])]
        ret_matrix = np.array([r["returns"] for r in runs])
        upr_matrix = np.array([r["upright_steps"] for r in runs])
        cum_ret_matrix = np.cumsum(ret_matrix, axis=1)
        cum_upr_matrix = np.cumsum(upr_matrix, axis=1)
        total_time = sum(r["elapsed"] for r in runs)

        results[name] = {
            "mean_returns": np.mean(ret_matrix, axis=0).tolist(),
            "stderr_returns": (np.std(ret_matrix, axis=0) / np.sqrt(args.seeds)).tolist(),
            "mean_cumulative_returns": np.mean(cum_ret_matrix, axis=0).tolist(),
            "stderr_cumulative_returns": (np.std(cum_ret_matrix, axis=0) / np.sqrt(args.seeds)).tolist(),
            "mean_upright_steps": np.mean(upr_matrix, axis=0).tolist(),
            "stderr_upright_steps": (np.std(upr_matrix, axis=0) / np.sqrt(args.seeds)).tolist(),
            "mean_cumulative_upright_steps": np.mean(cum_upr_matrix, axis=0).tolist(),
            "stderr_cumulative_upright_steps": (np.std(cum_upr_matrix, axis=0) / np.sqrt(args.seeds)).tolist(),
            "final_50ep_mean_return": float(np.mean(ret_matrix[:, -50:])),
            "final_50ep_stderr_return": float(np.std(np.mean(ret_matrix[:, -50:], axis=1)) / np.sqrt(args.seeds)),
            "final_50ep_mean_upright": float(np.mean(upr_matrix[:, -50:])),
            "final_50ep_stderr_upright": float(np.std(np.mean(upr_matrix[:, -50:], axis=1)) / np.sqrt(args.seeds)),
            "final_cumulative_return": float(np.mean(cum_ret_matrix[:, -1])),
            "final_cumulative_upright": float(np.mean(cum_upr_matrix[:, -1])),
            "total_time_sec": float(total_time),
            "wall_clock_time_sec": float(time.time() - start_total)
        }

        print(f"\n--> {name} Final 50-Ep Return: {results[name]['final_50ep_mean_return']:.2f} +/- {results[name]['final_50ep_stderr_return']:.2f} | "
              f"Upright Steps: {results[name]['final_50ep_mean_upright']:.1f} | "
              f"Total Cum Return: {results[name]['final_cumulative_return']:.1f} | "
              f"Total Cum Upright: {results[name]['final_cumulative_upright']:.0f}")

    out_file = os.path.join(args.out_dir, "cartpole_swingup_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Saved benchmark results to {out_file}")
    print(f"Total Wall-Clock Time: {time.time() - start_total:.1f}s")


if __name__ == "__main__":
    main()
