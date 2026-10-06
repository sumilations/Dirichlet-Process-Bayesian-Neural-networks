"""Head-to-head 3000-episode comparison: DP-DQN With vs Without Episodic Pre-sampling on Deep Sea N=20."""

import json
import os
import sys
import time
from typing import Any, Dict
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from bsuite.environments.deep_sea import DeepSea
from src.rl.batched_dp_dqn import BatchedDPDQNDeepSeaAgent

torch.set_num_threads(1)

def run_3000_episodes(mode: bool, name: str, seed: int = 42, num_episodes: int = 3000) -> Dict[str, Any]:
    print(f"\n" + "=" * 70)
    print(f"RUNNING: {name} (Deep Sea N=20, Seed={seed}, Episodes={num_episodes})")
    print("=" * 70, flush=True)

    env = DeepSea(size=20, deterministic=True, randomize_actions=True, seed=seed, mapping_seed=seed)
    agent = BatchedDPDQNDeepSeaAgent(
        size=20,
        hidden_dim=20,
        alpha=5.0,
        prior_reward_mean=1.0,
        candidate_batch_size=128,
        batch_size=32,
        gamma=0.99,
        tau=0.05,
        lr=1e-3,
        sgd_period=2,
        capacity=30000,
        use_layer_norm=True,
        presample_and_batch=mode,
        seed=seed,
    )

    t0 = time.time()
    returns = []
    regrets = []
    opt_return = env._optimal_return
    first_discovery = None
    solved_ep = None

    for ep in range(num_episodes):
        agent.start_episode()
        ts = env.reset()
        obs = ts.observation
        ep_ret = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_ts = env.step(action)
            reward = float(next_ts.reward or 0.0)
            done = next_ts.last()
            agent.step_update(obs, action, reward, next_ts.observation, done)
            obs = next_ts.observation
            ep_ret += reward

        agent.end_episode()
        returns.append(ep_ret)
        regret = float(opt_return - ep_ret)
        regrets.append(regret)

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep + 1
            print(f"  >>> [{name}] FIRST TREASURE DISCOVERED at Episode {first_discovery} | Time: {time.time()-t0:.2f}s", flush=True)

        if len(returns) >= 20 and np.mean(returns[-20:]) > 0.5 and solved_ep is None:
            solved_ep = ep + 1 - 10
            print(f"  >>> [{name}] SOLVED / CONVERGED at Episode {solved_ep} | Time: {time.time()-t0:.2f}s", flush=True)

        if (ep + 1) % 500 == 0:
            rec_ret = np.mean(returns[-50:])
            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / elapsed
            print(f"  [{name:28s}] Ep {ep+1:4d}/{num_episodes} | Return: {rec_ret:.4f} | Time: {elapsed:5.1f}s | Speed: {eps_per_sec:5.1f} eps/s", flush=True)

    elapsed_total = time.time() - t0
    episodes_per_sec = num_episodes / elapsed_total

    # Subsample regret every 10 episodes for smooth plotting
    subsampled_regrets = [float(np.mean(regrets[i:i+10])) for i in range(0, len(regrets), 10)]

    print(f"\nFinished {name}: Total Time: {elapsed_total:.2f}s | Avg Speed: {episodes_per_sec:.1f} eps/s")
    print(f"  First Discovery: {first_discovery} | Solved: {solved_ep} | Cumulative Regret: {sum(regrets):.1f}")

    return {
        "agent": name,
        "mode": "With Pre-sampling" if mode else "Without Pre-sampling",
        "seed": seed,
        "num_episodes": num_episodes,
        "solved": solved_ep is not None,
        "learn_time": solved_ep,
        "first_discovery": first_discovery,
        "elapsed_seconds": elapsed_total,
        "episodes_per_sec": episodes_per_sec,
        "cumulative_regret": float(np.sum(regrets)),
        "final_50ep_return": float(np.mean(returns[-50:])),
        "returns": returns,
        "regrets": regrets,
        "subsampled_regrets": subsampled_regrets,
    }


def main():
    os.makedirs("results_deepsea", exist_ok=True)
    out_file = "results_deepsea/comparison_3000ep_n20.json"

    # 1. Run Without Pre-sampling (Sequential)
    res_without = run_3000_episodes(
        mode=False,
        name="DP-DQN (Without Pre-sampling)",
        seed=42,
        num_episodes=3000
    )

    # 2. Run With Pre-sampling (Batched)
    res_with = run_3000_episodes(
        mode=True,
        name="DP-DQN (With Pre-sampling)",
        seed=42,
        num_episodes=3000
    )

    results = [res_without, res_with]
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_file}")

if __name__ == "__main__":
    main()
