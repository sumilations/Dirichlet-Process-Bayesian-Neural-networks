"""Test convergence speed of DP-DQN vs BootDQN-RP on Deep Sea."""

import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from bsuite.environments.deep_sea import DeepSea
from src.rl.deep_sea_dp_dqn import DPDQNDeepSeaAgent
from src.rl.deep_sea_boot_dqn import BootDQNRPDeepSeaAgent


def evaluate_agent(agent_type, size, seed=42, max_episodes=5000):
    env = DeepSea(size=size, deterministic=True, randomize_actions=True, seed=seed)
    if agent_type == "DP-DQN":
        agent = DPDQNDeepSeaAgent(size=size, hidden_dim=20, alpha=5.0, seed=seed)
    elif agent_type == "BootDQN-RP":
        agent = BootDQNRPDeepSeaAgent(size=size, num_heads=20, hidden_dim=20, prior_scale=10.0, seed=seed)

    returns = []
    t0 = time.time()
    solved_ep = None

    for ep in range(max_episodes):
        agent.start_episode()
        ts = env.reset()
        obs = ts.observation
        ep_ret = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_ts = env.step(action)
            reward = next_ts.reward
            done = next_ts.last()
            next_obs = next_ts.observation

            agent.step_update(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_ret += reward

        agent.end_episode()
        returns.append(ep_ret)

        if len(returns) >= 20 and np.mean(returns[-20:]) > 0.5:
            solved_ep = ep + 1 - 10  # episode where consistent optimal policy started
            elapsed = time.time() - t0
            print(f"[{agent_type:10s}] N={size:2d} | Seed {seed} | SOLVED at Ep {solved_ep:4d} | Time: {elapsed:.2f}s | Recent Return: {np.mean(returns[-20:]):.4f}")
            return solved_ep, elapsed

        if (ep + 1) % 500 == 0:
            print(f"  [{agent_type:10s}] N={size:2d} | Ep {ep+1:4d}/{max_episodes} | Recent Ret(20): {np.mean(returns[-20:]):.4f}")

    elapsed = time.time() - t0
    print(f"[{agent_type:10s}] N={size:2d} | TIMEOUT at Ep {max_episodes} | Time: {elapsed:.2f}s")
    return max_episodes, elapsed


if __name__ == "__main__":
    for N in [6, 8, 10]:
        print(f"\n--- Testing N={N} ---")
        evaluate_agent("DP-DQN", N, seed=42)
        evaluate_agent("BootDQN-RP", N, seed=42)
