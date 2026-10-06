"""Unit tests for Deep Sea RL Agents: DP-DQN, BootDQN-RP, and DQN Dithering."""

import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bsuite.environments.deep_sea import DeepSea

from src.rl.deep_sea_dp_dqn import DPDQNDeepSeaAgent
from src.rl.deep_sea_boot_dqn import BootDQNRPDeepSeaAgent
from src.rl.deep_sea_dithering import DQNDitheringAgent


def test_deep_sea_agents():
    N = 6
    env = DeepSea(size=N, deterministic=True, randomize_actions=True, seed=42)

    agents = [
        ("DP-DQN", DPDQNDeepSeaAgent(size=N, hidden_dim=20, alpha=5.0, truncation_K=16, seed=42)),
        ("BootDQN-RP", BootDQNRPDeepSeaAgent(size=N, num_heads=5, hidden_dim=20, batch_size=16, seed=42)),
        ("DQN-Dithering", DQNDitheringAgent(size=N, hidden_dim=20, batch_size=16, seed=42)),
    ]

    for name, agent in agents:
        print(f"Testing {name} on DeepSea N={N}...")
        for ep in range(3):
            agent.start_episode()
            ts = env.reset()
            obs = ts.observation
            done = False
            step_count = 0

            while not done:
                action = agent.select_action(obs)
                next_ts = env.step(action)
                reward = next_ts.reward
                done = next_ts.last()
                next_obs = next_ts.observation

                agent.step_update(obs, action, reward, next_obs, done)
                obs = next_obs
                step_count += 1

            agent.end_episode()
            assert step_count == N, f"Expected {N} steps in episode, got {step_count}"

        print(f"--> {name} passed successfully!")


if __name__ == "__main__":
    test_deep_sea_agents()
