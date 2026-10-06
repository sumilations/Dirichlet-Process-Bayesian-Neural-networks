import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.envs.cartpole_jmlr import CartpoleSwingupJMLR
from src.rl.dp_dqn import DPDQNAgent
from src.rl.baselines_rl import BootDQNRPAgent, StandardDQNAgent


def test_rl_pipeline():
    env = CartpoleSwingupJMLR(seed=42)
    print(f"JMLR Cart-Pole Env initialized: State dim = {env.state_dim}, Action dim = {env.action_dim}")

    agents = [
        ("DP-DQN", DPDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, truncation_K=32, sgd_period=2, seed=42)),
        ("BootDQN-RP (Ensemble RLSVI)", BootDQNRPAgent(state_dim=env.state_dim, action_dim=env.action_dim, num_models=3, batch_size=32, sgd_period=2, seed=42)),
        ("Standard-DQN", StandardDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, batch_size=32, sgd_period=2, seed=42))
    ]

    for name, agent in agents:
        print(f"Testing 1 episode with {name}...")
        agent.start_episode()
        state = env.reset()
        done = False
        step_count = 0
        total_rew = 0.0

        while not done and step_count < 100:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.step_update(state, action, reward, next_state, done)
            state = next_state
            total_rew += reward
            step_count += 1

        agent.end_episode()
        print(f"  {name} completed episode: {step_count} steps, reward = {total_rew:.2f}, peak cos(theta) = {info['cos_theta']:.2f}")

    print("\nAll RL component tests passed successfully!")


if __name__ == "__main__":
    test_rl_pipeline()
