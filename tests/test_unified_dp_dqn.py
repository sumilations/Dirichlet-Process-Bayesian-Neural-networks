"""Unit tests for the unified environment-agnostic DP-DQN framework."""

import os
import sys
import unittest
import numpy as np
import torch

# Put repo root on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unified_dp_dqn import (
    DPDQNAgent,
    DPDQNConfig,
    make_env,
    BootDQNRPAgent,
    BayesianDeepQNetworkAgent,
    StandardDQNAgent,
)


class TestUnifiedDPDQN(unittest.TestCase):

    def test_cartpole_dp_dqn_haar(self):
        """Test DP-DQN on Cart-Pole with Haar manifold base measure."""
        env = make_env("cartpole_swingup", seed=42)
        cfg = DPDQNConfig(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            alpha=3.0,
            base_measure_type="haar",
            haar_angle=True,
            buffer_capacity=10000,
            batch_size=16,
            sgd_period=2,
            seed=42
        )
        agent = DPDQNAgent(cfg)

        for ep in range(2):
            agent.reset_episode()
            s = env.reset()
            done = False
            steps = 0
            while not done and steps < 20:
                a = agent.act(s)
                self.assertIn(a, [0, 1, 2])
                ns, r, done, info = env.step(a)
                agent.step(s, a, r, ns, done)
                s = ns
                steps += 1

        self.assertGreater(agent.total_steps, 0)
        self.assertGreater(len(agent.replay), 0)

    def test_deep_sea_dp_dqn(self):
        """Test the EXACT SAME DP-DQN agent on Deep Sea without any modification."""
        env = make_env("deep_sea", size=8, seed=42)
        cfg = DPDQNConfig(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            alpha=3.0,
            base_measure_type="uniform",
            haar_angle=False,
            buffer_capacity=10000,
            batch_size=16,
            sgd_period=2,
            seed=42
        )
        agent = DPDQNAgent(cfg)

        for ep in range(3):
            agent.reset_episode()
            s = env.reset()
            done = False
            while not done:
                a = agent.act(s)
                self.assertIn(a, [0, 1])
                ns, r, done, info = env.step(a)
                agent.step(s, a, r, ns, done)
                s = ns

        self.assertGreater(agent.total_steps, 0)

    def test_baselines_on_cartpole(self):
        """Verify BootDQN, BDQN, and StandardDQN share the identical interface."""
        env = make_env("cartpole_swingup", seed=42)
        agents = [
            ("BootDQN", BootDQNRPAgent(env.state_dim, env.action_dim, num_models=3, batch_size=8, seed=42)),
            ("BDQN", BayesianDeepQNetworkAgent(env.state_dim, env.action_dim, batch_size=8, seed=42)),
            ("StandardDQN", StandardDQNAgent(env.state_dim, env.action_dim, batch_size=8, seed=42)),
        ]

        for name, agent in agents:
            for ep in range(2):
                agent.reset_episode()
                s = env.reset()
                done = False
                steps = 0
                while not done and steps < 15:
                    a = agent.act(s)
                    self.assertIn(a, [0, 1, 2])
                    ns, r, done, info = env.step(a)
                    agent.step(s, a, r, ns, done)
                    s = ns
                    steps += 1
            self.assertGreater(len(agent.replay), 0, f"{name} replay is empty!")


if __name__ == "__main__":
    unittest.main()
