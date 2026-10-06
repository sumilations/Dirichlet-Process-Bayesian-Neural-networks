import unittest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.dp_bnn_ids import BatchedMLP, DPBNN_IDSAgent
from src.envs.information_mismatch_bandit import InformationMismatchBandit


class TestDPBNNIDS(unittest.TestCase):

    def test_batched_mlp_dimensions(self):
        M = 10
        in_dim = 4
        out_dim = 3
        mlp = BatchedMLP(in_dim=in_dim, out_dim=out_dim, hidden_dim=32, num_models=M)

        # 1D context
        x_1d = torch.randn(in_dim)
        out_1d = mlp(x_1d)
        self.assertEqual(out_1d.shape, (M, out_dim))

        # 2D batch context
        B = 16
        x_2d = torch.randn(B, in_dim)
        out_2d = mlp(x_2d)
        self.assertEqual(out_2d.shape, (M, B, out_dim))

    def test_dp_bnn_ids_agent_step(self):
        agent = DPBNN_IDSAgent(
            context_dim=2,
            num_arms=3,
            num_models=10,
            hidden_dim=32,
            alpha=5.0,
            truncation_K=20,
            seed=42
        )

        ctx = np.array([0.5, -0.5], dtype=np.float32)

        # Initial random rounds
        for a in range(3):
            action = agent.select_action(ctx)
            self.assertEqual(action, a)
            agent.update(ctx, action, 1.0)

        # Normal IDS action selection
        action = agent.select_action(ctx)
        self.assertIn(action, [0, 1, 2])
        self.assertIsNotNone(agent.last_regret_vec)
        self.assertIsNotNone(agent.last_info_vec)
        self.assertIsNotNone(agent.last_sampling_dist)

        # Probabilities sum to 1.0
        self.assertAlmostEqual(float(agent.last_sampling_dist.sum()), 1.0, places=4)
        # All info gains non-negative
        self.assertTrue(np.all(agent.last_info_vec >= 0.0))
        # Greedy arm has 0 regret
        self.assertAlmostEqual(float(np.min(agent.last_regret_vec)), 0.0, places=5)

    def test_information_mismatch_bandit(self):
        env = InformationMismatchBandit(delta=1.0, epsilon=0.05, seed=123)
        ctx = env.reset()
        self.assertEqual(ctx.shape, (2,))

        next_ctx, r, opt_r, info = env.step(0)
        self.assertEqual(next_ctx.shape, (2,))
        self.assertIn('theta', info)
        self.assertIn('optimal_arm', info)
        self.assertIn('regret', info)


if __name__ == '__main__':
    unittest.main()
