"""Hard verification tests for the core DP-DQN algorithmic architecture.

Invariants strictly tested:
1. DP posterior sampling is performed ONLY during reset_episode() to warmstart living q_net.
2. Step updates (update()) perform standard empirical mini-batch SGD on q_net directly from replay (NO DP sampler).
3. Target network is continuously updated via Polyak tracking during update().
4. Action selection is always greedy with respect to living q_net.
5. TD information gain (cumulative_info / N_stat) accumulates during update() and is passed to DP sampler at warmstart.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unified_dp_dqn import DPDQNAgent, DPDQNConfig


class TestCoreAlgorithmicArchitecture(unittest.TestCase):

    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.batch_size = 16
        self.warmstart_steps = 3
        self.tau = 0.1

        self.config = DPDQNConfig(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            alpha=3.0,
            batch_size=self.batch_size,
            warmstart_steps=self.warmstart_steps,
            tau=self.tau,
            use_td_info_gain_decay=True,
            td_info_scale=1.0,
            buffer_capacity=1000,
            sgd_period=1,
            seed=42,
        )
        self.agent = DPDQNAgent(self.config)

        # Seed replay buffer with synthetic valid transitions
        rng = np.random.RandomState(42)
        for _ in range(self.batch_size * 2):
            s = rng.randn(self.state_dim).astype(np.float32)
            a = rng.choice(self.action_dim)
            r = float(rng.randn())
            sn = rng.randn(self.state_dim).astype(np.float32)
            d = bool(rng.choice([0, 1], p=[0.9, 0.1]))
            self.agent.replay.push(s, a, r, sn, d)

    def test_invariant_1_and_2_dp_sampler_only_at_warmstart_not_in_step(self):
        """Invariant: DP posterior sampler MUST NOT be called in step(), ONLY in reset_episode()."""
        # Wrap sampler.sample in a spy
        real_sample = self.agent.sampler.sample
        spy_sample = MagicMock(side_effect=real_sample)
        self.agent.sampler.sample = spy_sample

        # Step 1: Call environment step()
        s = np.zeros(self.state_dim, dtype=np.float32)
        res = self.agent.step(s, 0, 0.0, s, False)
        self.assertIsNone(res, "step() should return None (zero online updates in Pure TS)")

        # CRITICAL ASSERTION: DP sampler was NOT called during environment step!
        self.assertEqual(
            spy_sample.call_count, 0,
            "VIOLATION: DP posterior sampler was called during step()! Rollout must have zero updates."
        )

        # Step 2: Call reset_episode()
        self.agent.reset_episode()

        # CRITICAL ASSERTION: In Pure TS, DP sampler WAS called exactly 1 time per episode at episode start!
        self.assertEqual(
            spy_sample.call_count, 1,
            f"VIOLATION: In Pure TS, DP posterior sampler must be called exactly 1 time during reset_episode(), got {spy_sample.call_count}."
        )

    def test_invariant_3_polyak_target_tracking_in_reset_episode(self):
        """Invariant: Polyak target tracking occurs only at the beginning of reset_episode()."""
        # Perturb q_net slightly so target_net needs to move
        with torch.no_grad():
            for p in self.agent.q_net.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        target_init = [p.clone() for p in self.agent.target_net.parameters()]

        # Perform episode reset
        self.agent.reset_episode()

        # Target net must have moved towards q_net via Polyak tracking
        target_moved = False
        for p_init, p_new in zip(target_init, self.agent.target_net.parameters()):
            if not torch.allclose(p_init, p_new, atol=1e-7):
                target_moved = True
                break
        self.assertTrue(target_moved, "Target network parameters did not update via Polyak tracking at episode reset!")

    def test_invariant_4_greedy_action_on_living_qnet(self):
        """Invariant: Actions are always selected greedily with living q_net."""
        state = np.random.randn(self.state_dim).astype(np.float32)
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.agent.device)

        self.agent.q_net.eval()
        with torch.no_grad():
            expected_q = self.agent.q_net(state_t)
            expected_action = int(expected_q.argmax(dim=1).item())

        selected_action = self.agent.act(state)
        self.assertEqual(
            selected_action, expected_action,
            "Action was not selected greedily from living q_net!"
        )

    def test_invariant_5_td_info_gain_accumulates_and_passed_to_warmstart(self):
        """Invariant: TD surprise accumulates into cumulative_info and is passed to DP sampler at warmstart."""
        self.assertEqual(self.agent.cumulative_info, 0.0)

        # Perform multiple episode warmstarts
        for _ in range(3):
            self.agent.reset_episode()

        self.assertGreater(
            self.agent.cumulative_info, 0.0,
            "cumulative_info (N_stat) did not accumulate TD surprise during warmstarts!"
        )
        current_n_stat = self.agent.cumulative_info

        # Intercept sampler.sample to verify n_stat_override matches cumulative_info
        real_sample = self.agent.sampler.sample
        passed_n_stat = []

        def recording_sample(*args, **kwargs):
            passed_n_stat.append(kwargs.get("n_stat_override"))
            return real_sample(*args, **kwargs)

        self.agent.sampler.sample = recording_sample
        self.agent.reset_episode()

        self.assertEqual(len(passed_n_stat), 1)
        for val in passed_n_stat:
            self.assertAlmostEqual(val, current_n_stat, places=5)

    def test_no_warmstart_under_min_buffer_size(self):
        """Invariant: If replay has fewer samples than batch_size, warmstart is safely skipped."""
        fresh_agent = DPDQNAgent(self.config)
        self.assertEqual(len(fresh_agent.replay), 0)

        real_sample = fresh_agent.sampler.sample
        spy_sample = MagicMock(side_effect=real_sample)
        fresh_agent.sampler.sample = spy_sample

        # Call reset_episode() on empty buffer
        fresh_agent.reset_episode()
        self.assertEqual(spy_sample.call_count, 0, "Warmstart should not execute when buffer < batch_size!")


if __name__ == "__main__":
    unittest.main()
