"""
Unit Tests for DP-BNN REINFORCE.
Verifies policy networks, baseline estimators, stick-breaking rollout weighting,
policy-space Thompson sampling, and episode updates.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dp_reinforce.policies import CategoricalPolicyNet, GaussianPolicyNet, ValueBaselineNet
from src.dp_reinforce.dp_reinforce_agent import DP_BNN_REINFORCE


def test_categorical_policy():
    state_dim = 4
    action_dim = 2
    net = CategoricalPolicyNet(state_dim, action_dim, hidden_dim=32, use_layer_norm=True)
    
    s = torch.randn(state_dim)
    action, log_prob = net.get_action_and_log_prob(s)
    assert action in [0, 1]
    assert log_prob.numel() == 1
    
    batch_s = torch.randn(10, state_dim)
    batch_a = torch.randint(0, action_dim, (10,))
    log_probs, entropy = net.evaluate_actions(batch_s, batch_a)
    assert log_probs.shape == (10,)
    assert entropy.shape == (10,)
    assert torch.all(entropy >= 0.0)


def test_gaussian_policy():
    state_dim = 3
    action_dim = 1
    net = GaussianPolicyNet(state_dim, action_dim, hidden_dim=32, use_layer_norm=True)
    
    s = torch.randn(state_dim)
    action, log_prob = net.get_action_and_log_prob(s)
    assert action.numel() == 1
    assert log_prob.numel() == 1
    
    batch_s = torch.randn(10, state_dim)
    batch_a = torch.randn(10, action_dim)
    log_probs, entropy = net.evaluate_actions(batch_s, batch_a)
    assert log_probs.shape == (10,)
    assert entropy.shape == (10,)


def test_value_baseline():
    state_dim = 4
    net = ValueBaselineNet(state_dim, hidden_dim=32, use_layer_norm=True)
    s = torch.randn(state_dim)
    v = net(s)
    assert v.numel() == 1
    
    batch_s = torch.randn(15, state_dim)
    batch_v = net(batch_s)
    assert batch_v.shape == (15,)


def test_dp_bnn_reinforce_discrete():
    agent = DP_BNN_REINFORCE(
        state_dim=4,
        action_dim=2,
        is_continuous=False,
        hidden_dim=32,
        use_layer_norm=True,
        use_baseline=True,
        use_dp_weights=True,
        use_policy_ts=True,
        alpha_dp=3.0,
        k_trunc_prior=10,
        f0_return_mean=1.0,
        seed=42
    )

    # Run 5 synthetic episodes
    for ep in range(5):
        agent.start_episode()
        states = []
        actions = []
        rewards = []
        s = np.random.randn(4).astype(np.float32)
        for t in range(10):
            a, _ = agent.select_action(s)
            r = 1.0 if t == 9 else 0.0
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s + np.random.randn(4).astype(np.float32) * 0.1

        metrics = agent.update_with_episode(states, actions, rewards)
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "grad_norm" in metrics

    assert agent.total_episodes == 5
    assert len(agent.buffer_states) == 50
    assert agent.total_steps == 50


def test_dp_posterior_batch_weights():
    agent = DP_BNN_REINFORCE(
        state_dim=4,
        action_dim=2,
        is_continuous=False,
        alpha_dp=5.0,
        k_trunc_prior=15,
        seed=42
    )
    
    # Fill buffer
    for _ in range(50):
        agent.buffer_states.append(np.random.randn(4).astype(np.float32))
        agent.buffer_actions.append(int(np.random.choice(2)))
        agent.buffer_returns.append(float(np.random.randn()))
        
    w, s, a, g = agent._sample_dp_posterior_batch(batch_size=30)
    assert len(w) == 30 + 15  # 30 empirical + 15 prior atoms
    assert np.isclose(torch.sum(w).item(), 1.0, atol=1e-5)
    assert torch.all(w >= 0.0)
    assert s.shape[0] == 45
    assert a.shape[0] == 45
    assert g.shape[0] == 45


if __name__ == "__main__":
    test_categorical_policy()
    print("test_categorical_policy passed")
    test_gaussian_policy()
    print("test_gaussian_policy passed")
    test_value_baseline()
    print("test_value_baseline passed")
    test_dp_bnn_reinforce_discrete()
    print("test_dp_bnn_reinforce_discrete passed")
    test_dp_posterior_batch_weights()
    print("test_dp_posterior_batch_weights passed")
    print("ALL DP-BNN REINFORCE UNIT TESTS PASSED!")
