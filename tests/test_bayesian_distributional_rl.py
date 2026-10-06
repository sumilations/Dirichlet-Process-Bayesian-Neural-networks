"""
Unit Tests for Bayesian Distributional Reinforcement Learning with Dirichlet Processes
(Algorithm 5 & Section 4.5 of Vashishtha PhD Thesis)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bayesian_distributional_rl.dp_posterior import (
    BaseMeasure,
    sample_dp_prior,
    sample_dp_posterior,
    evaluate_functional
)
from src.bayesian_distributional_rl.agent import BayesianDistributionalRL_DP
from src.bayesian_distributional_rl.environments import RiverSwimEnv, CliffWalkingEnv


def test_dp_prior_sampling():
    base = BaseMeasure("gaussian", loc=2.0, scale=0.5)
    weights, atoms = sample_dp_prior(alpha=5.0, base_measure=base, k_trunc=50)
    
    assert len(weights) == 50
    assert len(atoms) == 50
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5)
    assert np.all(weights >= 0.0)
    
    # Check empirical mean across multiple draws concentrates near prior mean
    means = []
    for _ in range(200):
        w, a = sample_dp_prior(alpha=5.0, base_measure=base, k_trunc=50)
        means.append(np.dot(w, a))
    assert np.isclose(np.mean(means), 2.0, atol=0.2)


def test_dp_posterior_stickbreaking():
    base = BaseMeasure("gaussian", loc=0.0, scale=1.0)
    history = [10.0] * 50  # Strongly positive observations
    
    weights, atoms = sample_dp_posterior(
        history=history,
        alpha_0=1.0,
        base_measure=base,
        k_trunc_prior=30
    )
    
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5)
    assert np.all(weights >= 0.0)
    
    # Posterior mean should shift dramatically toward 10.0
    post_mean = np.dot(weights, atoms)
    assert post_mean > 8.0, f"Expected posterior mean > 8.0, got {post_mean}"


def test_functionals():
    weights = np.array([0.2, 0.5, 0.3])
    atoms = np.array([1.0, 2.0, 5.0])
    
    # Mean: 0.2*1 + 0.5*2 + 0.3*5 = 0.2 + 1.0 + 1.5 = 2.7
    mean_val = evaluate_functional(weights, atoms, functional="mean")
    assert np.isclose(mean_val, 2.7)
    
    # Quantile at 0.5 (median): cumulative weights are 0.2, 0.7, 1.0 -> atom 2.0
    q50 = evaluate_functional(weights, atoms, functional="quantile", tau=0.5)
    assert np.isclose(q50, 2.0)
    
    # Variance
    var_val = evaluate_functional(weights, atoms, functional="variance")
    expected_var = 0.2*(1-2.7)**2 + 0.5*(2-2.7)**2 + 0.3*(5-2.7)**2
    assert np.isclose(var_val, expected_var)


def test_bayesian_distributional_rl_agent_riverswim():
    env = RiverSwimEnv(n_states=5, max_steps=20, seed=42)
    agent = BayesianDistributionalRL_DP(
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha_0=5.0,
        base_measure=BaseMeasure("optimistic", loc=1.0, scale=0.2),
        phi_1="mean",
        phi_2="mean",
        synthesis_mode="direct_union",
        seed=42
    )
    
    for episode in range(10):
        s = env.reset()
        done = False
        while not done:
            a = agent.select_action(s)
            s_next, r, done, _ = env.step(a)
            agent.update_step(s, a, r, s_next if not done else None, done)
            s = s_next
        agent.episodes += 1
        agent.record_diagnostics()
        
    assert agent.total_steps > 0
    assert len(agent.contraction_history) == 10
    
    # Check concentration parameter update: alpha(s, a) = alpha_0 + len(H)
    for s in range(env.num_states):
        for a in range(env.num_actions):
            expected_alpha = 5.0 + len(agent.H[(s, a)])
            assert np.isclose(agent.get_concentration(s, a), expected_alpha)


def test_cliffwalking():
    env = CliffWalkingEnv(height=3, width=6, max_steps=30, seed=42)
    agent = BayesianDistributionalRL_DP(
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha_0=5.0,
        phi_1="mean",
        phi_2="mean",
        synthesis_mode="distributional_bellman",
        seed=42
    )
    
    s = env.reset()
    for _ in range(25):
        a = agent.select_action(s)
        s_next, r, done, _ = env.step(a)
        agent.update_step(s, a, r, s_next if not done else None, done)
        if done:
            break
        s = s_next
        
    assert agent.total_steps > 0
