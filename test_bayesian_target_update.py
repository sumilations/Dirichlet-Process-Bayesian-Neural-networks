#!/usr/bin/env python3
"""Test Bayesian Dynamic Target Network Updates on DeepSea-20:
Comparing:
1. Baseline: Periodic Hard Snapshot K=5
2. Bayesian Surprise Polyak: tau = Delta_n_stat / (alpha + n_stat)
3. Bayesian Robbins-Monro Polyak: tau = (1 + Delta_n_stat) / (alpha + n_stat)
4. Bayesian Convex Polyak: tau = min(0.5, n_stat / (alpha + n_stat))
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from test_option_c_dag_vs_nondag import TrueSupportDAGBaseMeasure, TrueSupportNonDAGBaseMeasure
from src.dp_dqn.agent import DPDQNAgent

class BayesianTargetAgent(DPDQNAgent):
    def __init__(self, config, tau_mode="bayesian_surprise", **kwargs):
        super().__init__(config, **kwargs)
        self.tau_mode = tau_mode
        self.last_cumulative_info = 0.0

    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        delta_info = max(0.0, self.cumulative_info - self.last_cumulative_info)
        self.last_cumulative_info = self.cumulative_info
        n_stat = self.cumulative_info

        # --- Dynamic Bayesian Target Update at Episode Boundary ---
        if self.tau_mode == "hard_k5":
            if self.episodes_completed % 5 == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        elif self.tau_mode == "bayesian_surprise":
            # Tau proportional to new surprise / total concentration
            tau = float(np.clip(delta_info / (self.config.alpha + n_stat), 0.001, 1.0))
            with torch.no_grad():
                for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                    tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        elif self.tau_mode == "bayesian_robbins_monro":
            # Step size = (1 + Delta_info) / (alpha + n_stat)
            tau = float(np.clip((1.0 + delta_info) / (self.config.alpha + n_stat), 0.005, 0.5))
            with torch.no_grad():
                for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                    tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        elif self.tau_mode == "bayesian_confidence":
            # Confidence in empirical network grows as n_stat / (alpha + n_stat)
            # Apply as soft tracking with tau scaled by empirical credibility
            credibility = n_stat / (self.config.alpha + n_stat)
            tau = float(np.clip(0.01 + 0.2 * credibility, 0.01, 0.25))
            with torch.no_grad():
                for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                    tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

        # Warmstart: Fresh sampling per step (Option C)
        for _ in range(self.config.warmstart_steps):
            s, a, r, sn, done, q_weights = self.sampler.sample(
                self.replay, device=self.device, n_stat_override=n_stat
            )
            with torch.no_grad():
                q_next = self.target_net(sn).max(dim=1)[0]
                target_y = r + self.config.gamma * q_next * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * s.shape[0]).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

            with torch.no_grad():
                n_emp = getattr(self.sampler, "last_n_emp", s.shape[0])
                if n_emp > 0:
                    emp_delta = (target_y[:n_emp] - pred_q[:n_emp]).detach()
                    surprise = torch.clamp(emp_delta.abs() / 1.0, max=1.0).sum().item()
                    self.cumulative_info += float(surprise)


def run_single(tau_mode, seed=42, size=20, max_episodes=600):
    r_min = -0.01 / size
    mean = (r_min + 1.0) / 2.0
    std = (1.0 - r_min) / np.sqrt(12)

    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=size * size,
        action_dim=2,
        seed=seed,
        alpha=5.0,
        batch_size=64,
        candidate_batch_size=128,
        base_measure="deep_sea",
        prior_reward_mean=float(mean),
        prior_reward_std=float(std),
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.0,
        target_warmstart=False,
        warmstart_steps=size // 2,
        sample_once_per_episode=False,
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        episodic_sgd=False,
        use_td_info_gain_decay=True,
        priority_positive_slot=True,
    )

    env = make_env("deep_sea", seed=seed, size=size)
    agent = BayesianTargetAgent(cfg, tau_mode=tau_mode)
    agent.base_measure = TrueSupportDAGBaseMeasure(size)
    agent.sampler.base_measure = agent.base_measure

    returns = []
    first_discovery = None
    solved_ep = None
    cum_regret = 0.0

    t0 = time.time()
    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        s = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(s)
            sn, r, done, _ = env.step(a)
            agent.replay.push(s, a, float(r), sn, done)
            s = sn
            ep_ret += float(r)

        returns.append(ep_ret)
        cum_regret += max(0.0, 0.99 - ep_ret)

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep

        if ep >= 20 and solved_ep is None:
            if np.mean(returns[-20:]) >= 0.8:
                solved_ep = ep
                break

    dt = time.time() - t0
    print(f"[{tau_mode:25s}] Seed {seed:2d} -> Disc: {str(first_discovery):5s} | Solved: {str(solved_ep):5s} | Regret: {cum_regret:6.1f} | Time: {dt:4.1f}s")
    return first_discovery, solved_ep, cum_regret

if __name__ == "__main__":
    modes = ["hard_k5", "bayesian_surprise", "bayesian_robbins_monro", "bayesian_confidence"]
    print("=" * 85)
    print("TESTING BAYESIAN DYNAMIC TARGET UPDATES ON DEEPSEA-20 (Seeds 42, 45)")
    print("=" * 85)
    for m in modes:
        for s in [42, 45]:
            run_single(m, seed=s)
