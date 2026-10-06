#!/usr/bin/env python3
"""Test Normal base measure with mean = 1/N on DeepSea-20:
Comparing:
1. Normal(mean = 1/N = 0.05, std = 1.0) on DAG
2. Normal(mean = 1/N = 0.05, std = 1.0) on Non-DAG
with Bayesian Confidence Target Updates on Seeds 42, 43, 44, 45, 46.
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import BaseMeasure
from src.dp_dqn.agent import DPDQNAgent

class DAGNormalOneOverNBaseMeasure(BaseMeasure):
    def __init__(self, size: int):
        self.size = size
        state_dim = size * size
        mean = 1.0 / size
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=mean, prior_reward_std=1.0)

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        r = rng.normal(self.prior_reward_mean, self.prior_reward_std, size=count).astype(np.float32)

        for i in range(count):
            row = rng.randint(0, self.size)
            col = rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                next_col = min(self.size - 1, col + 1) if a[i] == 1 else max(0, col - 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


class NonDAGNormalOneOverNBaseMeasure(BaseMeasure):
    def __init__(self, size: int):
        self.size = size
        state_dim = size * size
        mean = 1.0 / size
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=mean, prior_reward_std=1.0)

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        r = rng.normal(self.prior_reward_mean, self.prior_reward_std, size=count).astype(np.float32)

        for i in range(count):
            row = rng.randint(0, self.size)
            col = rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                next_col = rng.randint(0, next_row + 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


class BayesianConfidenceTargetAgent(DPDQNAgent):
    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        n_stat = self.cumulative_info if getattr(self.config, "use_td_info_gain_decay", False) else 0.0

        credibility = n_stat / (self.config.alpha + n_stat)
        tau = float(np.clip(0.01 + 0.20 * credibility, 0.01, 0.25))

        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

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


def run_test(variant="dag", seed=42, size=20, max_episodes=800):
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
        prior_reward_mean=1.0 / size,
        prior_reward_std=1.0,
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
    agent = BayesianConfidenceTargetAgent(cfg)

    if variant == "dag":
        agent.base_measure = DAGNormalOneOverNBaseMeasure(size)
    else:
        agent.base_measure = NonDAGNormalOneOverNBaseMeasure(size)
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
    print(f"[{variant.upper():6s} Normal(1/N)] Seed {seed:2d} -> Disc: {str(first_discovery):5s} | Solved: {str(solved_ep):5s} | Regret: {cum_regret:6.1f} | Time: {dt:4.1f}s")
    return first_discovery, solved_ep, cum_regret

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING NORMAL WITH MEAN = 1/N = 0.05 ON DEEPSEA-20 (Seeds 42, 45, 46)")
    print("=" * 80)
    for v in ["dag", "nondag"]:
        for s in [42, 45, 46]:
            run_test(v, seed=s)
