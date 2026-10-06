#!/usr/bin/env python3
"""Test Standard Normal N(0, 1) with:
1. Prior sampled ONCE per episode (preserving function variance against SGD washing)
2. Fresh empirical batches at EACH warmstart step
3. Bayesian Confidence Target Update: tau = 0.01 + 0.20 * (n_stat / (alpha + n_stat))
4. TD Information Gain decay for n_stat
Tested on DeepSea-20 across seeds 42, 43, 44, 45, 46 for DAG and Non-DAG.
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

class DAGStandardNormalBaseMeasure(BaseMeasure):
    def __init__(self, size: int):
        self.size = size
        state_dim = size * size
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=0.0, prior_reward_std=1.0)

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        r = rng.normal(0.0, 1.0, size=count).astype(np.float32)

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


class NonDAGStandardNormalBaseMeasure(BaseMeasure):
    def __init__(self, size: int):
        self.size = size
        state_dim = size * size
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=0.0, prior_reward_std=1.0)

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        r = rng.normal(0.0, 1.0, size=count).astype(np.float32)

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


class EpisodicNormalBayesianTargetAgent(DPDQNAgent):
    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        n_stat = self.cumulative_info if getattr(self.config, "use_td_info_gain_decay", False) else 0.0

        # Bayesian Confidence Target Update:
        credibility = n_stat / (self.config.alpha + n_stat)
        tau = float(np.clip(0.01 + 0.20 * credibility, 0.01, 0.25))

        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

        # -------------------------------------------------------------
        # DRAW PRIOR SAMPLES ONCE FOR THIS EPISODE (PRESERVES VARIANCE)
        # -------------------------------------------------------------
        K_prior = max(8, int(getattr(self.config, "vm_prior_multiplier", 10.0) * self.config.alpha))
        syn_s, syn_a, syn_r, syn_sn, syn_d = self.base_measure.sample(K_prior, self.rng)

        # Draw stick-breaking weights once for prior atoms
        V_prior = self.rng.beta(1.0, max(self.config.alpha, 1e-3), size=K_prior).astype(np.float32)
        U_prior = 1.0 - V_prior
        q_prior = np.zeros(K_prior, dtype=np.float32)
        q_prior[0] = V_prior[0]
        if K_prior > 1:
            prefix_U = np.cumprod(U_prior[:-1])
            q_prior[1:-1] = V_prior[1:-1] * prefix_U[:-1]
            q_prior[-1] = prefix_U[-1]
        q_prior = q_prior / q_prior.sum()

        syn_s_t = torch.from_numpy(syn_s).to(self.device)
        syn_a_t = torch.from_numpy(syn_a).to(self.device)
        syn_r_t = torch.from_numpy(syn_r).to(self.device)
        syn_sn_t = torch.from_numpy(syn_sn).to(self.device)
        syn_d_t = torch.from_numpy(syn_d).to(self.device)

        # -------------------------------------------------------------
        # WARMSTART WITH FRESH EMPIRICAL BATCHES AT EACH STEP
        # -------------------------------------------------------------
        n_emp_avail = len(self.replay)
        N_emp = min(n_emp_avail, self.config.batch_size)

        for step in range(self.config.warmstart_steps):
            emp_s, emp_a, emp_r, emp_sn, emp_d = self.replay.sample_priority(
                N_emp, self.rng, positive_slot=getattr(self.config, "priority_positive_slot", True)
            )

            # Compute empirical stick weights & W_prior dynamically
            N_stat = max(float(N_emp), float(n_stat))
            W_prior = float(self.rng.beta(self.config.alpha + 1.0, float(N_stat)))

            i_arr = np.arange(1, N_emp + 1, dtype=np.float32)
            V_emp = self.rng.beta(1.0, self.config.alpha + i_arr).astype(np.float32)
            U_emp = 1.0 - V_emp
            suffix_U = np.ones(N_emp + 1, dtype=np.float32)
            suffix_U[:-1] = np.cumprod(U_emp[::-1])[::-1]

            w_emp_raw = np.zeros(N_emp, dtype=np.float32)
            if N_emp > 1:
                w_emp_raw[:-1] = V_emp[:-1] * suffix_U[1:-1]
            w_emp_raw[-1] = V_emp[-1]

            emp_sum = w_emp_raw.sum()
            if emp_sum > 0:
                w_emp = (1.0 - W_prior) * (w_emp_raw / emp_sum)
            else:
                w_emp = np.full(N_emp, (1.0 - W_prior) / N_emp, dtype=np.float32)

            w_prior = W_prior * q_prior
            total_weights = np.concatenate([w_emp, w_prior])
            total_weights = total_weights / total_weights.sum()

            s = torch.cat([torch.from_numpy(emp_s).to(self.device), syn_s_t], dim=0)
            a = torch.cat([torch.from_numpy(emp_a).to(self.device), syn_a_t], dim=0)
            r = torch.cat([torch.from_numpy(emp_r).to(self.device), syn_r_t], dim=0)
            sn = torch.cat([torch.from_numpy(emp_sn).to(self.device), syn_sn_t], dim=0)
            done = torch.cat([torch.from_numpy(emp_d).to(self.device), syn_d_t], dim=0)
            weights = torch.from_numpy(total_weights).to(self.device)

            with torch.no_grad():
                q_next = self.target_net(sn).max(dim=1)[0]
                target_y = r + self.config.gamma * q_next * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * weights * s.shape[0]).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

            with torch.no_grad():
                if N_emp > 0:
                    emp_delta = (target_y[:N_emp] - pred_q[:N_emp]).detach()
                    surprise = torch.clamp(emp_delta.abs() / 1.0, max=1.0).sum().item()
                    self.cumulative_info += float(surprise)


def run_single(variant="dag", seed=42, size=20, max_episodes=800):
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
        prior_reward_mean=0.0,
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
    agent = EpisodicNormalBayesianTargetAgent(cfg)

    if variant == "dag":
        agent.base_measure = DAGStandardNormalBaseMeasure(size)
    else:
        agent.base_measure = NonDAGStandardNormalBaseMeasure(size)
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
    print(f"[{variant.upper():6s} StdNormal EpPrior] Seed {seed:2d} -> Disc: {str(first_discovery):5s} | Solved: {str(solved_ep):5s} | Regret: {cum_regret:6.1f} | Time: {dt:4.1f}s", flush=True)
    return first_discovery, solved_ep, cum_regret

if __name__ == "__main__":
    print("=" * 85)
    print("TESTING STANDARD NORMAL N(0, 1): PRIOR ONCE PER EPISODE + BAYESIAN TARGET UPDATE")
    print("=" * 85)
    for v in ["dag", "nondag"]:
        for s in [42, 43, 44, 45, 46]:
            run_single(v, seed=s)
