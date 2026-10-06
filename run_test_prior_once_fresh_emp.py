#!/usr/bin/env python3
"""Benchmark DP-DQN on DeepSea-20 with:
- Prior samples drawn ONCE before episode (one coherent prior realization Q_0 per episode)
- Fresh empirical mini-batches sampled at EACH warmstart gradient step (covering replay buffer)
- True-Support Uniform Base Measure: U(-0.01/N, 1.0)
- Hard Snapshot Target Network K=5
- TD Information Gain decay
Tested across 5 seeds (42-46) on both DAG and Non-DAG priors.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import BaseMeasure
from src.dp_dqn.agent import DPDQNAgent

class TrueSupportDAGBaseMeasure(BaseMeasure):
    def __init__(self, size: int):
        self.size = size
        state_dim = size * size
        self.r_min = -0.01 / size
        self.r_max = 1.0
        mean = (self.r_min + self.r_max) / 2.0
        std = (self.r_max - self.r_min) / np.sqrt(12)
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=mean, prior_reward_std=std)

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        r = rng.uniform(self.r_min, self.r_max, size=count).astype(np.float32)

        for i in range(count):
            row = rng.randint(0, self.size)
            col = rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                act = a[i]
                next_col = min(self.size - 1, col + 1) if act == 1 else max(0, col - 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


class TrueSupportNonDAGBaseMeasure(BaseMeasure):
    def __init__(self, size: int):
        self.size = size
        state_dim = size * size
        self.r_min = -0.01 / size
        self.r_max = 1.0
        mean = (self.r_min + self.r_max) / 2.0
        std = (self.r_max - self.r_min) / np.sqrt(12)
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=mean, prior_reward_std=std)

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        r = rng.uniform(self.r_min, self.r_max, size=count).astype(np.float32)

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


class EpisodicPriorFreshEmpiricalAgent(DPDQNAgent):
    """Samples prior transitions ONCE per episode; samples empirical batches FRESH each step."""
    def __init__(self, config, period_k=5, **kwargs):
        super().__init__(config, **kwargs)
        self.period_k = period_k
        self.rng = np.random.RandomState(config.seed)

    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        # Hard snapshot every K episodes
        if self.episodes_completed % self.period_k == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # -------------------------------------------------------------
        # 1. DRAW PRIOR SAMPLE ONCE FOR THIS ENTIRE EPISODE
        # -------------------------------------------------------------
        K_prior = max(8, int(getattr(self.config, "vm_prior_multiplier", 10.0) * self.config.alpha))
        syn_s, syn_a, syn_r, syn_sn, syn_d = self.base_measure.sample(K_prior, self.rng)

        # Draw prior stick-breaking weights once
        V_prior = self.rng.beta(1.0, max(self.config.alpha, 1e-3), size=K_prior).astype(np.float32)
        U_prior = 1.0 - V_prior
        q_prior = np.zeros(K_prior, dtype=np.float32)
        q_prior[0] = V_prior[0]
        if K_prior > 1:
            prefix_U = np.cumprod(U_prior[:-1])
            q_prior[1:-1] = V_prior[1:-1] * prefix_U[:-1]
            q_prior[-1] = prefix_U[-1]
        q_prior_sum = q_prior.sum()
        if q_prior_sum > 0:
            q_prior = q_prior / q_prior_sum

        # Pre-convert prior tensors
        syn_s_t = torch.from_numpy(syn_s).to(self.device)
        syn_a_t = torch.from_numpy(syn_a).to(self.device)
        syn_r_t = torch.from_numpy(syn_r).to(self.device)
        syn_sn_t = torch.from_numpy(syn_sn).to(self.device)
        syn_d_t = torch.from_numpy(syn_d).to(self.device)

        # -------------------------------------------------------------
        # 2. RUN WARMSTART STEPS WITH FRESH EMPIRICAL MINI-BATCHES
        # -------------------------------------------------------------
        n_stat = self.cumulative_info if getattr(self.config, "use_td_info_gain_decay", False) else None
        n_emp_avail = len(self.replay)
        N_emp = min(n_emp_avail, self.config.batch_size)

        for step in range(self.config.warmstart_steps):
            # FRESH empirical sample from replay buffer
            emp_s, emp_a, emp_r, emp_sn, emp_d = self.replay.sample_priority(
                N_emp, self.rng, positive_slot=getattr(self.config, "priority_positive_slot", True)
            )

            # Compute empirical stick-breaking weights & W_prior
            if n_stat is not None:
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
            else:
                i_arr = np.arange(1, N_emp + 1, dtype=np.float32)
                V_emp = self.rng.beta(1.0, self.config.alpha + i_arr).astype(np.float32)
                U_emp = 1.0 - V_emp
                suffix_U = np.ones(N_emp + 1, dtype=np.float32)
                suffix_U[:-1] = np.cumprod(U_emp[::-1])[::-1]
                w_emp = np.zeros(N_emp, dtype=np.float32)
                if N_emp > 1:
                    w_emp[:-1] = V_emp[:-1] * suffix_U[1:-1]
                w_emp[-1] = V_emp[-1]
                W_prior = float(suffix_U[0])

            w_prior = W_prior * q_prior
            total_weights = np.concatenate([w_emp, w_prior])
            t_sum = total_weights.sum()
            if t_sum > 0:
                total_weights = total_weights / t_sum

            # Convert to torch
            emp_s_t = torch.from_numpy(emp_s).to(self.device)
            emp_a_t = torch.from_numpy(emp_a).to(self.device)
            emp_r_t = torch.from_numpy(emp_r).to(self.device)
            emp_sn_t = torch.from_numpy(emp_sn).to(self.device)
            emp_d_t = torch.from_numpy(emp_d).to(self.device)

            s = torch.cat([emp_s_t, syn_s_t], dim=0)
            a = torch.cat([emp_a_t, syn_a_t], dim=0)
            r = torch.cat([emp_r_t, syn_r_t], dim=0)
            sn = torch.cat([emp_sn_t, syn_sn_t], dim=0)
            done = torch.cat([emp_d_t, syn_d_t], dim=0)
            q_weights = torch.from_numpy(total_weights).to(self.device)

            with torch.no_grad():
                q_next = self.target_net(sn).max(dim=1)[0]
                target_y = r + self.config.gamma * q_next * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * s.shape[0]).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate TD Information Gain on empirical data
            with torch.no_grad():
                if N_emp > 0:
                    emp_delta = (target_y[:N_emp] - pred_q[:N_emp]).detach()
                    surprise = torch.clamp(emp_delta.abs() / 1.0, max=1.0).sum().item()
                    self.cumulative_info += float(surprise)


def run_experiment(variant="dag", size=20, seeds=[42, 43, 44, 45, 46], max_episodes=800):
    results = []
    print("=" * 80)
    print(f"RUNNING EPISODIC PRIOR + FRESH EMPIRICAL: Variant={variant.upper()}, DeepSea-{size}")
    print(f"Seeds: {seeds}, Max Episodes: {max_episodes}")
    print("=" * 80)

    state_dim = size * size
    r_min = -0.01 / size
    mean = (r_min + 1.0) / 2.0
    std = (1.0 - r_min) / np.sqrt(12)

    for seed in seeds:
        cfg = DPDQNConfig(
            env_name="deep_sea",
            deep_sea_size=size,
            state_dim=state_dim,
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
        agent = EpisodicPriorFreshEmpiricalAgent(cfg, period_k=5)

        if variant == "dag":
            agent.base_measure = TrueSupportDAGBaseMeasure(size)
        else:
            agent.base_measure = TrueSupportNonDAGBaseMeasure(size)
        agent.sampler.base_measure = agent.base_measure

        returns = []
        cum_regrets = []
        cum_regret = 0.0
        t0 = time.time()
        first_discovery = None
        solved_ep = None

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
            regret = max(0.0, 0.99 - ep_ret)
            cum_regret += regret
            cum_regrets.append(cum_regret)

            if ep_ret > 0.5 and first_discovery is None:
                first_discovery = ep
                print(f"[{variant.upper()} Seed {seed}] DISCOVERY at Ep {ep}! Time: {time.time()-t0:.1f}s", flush=True)

            if ep >= 20 and solved_ep is None:
                recent_ret = np.mean(returns[-20:])
                if recent_ret >= 0.8:
                    solved_ep = ep
                    print(f"[{variant.upper()} Seed {seed}] SOLVED at Ep {ep}! (Recent ret: {recent_ret:.3f}) Time: {time.time()-t0:.1f}s", flush=True)
                    break

            if ep % 100 == 0:
                print(f"[{variant.upper()} Seed {seed}] Ep {ep:3d} | Ret: {np.mean(returns[-20:]):.3f} | Regret: {cum_regret:.1f}", flush=True)

        res = {
            "variant": variant,
            "seed": seed,
            "first_discovery": first_discovery,
            "solved_episode": solved_ep,
            "final_cum_regret": cum_regret,
            "total_time_sec": time.time() - t0,
        }
        results.append(res)
        print(f"--> Seed {seed} Finished: Disc={first_discovery}, Solved={solved_ep}, Regret={cum_regret:.1f}\n")

    return results

if __name__ == "__main__":
    dag_results = run_experiment("dag", size=20, seeds=[42, 43, 44, 45, 46])
    nondag_results = run_experiment("nondag", size=20, seeds=[42, 43, 44, 45, 46])

    summary = {
        "config_description": "Prior sampled ONCE before episode, Fresh empirical samples at EACH warmstart step",
        "dag": dag_results,
        "nondag": nondag_results,
    }
    with open("results_prior_once_fresh_emp.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print("Saved to results_prior_once_fresh_emp.json")
