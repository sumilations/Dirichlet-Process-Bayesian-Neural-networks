#!/usr/bin/env python3
"""Run 10 seeds (42-51) on DeepSea-20 with:
1. Standard Normal Base Measure N(0, 1) (mean=0.0, std=1.0)
2. Bayesian Confidence Target Update: tau = 0.01 + 0.20 * (n_stat / (alpha + n_stat))
3. Both DAG and Non-DAG transition structures
"""

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


class BayesianConfidenceTargetAgent(DPDQNAgent):
    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        n_stat = self.cumulative_info if getattr(self.config, "use_td_info_gain_decay", False) else 0.0

        # Bayesian Confidence Target Update:
        # tau scales smoothly from 0.01 (frozen during early search) to 0.21 (fast tracking in exploitation)
        credibility = n_stat / (self.config.alpha + n_stat)
        tau = float(np.clip(0.01 + 0.20 * credibility, 0.01, 0.25))

        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

        # Fresh empirical + prior sampling at each step (Option C)
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


def run_experiment(variant="dag", size=20, seeds=range(42, 52), max_episodes=800):
    results = []
    print("=" * 80)
    print(f"RUNNING 10 SEEDS: Variant={variant.upper()}, Standard Normal Base Measure N(0, 1)")
    print(f"Algorithm: Bayesian Confidence Target Update tau(n_stat) on DeepSea-{size}")
    print(f"Seeds: {list(seeds)}, Max Episodes: {max_episodes}")
    print("=" * 80)

    for seed in seeds:
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
            prior_reward_mean=0.0,          # Standard Normal Mean = 0.0
            prior_reward_std=1.0,           # Standard Normal Std = 1.0
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
            agent.base_measure = DAGStandardNormalBaseMeasure(size)
        else:
            agent.base_measure = NonDAGStandardNormalBaseMeasure(size)
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
                print(f"[{variant.upper()} Seed {seed:2d}] >>> DISCOVERY at Episode {ep}! (t={time.time()-t0:.1f}s)", flush=True)

            if ep >= 20 and solved_ep is None:
                recent_ret = np.mean(returns[-20:])
                if recent_ret >= 0.8:
                    solved_ep = ep
                    print(f"[{variant.upper()} Seed {seed:2d}] *** SOLVED at Episode {ep}! (Return={recent_ret:.3f}, t={time.time()-t0:.1f}s) ***", flush=True)
                    break

            if ep % 200 == 0:
                print(f"[{variant.upper()} Seed {seed:2d}] Ep {ep:3d} | Recent Ret: {np.mean(returns[-20:]):.3f} | Regret: {cum_regret:.1f}", flush=True)

        dt = time.time() - t0
        res = {
            "variant": variant,
            "seed": seed,
            "first_discovery": first_discovery,
            "solved_episode": solved_ep,
            "final_cum_regret": cum_regret,
            "total_time_sec": dt,
        }
        results.append(res)
        print(f"--> Finished {variant.upper()} Seed {seed:2d}: Disc={first_discovery}, Solved={solved_ep}, Regret={cum_regret:.1f} ({dt:.1f}s)\n", flush=True)

    return results

if __name__ == "__main__":
    seeds = list(range(42, 52))
    dag_results = run_experiment("dag", size=20, seeds=seeds)
    nondag_results = run_experiment("nondag", size=20, seeds=seeds)

    summary = {
        "base_measure": "Standard Normal N(0, 1)",
        "target_update": "Bayesian Confidence tau(n_stat) = 0.01 + 0.20 * credibility",
        "seeds": seeds,
        "dag": dag_results,
        "nondag": nondag_results,
    }
    with open("results_10seeds_standard_normal.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print("Saved to results_10seeds_standard_normal.json")
