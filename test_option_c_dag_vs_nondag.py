import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import BaseMeasure
from test_td_info_deepsea20 import HardSnapshotTDInfoAgent

class TrueSupportDAGBaseMeasure(BaseMeasure):
    """Causal DAG Prior with True Physical Support Uniform Rewards [r_min, R_max]."""
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
                # Physical grid neighbor transition
                act = a[i]
                next_col = min(self.size - 1, col + 1) if act == 1 else max(0, col - 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


class TrueSupportNonDAGBaseMeasure(BaseMeasure):
    """Model-Free Uniform Base Measure (Zero DAG Constraint) with True Physical Support Rewards."""
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
                # Generic unconstrained next column (no DAG neighbor constraint)
                next_col = rng.randint(0, next_row + 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


def run_test(variant="dag", size=20, seed=42, max_episodes=800):
    print("=" * 80)
    print(f"RUNNING {variant.upper()} PRIOR (Option C: True Support Uniform + TD Info Gain)")
    print(f"DeepSea N={size}, Seed={seed}, Fresh Sampling W={size//2}, Hard Snapshot K=5")
    print("=" * 80)

    state_dim = size * size
    r_min = -0.01 / size
    mean = (r_min + 1.0) / 2.0
    std = (1.0 - r_min) / np.sqrt(12)

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
        sample_once_per_episode=False,  # Fresh per step!
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
    agent = HardSnapshotTDInfoAgent(cfg, period_k=5)

    if variant == "dag":
        agent.base_measure = TrueSupportDAGBaseMeasure(size)
    else:
        agent.base_measure = TrueSupportNonDAGBaseMeasure(size)
    agent.sampler.base_measure = agent.base_measure

    returns = []
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

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep
            print(f">>> [{variant.upper()} DISCOVERY!] Goal reached at Episode {ep}! Info: {agent.cumulative_info:.1f} ({time.time()-t0:.2f}s)", flush=True)

        if ep >= 20 and solved_ep is None:
            recent_ret = np.mean(returns[-20:])
            if recent_ret >= 0.8:
                solved_ep = ep
                print(f"*** [{variant.upper()} SOLVED!] Solved at Episode {ep}! (Return: {recent_ret:.3f}, Time: {time.time()-t0:.2f}s) ***", flush=True)
                break

        if ep % 50 == 0:
            print(f"Episode {ep:3d} | Return: {np.mean(returns[-20:]):.4f} | Info: {agent.cumulative_info:.1f} | Time: {time.time()-t0:.1f}s", flush=True)

    print(f"--> Result {variant.upper()}: Discovery: {first_discovery}, Solved: {solved_ep}, Cum Regret: {cum_regret:.1f}")
    return first_discovery, solved_ep, cum_regret

if __name__ == "__main__":
    run_test(variant="dag", seed=42)
    run_test(variant="nondag", seed=42)
