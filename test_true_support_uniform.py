import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import BaseMeasure
from src.dp_dqn.agent import DPDQNAgent

class TrueSupportUniform(BaseMeasure):
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
        # Uniform on the EXACT physical support [r_min, r_max]!
        r = rng.uniform(self.r_min, self.r_max, size=count).astype(np.float32)

        for i in range(count):
            row = rng.randint(0, self.size)
            col = rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0
            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                step_dir = 1 if rng.rand() < 0.5 else -1
                next_col = max(0, min(self.size - 1, col + step_dir))
                sn[i, next_row * self.size + next_col] = 1.0
        return s, a, r, sn, done

def test_true_support(size=10, seed=42):
    print("=" * 70)
    print(f"TESTING TRUE-SUPPORT UNIFORM [r_min, R_max] ON DEEPSEA N={size}, Seed={seed}")
    print(f"r_min = {-0.01/size:.5f}, r_max = 1.0, Mean = {(-0.01/size + 1.0)/2:.4f}")
    print("=" * 70)
    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=size*size,
        action_dim=2,
        seed=seed,
        alpha=5.0,
        batch_size=64,
        candidate_batch_size=128,
        base_measure="deep_sea_dag_maxent",
        prior_reward_mean=float((-0.01/size + 1.0)/2),
        prior_reward_std=float((1.0 - (-0.01/size))/np.sqrt(12)),
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.05,
        target_warmstart=True,
        warmstart_steps=size // 2,
        sample_once_per_episode=False,
        num_episodes=500,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        episodic_sgd=False,
    )
    env = make_env("deep_sea", seed=seed, size=size)
    agent = DPDQNAgent(cfg)
    agent.base_measure = TrueSupportUniform(size=size)
    agent.sampler.base_measure = agent.base_measure

    t0 = time.time()
    for ep in range(1, 501):
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

        if ep_ret > 0.5:
            print(f">>> [DISCOVERY!] Reached goal at Episode {ep}! ({time.time()-t0:.2f}s)")
            return ep

    print("Did not reach goal in 500 episodes.")
    return None

if __name__ == "__main__":
    for s in [42, 43, 44]:
        test_true_support(size=10, seed=s)
