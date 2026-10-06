import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import BaseMeasure
from src.dp_dqn.agent import DPDQNAgent

class SupportRespectingUniform(BaseMeasure):
    def __init__(self, size: int):
        self.size = size
        state_dim = size * size
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=0.0, prior_reward_std=0.577)

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        # Uniform on [-1, 1] - strictly respects support!
        r = rng.uniform(-1.0, 1.0, size=count).astype(np.float32)

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

class SnapshotAgent(DPDQNAgent):
    def __init__(self, config, period_k=5, **kwargs):
        super().__init__(config, **kwargs)
        self.period_k = period_k

    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        if self.episodes_completed % self.period_k == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        for _ in range(self.config.warmstart_steps):
            s, a, r, sn, done, q_weights = self.sampler.sample(self.replay, device=self.device)
            with torch.no_grad():
                q_next = self.target_net(sn).max(dim=1)[0]
                target_y = r + self.config.gamma * q_next * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * s.shape[0]).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

def test_u(size=10, seed=42):
    print("=" * 70)
    print(f"TESTING UNIFORM[-1, 1] ON DEEPSEA N={size}, Seed={seed}")
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
        prior_reward_mean=0.0,
        prior_reward_std=0.577,
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
        num_episodes=500,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        episodic_sgd=False,
    )
    env = make_env("deep_sea", seed=seed, size=size)
    agent = SnapshotAgent(cfg, period_k=5)
    agent.base_measure = SupportRespectingUniform(size=size)
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
        test_u(size=10, seed=s)
