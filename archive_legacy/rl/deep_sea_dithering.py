"""Standard DQN with Epsilon-Greedy Dithering for Deep Sea Exploration.

Illustrates the failure of dithering exploration:
- Action selection: epsilon-greedy with linear decay from epsilon_start to epsilon_end.
- Q-network: 20-unit MLP.
- Theoretical lower bound: takes Omega(2^N) episodes to reach the chest.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNDitheringAgent:
    """Standard DQN with epsilon-greedy dithering."""

    def __init__(
        self,
        size: int,
        hidden_dim: int = 20,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        decay_episodes: int = 2000,
        gamma: float = 0.99,
        tau: float = 0.05,
        lr: float = 1e-3,
        sgd_period: int = 2,
        batch_size: int = 64,
        capacity: int = 50000,
        seed: Optional[int] = None
    ):
        self.size = size
        self.in_dim = size * size
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_episodes = decay_episodes
        self.gamma = gamma
        self.tau = tau
        self.sgd_period = sgd_period
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.q_net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.target_net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.capacity = capacity
        self.replay_s = np.zeros((capacity, self.in_dim), dtype=np.float32)
        self.replay_a = np.zeros(capacity, dtype=np.int64)
        self.replay_r = np.zeros(capacity, dtype=np.float32)
        self.replay_sn = np.zeros((capacity, self.in_dim), dtype=np.float32)
        self.replay_d = np.zeros(capacity, dtype=np.float32)
        self.replay_size = 0
        self.replay_idx = 0
        self.total_steps = 0
        self.episode_count = 0

    def start_episode(self):
        """Update epsilon decay."""
        self.episode_count += 1
        frac = min(1.0, self.episode_count / self.decay_episodes)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, obs: np.ndarray) -> int:
        if self.rng.rand() < self.epsilon:
            return int(self.rng.randint(0, 2))

        self.q_net.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(obs.flatten()).float().unsqueeze(0)
            q_vals = self.q_net(s_t).squeeze(0).numpy()
            return int(np.argmax(q_vals))

    def step_update(self, s: np.ndarray, a: int, r: float, sn: np.ndarray, done: bool):
        self.replay_s[self.replay_idx] = s.flatten()
        self.replay_a[self.replay_idx] = a
        self.replay_r[self.replay_idx] = r
        self.replay_sn[self.replay_idx] = sn.flatten()
        self.replay_d[self.replay_idx] = float(done)

        self.replay_idx = (self.replay_idx + 1) % self.capacity
        if self.replay_size < self.capacity:
            self.replay_size += 1
        self.total_steps += 1

        if self.replay_size >= self.batch_size and self.total_steps % self.sgd_period == 0:
            self._train_step()

    def _train_step(self):
        idx = self.rng.choice(self.replay_size, size=self.batch_size, replace=True)

        sb = torch.from_numpy(self.replay_s[idx])
        ab = torch.from_numpy(self.replay_a[idx])
        rb = torch.from_numpy(self.replay_r[idx])
        snb = torch.from_numpy(self.replay_sn[idx])
        db = torch.from_numpy(self.replay_d[idx])

        with torch.no_grad():
            next_q = self.target_net(snb)
            max_next_q, _ = torch.max(next_q, dim=1)
            targets = rb + (1.0 - db) * self.gamma * max_next_q

        self.optimizer.zero_grad()
        self.q_net.train()
        q_vals = self.q_net(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
        loss = torch.mean((q_vals - targets) ** 2)
        loss.backward()
        self.optimizer.step()

        for tgt_p, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tgt_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * tgt_p.data)

    def end_episode(self):
        pass
