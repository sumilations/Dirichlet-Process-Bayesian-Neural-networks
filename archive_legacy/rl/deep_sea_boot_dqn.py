"""Bootstrapped DQN with Additive Randomized Prior Functions (BSP).

Implements Algorithm 8 / Section 4.2.1 of arXiv:1806.03335 (Osband, Aslanides, Cassirer, NeurIPS 2018):
- K=20 ensemble heads with 20 hidden units.
- Fixed random untrainable prior network p_k for each head, scaled by beta = 10.0:
    Q_k(s, a) = f_{theta_k}(s, a) + beta * p_k(s, a)
- Online double-or-nothing bootstrap masking (Bernoulli p=0.5).
- Episode-horizon Thompson sampling: sample k ~ Uniform(1..K) at episode start.
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DeepSeaMLP(nn.Module):
    """20-unit MLP for Deep Sea matching Osband et al. 2018."""

    def __init__(self, in_dim: int, hidden_dim: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BootDQNRPDeepSeaAgent:
    """BootDQN with Randomized Priors (BSP) for Deep Sea."""

    def __init__(
        self,
        size: int,
        num_heads: int = 20,
        hidden_dim: int = 20,
        prior_scale: float = 10.0,
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
        self.K = num_heads
        self.prior_scale = float(prior_scale)
        self.gamma = gamma
        self.tau = tau
        self.sgd_period = sgd_period
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        self.trainable_nets: List[nn.Module] = []
        self.prior_nets: List[nn.Module] = []
        self.target_nets: List[nn.Module] = []
        self.optimizers: List[optim.Optimizer] = []

        for k in range(self.K):
            head_seed = (seed + k * 1000) if seed is not None else None
            if head_seed is not None:
                torch.manual_seed(head_seed)
            t_net = DeepSeaMLP(self.in_dim, hidden_dim)
            tgt_net = DeepSeaMLP(self.in_dim, hidden_dim)
            tgt_net.load_state_dict(t_net.state_dict())
            for p in tgt_net.parameters():
                p.requires_grad = False

            if head_seed is not None:
                torch.manual_seed(head_seed + 777)
            p_net = DeepSeaMLP(self.in_dim, hidden_dim)
            for p in p_net.parameters():
                p.requires_grad = False
            p_net.eval()

            opt = optim.Adam(t_net.parameters(), lr=lr)

            self.trainable_nets.append(t_net)
            self.prior_nets.append(p_net)
            self.target_nets.append(tgt_net)
            self.optimizers.append(opt)

        self.capacity = capacity
        self.replay_s = np.zeros((capacity, self.in_dim), dtype=np.float32)
        self.replay_a = np.zeros(capacity, dtype=np.int64)
        self.replay_r = np.zeros(capacity, dtype=np.float32)
        self.replay_sn = np.zeros((capacity, self.in_dim), dtype=np.float32)
        self.replay_d = np.zeros(capacity, dtype=np.float32)
        self.replay_m = np.zeros((capacity, self.K), dtype=np.float32)
        self.replay_size = 0
        self.replay_idx = 0
        self.active_head = 0
        self.total_steps = 0

    def start_episode(self):
        """Thompson sampling: sample a single head uniformly for the episode."""
        self.active_head = self.rng.randint(0, self.K)

    def select_action(self, obs: np.ndarray) -> int:
        """Greedy action selection under active ensemble head."""
        t_net = self.trainable_nets[self.active_head]
        p_net = self.prior_nets[self.active_head]

        t_net.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(obs.flatten()).float().unsqueeze(0)
            q_val = t_net(s_t).squeeze(0) + self.prior_scale * p_net(s_t).squeeze(0)
            return int(torch.argmax(q_val).item())

    def step_update(self, s: np.ndarray, a: int, r: float, sn: np.ndarray, done: bool):
        """Record transition with independent bootstrap mask and trigger periodic SGD."""
        mask = self.rng.binomial(1, 0.5, size=self.K).astype(np.float32)
        self.replay_s[self.replay_idx] = s.flatten()
        self.replay_a[self.replay_idx] = a
        self.replay_r[self.replay_idx] = r
        self.replay_sn[self.replay_idx] = sn.flatten()
        self.replay_d[self.replay_idx] = float(done)
        self.replay_m[self.replay_idx] = mask

        self.replay_idx = (self.replay_idx + 1) % self.capacity
        if self.replay_size < self.capacity:
            self.replay_size += 1
        self.total_steps += 1

        if self.replay_size >= self.batch_size and self.total_steps % self.sgd_period == 0:
            self._train_step()

    def _train_step(self):
        """One step of SGD across ensemble heads."""
        idx = self.rng.choice(self.replay_size, size=self.batch_size, replace=True)

        sb = torch.from_numpy(self.replay_s[idx])
        ab = torch.from_numpy(self.replay_a[idx])
        rb = torch.from_numpy(self.replay_r[idx])
        snb = torch.from_numpy(self.replay_sn[idx])
        db = torch.from_numpy(self.replay_d[idx])
        mb = self.replay_m[idx]

        # Update active head plus 1 random head to maintain diverse ensemble
        other_head = self.rng.randint(0, self.K)
        heads_to_update = [self.active_head] if other_head == self.active_head else [self.active_head, other_head]

        for k in heads_to_update:
            k_mask = torch.from_numpy(mb[:, k])
            if k_mask.sum() == 0:
                continue

            with torch.no_grad():
                tgt_t = self.target_nets[k](snb)
                tgt_p = self.prior_nets[k](snb)
                next_q = tgt_t + self.prior_scale * tgt_p
                max_next_q, _ = torch.max(next_q, dim=1)
                targets = rb + (1.0 - db) * self.gamma * max_next_q

            self.optimizers[k].zero_grad()
            self.trainable_nets[k].train()
            q_t = self.trainable_nets[k](sb).gather(1, ab.unsqueeze(1)).squeeze(1)
            p_t = self.prior_nets[k](sb).gather(1, ab.unsqueeze(1)).squeeze(1)
            pred_q = q_t + self.prior_scale * p_t

            loss = torch.sum(k_mask * (pred_q - targets) ** 2) / (k_mask.sum() + 1e-6)
            loss.backward()
            self.optimizers[k].step()

            for tgt_p_param, p_param in zip(self.target_nets[k].parameters(), self.trainable_nets[k].parameters()):
                tgt_p_param.data.copy_(self.tau * p_param.data + (1.0 - self.tau) * tgt_p_param.data)

    def end_episode(self):
        pass
