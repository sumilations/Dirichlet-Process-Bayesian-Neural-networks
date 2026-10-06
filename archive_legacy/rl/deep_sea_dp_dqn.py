"""DP-DQN Agent for Deep Sea Exploration Benchmark (arXiv:1806.03335).

Implements Data-Space Dirichlet Process Value Learning for Deep Sea:
1. Dirichlet Process prior directly on the transition data space: F ~ DP(alpha, F_0).
2. Base measure F_0 generates synthetic transitions with stochastic optimism for unvisited states,
   counteracting the -0.01/N movement penalty and preventing the inaction trap (drifting left).
3. Mini-batched DP posterior sampling via Sethuraman stick-breaking weights q_k ~ GEM(alpha + B).
4. Episode-horizon Thompson sampling provides temporally coherent deep exploration.
5. Single Q-network with Layer Normalization matching the paper's 20-unit hidden layer.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DeepSeaQNet(nn.Module):
    """20-unit MLP with Layer Normalization for Deep Sea matching arXiv:1806.03335."""

    def __init__(self, size: int, hidden_dim: int = 20, use_layer_norm: bool = True):
        super().__init__()
        self.size = size
        self.in_dim = size * size
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DPDQNDeepSeaAgent:
    """Single-network Dirichlet Process DQN for Deep Sea Exploration."""

    def __init__(
        self,
        size: int,
        hidden_dim: int = 20,
        alpha: float = 10.0,
        prior_reward_mean: float = 1.0,
        candidate_batch_size: int = 128,
        truncation_K: int = 64,
        gamma: float = 0.99,
        tau: float = 0.05,
        lr: float = 1e-3,
        sgd_period: int = 2,
        capacity: int = 50000,
        use_layer_norm: bool = True,
        episode_target_warmstart: bool = True,
        base_measure_type: str = "structured",
        seed: Optional[int] = None,
    ):
        self.size = size
        self.in_dim = size * size
        self.alpha = float(alpha)
        self.prior_reward_mean = float(prior_reward_mean)
        self.candidate_batch_size = int(candidate_batch_size)
        self.truncation_K = int(truncation_K)
        self.gamma = gamma
        self.tau = tau
        self.sgd_period = sgd_period
        self.episode_target_warmstart = episode_target_warmstart
        self.base_measure_type = base_measure_type

        self.rng = np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)

        # Single Trainable Q-network and Target Network
        self.q_net = DeepSeaQNet(size, hidden_dim, use_layer_norm=use_layer_norm)
        self.target_net = DeepSeaQNet(size, hidden_dim, use_layer_norm=use_layer_norm)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = episode_target_warmstart

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        if episode_target_warmstart:
            self.target_optimizer = optim.Adam(self.target_net.parameters(), lr=lr * 0.5)

        # Replay storage
        self.replay_s = np.zeros((capacity, self.in_dim), dtype=np.float32)
        self.replay_a = np.zeros(capacity, dtype=np.int64)
        self.replay_r = np.zeros(capacity, dtype=np.float32)
        self.replay_sn = np.zeros((capacity, self.in_dim), dtype=np.float32)
        self.replay_d = np.zeros(capacity, dtype=np.float32)
        self.replay_size = 0
        self.replay_idx = 0
        self.capacity = capacity
        self.total_steps = 0

        # Episode Thompson sampling bias
        self.episode_bias = np.zeros(2, dtype=np.float32)

    def sample_base_measure_batch(self, count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Synthetic transition atoms from a uniformly optimistic base measure F_0."""
        s = np.zeros((count, self.in_dim), dtype=np.float32)
        sn = np.zeros((count, self.in_dim), dtype=np.float32)
        a = self.rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)

        # Uniformly optimistic reward everywhere (no special-cased columns or coordinates)
        r = self.rng.normal(self.prior_reward_mean, 0.05, size=count).astype(np.float32)

        for i in range(count):
            row = self.rng.randint(0, self.size)
            col = self.rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                next_col = self.rng.randint(0, next_row + 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done

    def sample_posterior_measure(self, batch_size: int = 64):
        """Vectorized Sethuraman stick-breaking sample construction."""
        n_emp = self.replay_size
        B = min(n_emp, self.candidate_batch_size)
        total_mass = self.alpha + B

        V = self.rng.beta(1.0, total_mass, size=batch_size).astype(np.float32)
        V[-1] = 1.0
        cum_prod = np.cumprod(1.0 - V)
        q = np.empty_like(V)
        q[0] = V[0]
        q[1:] = V[1:] * cum_prod[:-1]
        q_sum = q.sum()
        q = q / q_sum if q_sum > 0 else np.full(batch_size, 1.0 / batch_size, dtype=np.float32)

        prob_emp = B / total_mass if total_mass > 0 else 0.0
        is_emp = (self.rng.rand(batch_size) < prob_emp) if n_emp > 0 else np.zeros(batch_size, dtype=bool)
        k_emp = int(np.sum(is_emp))
        k_syn = batch_size - k_emp

        atom_s = np.zeros((batch_size, self.in_dim), dtype=np.float32)
        atom_a = np.zeros(batch_size, dtype=np.int64)
        atom_r = np.zeros(batch_size, dtype=np.float32)
        atom_sn = np.zeros((batch_size, self.in_dim), dtype=np.float32)
        atom_d = np.zeros(batch_size, dtype=np.float32)

        if k_emp > 0:
            idx = self.rng.choice(n_emp, size=k_emp, replace=True)
            atom_s[is_emp] = self.replay_s[idx]
            atom_a[is_emp] = self.replay_a[idx]
            atom_r[is_emp] = self.replay_r[idx]
            atom_sn[is_emp] = self.replay_sn[idx]
            atom_d[is_emp] = self.replay_d[idx]

        if k_syn > 0:
            syn_s, syn_a, syn_r, syn_sn, syn_d = self.sample_base_measure_batch(k_syn)
            atom_s[~is_emp] = syn_s
            atom_a[~is_emp] = syn_a
            atom_r[~is_emp] = syn_r
            atom_sn[~is_emp] = syn_sn
            atom_d[~is_emp] = syn_d

        return atom_s, atom_a, atom_r, atom_sn, atom_d, q

    def start_episode(self):
        """Sample episode Thompson exploration hypothesis and perform Target Warm-Start."""
        self.episode_bias = self.rng.randn(2).astype(np.float32) * 0.15

        if self.episode_target_warmstart and self.replay_size >= self.truncation_K:
            self.target_net.train()
            for _ in range(5):
                atom_s, atom_a, atom_r, atom_sn, atom_d, q = self.sample_posterior_measure(batch_size=self.truncation_K)
                s_t = torch.from_numpy(atom_s)
                a_t = torch.from_numpy(atom_a)
                r_t = torch.from_numpy(atom_r)
                sn_t = torch.from_numpy(atom_sn)
                d_t = torch.from_numpy(atom_d)
                q_t = torch.from_numpy(q)

                with torch.no_grad():
                    nq = self.target_net(sn_t)
                    mnq, _ = torch.max(nq, dim=1)
                    targets = r_t + (1.0 - d_t) * self.gamma * mnq

                self.target_optimizer.zero_grad()
                tgt_q = self.target_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                loss = torch.sum(q_t * (tgt_q - targets) ** 2)
                loss.backward()
                self.target_optimizer.step()

    def select_action(self, obs: np.ndarray) -> int:
        """Select action using current Q-network and Thompson bias."""
        self.q_net.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(obs.flatten()).float().unsqueeze(0)
            q_vals = self.q_net(s_t).squeeze(0).numpy()
            decay = max(0.01, 500.0 / (500.0 + self.replay_size))
            return int(np.argmax(q_vals + decay * self.episode_bias))

    def step_update(self, s: np.ndarray, a: int, r: float, sn: np.ndarray, done: bool):
        """Record transition and trigger periodic DP-weighted SGD."""
        self.replay_s[self.replay_idx] = s.flatten()
        self.replay_a[self.replay_idx] = a
        self.replay_r[self.replay_idx] = r
        self.replay_sn[self.replay_idx] = sn.flatten()
        self.replay_d[self.replay_idx] = float(done)

        self.replay_idx = (self.replay_idx + 1) % self.capacity
        if self.replay_size < self.capacity:
            self.replay_size += 1
        self.total_steps += 1

        if self.replay_size >= self.truncation_K and self.total_steps % self.sgd_period == 0:
            self._train_step()

    def _train_step(self):
        """One step of SGD on stick-breaking weighted Bellman loss."""
        atom_s, atom_a, atom_r, atom_sn, atom_d, q = self.sample_posterior_measure(batch_size=self.truncation_K)

        s_t = torch.from_numpy(atom_s)
        a_t = torch.from_numpy(atom_a)
        r_t = torch.from_numpy(atom_r)
        sn_t = torch.from_numpy(atom_sn)
        d_t = torch.from_numpy(atom_d)
        q_t = torch.from_numpy(q)

        with torch.no_grad():
            next_q = self.target_net(sn_t)
            max_next_q, _ = torch.max(next_q, dim=1)
            targets = r_t + (1.0 - d_t) * self.gamma * max_next_q

        self.optimizer.zero_grad()
        self.q_net.train()
        q_vals = self.q_net(s_t)
        chosen_q = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)

        loss = torch.sum(q_t * (chosen_q - targets) ** 2)
        loss.backward()
        self.optimizer.step()

        for tgt_p, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tgt_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * tgt_p.data)

    def end_episode(self):
        pass
