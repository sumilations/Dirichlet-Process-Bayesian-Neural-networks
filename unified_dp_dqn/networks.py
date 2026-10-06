"""Neural Network Architectures and Replay Buffer for Unified DP-DQN."""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Generic, environment-agnostic MLP with optional Layer Normalization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 50,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        act_cls = nn.ReLU if activation.lower() == "relu" else nn.Tanh

        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Pre-allocated circular buffer for transition storage."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.idx = 0
        self.size = 0
        self.total_count = 0
        self.num_positive_rewards = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = float(done)

        if reward > 0:
            self.num_positive_rewards += 1

        self.idx = (self.idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        self.total_count += 1

    def sample(self, batch_size: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


class StratifiedReplayBuffer:
    """Replay buffer that balances exploration transitions (r <= 0) and goal/balance transitions (r > 0).

    Prevents catastrophic forgetting: guarantees that both swing-up trajectories
    and upright stabilization trajectories remain equally represented in every SGD batch.
    """

    def __init__(self, capacity: int, state_dim: int, pos_ratio: float = 0.5):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.pos_ratio = float(pos_ratio)

        self.neg_buffer = ReplayBuffer(capacity=capacity, state_dim=state_dim)
        self.pos_buffer = ReplayBuffer(capacity=capacity, state_dim=state_dim)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        if reward > 0.0:
            self.pos_buffer.push(state, action, reward, next_state, done)
        else:
            self.neg_buffer.push(state, action, reward, next_state, done)

    def sample(self, batch_size: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_pos_avail = len(self.pos_buffer)
        n_neg_avail = len(self.neg_buffer)

        if n_pos_avail == 0:
            return self.neg_buffer.sample(batch_size, rng)
        if n_neg_avail == 0:
            return self.pos_buffer.sample(batch_size, rng)

        n_pos = max(1, min(int(batch_size * self.pos_ratio), batch_size - 1))
        n_neg = batch_size - n_pos

        s_pos, a_pos, r_pos, sn_pos, d_pos = self.pos_buffer.sample(n_pos, rng)
        s_neg, a_neg, r_neg, sn_neg, d_neg = self.neg_buffer.sample(n_neg, rng)

        return (
            np.concatenate([s_pos, s_neg], axis=0),
            np.concatenate([a_pos, a_neg], axis=0),
            np.concatenate([r_pos, r_neg], axis=0),
            np.concatenate([sn_pos, sn_neg], axis=0),
            np.concatenate([d_pos, d_neg], axis=0),
        )

    def __len__(self) -> int:
        return len(self.pos_buffer) + len(self.neg_buffer)
