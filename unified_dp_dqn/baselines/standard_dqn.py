"""Standard DQN with Linear Epsilon-Greedy Annealing."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..networks import QNetwork, ReplayBuffer


class StandardDQNAgent:
    """Vanilla DQN with Linear Epsilon Annealing matching Section 7.2.2 of arXiv:1703.07608."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 50,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        anneal_episodes: int = 500,
        gamma: float = 0.99,
        tau: float = 0.05,
        lr: float = 1e-3,
        sgd_period: int = 2,
        batch_size: int = 64,
        capacity: int = 1000000,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.anneal_episodes = anneal_episodes
        self.gamma = gamma
        self.tau = tau
        self.sgd_period = sgd_period
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        agent_seed = (seed * 10007 + 997) if seed is not None else None
        self.rng = np.random.RandomState(agent_seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity=capacity, state_dim=state_dim)
        self.total_steps = 0
        self.episodes_completed = 0

    def reset_episode(self) -> None:
        self.episodes_completed += 1
        frac = min(1.0, self.episodes_completed / float(self.anneal_episodes))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and (len(self.replay) < self.batch_size or self.rng.rand() < self.epsilon):
            return self.rng.randint(0, self.action_dim)

        self.q_net.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            q_vals = self.q_net(s_t).squeeze(0)
            return int(torch.argmax(q_vals).item())

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        self.replay.push(state, action, reward, next_state, done)
        self.total_steps += 1

        if len(self.replay) >= self.batch_size and self.total_steps % self.sgd_period == 0:
            return self._train_step()
        return None

    def _train_step(self) -> float:
        s_b, a_b, r_b, sn_b, d_b = self.replay.sample(self.batch_size, self.rng)
        s_t = torch.from_numpy(s_b).to(self.device)
        a_t = torch.from_numpy(a_b).to(self.device)
        r_t = torch.from_numpy(r_b).to(self.device)
        sn_t = torch.from_numpy(sn_b).to(self.device)
        d_t = torch.from_numpy(d_b).to(self.device)

        with torch.no_grad():
            next_q = self.target_net(sn_t)
            max_next_q, _ = torch.max(next_q, dim=1)
            targets = r_t + (1.0 - d_t) * self.gamma * max_next_q

        self.optimizer.zero_grad()
        self.q_net.train()
        q_vals = self.q_net(s_t)
        chosen_q = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)

        loss = nn.functional.mse_loss(chosen_q, targets)
        loss.backward()
        self.optimizer.step()

        # Polyak target tracking
        with torch.no_grad():
            for tgt_p, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tgt_p.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return float(loss.item())
