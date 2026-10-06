"""Bootstrapped DQN with Additive Randomized Prior Functions (Osband et al., 2018)."""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..networks import QNetwork, ReplayBuffer


class BootDQNRPAgent:
    """Ensemble RLSVI / Bootstrapped DQN with Additive Prior Functions.

    Reference: Osband et al., NeurIPS 2018 / arXiv:1703.07608:
    - Maintains K ensemble members: Q_k(s, a) = f_{theta_k}(s, a) + beta * f_{theta_k^0}(s, a).
    - Fixed, untrainable randomized prior network f_{theta_k^0}.
    - Online bootstrap masking (Bernoulli p=0.5).
    - Thompson sampling: selects active head uniformly at episode start.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_models: int = 20,
        hidden_dim: int = 50,
        prior_scale: float = 1.0,
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
        self.num_models = num_models
        self.prior_scale = float(prior_scale)
        self.gamma = gamma
        self.tau = tau
        self.sgd_period = sgd_period
        self.batch_size = batch_size
        self.capacity = capacity
        self.device = device or torch.device("cpu")
        self.rng = np.random.RandomState(seed)

        self.trainable_nets: List[nn.Module] = []
        self.prior_nets: List[nn.Module] = []
        self.target_nets: List[nn.Module] = []
        self.optimizers: List[optim.Optimizer] = []

        for i in range(num_models):
            if seed is not None:
                torch.manual_seed(seed + i * 1000)
            t_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            tgt_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            tgt_net.load_state_dict(t_net.state_dict())
            for p in tgt_net.parameters():
                p.requires_grad = False

            if seed is not None:
                torch.manual_seed(seed + i * 1000 + 777)
            p_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            for p in p_net.parameters():
                p.requires_grad = False
            p_net.eval()

            opt = optim.Adam(t_net.parameters(), lr=lr)

            self.trainable_nets.append(t_net)
            self.prior_nets.append(p_net)
            self.target_nets.append(tgt_net)
            self.optimizers.append(opt)

        self.replay = ReplayBuffer(capacity=capacity, state_dim=state_dim)
        self.masks = np.zeros((capacity, num_models), dtype=np.float32)
        self.active_head = 0
        self.total_steps = 0
        self.episodes_completed = 0

    def reset_episode(self) -> None:
        """Sample active head uniformly at episode start (Thompson sampling)."""
        self.active_head = self.rng.randint(0, self.num_models)
        self.episodes_completed += 1

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action greedily under active ensemble member."""
        if len(self.replay) < self.batch_size:
            return self.rng.randint(0, self.action_dim)

        t_net = self.trainable_nets[self.active_head]
        p_net = self.prior_nets[self.active_head]
        t_net.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            q_val = t_net(s_t).squeeze(0) + self.prior_scale * p_net(s_t).squeeze(0)
            return int(torch.argmax(q_val).item())

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        cur_idx = self.replay.idx
        self.masks[cur_idx] = self.rng.binomial(1, 0.5, size=self.num_models).astype(np.float32)
        self.replay.push(state, action, reward, next_state, done)
        self.total_steps += 1

        if len(self.replay) >= self.batch_size and self.total_steps % self.sgd_period == 0:
            return self._train_step()
        return None

    def _train_step(self) -> float:
        n = len(self.replay)
        batch_idx = self.rng.choice(n, size=self.batch_size, replace=True)

        s_b = torch.from_numpy(self.replay.states[batch_idx]).to(self.device)
        a_b = torch.from_numpy(self.replay.actions[batch_idx]).to(self.device)
        r_b = torch.from_numpy(self.replay.rewards[batch_idx]).to(self.device)
        sn_b = torch.from_numpy(self.replay.next_states[batch_idx]).to(self.device)
        d_b = torch.from_numpy(self.replay.dones[batch_idx]).to(self.device)
        m_b = self.masks[batch_idx]

        # Update active head plus an auxiliary head
        other_head = self.rng.randint(0, self.num_models)
        heads_to_update = [self.active_head] if other_head == self.active_head else [self.active_head, other_head]

        total_loss = 0.0
        for k in heads_to_update:
            t_net = self.trainable_nets[k]
            p_net = self.prior_nets[k]
            tgt_net = self.target_nets[k]
            opt = self.optimizers[k]
            mask_k = torch.from_numpy(m_b[:, k]).to(self.device)

            if mask_k.sum() == 0:
                continue

            t_net.train()
            with torch.no_grad():
                next_q_tot = tgt_net(sn_b) + self.prior_scale * p_net(sn_b)
                max_next_q, _ = torch.max(next_q_tot, dim=1)
                targets = r_b + (1.0 - d_b) * self.gamma * max_next_q

            opt.zero_grad()
            q_trainable = t_net(s_b)
            with torch.no_grad():
                q_prior = p_net(s_b)
            q_tot = q_trainable + self.prior_scale * q_prior
            chosen_q = q_tot.gather(1, a_b.unsqueeze(1)).squeeze(1)

            loss = torch.sum(mask_k * (chosen_q - targets) ** 2) / (mask_k.sum() + 1e-8)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

            # Polyak target tracking
            for tgt_p, p in zip(tgt_net.parameters(), t_net.parameters()):
                tgt_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * tgt_p.data)

        return total_loss
