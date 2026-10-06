"""Baseline RL Algorithms for Cart-Pole Swing-Up Benchmark.

References:
1. "Deep Exploration via Randomized Value Functions" (Osband, Van Roy, Russo, Wen, 2017/2019 - arXiv:1703.07608).
2. "Deep Exploration via Bootstrapped DQN" (Osband et al., NeurIPS 2016).
3. "Randomized Prior Functions for Deep Reinforcement Learning" (Osband et al., NeurIPS 2018).
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dp_dqn import QNetwork, ReplayBuffer


class BootDQNRPAgent:
    """Ensemble RLSVI / Bootstrapped DQN with Additive Prior Functions.

    Implements Algorithm 8 (learn_ensemble_rlsvi) and Section 7.2.1/7.2.2 of arXiv:1703.07608:
    - Maintains K ensemble members where each member computes:
        Q_k(s, a) = f_{theta_k}(s, a) + beta * f_{theta_k^0}(s, a)
      with f_{theta_k^0} being a fixed, randomly initialized prior network.
    - Uses online bootstrap masking (Bernoulli p=0.5).
    - Samples a single head at episode start for temporally consistent deep exploration.
    - Executes online step-wise SGD updates every `sgd_period` steps.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        num_models: int = 20,
        hidden_dim: int = 50,
        prior_scale: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.05,
        lr: float = 1e-3,
        sgd_period: int = 4,
        batch_size: int = 128,
        capacity: int = 100000,
        seed: Optional[int] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_models = num_models
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

        for i in range(num_models):
            if seed is not None:
                torch.manual_seed(seed + i * 1000)
            t_net = QNetwork(state_dim, action_dim, hidden_dim)
            tgt_net = QNetwork(state_dim, action_dim, hidden_dim)
            tgt_net.load_state_dict(t_net.state_dict())
            for p in tgt_net.parameters():
                p.requires_grad = False

            if seed is not None:
                torch.manual_seed(seed + i * 1000 + 777)
            p_net = QNetwork(state_dim, action_dim, hidden_dim)
            for p in p_net.parameters():
                p.requires_grad = False
            p_net.eval()

            opt = optim.Adam(t_net.parameters(), lr=lr)

            self.trainable_nets.append(t_net)
            self.prior_nets.append(p_net)
            self.target_nets.append(tgt_net)
            self.optimizers.append(opt)

        self.replay = ReplayBuffer(capacity=capacity, state_dim=state_dim)
        # Preallocated numpy array for masks: [capacity, num_models]
        self.masks = np.zeros((capacity, num_models), dtype=np.float32)
        self.active_head = 0
        self.total_steps = 0

    def start_episode(self):
        """Thompson sampling: select one ensemble head uniformly for the episode."""
        self.active_head = self.rng.randint(0, self.num_models)

    def select_action(self, state: np.ndarray) -> int:
        """Greedy action selection under active ensemble head."""
        if len(self.replay) < self.batch_size:
            return self.rng.randint(0, self.action_dim)

        t_net = self.trainable_nets[self.active_head]
        p_net = self.prior_nets[self.active_head]

        t_net.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(state).unsqueeze(0)
            q_val = t_net(s_t).squeeze(0) + self.prior_scale * p_net(s_t).squeeze(0)
            return int(torch.argmax(q_val).item())

    def step_update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Record transition with independent bootstrap mask and trigger step-wise SGD."""
        cur_idx = self.replay.idx
        self.masks[cur_idx] = self.rng.binomial(1, 0.5, size=self.num_models).astype(np.float32)
        self.replay.push(state, action, reward, next_state, done)
        self.total_steps += 1

        if len(self.replay) >= self.batch_size and self.total_steps % self.sgd_period == 0:
            self._train_step()

    def _train_step(self):
        """Execute one step of SGD across ensemble heads."""
        n = len(self.replay)
        batch_idx = self.rng.choice(n, size=self.batch_size, replace=True)

        s_b = torch.from_numpy(self.replay.states[batch_idx])
        a_b = torch.from_numpy(self.replay.actions[batch_idx])
        r_b = torch.from_numpy(self.replay.rewards[batch_idx])
        sn_b = torch.from_numpy(self.replay.next_states[batch_idx])
        d_b = torch.from_numpy(self.replay.dones[batch_idx])
        m_b = self.masks[batch_idx]  # [B, K]

        # Always update the active head making decisions this episode plus an auxiliary head
        other_head = self.rng.randint(0, self.num_models)
        heads_to_update = [self.active_head] if other_head == self.active_head else [self.active_head, other_head]

        for k in heads_to_update:
            t_net = self.trainable_nets[k]
            p_net = self.prior_nets[k]
            tgt_net = self.target_nets[k]
            opt = self.optimizers[k]
            mask_k = torch.from_numpy(m_b[:, k])

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

            # Polyak update
            for tgt_p, p in zip(tgt_net.parameters(), t_net.parameters()):
                tgt_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * tgt_p.data)

    def end_episode(self):
        pass


class StandardDQNAgent:
    """Standard DQN Agent with Linear Epsilon-Greedy Annealing.

    Matches Section 7.2.2 and Figure 16 of arXiv:1703.07608:
    - 50-50 MLP with ReLU.
    - Epsilon annealed linearly from 1.0 to 0.0 over training.
    - Evaluates the dithering failure mode in Cart-Pole Swing-Up.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dim: int = 50,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.0,
        anneal_episodes: int = 500,
        gamma: float = 0.99,
        tau: float = 0.05,
        lr: float = 1e-3,
        sgd_period: int = 4,
        batch_size: int = 128,
        capacity: int = 100000,
        seed: Optional[int] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.anneal_episodes = anneal_episodes
        self.current_episode = 0

        self.gamma = gamma
        self.tau = tau
        self.sgd_period = sgd_period
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.q_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity=capacity, state_dim=state_dim)
        self.total_steps = 0

    def start_episode(self):
        pass

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if len(self.replay) < self.batch_size or self.rng.rand() < self.epsilon:
            return self.rng.randint(0, self.action_dim)

        self.q_net.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(state).unsqueeze(0)
            q_vals = self.q_net(s_t).squeeze(0).numpy()
            return int(np.argmax(q_vals))

    def step_update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.replay.push(state, action, reward, next_state, done)
        self.total_steps += 1

        if len(self.replay) >= self.batch_size and self.total_steps % self.sgd_period == 0:
            self._train_step()

    def _train_step(self):
        n = len(self.replay)
        batch_idx = self.rng.choice(n, size=self.batch_size, replace=True)

        s_b = torch.from_numpy(self.replay.states[batch_idx])
        a_b = torch.from_numpy(self.replay.actions[batch_idx])
        r_b = torch.from_numpy(self.replay.rewards[batch_idx])
        sn_b = torch.from_numpy(self.replay.next_states[batch_idx])
        d_b = torch.from_numpy(self.replay.dones[batch_idx])

        with torch.no_grad():
            next_q = self.target_net(sn_b)
            max_next_q, _ = torch.max(next_q, dim=1)
            targets = r_b + (1.0 - d_b) * self.gamma * max_next_q

        self.optimizer.zero_grad()
        self.q_net.train()
        q_vals = self.q_net(s_b)
        chosen_q = q_vals.gather(1, a_b.unsqueeze(1)).squeeze(1)

        loss = nn.functional.mse_loss(chosen_q, targets)
        loss.backward()
        self.optimizer.step()

        for tgt_p, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tgt_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * tgt_p.data)

    def end_episode(self):
        """Linear annealing of epsilon as described in Figure 16."""
        self.current_episode += 1
        frac = min(1.0, self.current_episode / float(self.anneal_episodes))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
