"""DP-DQN Agent implementing Data-Space Dirichlet Process Value Learning.

Based on the Vashishtha formulation of DP-BNNs:
1. Dirichlet Process prior directly on the transition data space: F ~ DP(alpha, F_0).
2. Base measure F_0 provides optimistic exploration anchors across actions and upright states,
   counteracting the action penalty and preventing the inaction trap (Q=0).
3. Mini-batched DP posterior sampling via Sethuraman stick-breaking weights q_k ~ GEM(alpha + B).
4. Episode-horizon Thompson sampling provides temporally coherent momentum pumping.
5. Online step-wise DP-weighted Bellman optimization.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """50-50 MLP with Layer Normalization for state-action value estimation matching arXiv:1703.07608."""

    def __init__(self, state_dim: int = 6, action_dim: int = 3, hidden_dim: int = 50, use_layer_norm: bool = False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """High-performance contiguous numpy Replay Buffer for MDP transitions."""

    def __init__(self, capacity: int = 100000, state_dim: int = 6):
        self.capacity = capacity
        self.state_dim = state_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def __len__(self):
        return self.size


class DPDQNAgent:
    """DP-DQN Agent implementing Mini-Batch Dirichlet Process Value Learning with Layer Normalization.

    Based on the Vashishtha formulation of DP-BNNs:
    1. Dirichlet Process prior directly on the transition data space: F ~ DP(alpha, F_0).
    2. Base measure F_0 provides optimistic exploration anchors across actions and upright states,
       counteracting the action penalty and preventing the inaction trap (Q=0).
    3. Mini-batched DP posterior sampling via Sethuraman stick-breaking weights q_k ~ GEM(alpha + B).
    4. Episode-horizon Thompson directional bias for temporally consistent exploration.
    5. Online step-wise DP stick-breaking weighted Bellman optimization with Layer Normalization.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dim: int = 50,
        alpha: float = 50.0,
        candidate_batch_size: int = 256,
        truncation_K: int = 128,
        prior_reward_mean: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.01,
        lr: float = 1e-3,
        sgd_period: int = 2,
        capacity: int = 100000,
        use_layer_norm: bool = True,
        base_measure_type: str = "default",
        episode_target_warmstart: bool = False,
        episodic_burst_steps: int = 0,
        seed: Optional[int] = None,
        **kwargs
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = float(alpha)
        self.candidate_batch_size = int(candidate_batch_size)
        self.truncation_K = int(truncation_K)
        self.prior_reward_mean = float(prior_reward_mean)
        self.gamma = gamma
        self.tau = tau
        self.sgd_period = sgd_period
        self.use_layer_norm = use_layer_norm
        self.base_measure_type = base_measure_type
        self.episode_target_warmstart = episode_target_warmstart
        self.episodic_burst_steps = episodic_burst_steps

        self.rng = np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)

        # Trainable Q-network and target network with Layer Normalization
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim, use_layer_norm=use_layer_norm)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim, use_layer_norm=use_layer_norm)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = episode_target_warmstart

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        if episode_target_warmstart:
            self.target_optimizer = optim.Adam(self.target_net.parameters(), lr=lr * 0.5)

        self.replay = ReplayBuffer(capacity=capacity, state_dim=state_dim)
        self.total_steps = 0

        # Episode-horizon Thompson sampling directional bias
        self.episode_directional_bias = np.zeros(action_dim, dtype=np.float32)

    def sample_base_measure_batch(self, count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized generation of synthetic transition atoms from domain-informed base measure F_0."""
        s = np.zeros((count, self.state_dim), dtype=np.float32)

        if self.base_measure_type in ("uniform_max_entropy", "uniform_entropy", "max_entropy"):
            # Jaynes' Principle of Maximum Entropy: Uniform state bounds + Uniform(0, R_max) reward
            s[:, 0] = self.rng.uniform(-1.0, 1.0, size=count)       # cos(theta)
            s[:, 1] = self.rng.uniform(-1.0, 1.0, size=count)       # sin(theta)
            s[:, 2] = self.rng.uniform(-10.0, 10.0, size=count)     # theta_dot
            s[:, 3] = self.rng.uniform(-5.0, 5.0, size=count)       # x
            s[:, 4] = self.rng.uniform(-5.0, 5.0, size=count)       # x_dot
            if self.state_dim >= 6:
                s[:, 5] = self.rng.uniform(0.0, 1.0, size=count)    # normalized time
            a = self.rng.randint(0, self.action_dim, size=count)
            # Maximum-entropy uniform reward: Uniform(0, R_max) where R_max = 1.0 (mean = 0.5)
            r = self.rng.uniform(0.0, 1.0, size=count).astype(np.float32)
            sn = s + self.rng.normal(0.0, 0.05, size=(count, self.state_dim)).astype(np.float32)
            done = np.zeros(count, dtype=np.float32)
            return s, a, r, sn, done

        if self.base_measure_type == "uniform_gaussian_optimistic":
            # Completely uninformed uniform state bounds + Gaussian optimistic reward
            s[:, 0] = self.rng.uniform(-1.0, 1.0, size=count)       # cos(theta)
            s[:, 1] = self.rng.uniform(-1.0, 1.0, size=count)       # sin(theta)
            s[:, 2] = self.rng.uniform(-10.0, 10.0, size=count)     # theta_dot
            s[:, 3] = self.rng.uniform(-5.0, 5.0, size=count)       # x
            s[:, 4] = self.rng.uniform(-5.0, 5.0, size=count)       # x_dot
            s[:, 5] = self.rng.uniform(0.0, 1.0, size=count)        # time
            a = self.rng.randint(0, self.action_dim, size=count)
            r = self.rng.normal(self.prior_reward_mean, 0.5, size=count).astype(np.float32)
            sn = s.copy()
            done = np.zeros(count, dtype=np.float32)
            return s, a, r, sn, done

        if self.base_measure_type == "pure_upright":
            # 100% Upright Target Anchors
            s[:, 0] = self.rng.uniform(0.95, 1.0, size=count)
            s[:, 1] = self.rng.uniform(-0.3, 0.3, size=count)
            s[:, 2] = self.rng.uniform(-0.5, 0.5, size=count)
            s[:, 3] = self.rng.uniform(-0.8, 0.8, size=count)
            s[:, 4] = self.rng.uniform(-0.5, 0.5, size=count)
            s[:, 5] = self.rng.uniform(0.0, 1.0, size=count)
            a = self.rng.randint(0, self.action_dim, size=count)
            r = self.rng.normal(self.prior_reward_mean, 0.1, size=count).astype(np.float32)
            sn = s.copy()
            done = np.zeros(count, dtype=np.float32)
            return s, a, r, sn, done

        n_upright = count // 2
        n_swing = count - n_upright

        # Upright states: cos(theta) > 0.95, |theta_dot| <= 0.5, |x| <= 0.8, |x_dot| <= 0.5
        s[:n_upright, 0] = self.rng.uniform(0.95, 1.0, size=n_upright)       # cos(theta) > 0.95
        s[:n_upright, 1] = self.rng.uniform(-0.3, 0.3, size=n_upright)       # sin(theta)
        s[:n_upright, 2] = self.rng.uniform(-0.5, 0.5, size=n_upright)       # theta_dot
        s[:n_upright, 3] = self.rng.uniform(-0.8, 0.8, size=n_upright)       # x
        s[:n_upright, 4] = self.rng.uniform(-0.5, 0.5, size=n_upright)       # x_dot
        s[:n_upright, 5] = self.rng.uniform(0.0, 1.0, size=n_upright)        # normalized time

        # Active swing states: covers the dynamic swinging range
        s[n_upright:, 0] = self.rng.uniform(-1.0, 1.0, size=n_swing)
        s[n_upright:, 1] = self.rng.uniform(-1.0, 1.0, size=n_swing)
        s[n_upright:, 2] = self.rng.uniform(-2.0, 2.0, size=n_swing)
        s[n_upright:, 3] = self.rng.uniform(-3.0, 3.0, size=n_swing)
        s[n_upright:, 4] = self.rng.uniform(-2.0, 2.0, size=n_swing)
        s[n_upright:, 5] = self.rng.uniform(0.0, 1.0, size=n_swing)

        a = np.zeros(count, dtype=np.int64)
        a[:n_upright] = self.rng.randint(0, self.action_dim, size=n_upright)

        r = np.zeros(count, dtype=np.float32)

        if self.base_measure_type == "stochastic_sparks":
            # High-variance stochastic optimism: 30% jackpot sparks, 70% standard
            sparks = self.rng.rand(n_upright) < 0.3
            r_up = np.where(sparks, self.rng.uniform(1.5, 3.0, size=n_upright), self.rng.uniform(0.8, 1.2, size=n_upright))
            r[:n_upright] = r_up.astype(np.float32)
            a[n_upright:] = self.rng.choice([0, 2], size=n_swing)
            r[n_upright:] = self.rng.uniform(0.1, 0.45, size=n_swing).astype(np.float32)

        elif self.base_measure_type == "energy_coupled":
            # Swing actions coupled to angular velocity to encourage resonant pumping
            # if theta_dot < 0 (moving left), push left (a=0); else push right (a=2)
            a[n_upright:] = np.where(s[n_upright:, 2] < 0, 0, 2)
            r[:n_upright] = self.rng.normal(self.prior_reward_mean, 0.1, size=n_upright)
            r[n_upright:] = self.rng.normal(0.25, 0.05, size=n_swing)

        else:  # "default" 50/50
            a[n_upright:] = self.rng.choice([0, 2], size=n_swing)
            r[:n_upright] = self.rng.normal(self.prior_reward_mean, 0.1, size=n_upright)
            r[n_upright:] = self.rng.normal(0.2, 0.05, size=n_swing)

        sn = s.copy()
        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done

    def sample_posterior_measure(self, batch_size: int = 128):
        """Vectorized Sethuraman stick-breaking sample construction P_j = sum_{k=1}^K q_k delta_{z_k}."""
        n_total = len(self.replay)
        B = min(n_total, self.candidate_batch_size)
        total_mass = self.alpha + B

        # 1. Stick-breaking weights: V_k ~ Beta(1, alpha + B)
        V = self.rng.beta(1.0, total_mass, size=batch_size).astype(np.float32)
        V[-1] = 1.0
        cum_prod = np.cumprod(1.0 - V)
        remaining = np.empty_like(V)
        remaining[0] = 1.0
        remaining[1:] = cum_prod[:-1]
        q = V * remaining
        q_sum = q.sum()
        q = q / q_sum if q_sum > 0 else np.full(batch_size, 1.0 / batch_size, dtype=np.float32)

        # 2. Probability of empirical vs base measure anchor
        prob_emp = B / total_mass if total_mass > 0 else 0.0
        is_emp = (self.rng.rand(batch_size) < prob_emp) if n_total > 0 else np.zeros(batch_size, dtype=bool)
        n_emp = int(np.sum(is_emp))
        n_syn = batch_size - n_emp

        atom_s = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        atom_a = np.zeros(batch_size, dtype=np.int64)
        atom_r = np.zeros(batch_size, dtype=np.float32)
        atom_sn = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        atom_done = np.zeros(batch_size, dtype=np.float32)

        if n_emp > 0:
            cand_idx = self.rng.choice(n_total, size=n_emp, replace=True)
            atom_s[is_emp] = self.replay.states[cand_idx]
            atom_a[is_emp] = self.replay.actions[cand_idx]
            atom_r[is_emp] = self.replay.rewards[cand_idx]
            atom_sn[is_emp] = self.replay.next_states[cand_idx]
            atom_done[is_emp] = self.replay.dones[cand_idx]

        if n_syn > 0:
            syn_s, syn_a, syn_r, syn_sn, syn_d = self.sample_base_measure_batch(n_syn)
            atom_s[~is_emp] = syn_s
            atom_a[~is_emp] = syn_a
            atom_r[~is_emp] = syn_r
            atom_sn[~is_emp] = syn_sn
            atom_done[~is_emp] = syn_d

        return atom_s, atom_a, atom_r, atom_sn, atom_done, q

    def start_episode(self):
        """Draw an episode Thompson sampling directional hypothesis and optional target warm-start."""
        self.episode_directional_bias = self.rng.randn(self.action_dim).astype(np.float32) * 0.3
        self.episode_directional_bias[1] = 0.0

        # Episodic burst learning (Pure PSRL paradigm: 0 online updates, burst update at boundary)
        if self.episodic_burst_steps > 0 and len(self.replay) >= self.truncation_K:
            for _ in range(self.episodic_burst_steps):
                self._train_step()

        if self.episode_target_warmstart and len(self.replay) >= self.truncation_K:
            self.target_net.train()
            for _ in range(10):
                atom_s, atom_a, atom_r, atom_sn, atom_done, q = self.sample_posterior_measure(batch_size=self.truncation_K)
                s_t = torch.from_numpy(atom_s)
                a_t = torch.from_numpy(atom_a)
                r_t = torch.from_numpy(atom_r)
                sn_t = torch.from_numpy(atom_sn)
                d_t = torch.from_numpy(atom_done)
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

    def select_action(self, state: np.ndarray) -> int:
        """Greedy action selection under currently active Thompson hypothesis."""
        if len(self.replay) < self.truncation_K:
            return int(self.rng.choice([0, 2]))

        self.q_net.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(state).unsqueeze(0)
            q_vals = self.q_net(s_t).squeeze(0).numpy()
            decay_factor = max(0.01, 10000.0 / (10000.0 + len(self.replay)))
            biased_q = q_vals + decay_factor * self.episode_directional_bias
            return int(np.argmax(biased_q))

    def step_update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer and execute periodic DP-weighted SGD."""
        self.replay.push(state, action, reward, next_state, done)
        self.total_steps += 1

        # Online updates only when not in pure episodic burst mode
        if self.episodic_burst_steps == 0 and len(self.replay) >= self.truncation_K and self.total_steps % self.sgd_period == 0:
            self._train_step()

    def _train_step(self):
        """Execute one step of SGD on stick-breaking weighted Bellman loss."""
        atom_s, atom_a, atom_r, atom_sn, atom_done, q = self.sample_posterior_measure(batch_size=self.truncation_K)

        s_t = torch.from_numpy(atom_s)
        a_t = torch.from_numpy(atom_a)
        r_t = torch.from_numpy(atom_r)
        sn_t = torch.from_numpy(atom_sn)
        d_t = torch.from_numpy(atom_done)
        q_t = torch.from_numpy(q)

        # Bellman targets using frozen target network
        with torch.no_grad():
            next_q = self.target_net(sn_t)
            max_next_q, _ = torch.max(next_q, dim=1)
            targets = r_t + (1.0 - d_t) * self.gamma * max_next_q

        self.optimizer.zero_grad()
        self.q_net.train()
        q_vals = self.q_net(s_t)
        chosen_q = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)

        # Stick-breaking weighted Bellman error: sum_k q_k * (Q(s_k, a_k) - y_k)^2
        loss = torch.sum(q_t * (chosen_q - targets) ** 2)
        loss.backward()
        self.optimizer.step()

        # Polyak update of target network
        for tgt_p, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tgt_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * tgt_p.data)

    def end_episode(self):
        pass

