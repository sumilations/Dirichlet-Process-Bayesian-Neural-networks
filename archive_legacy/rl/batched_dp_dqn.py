"""Batched and Pre-Sampled DP-DQN Agent for Deep Sea Exploration.

Supports two operational modes for rigorous head-to-head comparison:
1. Standard Mode (presample_and_batch=False):
   - Traditional step-by-step sampling: Every sgd_period steps, samples 1 batch from
     the live replay buffer and takes a sequential SGD update.
2. Episodic Pre-Sampled & Batched Mode (presample_and_batch=True):
   - At start_episode(), pre-samples all K stick-breaking random measures and transition
     batches in a single vectorized operation from D_{m-1}.
   - During the episode, runs with zero sampling overhead in the step loop.
   - At end_episode(), executes all K updates in a single parallel batched tensor pass
     and flushes the new transitions into replay storage.
"""

from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .deep_sea_dp_dqn import DeepSeaQNet


class BatchedDPDQNDeepSeaAgent:
    """DP-DQN Agent supporting both step-by-step sequential and episodic batched execution."""

    def __init__(
        self,
        size: int = 20,
        hidden_dim: int = 20,
        alpha: float = 5.0,
        prior_reward_mean: float = 0.1,
        prior_reward_std: float = 0.05,
        candidate_batch_size: int = 128,
        batch_size: int = 32,
        gamma: float = 0.99,
        tau: float = 0.05,
        lr: float = 1e-3,
        sgd_period: int = 2,
        capacity: int = 50000,
        use_layer_norm: bool = True,
        presample_and_batch: bool = True,
        sgd_per_episode: Optional[int] = None,
        warm_start_target: bool = True,
        target_update_freq: int = 1,
        target_tau: Optional[float] = None,
        use_thompson_bias: bool = True,
        seed: Optional[int] = None,
    ):
        self.size = size
        self.in_dim = size * size
        self.alpha = float(alpha)
        self.prior_reward_mean = float(prior_reward_mean)
        self.prior_reward_std = float(prior_reward_std)
        self.candidate_batch_size = int(candidate_batch_size)
        self.batch_size = int(batch_size)
        self.gamma = gamma
        self.tau = tau
        self.sgd_period = sgd_period
        self.presample_and_batch = presample_and_batch
        self.sgd_per_episode = sgd_per_episode
        self.warm_start_target = warm_start_target
        self.target_update_freq = target_update_freq
        self.target_tau = target_tau if target_tau is not None else (1.0 if target_update_freq > 1 else tau)
        self.use_thompson_bias = use_thompson_bias
        self.episode_count = 0

        # Number of updates per episode: K = sgd_per_episode if specified, else ceil(size / sgd_period)
        if sgd_per_episode is not None:
            self.K_updates = max(1, sgd_per_episode)
        else:
            self.K_updates = max(1, size // sgd_period)
        self.tau_eff = 1.0 - (1.0 - tau) ** self.K_updates

        self.rng = np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)

        # Single Trainable Q-network and Target Network with Layer Normalization
        self.q_net = DeepSeaQNet(size, hidden_dim, use_layer_norm=use_layer_norm)
        self.target_net = DeepSeaQNet(size, hidden_dim, use_layer_norm=use_layer_norm)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = True

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.target_optimizer = optim.Adam(self.target_net.parameters(), lr=lr * 0.5)

        # Replay storage
        self.capacity = capacity
        self.replay_s = np.zeros((capacity, self.in_dim), dtype=np.float32)
        self.replay_a = np.zeros(capacity, dtype=np.int64)
        self.replay_r = np.zeros(capacity, dtype=np.float32)
        self.replay_sn = np.zeros((capacity, self.in_dim), dtype=np.float32)
        self.replay_d = np.zeros(capacity, dtype=np.float32)
        self.replay_size = 0
        self.replay_idx = 0
        self.total_steps = 0

        # Thompson exploration bias
        self.episode_bias = np.zeros(2, dtype=np.float32)

        # Episode temporary buffer for transitions collected during the current episode
        self.episode_buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

        # Pre-sampled batch containers (when presample_and_batch=True)
        self.presampled_s: Optional[torch.Tensor] = None
        self.presampled_a: Optional[torch.Tensor] = None
        self.presampled_r: Optional[torch.Tensor] = None
        self.presampled_sn: Optional[torch.Tensor] = None
        self.presampled_d: Optional[torch.Tensor] = None
        self.presampled_q: Optional[torch.Tensor] = None

    def sample_base_measure_batch(self, count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample synthetic transition atoms from a uniformly optimistic base measure F_0.
        
        Uniform everywhere & optimistic:
        - States s are sampled uniformly across all valid grid positions.
        - Actions a are sampled uniformly from {0, 1} with equal probability.
        - Next states sn advance to the next row with unbiased next column selection (no rightward bias).
        - Terminal states (row == size - 1) terminate with done = 1.0.
        - Reward r is uniformly optimistic everywhere: r ~ N(prior_reward_mean, sigma^2) with NO
          special-cased columns, no oracle knowledge of the chest location, and no directional bias.
        """
        s = np.zeros((count, self.in_dim), dtype=np.float32)
        sn = np.zeros((count, self.in_dim), dtype=np.float32)
        a = self.rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)

        # Uniformly optimistic reward everywhere (no special-cased columns or coordinates)
        r = self.rng.normal(self.prior_reward_mean, self.prior_reward_std, size=count).astype(np.float32)

        for i in range(count):
            row = self.rng.randint(0, self.size)
            col = self.rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                # Unbiased next column (no preference for right vs left)
                next_col = self.rng.randint(0, next_row + 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done

    def sample_posterior_measure(self, batch_size: int = 32):
        """Draw 1 stick-breaking sample from Dirichlet Process posterior conditioned on a candidate mini-batch."""
        n_total = self.replay_size
        B = min(n_total, self.candidate_batch_size)
        total_mass = self.alpha + B

        V = self.rng.beta(1.0, max(total_mass, 1e-3), size=batch_size).astype(np.float32)
        V[-1] = 1.0
        cum_prod = np.cumprod(1.0 - V)
        q = np.empty_like(V)
        q[0] = V[0]
        q[1:] = V[1:] * cum_prod[:-1]
        q_sum = q.sum()
        q = q / q_sum if q_sum > 0 else np.full(batch_size, 1.0 / batch_size, dtype=np.float32)

        prob_emp = B / total_mass if total_mass > 0 else 0.0
        is_emp = (self.rng.rand(batch_size) < prob_emp) if n_total > 0 else np.zeros(batch_size, dtype=bool)
        k_emp = int(np.sum(is_emp))
        k_syn = batch_size - k_emp

        atom_s = np.zeros((batch_size, self.in_dim), dtype=np.float32)
        atom_a = np.zeros(batch_size, dtype=np.int64)
        atom_r = np.zeros(batch_size, dtype=np.float32)
        atom_sn = np.zeros((batch_size, self.in_dim), dtype=np.float32)
        atom_d = np.zeros(batch_size, dtype=np.float32)

        if k_emp > 0:
            # 1. Sample candidate mini-batch of size B from replay buffer WITH replacement (fast randint)
            cand_batch = self.rng.randint(0, n_total, size=B)
            # 2. Draw empirical atoms from this candidate batch WITH replacement
            sample_in_cand = self.rng.randint(0, B, size=k_emp)
            idx = cand_batch[sample_in_cand]
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

    def presample_episodic_batches(self):
        """Vectorized pre-sampling of all K stick-breaking measures conditioned on candidate batch at t=0."""
        K = self.K_updates
        B_sub = self.batch_size
        total_atoms = K * B_sub

        n_total = self.replay_size
        B = min(n_total, self.candidate_batch_size)
        total_mass = self.alpha + B

        # 1. Vectorized stick-breaking weights GEM(alpha + B) of shape (K, B_sub)
        V = self.rng.beta(1.0, max(total_mass, 1e-3), size=(K, B_sub)).astype(np.float32)
        V[:, -1] = 1.0
        cum_prod = np.cumprod(1.0 - V, axis=1)
        Q = np.empty_like(V)
        Q[:, 0] = V[:, 0]
        Q[:, 1:] = V[:, 1:] * cum_prod[:, :-1]
        Q_sum = Q.sum(axis=1, keepdims=True)
        Q = Q / Q_sum

        # 2. Vectorized transition atom assembly strictly from candidate batch
        prob_emp = B / total_mass if total_mass > 0 else 0.0
        is_emp = (self.rng.rand(total_atoms) < prob_emp) if n_total > 0 else np.zeros(total_atoms, dtype=bool)
        k_emp = int(np.sum(is_emp))
        k_syn = total_atoms - k_emp

        atom_s = np.zeros((total_atoms, self.in_dim), dtype=np.float32)
        atom_a = np.zeros(total_atoms, dtype=np.int64)
        atom_r = np.zeros(total_atoms, dtype=np.float32)
        atom_sn = np.zeros((total_atoms, self.in_dim), dtype=np.float32)
        atom_d = np.zeros(total_atoms, dtype=np.float32)

        if k_emp > 0:
            # Subsample candidate mini-batch from replay WITH replacement (fast randint)
            cand_batch = self.rng.randint(0, n_total, size=B)
            sample_in_cand = self.rng.randint(0, B, size=k_emp)
            idx = cand_batch[sample_in_cand]
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

        # Reshape Q into flat (K*B_sub,) and convert to tensors
        self.presampled_s = torch.from_numpy(atom_s)
        self.presampled_a = torch.from_numpy(atom_a).unsqueeze(1)
        self.presampled_r = torch.from_numpy(atom_r)
        self.presampled_sn = torch.from_numpy(atom_sn)
        self.presampled_d = torch.from_numpy(atom_d)
        self.presampled_q = torch.from_numpy(Q.reshape(-1))

    def start_episode(self):
        """Sample episode Thompson exploration hypothesis and perform Target Warm-Start."""
        self.episode_bias = self.rng.randn(2).astype(np.float32) * 0.15

        if self.warm_start_target and self.replay_size >= self.batch_size:
            self.target_net.train()
            for _ in range(5):
                atom_s, atom_a, atom_r, atom_sn, atom_d, q = self.sample_posterior_measure(batch_size=self.batch_size)
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

        # Reset episode buffer
        self.episode_buffer = []

        # Pre-sample all random measures for the episode (if in batched mode)
        if self.presample_and_batch:
            self.presample_episodic_batches()

    def select_action(self, obs: np.ndarray) -> int:
        """Select action using current Q-network (pure value network or with Thompson bias)."""
        self.q_net.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(obs.flatten()).float().unsqueeze(0)
            q_vals = self.q_net(s_t).squeeze(0).numpy()
            if self.use_thompson_bias:
                decay = max(0.01, 500.0 / (500.0 + self.replay_size))
                return int(np.argmax(q_vals + decay * self.episode_bias))
            else:
                return int(np.argmax(q_vals))

    def step_update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Record step."""
        s_flat = np.ascontiguousarray(state, dtype=np.float32).reshape(-1)
        sn_flat = np.ascontiguousarray(next_state, dtype=np.float32).reshape(-1)
        self.total_steps += 1

        if self.presample_and_batch:
            # Append to episode temporary buffer (zero sampling overhead during interaction!)
            self.episode_buffer.append((s_flat, action, reward, sn_flat, done))
        else:
            # Traditional sequential mode: store immediately
            self._push_to_replay(s_flat, action, reward, sn_flat, done)
            if self.sgd_per_episode == 1:
                if done and self.replay_size >= self.batch_size:
                    self._train_step_sequential()
            else:
                if self.replay_size >= self.batch_size and self.total_steps % self.sgd_period == 0:
                    self._train_step_sequential()

    def _push_to_replay(self, s, a, r, sn, done):
        self.replay_s[self.replay_idx] = s
        self.replay_a[self.replay_idx] = a
        self.replay_r[self.replay_idx] = r
        self.replay_sn[self.replay_idx] = sn
        self.replay_d[self.replay_idx] = float(done)
        self.replay_idx = (self.replay_idx + 1) % self.capacity
        if self.replay_size < self.capacity:
            self.replay_size += 1

    def _train_step_sequential(self):
        """Standard sequential update."""
        atom_s, atom_a, atom_r, atom_sn, atom_d, q = self.sample_posterior_measure(self.batch_size)
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

        # Step-wise Polyak update only if target_update_freq == 1
        if self.target_update_freq == 1:
            for tgt_p, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tgt_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * tgt_p.data)

    def end_episode(self):
        """End of episode: flush transitions, execute batched update, and periodic target update."""
        self.episode_count += 1

        if self.presample_and_batch:
            # 1. Flush episode transitions into replay buffer
            for s, a, r, sn, done in self.episode_buffer:
                self._push_to_replay(s, a, r, sn, done)
            self.episode_buffer = []

            # 2. Execute K updates on all K pre-sampled measures
            if self.replay_size >= self.batch_size and self.presampled_s is not None:
                B = self.batch_size
                self.q_net.train()
                for k in range(self.K_updates):
                    s_t = self.presampled_s[k * B : (k + 1) * B]
                    a_t = self.presampled_a[k * B : (k + 1) * B]
                    r_t = self.presampled_r[k * B : (k + 1) * B]
                    sn_t = self.presampled_sn[k * B : (k + 1) * B]
                    d_t = self.presampled_d[k * B : (k + 1) * B]
                    q_t = self.presampled_q[k * B : (k + 1) * B]

                    with torch.no_grad():
                        next_q = self.target_net(sn_t)
                        max_next_q, _ = torch.max(next_q, dim=1)
                        targets = r_t + (1.0 - d_t) * self.gamma * max_next_q

                    self.optimizer.zero_grad()
                    q_vals = self.q_net(s_t)
                    chosen_q = q_vals.gather(1, a_t).squeeze(1)
                    loss = torch.sum(q_t * (chosen_q - targets) ** 2)

                    loss.backward()
                    self.optimizer.step()

        # 3. Periodic Target Network update
        if self.replay_size >= self.batch_size and (self.episode_count % self.target_update_freq == 0):
            for tgt_p, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tgt_p.data.copy_(self.target_tau * p.data + (1.0 - self.target_tau) * tgt_p.data)
