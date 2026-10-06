"""Sethuraman Stick-Breaking Posterior Sampler and Replay Buffer for DP-DQN."""

from typing import Tuple, Optional
import numpy as np
import torch
from .base_measures import BaseMeasure


class ReplayBuffer:
    """Pre-allocated contiguous NumPy Replay Buffer for MDP transitions."""

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
        self.total_count = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.total_count += 1
        if self.size < self.capacity:
            self.size += 1

    @property
    def num_positive_rewards(self) -> int:
        """Count how many transitions in the buffer have r > 0."""
        return int(np.sum(self.rewards[:self.size] > 0.0))

    def sample(self, count: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.randint(0, self.size, size=count)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def sample_priority(
        self,
        count: int,
        rng: np.random.RandomState,
        positive_slot: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample transitions, optionally dedicating 1 slot to a rare positive reward."""
        if not positive_slot or self.size == 0:
            return self.sample(count, rng)

        pos_indices = np.where(self.rewards[:self.size] > 0.0)[0]
        if len(pos_indices) > 0 and count > 1:
            lucky_idx = rng.choice(pos_indices, size=1)
            other_idx = rng.randint(0, self.size, size=count - 1)
            idx = np.concatenate([lucky_idx, other_idx])
        else:
            idx = rng.randint(0, self.size, size=count)

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def sample_candidate(self, candidate_size: int, count: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        B = min(self.size, candidate_size)
        cand_idx = rng.randint(0, self.size, size=B)
        sample_in_cand = rng.randint(0, B, size=count)
        idx = cand_idx[sample_in_cand]
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size


class BayesianBetaAlphaPrior:
    """Conjugate Beta Hyper-Prior on Dirichlet Process Synthetic Injection Probability theta.

    Reparameterization:
        theta = P(syn) = alpha / (alpha + B) in (0, 1)
        alpha = B * (theta / (1 - theta))  [Beta-Prime scaled by B]

    Prior:
        theta ~ Beta(a_0, b_0)
        E[theta] = a_0 / (a_0 + b_0)
        E[alpha] = B * (a_0 / (b_0 - 1))

    Conjugate Bayesian Posterior Update:
        b_t = b_0 + (n_transitions / (B * evidence_scale))
        theta | D_t ~ Beta(a_0, b_t)
        alpha_sample = B * (theta_sample / (1 - theta_sample))
    """

    def __init__(
        self,
        a_0: float = 1.0,
        b_0: Optional[float] = None,
        initial_alpha: float = 5.0,
        candidate_batch_size: int = 256,
        evidence_scale: float = 1.0,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.a_0 = float(a_0)
        self.candidate_batch_size = int(candidate_batch_size)
        self.evidence_scale = max(float(evidence_scale), 1e-4)
        self.rng = rng if rng is not None else np.random.RandomState(42)

        if b_0 is not None and b_0 > 0:
            self.b_0 = float(b_0)
        else:
            # Auto-align b_0 so that prior E[alpha] matches initial_alpha
            # alpha_0 = B * (a_0 / b_0) => b_0 = (B / alpha_0) * a_0
            self.b_0 = (float(self.candidate_batch_size) / max(float(initial_alpha), 1e-3)) * self.a_0

        self.last_theta = self.a_0 / (self.a_0 + self.b_0)
        self.last_alpha = float(initial_alpha)

    def sample_alpha(self, total_transitions: int) -> Tuple[float, float]:
        """Sample posterior theta ~ Beta(a_0, b_t) and compute alpha = B * (theta / (1 - theta)).

        Returns:
            (alpha_sample, theta_sample)
        """
        # Evidence accumulation directly in transition counts (no B division)
        b_t = self.b_0 + (float(total_transitions) / self.evidence_scale)

        # Sample theta from conjugate Beta posterior
        theta = self.rng.beta(self.a_0, b_t)
        theta = float(np.clip(theta, 1e-6, 1.0 - 1e-6))

        # Compute induced alpha
        alpha = float(self.candidate_batch_size * (theta / (1.0 - theta)))

        self.last_theta = theta
        self.last_alpha = alpha
        return alpha, theta


class SethuramanStickBreakingSampler:
    """Draws posterior samples F ~ DP(alpha + B, (alpha F_0 + sum delta_i) / (alpha + B))

    Uses Sethuraman's (1994) stick-breaking construction:
        V_k ~ Beta(1, alpha + B)
        q_k = V_k * prod_{j < k} (1 - V_j)
    """

    def __init__(
        self,
        alpha: float,
        candidate_batch_size: int,
        batch_size: int,
        base_measure: BaseMeasure,
        rng: np.random.RandomState,
        contraction_C: Optional[float] = None,
        use_buffer_size_denominator: bool = False,
        trajectory_C: Optional[float] = None,
        direct_replay_sample: bool = False,
        use_bayesian_alpha: bool = False,
        alpha_prior_a: float = 1.0,
        alpha_prior_b: Optional[float] = None,
        alpha_evidence_scale: float = 1.0,
    ):
        self.alpha = float(alpha)
        self.candidate_batch_size = int(candidate_batch_size)
        self.batch_size = int(batch_size)
        self.base_measure = base_measure
        self.rng = rng
        self.contraction_C = float(contraction_C) if contraction_C is not None else None
        self.trajectory_C = float(trajectory_C) if trajectory_C is not None else None
        self.use_buffer_size_denominator = bool(use_buffer_size_denominator)
        self.direct_replay_sample = bool(direct_replay_sample)
        self.use_bayesian_alpha = bool(use_bayesian_alpha)

        if self.use_bayesian_alpha:
            self.bayesian_alpha_prior = BayesianBetaAlphaPrior(
                a_0=alpha_prior_a,
                b_0=alpha_prior_b,
                initial_alpha=alpha,
                candidate_batch_size=candidate_batch_size,
                evidence_scale=alpha_evidence_scale,
                rng=rng,
            )
        else:
            self.bayesian_alpha_prior = None

        self.last_alpha = self.alpha
        self.last_prob_syn = 0.0
        self.last_n_emp = 0

    def sample(
        self,
        replay: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        n_stat_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (states, actions, rewards, next_states, dones, q_weights) as PyTorch tensors."""
        n_emp = len(replay)
        B = min(n_emp, self.candidate_batch_size)
        self.last_n_emp = B

        # Determine effective alpha and synthetic injection probability
        if self.bayesian_alpha_prior is not None:
            effective_alpha, prob_syn = self.bayesian_alpha_prior.sample_alpha(replay.total_count)
            total_mass = effective_alpha + B
        elif self.trajectory_C is not None and self.trajectory_C > 0:
            effective_alpha = self.alpha
            total_mass = self.alpha + B
            effective_trajectories = float(replay.total_count) / float(self.trajectory_C)
            prob_syn = self.alpha / (self.alpha + effective_trajectories) if (self.alpha + effective_trajectories) > 0 else 0.0
        elif self.use_buffer_size_denominator:
            effective_alpha = self.alpha
            total_mass = self.alpha + B
            prob_syn = self.alpha / (self.alpha + float(n_emp)) if (self.alpha + n_emp) > 0 else 0.0
        elif self.contraction_C is not None and self.contraction_C > 0:
            effective_alpha = self.alpha / (1.0 + float(replay.total_count) / float(self.contraction_C))
            total_mass = effective_alpha + B
            prob_syn = effective_alpha / total_mass if total_mass > 0 else 0.0
        else:
            effective_alpha = self.alpha
            total_mass = self.alpha + B
            prob_syn = self.alpha / total_mass if total_mass > 0 else 0.0

        self.last_alpha = effective_alpha
        self.last_prob_syn = prob_syn

        # 1. Compute Sethuraman stick-breaking categorical weights q_k ~ GEM(alpha + B)
        V = self.rng.beta(1.0, max(total_mass, 1e-3), size=self.batch_size).astype(np.float32)
        V[-1] = 1.0  # Truncate final stick fragment
        cum_prod = np.cumprod(1.0 - V)
        q = np.empty_like(V)
        q[0] = V[0]
        q[1:] = V[1:] * cum_prod[:-1]
        q_sum = q.sum()
        q = q / q_sum if q_sum > 0 else np.full(self.batch_size, 1.0 / self.batch_size, dtype=np.float32)

        prob_emp = 1.0 - prob_syn
        is_emp = (self.rng.rand(self.batch_size) < prob_emp) if n_emp > 0 else np.zeros(self.batch_size, dtype=bool)
        k_emp = int(np.sum(is_emp))
        k_syn = self.batch_size - k_emp

        state_dim = self.base_measure.state_dim
        atom_s = np.zeros((self.batch_size, state_dim), dtype=np.float32)
        atom_a = np.zeros(self.batch_size, dtype=np.int64)
        atom_r = np.zeros(self.batch_size, dtype=np.float32)
        atom_sn = np.zeros((self.batch_size, state_dim), dtype=np.float32)
        atom_d = np.zeros(self.batch_size, dtype=np.float32)

        # Fill empirical transitions
        if k_emp > 0:
            if getattr(self, "priority_positive_slot", False):
                emp_s, emp_a, emp_r, emp_sn, emp_d = replay.sample_priority(k_emp, self.rng, positive_slot=True)
            elif self.direct_replay_sample:
                emp_s, emp_a, emp_r, emp_sn, emp_d = replay.sample(k_emp, self.rng)
            else:
                emp_s, emp_a, emp_r, emp_sn, emp_d = replay.sample_candidate(self.candidate_batch_size, k_emp, self.rng)
            atom_s[is_emp] = emp_s
            atom_a[is_emp] = emp_a
            atom_r[is_emp] = emp_r
            atom_sn[is_emp] = emp_sn
            atom_d[is_emp] = emp_d

        # Fill synthetic transitions from base measure F_0
        if k_syn > 0:
            syn_s, syn_a, syn_r, syn_sn, syn_d = self.base_measure.sample(k_syn, self.rng)
            atom_s[~is_emp] = syn_s
            atom_a[~is_emp] = syn_a
            atom_r[~is_emp] = syn_r
            atom_sn[~is_emp] = syn_sn
            atom_d[~is_emp] = syn_d

        # Convert to PyTorch tensors
        s_t = torch.from_numpy(atom_s).to(device)
        a_t = torch.from_numpy(atom_a).to(device)
        r_t = torch.from_numpy(atom_r).to(device)
        sn_t = torch.from_numpy(atom_sn).to(device)
        d_t = torch.from_numpy(atom_d).to(device)
        q_t = torch.from_numpy(q).to(device)

        return s_t, a_t, r_t, sn_t, d_t, q_t


class FixedBudgetVashishthaMaillardSampler:
    """Draws posterior samples from DP according to Eq. (288) of Vashishtha & Maillard (2025):

        Q_N = sum_{i=1}^{N} [V_i prod_{j=i+1}^N (1 - V_j)] delta_{X_i} + [prod_{i=1}^N (1 - V_i)] Q_0

    with:
        - N_emp = batch_size empirical observations from replay buffer.
        - K_prior = max(8, int(vm_prior_multiplier * alpha)) prior observations from F_0.
        - Q_0 = sum_{k=1}^{K_prior} q_k^{(0)} delta_{Z_k} with Z_k ~ F_0 and q_k^{(0)} ~ GEM(alpha).
        - Exact analytical stick weights summing to 1.0.
    """

    def __init__(
        self,
        alpha: float,
        batch_size: int,
        base_measure: BaseMeasure,
        rng: np.random.RandomState,
        vm_prior_multiplier: float = 10.0,
        priority_positive_slot: bool = False,
        vm_replay_scaled_n: bool = False,
        vm_scale_post_discovery_only: bool = False,
    ):
        self.alpha = float(alpha)
        self.batch_size = int(batch_size)
        self.base_measure = base_measure
        self.rng = rng
        self.vm_prior_multiplier = float(vm_prior_multiplier)
        self.priority_positive_slot = bool(priority_positive_slot)
        self.vm_replay_scaled_n = bool(vm_replay_scaled_n)
        self.vm_scale_post_discovery_only = bool(vm_scale_post_discovery_only)
        self.last_alpha = self.alpha
        self.last_prob_syn = 0.0
        self.last_n_emp = 0

    def sample(
        self,
        replay: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        n_stat_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_emp_avail = len(replay)
        K_prior = max(8, int(self.vm_prior_multiplier * self.alpha))

        if n_emp_avail == 0:
            self.last_n_emp = 0
            # Replay empty: draw everything from base measure Q_0
            syn_s, syn_a, syn_r, syn_sn, syn_d = self.base_measure.sample(K_prior, self.rng)
            V_prior = self.rng.beta(1.0, max(self.alpha, 1e-3), size=K_prior).astype(np.float32)
            U_prior = 1.0 - V_prior
            q_prior = np.zeros(K_prior, dtype=np.float32)
            q_prior[0] = V_prior[0]
            if K_prior > 1:
                prefix_U = np.cumprod(U_prior[:-1])
                q_prior[1:-1] = V_prior[1:-1] * prefix_U[:-1]
                q_prior[-1] = prefix_U[-1]
            q_prior_sum = q_prior.sum()
            if q_prior_sum > 0:
                q_prior = q_prior / q_prior_sum

            return (
                torch.from_numpy(syn_s).to(device),
                torch.from_numpy(syn_a).to(device),
                torch.from_numpy(syn_r).to(device),
                torch.from_numpy(syn_sn).to(device),
                torch.from_numpy(syn_d).to(device),
                torch.from_numpy(q_prior).to(device),
            )

        N_emp = min(n_emp_avail, self.batch_size)
        self.last_n_emp = N_emp

        # 1. Sample N_emp empirical transitions
        emp_s, emp_a, emp_r, emp_sn, emp_d = replay.sample_priority(
            N_emp, self.rng, positive_slot=self.priority_positive_slot
        )

        # 2. Sample K_prior synthetic transitions from base measure F_0
        syn_s, syn_a, syn_r, syn_sn, syn_d = self.base_measure.sample(K_prior, self.rng)

        # 3. Compute Vashishtha & Maillard (2025) recursive stick-breaking weights (Eq. 288)
        if n_stat_override is not None:
            # External statistical count override (e.g. TD Information Gain)
            N_stat = max(float(N_emp), float(n_stat_override))
            W_prior = float(self.rng.beta(self.alpha + 1.0, float(N_stat)))

            i_arr = np.arange(1, N_emp + 1, dtype=np.float32)
            V_emp = self.rng.beta(1.0, self.alpha + i_arr).astype(np.float32)
            U_emp = 1.0 - V_emp
            suffix_U = np.ones(N_emp + 1, dtype=np.float32)
            suffix_U[:-1] = np.cumprod(U_emp[::-1])[::-1]

            w_emp_raw = np.zeros(N_emp, dtype=np.float32)
            if N_emp > 1:
                w_emp_raw[:-1] = V_emp[:-1] * suffix_U[1:-1]
            w_emp_raw[-1] = V_emp[-1]

            emp_sum = w_emp_raw.sum()
            if emp_sum > 0:
                w_emp = (1.0 - W_prior) * (w_emp_raw / emp_sum)
            else:
                w_emp = np.full(N_emp, (1.0 - W_prior) / N_emp, dtype=np.float32)
        else:
            use_scaled_n = self.vm_replay_scaled_n
            if use_scaled_n and self.vm_scale_post_discovery_only:
                if replay.num_positive_rewards == 0:
                    use_scaled_n = False

            if use_scaled_n:
                # Statistical sample size N_stat scales with full replay buffer
                N_stat = max(N_emp, n_emp_avail)
                # Analytically: prod_{i=1}^{N_stat} (1 - V_i) ~ Beta(alpha + 1, N_stat)
                W_prior = float(self.rng.beta(self.alpha + 1.0, float(N_stat)))

                i_arr = np.arange(1, N_emp + 1, dtype=np.float32)
                V_emp = self.rng.beta(1.0, self.alpha + i_arr).astype(np.float32)
                U_emp = 1.0 - V_emp
                suffix_U = np.ones(N_emp + 1, dtype=np.float32)
                suffix_U[:-1] = np.cumprod(U_emp[::-1])[::-1]

                w_emp_raw = np.zeros(N_emp, dtype=np.float32)
                if N_emp > 1:
                    w_emp_raw[:-1] = V_emp[:-1] * suffix_U[1:-1]
                w_emp_raw[-1] = V_emp[-1]

                emp_sum = w_emp_raw.sum()
                if emp_sum > 0:
                    w_emp = (1.0 - W_prior) * (w_emp_raw / emp_sum)
                else:
                    w_emp = np.full(N_emp, (1.0 - W_prior) / N_emp, dtype=np.float32)
            else:
                # Fixed-budget V&M (Eq. 288 on N_emp)
                i_arr = np.arange(1, N_emp + 1, dtype=np.float32)
                V_emp = self.rng.beta(1.0, self.alpha + i_arr).astype(np.float32)
                U_emp = 1.0 - V_emp

                # Suffix products of U_emp: suffix_U[i] = prod_{j=i}^N U_emp[j]
                suffix_U = np.ones(N_emp + 1, dtype=np.float32)
                suffix_U[:-1] = np.cumprod(U_emp[::-1])[::-1]

                w_emp = np.zeros(N_emp, dtype=np.float32)
                if N_emp > 1:
                    w_emp[:-1] = V_emp[:-1] * suffix_U[1:-1]
                w_emp[-1] = V_emp[-1]
                W_prior = float(suffix_U[0])

        # 4. Truncated DP prior Q_0 stick-breaking weights
        V_prior = self.rng.beta(1.0, max(self.alpha, 1e-3), size=K_prior).astype(np.float32)
        U_prior = 1.0 - V_prior
        q_prior = np.zeros(K_prior, dtype=np.float32)
        q_prior[0] = V_prior[0]
        if K_prior > 1:
            prefix_U = np.cumprod(U_prior[:-1])
            q_prior[1:-1] = V_prior[1:-1] * prefix_U[:-1]
            q_prior[-1] = prefix_U[-1]
        q_prior_sum = q_prior.sum()
        if q_prior_sum > 0:
            q_prior = q_prior / q_prior_sum

        w_prior = W_prior * q_prior

        # Combine empirical and prior atoms
        total_weights = np.concatenate([w_emp, w_prior])
        total_weights_sum = total_weights.sum()
        if total_weights_sum > 0:
            total_weights = total_weights / total_weights_sum

        all_s = np.concatenate([emp_s, syn_s], axis=0)
        all_a = np.concatenate([emp_a, syn_a], axis=0)
        all_r = np.concatenate([emp_r, syn_r], axis=0)
        all_sn = np.concatenate([emp_sn, syn_sn], axis=0)
        all_d = np.concatenate([emp_d, syn_d], axis=0)

        self.last_alpha = self.alpha
        self.last_prob_syn = W_prior

        return (
            torch.from_numpy(all_s).to(device),
            torch.from_numpy(all_a).to(device),
            torch.from_numpy(all_r).to(device),
            torch.from_numpy(all_sn).to(device),
            torch.from_numpy(all_d).to(device),
            torch.from_numpy(total_weights).to(device),
        )
