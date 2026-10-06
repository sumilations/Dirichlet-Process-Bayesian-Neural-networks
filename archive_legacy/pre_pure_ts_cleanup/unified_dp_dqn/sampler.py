"""Dirichlet Process Posterior Samplers for Unified DP-DQN."""

from typing import Optional, Tuple
import numpy as np
import torch

from .networks import ReplayBuffer
from .base_measures import BaseMeasure


class VashishthaMaillardSampler:
    """Fixed-Budget Posterior Sampler based on Vashishtha & Maillard (2025) Eq. (288).

    Guarantees:
    - Bounded minibatch size: N_emp empirical transitions + K_prior base measure transitions.
    - Analytical total prior probability: W_prior ~ Beta(alpha + 1, N_stat).
    - As N_stat -> inf (via cumulative TD information gain), W_prior -> 0 with sublinear contraction.
    """

    def __init__(
        self,
        alpha: float,
        batch_size: int,
        base_measure: BaseMeasure,
        rng: np.random.RandomState,
        vm_prior_multiplier: float = 10.0,
        w_min: float = 0.0,
    ):
        self.alpha = float(alpha)
        self.batch_size = int(batch_size)
        self.base_measure = base_measure
        self.rng = rng
        self.vm_prior_multiplier = float(vm_prior_multiplier)
        self.w_min = float(w_min)

    def sample(
        self,
        replay: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        n_stat_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_emp_avail = len(replay)
        K_prior = max(8, int(self.vm_prior_multiplier * self.alpha))

        # Replay empty: sample entirely from base measure
        if n_emp_avail == 0:
            syn_s, syn_a, syn_r, syn_sn, syn_d = self.base_measure.sample(K_prior, self.rng)
            if self.alpha <= 1e-8:
                q_prior = np.full(K_prior, 1.0 / K_prior, dtype=np.float32)
            else:
                g_prior = self.rng.gamma(self.alpha / K_prior, 1.0, size=K_prior).astype(np.float32)
                q_sum = g_prior.sum()
                q_prior = g_prior / q_sum if q_sum > 0 else np.full(K_prior, 1.0 / K_prior, dtype=np.float32)

            return (
                torch.from_numpy(syn_s).to(device),
                torch.from_numpy(syn_a).to(device),
                torch.from_numpy(syn_r).to(device),
                torch.from_numpy(syn_sn).to(device),
                torch.from_numpy(syn_d).to(device),
                torch.from_numpy(q_prior).to(device),
            )

        N_emp = min(n_emp_avail, self.batch_size)

        # 1. Sample N_emp empirical transitions
        emp_s, emp_a, emp_r, emp_sn, emp_d = replay.sample(N_emp, self.rng)

        # 2. Sample K_prior synthetic transitions from base measure
        syn_s, syn_a, syn_r, syn_sn, syn_d = self.base_measure.sample(K_prior, self.rng)

        # 3. Exact Vashishtha & Maillard (2025) Eq. (288) Recursive Stick-Breaking:
        if self.alpha <= 1e-8:
            # Exact Bayesian Bootstrap Limit (Rubin, 1981): alpha -> 0 (Zero Prior Mass)
            g_emp = self.rng.standard_exponential(N_emp).astype(np.float32)
            total_g = g_emp.sum()
            w_emp = g_emp / total_g if total_g > 0 else np.full(N_emp, 1.0 / N_emp, dtype=np.float32)
            W_prior = 0.0
            w_prior = np.zeros(K_prior, dtype=np.float32)
        else:
            if n_stat_override is not None:
                # TD Information Gain override: N_stat = max(N_emp, n_stat_override)
                N_stat = max(float(N_emp), float(n_stat_override))
                W_prior_raw = float(self.rng.beta(self.alpha + 1.0, float(N_stat)))
                W_prior = max(self.w_min, W_prior_raw)

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

                suffix_U = np.ones(N_emp + 1, dtype=np.float32)
                suffix_U[:-1] = np.cumprod(U_emp[::-1])[::-1]

                w_emp_raw = np.zeros(N_emp, dtype=np.float32)
                if N_emp > 1:
                    w_emp_raw[:-1] = V_emp[:-1] * suffix_U[1:-1]
                w_emp_raw[-1] = V_emp[-1]
                W_prior_raw = float(suffix_U[0])
                W_prior = max(self.w_min, W_prior_raw)

                emp_sum = w_emp_raw.sum()
                if emp_sum > 0:
                    w_emp = (1.0 - W_prior) * (w_emp_raw / emp_sum)
                else:
                    w_emp = np.full(N_emp, (1.0 - W_prior) / N_emp, dtype=np.float32)

            # Prior atoms stick-breaking (Ishwaran & Zarepour, 2002 / Sethuraman)
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

        # Concatenate empirical and synthetic atoms
        all_s = np.concatenate([emp_s, syn_s], axis=0)
        all_a = np.concatenate([emp_a, syn_a], axis=0)
        all_r = np.concatenate([emp_r, syn_r], axis=0)
        all_sn = np.concatenate([emp_sn, syn_sn], axis=0)
        all_d = np.concatenate([emp_d, syn_d], axis=0)
        all_w = np.concatenate([w_emp, w_prior], axis=0)

        # Final normalization check
        w_total = all_w.sum()
        if w_total > 0:
            all_w = all_w / w_total

        return (
            torch.from_numpy(all_s).to(device),
            torch.from_numpy(all_a).to(device),
            torch.from_numpy(all_r).to(device),
            torch.from_numpy(all_sn).to(device),
            torch.from_numpy(all_d).to(device),
            torch.from_numpy(all_w).to(device),
        )


class DirichletPosteriorSampler:
    """Exact Unified Dirichlet Posterior Sampler (Ferguson, 1973).

    Joint distribution:
        (w_emp, w_prior) ~ Dirichlet(1, ..., 1, alpha/K, ..., alpha/K)
    with optional lower bound floor W_prior >= w_min.

    Properties:
    - Symmetric: every empirical atom has expected weight (1 - W_prior)/N_emp.
    - Symmetric: every prior atom has expected weight W_prior/K_prior.
    - Macro prior mass: W_prior ~ Beta(alpha + 1, N_stat) bounded by w_min.
    """

    def __init__(
        self,
        alpha: float,
        batch_size: int,
        base_measure: BaseMeasure,
        rng: np.random.RandomState,
        vm_prior_multiplier: float = 10.0,
        w_min: float = 0.0,
    ):
        self.alpha = float(alpha)
        self.batch_size = int(batch_size)
        self.base_measure = base_measure
        self.rng = rng
        self.vm_prior_multiplier = float(vm_prior_multiplier)
        self.w_min = float(w_min)

    def sample(
        self,
        replay: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        n_stat_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_emp_avail = len(replay)
        K_prior = max(8, int(self.vm_prior_multiplier * self.alpha))

        if n_emp_avail == 0:
            syn_s, syn_a, syn_r, syn_sn, syn_d = self.base_measure.sample(K_prior, self.rng)
            g_prior = self.rng.gamma(max(self.alpha / K_prior, 1e-4), 1.0, size=K_prior).astype(np.float32)
            q_sum = g_prior.sum()
            q_prior = g_prior / q_sum if q_sum > 0 else np.full(K_prior, 1.0 / K_prior, dtype=np.float32)
            return (
                torch.from_numpy(syn_s).to(device),
                torch.from_numpy(syn_a).to(device),
                torch.from_numpy(syn_r).to(device),
                torch.from_numpy(syn_sn).to(device),
                torch.from_numpy(syn_d).to(device),
                torch.from_numpy(q_prior).to(device),
            )

        N_emp = min(n_emp_avail, self.batch_size)

        # 1. Sample N_emp empirical and K_prior synthetic transitions
        emp_s, emp_a, emp_r, emp_sn, emp_d = replay.sample(N_emp, self.rng)
        syn_s, syn_a, syn_r, syn_sn, syn_d = self.base_measure.sample(K_prior, self.rng)

        # 2. Symmetric Intra-Empirical Dirichlet (Shape = 1.0 => Exponential(1))
        g_emp = self.rng.standard_exponential(N_emp).astype(np.float32)
        s_emp = g_emp.sum()
        norm_emp = g_emp / s_emp if s_emp > 0 else np.full(N_emp, 1.0 / N_emp, dtype=np.float32)

        # 3. Symmetric Intra-Prior Dirichlet (Shape = alpha / K_prior)
        g_prior = self.rng.gamma(max(self.alpha / K_prior, 1e-4), 1.0, size=K_prior).astype(np.float32)
        s_prior = g_prior.sum()
        norm_prior = g_prior / s_prior if s_prior > 0 else np.full(K_prior, 1.0 / K_prior, dtype=np.float32)

        # 4. Macro Mass Balance W_prior with W_min Floor
        if n_stat_override is not None:
            N_stat = max(float(N_emp), float(n_stat_override))
            W_prior_raw = float(self.rng.beta(self.alpha + 1.0, float(N_stat)))
        else:
            W_prior_raw = float(self.rng.beta(self.alpha + 1.0, float(N_emp)))

        W_prior = max(self.w_min, W_prior_raw)
        W_prior = min(W_prior, 1.0 - 1e-6)
        W_emp = 1.0 - W_prior

        w_emp = W_emp * norm_emp
        w_prior = W_prior * norm_prior

        # Combine atoms
        all_s = np.concatenate([emp_s, syn_s], axis=0)
        all_a = np.concatenate([emp_a, syn_a], axis=0)
        all_r = np.concatenate([emp_r, syn_r], axis=0)
        all_sn = np.concatenate([emp_sn, syn_sn], axis=0)
        all_d = np.concatenate([emp_d, syn_d], axis=0)
        all_w = np.concatenate([w_emp, w_prior], axis=0)

        w_total = all_w.sum()
        if w_total > 0:
            all_w = all_w / w_total

        return (
            torch.from_numpy(all_s).to(device),
            torch.from_numpy(all_a).to(device),
            torch.from_numpy(all_r).to(device),
            torch.from_numpy(all_sn).to(device),
            torch.from_numpy(all_d).to(device),
            torch.from_numpy(all_w).to(device),
        )

