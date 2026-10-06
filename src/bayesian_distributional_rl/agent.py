"""
Bayesian Distributional Reinforcement Learning with Dirichlet Processes (Algorithm 5)
Reference: Vashishtha PhD Thesis, Section 4.5
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from .dp_posterior import (
    BaseMeasure,
    sample_dp_prior,
    sample_dp_posterior,
    evaluate_functional
)


class BayesianDistributionalRL_DP:
    """
    Algorithm 5: Bayesian Distributional RL with Dirichlet Processes for Tabular MDPs.
    
    Learns action-value distribution Z(s, a) ~ DP(alpha(s, a), F_bar(s, a))
    governed by the Bayesian distributional equation (Eq. 4.15):
        Z(s, a) =^d V * delta_{r(s,a)} + (1 - V) * Z(s', a')
    where V ~ Beta(1, alpha(s, a)).
    """
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha_0: float = 5.0,
        base_measure: Optional[BaseMeasure] = None,
        phi_1: str = "mean",       # Action selection functional (e.g. 'mean' for Thompson sampling, 'quantile', 'cvar')
        phi_2: str = "mean",       # Target selection functional (e.g. 'mean' or 'greedy')
        phi_1_kwargs: Optional[Dict] = None,
        phi_2_kwargs: Optional[Dict] = None,
        synthesis_mode: str = "direct_union", # 'direct_union' (exact Alg 5) or 'distributional_bellman'
        k_trunc_prior: int = 50,
        max_history_len: int = 200,
        seed: Optional[int] = None
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha_0 = float(alpha_0)
        self.base_measure = base_measure if base_measure is not None else BaseMeasure("gaussian", loc=1.0, scale=0.5)
        
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.phi_1_kwargs = phi_1_kwargs or {}
        self.phi_2_kwargs = phi_2_kwargs or {}
        self.synthesis_mode = synthesis_mode
        self.k_trunc_prior = k_trunc_prior
        self.max_history_len = max_history_len
        
        self.rng = np.random.default_rng(seed)
        
        # Reward histories H(s, a) initialized to empty for all s, a pairs (Line 2)
        self.H: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        
        # Metrics and diagnostics
        self.total_steps = 0
        self.episodes = 0
        self.contraction_history: List[Dict] = []

    def get_concentration(self, s: int, a: int) -> float:
        """Eq. 4.16: alpha(s, a) = alpha_0(s, a) + length(H(s, a))."""
        return self.alpha_0 + len(self.H[(s, a)])

    def get_emergent_discount(self, s: int, a: int) -> float:
        """
        Expected natural emergent discount factor:
        E[1 - V] = alpha(s, a) / (1 + alpha(s, a)) where V ~ Beta(1, alpha(s, a)).
        """
        alpha = self.get_concentration(s, a)
        return float(alpha / (1.0 + alpha))

    def sample_Z(self, s: int, a: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a discrete random measure Z(s, a) ~ DP(alpha(s, a), F_bar(s, a))."""
        history = self.H[(s, a)]
        return sample_dp_posterior(
            history=history,
            alpha_0=self.alpha_0,
            base_measure=self.base_measure,
            k_trunc_prior=self.k_trunc_prior,
            max_history_len=self.max_history_len,
            rng=self.rng
        )

    def select_action(self, s: int, functional: Optional[str] = None, **kwargs) -> int:
        """
        Line 6: Select action a_t = argmax_a phi_1(Z(s_t, a) ~ DP(alpha(s_t, a), F_bar(s_t, a))).
        """
        func = functional or self.phi_1
        kw = {**self.phi_1_kwargs, **kwargs}
        
        q_samples = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            weights, atoms = self.sample_Z(s, a)
            q_samples[a] = evaluate_functional(weights, atoms, functional=func, **kw)
            
        # Tie-breaking with random choice among max
        max_val = np.max(q_samples)
        best_actions = np.where(np.isclose(q_samples, max_val, atol=1e-8))[0]
        return int(self.rng.choice(best_actions))

    def select_target_action(self, s_next: int, functional: Optional[str] = None, **kwargs) -> int:
        """
        Line 8: a' = argmax_a phi_2(Z(s_{t+1}, a) ~ DP(alpha(s_{t+1}, a), F_bar(s_{t+1}, a))).
        """
        func = functional or self.phi_2
        kw = {**self.phi_2_kwargs, **kwargs}
        
        q_samples = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            weights, atoms = self.sample_Z(s_next, a)
            q_samples[a] = evaluate_functional(weights, atoms, functional=func, **kw)
            
        max_val = np.max(q_samples)
        best_actions = np.where(np.isclose(q_samples, max_val, atol=1e-8))[0]
        return int(self.rng.choice(best_actions))

    def update_step(
        self,
        s: int,
        a: int,
        r: float,
        s_next: Optional[int],
        done: bool
    ) -> Optional[int]:
        """
        Execute lines 7-9 of Algorithm 5 for a single transition:
        - Record reward r_t
        - Select target action a' in s_{t+1}
        - Synthesize histories H(s_t, a_t) and H(s_{t+1}, a')
        """
        self.total_steps += 1
        
        if done or s_next is None:
            # Terminal state: only immediate reward is received
            self.H[(s, a)].append(float(r))
            if len(self.H[(s, a)]) > self.max_history_len:
                self.H[(s, a)].pop(0)
            return None

        # Line 8: a' = argmax_a phi_2(Z(s_{t+1}, a) ~ DP(...))
        a_next = self.select_target_action(s_next)
        
        # Line 9: Synthesize histories
        if self.synthesis_mode == "direct_union":
            # Exact Algorithm 5: Immediate reward r_t recorded, then histories synthesized
            new_elements = [float(r)]
            
            # Incorporate successor history H(s_{t+1}, a') to propagate downstream returns
            succ_hist = self.H[(s_next, a_next)]
            if len(succ_hist) > 0:
                # Subsample up to 5 atoms from successor to balance local and future rewards
                sample_size = min(5, len(succ_hist))
                sampled_succ = self.rng.choice(succ_hist, size=sample_size, replace=False).tolist()
                new_elements.extend(sampled_succ)
                
            self.H[(s, a)].extend(new_elements)
            
            # Truncate if exceeds max_history_len
            if len(self.H[(s, a)]) > self.max_history_len:
                excess = len(self.H[(s, a)]) - self.max_history_len
                del self.H[(s, a)][:excess]
                
        elif self.synthesis_mode == "distributional_bellman":
            # Eq. 4.15: Z(s, a) =^d V * delta_r + (1 - V) * Z(s', a')
            alpha = self.get_concentration(s, a)
            V = self.rng.beta(1.0, alpha)
            
            # Sample a return from successor distribution Z(s_next, a_next)
            w_next, a_next_atoms = self.sample_Z(s_next, a_next)
            r_next_sample = self.rng.choice(a_next_atoms, p=w_next)
            
            # Form mixture return
            return_sample = V * r + (1.0 - V) * r_next_sample
            self.H[(s, a)].append(float(return_sample))
            
            if len(self.H[(s, a)]) > self.max_history_len:
                self.H[(s, a)].pop(0)

        return a_next

    def update_trajectory(self, trajectory: List[Tuple[int, int, float]]):
        """
        Synthesize full episodic trajectory via recursive distributional equation (Eq. 4.15):
            R_t =^d V_t * r_t + (1 - V_t) * R_{t+1}
        where V_t ~ Beta(1, alpha(s_t, a_t)).
        Propagates returns backward along the visited path.
        """
        R_curr = 0.0
        for s, a, r in reversed(trajectory):
            self.total_steps += 1
            alpha = self.get_concentration(s, a)
            V = self.rng.beta(1.0, alpha)
            R_curr = V * r + (1.0 - V) * R_curr
            self.H[(s, a)].append(float(R_curr))
            if len(self.H[(s, a)]) > self.max_history_len:
                self.H[(s, a)].pop(0)

    def get_q_values(self, functional: str = "mean", **kwargs) -> np.ndarray:
        """Evaluate deterministic/expected Q-table for all (s, a)."""
        q_table = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                weights, atoms = self.sample_Z(s, a)
                q_table[s, a] = evaluate_functional(weights, atoms, functional=functional, **kwargs)
        return q_table

    def compute_posterior_variances(self) -> np.ndarray:
        """Compute the epistemic variance of Z(s, a) across all states and actions."""
        var_table = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                weights, atoms = self.sample_Z(s, a)
                var_table[s, a] = evaluate_functional(weights, atoms, functional="variance")
        return var_table

    def record_diagnostics(self) -> Dict:
        """Record contraction metrics and statistics."""
        variances = self.compute_posterior_variances()
        mean_var = float(np.mean(variances))
        max_var = float(np.max(variances))
        
        counts = np.array([[len(self.H[(s, a)]) for a in range(self.num_actions)] for s in range(self.num_states)])
        mean_counts = float(np.mean(counts))
        
        discounts = np.array([[self.get_emergent_discount(s, a) for a in range(self.num_actions)] for s in range(self.num_states)])
        mean_discount = float(np.mean(discounts))
        
        diag = {
            "episode": self.episodes,
            "total_steps": self.total_steps,
            "mean_variance": mean_var,
            "max_variance": max_var,
            "mean_history_length": mean_counts,
            "mean_emergent_discount": mean_discount
        }
        self.contraction_history.append(diag)
        return diag
