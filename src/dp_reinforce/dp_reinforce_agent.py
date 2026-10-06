"""
DP-BNN REINFORCE Agent: Policy Gradient with Dirichlet Process Bayesian Neural Networks.
Implements:
1. Dirichlet Process stick-breaking weighting over rollout trajectories (Algorithm 3 & Eq. 4.7)
2. Policy-Space Thompson Sampling (episodic warm-start on freshly drawn DP posterior measure)
3. Layer-Normalized DP-BNN Value Baseline Network
Reference: Vashishtha PhD Thesis Chapter 4 (Sections 4.1 - 4.4 & Algorithm 3)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, Union

from .policies import CategoricalPolicyNet, GaussianPolicyNet, ValueBaselineNet


class DP_BNN_REINFORCE:
    """
    Dirichlet Process Bayesian Neural Network REINFORCE Agent.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        is_continuous: bool = False,
        hidden_dim: int = 64,
        use_layer_norm: bool = True,
        use_baseline: bool = True,
        use_dp_weights: bool = True,
        use_policy_ts: bool = True,       # Policy-Space Thompson Sampling
        alpha_dp: float = 5.0,            # DP concentration parameter
        k_trunc_prior: int = 20,          # Truncation level for prior atoms
        f0_return_mean: float = 1.0,      # Prior base measure return mean (optimism)
        f0_return_std: float = 0.5,       # Prior base measure return std
        gamma: float = 0.99,
        lr_policy: float = 1e-3,
        lr_baseline: float = 2e-3,
        entropy_coef: float = 0.01,
        warm_start_steps: int = 3,        # Gradient adaptation steps for Thompson policy
        warm_start_lr: float = 1e-3,
        max_buffer_size: int = 2000,
        device: str = "cpu",
        seed: Optional[int] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_continuous = is_continuous
        self.use_baseline = use_baseline
        self.use_dp_weights = use_dp_weights
        self.use_policy_ts = use_policy_ts
        self.alpha_dp = float(alpha_dp)
        self.k_trunc_prior = k_trunc_prior
        self.f0_return_mean = float(f0_return_mean)
        self.f0_return_std = float(f0_return_std)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.warm_start_steps = warm_start_steps
        self.warm_start_lr = warm_start_lr
        self.max_buffer_size = max_buffer_size
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Policy Network
        if is_continuous:
            self.policy_net = GaussianPolicyNet(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                use_layer_norm=use_layer_norm
            ).to(self.device)
        else:
            self.policy_net = CategoricalPolicyNet(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                use_layer_norm=use_layer_norm
            ).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)

        # Value Baseline Network
        if self.use_baseline:
            self.baseline_net = ValueBaselineNet(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                use_layer_norm=use_layer_norm
            ).to(self.device)
            self.baseline_optimizer = optim.Adam(self.baseline_net.parameters(), lr=lr_baseline)
        else:
            self.baseline_net = None
            self.baseline_optimizer = None

        # Replay buffer of past transitions: (s, a, G, s_next, done)
        self.buffer_states: List[np.ndarray] = []
        self.buffer_actions: List[Union[int, np.ndarray]] = []
        self.buffer_returns: List[float] = []

        # Active episode policy (for Thompson sampling)
        self.active_policy = self.policy_net

        # Diagnostics
        self.total_episodes = 0
        self.total_steps = 0
        self.history_metrics: List[Dict] = []

    def start_episode(self):
        """
        Start of episode: Execute Policy-Space Thompson Sampling if enabled.
        Samples a fresh policy hypothesis theta_e ~ DP-Posterior(theta | D).
        """
        if self.use_policy_ts and len(self.buffer_states) >= 10:
            # Clone current policy network
            self.active_policy = copy.deepcopy(self.policy_net)
            optimizer_warm = optim.Adam(self.active_policy.parameters(), lr=self.warm_start_lr)
            
            # Draw DP posterior sample
            w_tensor, s_tensor, a_tensor, g_tensor = self._sample_dp_posterior_batch(batch_size=128)
            
            # Fast adaptation steps
            for _ in range(self.warm_start_steps):
                log_probs, entropy = self.active_policy.evaluate_actions(s_tensor, a_tensor)
                
                # Advantage
                if self.use_baseline and self.baseline_net is not None:
                    with torch.no_grad():
                        v = self.baseline_net(s_tensor)
                    adv = g_tensor - v
                else:
                    adv = g_tensor

                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    
                loss = -torch.sum(w_tensor * log_probs * adv) - self.entropy_coef * torch.sum(w_tensor * entropy)
                optimizer_warm.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.active_policy.parameters(), max_norm=1.0)
                optimizer_warm.step()
        else:
            self.active_policy = self.policy_net

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[Union[int, np.ndarray], float]:
        """Select action using the active episode policy."""
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, log_prob = self.active_policy.get_action_and_log_prob(s_tensor, deterministic=deterministic)
        return action, float(log_prob.item())

    def _sample_dp_posterior_batch(
        self,
        batch_size: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random measure from the DP posterior via stick-breaking (Algorithm 3 & Eq. 4.7).
        Combines empirical transitions from replay buffer with optimistic prior atoms from F_0.
        """
        n_buf = len(self.buffer_states)
        n_emp = min(batch_size, n_buf)
        idx = self.rng.choice(n_buf, size=n_emp, replace=False)
        
        emp_states = np.array([self.buffer_states[i] for i in idx], dtype=np.float32)
        if self.is_continuous:
            emp_actions = np.array([self.buffer_actions[i] for i in idx], dtype=np.float32)
        else:
            emp_actions = np.array([self.buffer_actions[i] for i in idx], dtype=np.int64)
        emp_returns = np.array([self.buffer_returns[i] for i in idx], dtype=np.float32)

        # Stick-breaking over empirical observations (Eq. 4.3):
        # V_i ~ Beta(1, alpha + i)
        alphas = self.alpha_dp + np.arange(1, n_emp + 1, dtype=np.float64)
        V = self.rng.beta(1.0, alphas)
        one_minus_V = 1.0 - V
        
        # Suffix cumulative products: suffix_prod[i] = prod_{j=i+1}^n (1 - V_j)
        rev_cum = np.cumprod(np.concatenate(([1.0], one_minus_V[::-1][:-1])))
        suffix_prod = rev_cum[::-1]
        w_emp = V * suffix_prod
        
        # Remainder stick for DP prior P_0: prod_{i=1}^n (1 - V_i)
        stick_prior = np.prod(one_minus_V)

        # Sample prior atoms from F_0 if k_trunc_prior > 0
        k_t = self.k_trunc_prior
        if k_t > 0:
            V_prior = self.rng.beta(1.0, self.alpha_dp, size=k_t)
            V_prior[-1] = 1.0
            one_minus_Vp = 1.0 - V_prior
            cum_rem_p = np.cumprod(np.concatenate(([1.0], one_minus_Vp[:-1])))
            w_prior = V_prior * cum_rem_p
            w_prior_scaled = stick_prior * w_prior

            prior_state_idx = self.rng.choice(n_buf, size=k_t, replace=True)
            prior_states = np.array([self.buffer_states[i] for i in prior_state_idx], dtype=np.float32)
            
            if self.is_continuous:
                prior_actions = self.rng.uniform(-1.0, 1.0, size=(k_t, self.action_dim)).astype(np.float32)
            else:
                prior_actions = self.rng.integers(0, self.action_dim, size=k_t, dtype=np.int64)
                
            prior_returns = self.rng.normal(self.f0_return_mean, self.f0_return_std, size=k_t).astype(np.float32)

            all_weights = np.concatenate([w_emp, w_prior_scaled])
            all_states = np.concatenate([emp_states, prior_states], axis=0)
            all_actions = np.concatenate([emp_actions, prior_actions], axis=0)
            all_returns = np.concatenate([emp_returns, prior_returns], axis=0)
        else:
            all_weights = w_emp
            all_states = emp_states
            all_actions = emp_actions
            all_returns = emp_returns

        w_tensor = torch.as_tensor(all_weights, dtype=torch.float32, device=self.device)
        s_tensor = torch.as_tensor(all_states, dtype=torch.float32, device=self.device)
        if self.is_continuous:
            a_tensor = torch.as_tensor(all_actions, dtype=torch.float32, device=self.device)
        else:
            a_tensor = torch.as_tensor(all_actions, dtype=torch.int64, device=self.device)
        g_tensor = torch.as_tensor(all_returns, dtype=torch.float32, device=self.device)

        return w_tensor, s_tensor, a_tensor, g_tensor

    def update_with_episode(
        self,
        states: List[np.ndarray],
        actions: List[Union[int, np.ndarray]],
        rewards: List[float]
    ) -> Dict:
        """
        Process completed episode trajectory, compute discounted returns G_t,
        add to buffer, and perform DP-BNN policy and baseline updates.
        """
        self.total_episodes += 1
        T = len(rewards)
        self.total_steps += T

        # Compute discounted returns G_t
        returns = np.zeros(T, dtype=np.float32)
        running_g = 0.0
        for t in reversed(range(T)):
            running_g = rewards[t] + self.gamma * running_g
            returns[t] = running_g

        # Append to rollout buffer
        for t in range(T):
            self.buffer_states.append(states[t])
            self.buffer_actions.append(actions[t])
            self.buffer_returns.append(float(returns[t]))

        # Manage buffer capacity
        if len(self.buffer_states) > self.max_buffer_size:
            excess = len(self.buffer_states) - self.max_buffer_size
            del self.buffer_states[:excess]
            del self.buffer_actions[:excess]
            del self.buffer_returns[:excess]

        # -----------------------------------------------------------------
        # Prepare Batch for Policy & Baseline Update
        # -----------------------------------------------------------------
        if self.use_dp_weights:
            # DP Posterior Sample (Algorithm 3)
            w_tensor, s_tensor, a_tensor, g_tensor = self._sample_dp_posterior_batch(batch_size=min(len(self.buffer_states), 128))
        else:
            # Standard uniform episode batch
            s_tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
            if self.is_continuous:
                a_tensor = torch.as_tensor(np.array(actions), dtype=torch.float32, device=self.device)
            else:
                a_tensor = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device)
            g_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            w_tensor = torch.full((T,), 1.0 / T, dtype=torch.float32, device=self.device)

        # -----------------------------------------------------------------
        # Update Value Baseline Network
        # -----------------------------------------------------------------
        baseline_loss_val = 0.0
        if self.use_baseline and self.baseline_net is not None:
            v_pred = self.baseline_net(s_tensor)
            baseline_loss = torch.sum(w_tensor * (v_pred - g_tensor) ** 2)
            
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.baseline_net.parameters(), max_norm=1.0)
            self.baseline_optimizer.step()
            baseline_loss_val = float(baseline_loss.item())

            with torch.no_grad():
                v_eval = self.baseline_net(s_tensor)
            advantage = g_tensor - v_eval
        else:
            advantage = g_tensor

        # Normalize advantages for numerical stability
        if len(advantage) > 1:
            adv_mean = torch.mean(advantage)
            adv_std = torch.std(advantage) + 1e-8
            advantage = (advantage - adv_mean) / adv_std

        # -----------------------------------------------------------------
        # Update Policy Network (DP-Weighted Policy Gradient)
        # -----------------------------------------------------------------
        log_probs, entropy = self.policy_net.evaluate_actions(s_tensor, a_tensor)
        policy_loss = -torch.sum(w_tensor * log_probs * advantage) - self.entropy_coef * torch.sum(w_tensor * entropy)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        metrics = {
            "episode": self.total_episodes,
            "total_steps": self.total_steps,
            "episode_return": float(np.sum(rewards)),
            "policy_loss": float(policy_loss.item()),
            "baseline_loss": baseline_loss_val,
            "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, 'item') else float(grad_norm),
            "mean_entropy": float(torch.mean(entropy).item())
        }
        self.history_metrics.append(metrics)
        return metrics
