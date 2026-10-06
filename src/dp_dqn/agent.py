"""Unified DP-DQN Agent."""

from typing import Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import DPDQNConfig
from .networks import QNetwork
from .base_measures import BaseMeasure, get_base_measure
from .sampler import ReplayBuffer, SethuramanStickBreakingSampler, FixedBudgetVashishthaMaillardSampler


class DPDQNAgent:
    """Unified Dirichlet Process Deep Q-Network Agent.

    Features:
    - Non-parametric data-space Dirichlet Process prior F ~ DP(alpha, F_0).
    - Modular base measures F_0 (Cart-Pole, Deep Sea, Gaussian, or custom).
    - Single neural network with Layer Normalization.
    - Episodic Thompson Sampling via Target Warm-Start.
    - Online stick-breaking importance-weighted TD learning.
    """

    def __init__(
        self,
        config: Optional[DPDQNConfig] = None,
        base_measure: Optional[Union[str, BaseMeasure]] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.config = config or DPDQNConfig(**kwargs)
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        self.device = device or torch.device("cpu")
        self.rng = np.random.RandomState(self.config.seed)
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        # 1. Neural Networks (Online and Target) with Layer Normalization
        self.q_net = QNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            use_layer_norm=self.config.use_layer_norm,
            activation=self.config.activation,
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            use_layer_norm=self.config.use_layer_norm,
            activation=self.config.activation,
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = self.config.target_warmstart or getattr(self.config, "dp_sampled_target", False)

        # 2. Optimizers
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        if self.config.target_warmstart or getattr(self.config, "dp_sampled_target", False):
            target_lr = self.config.lr * self.config.warmstart_lr_scale
            self.target_optimizer = optim.Adam(self.target_net.parameters(), lr=target_lr)
        else:
            self.target_optimizer = None

        # 3. Base Measure & Replay Storage
        bm_arg = base_measure or self.config.base_measure
        self.base_measure = get_base_measure(
            bm_arg,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            prior_reward_mean=self.config.prior_reward_mean,
            prior_reward_std=self.config.prior_reward_std,
            deep_sea_size=self.config.deep_sea_size,
        )

        self.replay = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            state_dim=self.config.state_dim,
        )

        # 4. Posterior Sampler (Sethuraman vs Vashishtha & Maillard Fixed-Budget)
        if getattr(self.config, "sampler_type", "sethuraman") == "vashishtha_maillard":
            self.sampler = FixedBudgetVashishthaMaillardSampler(
                alpha=self.config.alpha,
                batch_size=self.config.batch_size,
                base_measure=self.base_measure,
                rng=self.rng,
                vm_prior_multiplier=getattr(self.config, "vm_prior_multiplier", 10.0),
                priority_positive_slot=getattr(self.config, "priority_positive_slot", False),
                vm_replay_scaled_n=getattr(self.config, "vm_replay_scaled_n", False),
                vm_scale_post_discovery_only=getattr(self.config, "vm_scale_post_discovery_only", False),
            )
        else:
            self.sampler = SethuramanStickBreakingSampler(
                alpha=self.config.alpha,
                candidate_batch_size=self.config.candidate_batch_size,
                batch_size=self.config.batch_size,
                base_measure=self.base_measure,
                rng=self.rng,
                contraction_C=self.config.contraction_C,
                use_buffer_size_denominator=self.config.use_buffer_size_denominator,
                trajectory_C=self.config.trajectory_C,
                direct_replay_sample=self.config.direct_replay_sample,
                use_bayesian_alpha=self.config.use_bayesian_alpha,
                alpha_prior_a=self.config.alpha_prior_a,
                alpha_prior_b=self.config.alpha_prior_b,
                alpha_evidence_scale=self.config.alpha_evidence_scale,
            )
            self.sampler.priority_positive_slot = getattr(self.config, "priority_positive_slot", False)

        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.total_steps = 0
        self.episodes_completed = 0
        self.cumulative_info = 0.0

    @property
    def current_alpha(self) -> float:
        return self.sampler.last_alpha

    @property
    def current_prob_syn(self) -> float:
        return self.sampler.last_prob_syn

    def reset_episode(self):
        """Prepare agent for a new episode via Pure DP Thompson Sampling."""
        self.episodes_completed += 1

        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        # 1. Update target network at episode boundary via Polyak
        if self.config.tau > 0.0:
            with torch.no_grad():
                tau = self.config.tau
                for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

        n_stat = self.cumulative_info if getattr(self.config, "use_td_info_gain_decay", False) else None

        if getattr(self.config, "sample_once_per_episode", True):
            # Algorithm 2 (Paper): Prior transitions sampled ONCE at episode reset;
            # fresh empirical mini-batches drawn at EACH warmstart step w = 1..W.
            K_prior = max(8, int(getattr(self.config, "vm_prior_multiplier", 10.0) * self.config.alpha))
            syn_s, syn_a, syn_r, syn_sn, syn_d = self.base_measure.sample(K_prior, self.rng)

            # Prior stick-breaking weights once
            V_prior = self.rng.beta(1.0, max(self.config.alpha, 1e-3), size=K_prior).astype(np.float32)
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

            syn_s_t = torch.from_numpy(syn_s).to(self.device)
            syn_a_t = torch.from_numpy(syn_a).to(self.device)
            syn_r_t = torch.from_numpy(syn_r).to(self.device)
            syn_sn_t = torch.from_numpy(syn_sn).to(self.device)
            syn_d_t = torch.from_numpy(syn_d).to(self.device)

            n_emp_avail = len(self.replay)
            N_emp = min(n_emp_avail, self.config.batch_size)

            for _ in range(self.config.warmstart_steps):
                emp_s, emp_a, emp_r, emp_sn, emp_d = self.replay.sample_priority(
                    N_emp, self.rng, positive_slot=getattr(self.config, "priority_positive_slot", True)
                )

                if n_stat is not None:
                    N_stat = max(float(N_emp), float(n_stat))
                    W_prior = float(self.rng.beta(self.config.alpha + 1.0, float(N_stat)))
                    i_arr = np.arange(1, N_emp + 1, dtype=np.float32)
                    V_emp = self.rng.beta(1.0, self.config.alpha + i_arr).astype(np.float32)
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
                    i_arr = np.arange(1, N_emp + 1, dtype=np.float32)
                    V_emp = self.rng.beta(1.0, self.config.alpha + i_arr).astype(np.float32)
                    U_emp = 1.0 - V_emp
                    suffix_U = np.ones(N_emp + 1, dtype=np.float32)
                    suffix_U[:-1] = np.cumprod(U_emp[::-1])[::-1]
                    w_emp = np.zeros(N_emp, dtype=np.float32)
                    if N_emp > 1:
                        w_emp[:-1] = V_emp[:-1] * suffix_U[1:-1]
                    w_emp[-1] = V_emp[-1]
                    W_prior = float(suffix_U[0])

                w_prior = W_prior * q_prior
                total_weights = np.concatenate([w_emp, w_prior])
                t_sum = total_weights.sum()
                if t_sum > 0:
                    total_weights = total_weights / t_sum

                emp_s_t = torch.from_numpy(emp_s).to(self.device)
                emp_a_t = torch.from_numpy(emp_a).to(self.device)
                emp_r_t = torch.from_numpy(emp_r).to(self.device)
                emp_sn_t = torch.from_numpy(emp_sn).to(self.device)
                emp_d_t = torch.from_numpy(emp_d).to(self.device)

                s = torch.cat([emp_s_t, syn_s_t], dim=0)
                a = torch.cat([emp_a_t, syn_a_t], dim=0)
                r = torch.cat([emp_r_t, syn_r_t], dim=0)
                sn = torch.cat([emp_sn_t, syn_sn_t], dim=0)
                done = torch.cat([emp_d_t, syn_d_t], dim=0)
                q_weights = torch.from_numpy(total_weights).to(self.device)

                with torch.no_grad():
                    q_next = self.target_net(sn).max(dim=1)[0]
                    target_y = r + self.config.gamma * q_next * (1.0 - done)

                pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                effective_batch = s.shape[0]
                loss = (self.loss_fn(pred_q, target_y) * q_weights * effective_batch).mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                self.optimizer.step()
        else:
            # Multi-Sample: Fresh mini-batch and stick-breaking draw at each warmstart step
            for _ in range(self.config.warmstart_steps):
                s, a, r, sn, done, q_weights = self.sampler.sample(self.replay, device=self.device, n_stat_override=n_stat)
                effective_batch = s.shape[0]

                with torch.no_grad():
                    q_next = self.target_net(sn).max(dim=1)[0]
                    target_y = r + self.config.gamma * q_next * (1.0 - done)

                pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = (self.loss_fn(pred_q, target_y) * q_weights * effective_batch).mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                self.optimizer.step()

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action. In exploration, acts greedily w.r.t. target Thompson sample or living net."""
        state_t = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if getattr(self.config, "one_living_network", False):
                net = self.q_net
            else:
                net = self.q_net if eval_mode else self.target_net
            q_vals = net(state_t)
            return int(q_vals.argmax(dim=1).item())

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Record transition. If episodic_sgd is False, trigger online TD update every sgd_period steps."""
        self.replay.push(state, action, reward, next_state, done)
        self.total_steps += 1

        if getattr(self.config, "episodic_sgd", False):
            # Option B: Do not perform intra-episode updates; policy remains 100% frozen
            return None

        if self.total_steps % self.config.sgd_period == 0:
            return self.update()
        return None

    def end_episode(self, episode_steps: Optional[int] = None) -> List[float]:
        """Run consolidated SGD updates at episode completion (Option B)."""
        losses = []
        if getattr(self.config, "episodic_sgd", False):
            if len(self.replay) < self.config.batch_size:
                return losses
            num_updates = max(1, (episode_steps if episode_steps is not None else 1) // max(1, self.config.sgd_period))
            for _ in range(num_updates):
                loss = self.update()
                losses.append(loss)
        return losses

    def update(self) -> float:
        """Perform single online TD update weighted by stick-breaking posterior weights."""
        n_stat = self.cumulative_info if getattr(self.config, "use_td_info_gain_decay", False) else None
        s, a, r, sn, done, q_weights = self.sampler.sample(self.replay, device=self.device, n_stat_override=n_stat)

        with torch.no_grad():
            q_next = self.target_net(sn).max(dim=1)[0]
            target_y = r + self.config.gamma * q_next * (1.0 - done)

        pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        effective_batch = s.shape[0]
        loss = (self.loss_fn(pred_q, target_y) * q_weights * effective_batch).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # TD information gain accumulation (Option 3)
        if getattr(self.config, "use_td_info_gain_decay", False):
            with torch.no_grad():
                n_emp = getattr(self.sampler, "last_n_emp", s.shape[0])
                if n_emp > 0:
                    emp_delta = (target_y[:n_emp] - pred_q[:n_emp]).detach()
                    td_scale = max(getattr(self.config, "td_info_scale", 1.0), 1e-4)
                    surprise = torch.clamp(emp_delta.abs() / td_scale, max=1.0).mean().item()
                    self.cumulative_info += float(surprise)

        # Update Polyak tracking target network (disabled if dp_sampled_target)
        is_dp_target = getattr(self.config, "dp_sampled_target", False)
        is_living = getattr(self.config, "one_living_network", False)
        if not is_dp_target and (is_living or not self.config.target_warmstart) and self.config.tau > 0.0:
            with torch.no_grad():
                tau = self.config.tau
                for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

        return float(loss.item())
