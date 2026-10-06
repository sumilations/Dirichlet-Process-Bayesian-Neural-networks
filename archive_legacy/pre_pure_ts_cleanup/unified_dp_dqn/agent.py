"""Canonical Environment-Agnostic DP-DQN Agent."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import DPDQNConfig
from .networks import QNetwork, ReplayBuffer, StratifiedReplayBuffer
from .base_measures import get_base_measure
from .sampler import VashishthaMaillardSampler, DirichletPosteriorSampler


class DPDQNAgent:
    """Canonical, environment-agnostic Dirichlet Process Deep Q-Network Agent.

    Features:
    1. Single canonical architecture across all benchmarks (Cart-Pole, Deep Sea, Gym).
    2. Acts with decoupled episodic Thompson hypothesis (or master `q_net` in eval mode).
    3. Continuous Polyak tracking of target network on every SGD step.
    4. Dirichlet Process posterior sampling via Vashishtha & Maillard (2025) or Dirichlet Ferguson (1973).
    5. Pure TD Information Gain decay: W_prior ~ Beta(alpha + 1, N_TD) with zero heuristic constants.
    6. Optional Stratified 50/50 Replay Buffer preventing catastrophic forgetting of swing-up trajectories.
    7. Optional Double DQN Bellman targets preventing maximization bias and value overestimation.
    8. Optional DP-based online SGD updates during episodes.
    """

    def __init__(self, config: DPDQNConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cpu")
        self.rng = np.random.RandomState(config.seed)
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # 1. Living Q-Network and Target Network
        self.q_net = QNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            use_layer_norm=config.use_layer_norm,
            activation=config.activation,
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            use_layer_norm=config.use_layer_norm,
            activation=config.activation,
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        # Decoupled Episodic Acting Network (preserves master q_net uncorrupted by prior noise)
        if getattr(config, "use_decoupled_actor", False):
            self.acting_net = QNetwork(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                use_layer_norm=config.use_layer_norm,
                activation=config.activation,
            ).to(self.device)
            self.acting_net.load_state_dict(self.q_net.state_dict())
            self.acting_optimizer = optim.Adam(self.acting_net.parameters(), lr=config.lr)
        else:
            self.acting_net = self.q_net
            self.acting_optimizer = None

        # 2. Optimizer & Loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        # 3. Base Measure & Replay Storage
        bm_name = "haar" if config.haar_angle else config.base_measure_type
        self.base_measure = get_base_measure(
            name=bm_name,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
        )

        if getattr(config, "use_stratified_replay", False):
            self.replay = StratifiedReplayBuffer(
                capacity=config.buffer_capacity,
                state_dim=config.state_dim,
            )
        else:
            self.replay = ReplayBuffer(
                capacity=config.buffer_capacity,
                state_dim=config.state_dim,
            )

        if hasattr(self.base_measure, "set_replay"):
            self.base_measure.set_replay(self.replay)

        # 4. Posterior Sampler (Vashishtha & Maillard 2025 Eq. 288 vs Unified Dirichlet Ferguson 1973)
        if getattr(config, "sampler_type", "vashishtha_maillard") == "dirichlet":
            self.sampler = DirichletPosteriorSampler(
                alpha=config.alpha,
                batch_size=config.batch_size,
                base_measure=self.base_measure,
                rng=self.rng,
                vm_prior_multiplier=config.vm_prior_multiplier,
                w_min=getattr(config, "w_min", 0.0),
            )
        else:
            self.sampler = VashishthaMaillardSampler(
                alpha=config.alpha,
                batch_size=config.batch_size,
                base_measure=self.base_measure,
                rng=self.rng,
                vm_prior_multiplier=config.vm_prior_multiplier,
                w_min=getattr(config, "w_min", 0.0),
            )

        self.total_steps = 0
        self.episodes_completed = 0
        self.cumulative_info = 0.0

    def reset_episode(self) -> None:
        """Invoked at episode boundaries to draw an episodic DP Thompson hypothesis."""
        self.episodes_completed += 1

        # Optional linear annealing of w_min
        if getattr(self.config, "w_min_end", None) is not None:
            total_ep = getattr(self.config, "total_episodes", 2500)
            frac = min(1.0, float(self.episodes_completed) / float(total_ep))
            self.sampler.w_min = float(self.config.w_min + frac * (self.config.w_min_end - self.config.w_min))

        if getattr(self.config, "use_decoupled_actor", False):
            self.acting_net.load_state_dict(self.q_net.state_dict())
            net_to_train = self.acting_net
            opt_to_use = self.acting_optimizer
        else:
            net_to_train = self.q_net
            opt_to_use = self.optimizer

        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        net_to_train.train()
        n_stat = self.cumulative_info if self.config.use_td_info_gain_decay else None

        if getattr(self.config, "sample_once_per_episode", False):
            # Pure TS: Single DP posterior draw frozen for the entire episode
            s, a, r, sn, done, q_weights = self.sampler.sample(
                self.replay,
                device=self.device,
                n_stat_override=n_stat,
            )
            effective_batch = s.shape[0]
            for _ in range(self.config.warmstart_steps):
                with torch.no_grad():
                    if getattr(self.config, "use_double_dqn", False):
                        best_a = self.q_net(sn).argmax(dim=1, keepdim=True)
                        next_q = self.target_net(sn).gather(1, best_a).squeeze(1)
                    else:
                        next_q = self.target_net(sn).max(dim=1)[0]
                    target_y = r + (1.0 - done) * self.config.gamma * next_q

                pred_q = net_to_train(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = (self.loss_fn(pred_q, target_y) * q_weights * effective_batch).mean()

                opt_to_use.zero_grad()
                loss.backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(net_to_train.parameters(), self.config.grad_clip)
                opt_to_use.step()
        else:
            # Multi-sample warmstart (redraws DP posterior each warmstart step)
            for _ in range(self.config.warmstart_steps):
                s, a, r, sn, done, q_weights = self.sampler.sample(
                    self.replay,
                    device=self.device,
                    n_stat_override=n_stat,
                )

                with torch.no_grad():
                    if getattr(self.config, "use_double_dqn", False):
                        best_a = self.q_net(sn).argmax(dim=1, keepdim=True)
                        next_q = self.target_net(sn).gather(1, best_a).squeeze(1)
                    else:
                        next_q = self.target_net(sn).max(dim=1)[0]
                    target_y = r + (1.0 - done) * self.config.gamma * next_q

                pred_q = net_to_train(s).gather(1, a.unsqueeze(1)).squeeze(1)
                effective_batch = s.shape[0]

                # Weight by DP Dirichlet representation weights
                loss = (self.loss_fn(pred_q, target_y) * q_weights * effective_batch).mean()

                opt_to_use.zero_grad()
                loss.backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(net_to_train.parameters(), self.config.grad_clip)
                opt_to_use.step()

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select discrete action greedily w.r.t. acting_net (or q_net in eval mode)."""
        state_t = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        net = self.q_net if (eval_mode or not getattr(self.config, "use_decoupled_actor", False)) else self.acting_net
        net.eval()
        with torch.no_grad():
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
        """Record environment transition and execute online TD update periodically."""
        self.replay.push(state, action, reward, next_state, done)
        self.total_steps += 1

        if len(self.replay) >= self.config.batch_size and self.total_steps % self.config.sgd_period == 0:
            return self.update()
        return None

    def update(self) -> float:
        """Execute one online TD gradient step on living q_net."""
        self.q_net.train()

        if getattr(self.config, "dp_online_sgd", False):
            # DP based update of q_net in SGD during episodes
            n_stat = self.cumulative_info if self.config.use_td_info_gain_decay else None
            s, a, r, sn, done, q_weights = self.sampler.sample(
                self.replay,
                device=self.device,
                n_stat_override=n_stat,
            )
            effective_batch = s.shape[0]
            with torch.no_grad():
                if getattr(self.config, "use_double_dqn", False):
                    best_a = self.q_net(sn).argmax(dim=1, keepdim=True)
                    next_q = self.target_net(sn).gather(1, best_a).squeeze(1)
                else:
                    next_q = self.target_net(sn).max(dim=1)[0]
                target_y = r + (1.0 - done) * self.config.gamma * next_q

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * effective_batch).mean()
        else:
            # Standard uniform or stratified empirical batch from replay buffer
            s_np, a_np, r_np, sn_np, done_np = self.replay.sample(self.config.batch_size, self.rng)
            s = torch.from_numpy(s_np).to(self.device)
            a = torch.from_numpy(a_np).to(self.device)
            r = torch.from_numpy(r_np).to(self.device)
            sn = torch.from_numpy(sn_np).to(self.device)
            done = torch.from_numpy(done_np).to(self.device)

            with torch.no_grad():
                if getattr(self.config, "use_double_dqn", False):
                    best_a = self.q_net(sn).argmax(dim=1, keepdim=True)
                    next_q = self.target_net(sn).gather(1, best_a).squeeze(1)
                else:
                    next_q = self.target_net(sn).max(dim=1)[0]
                target_y = r + (1.0 - done) * self.config.gamma * next_q

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = self.loss_fn(pred_q, target_y).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip)
        self.optimizer.step()

        # Track Bellman TD Information Gain accumulation (TD surprise on empirical transitions)
        if self.config.use_td_info_gain_decay:
            with torch.no_grad():
                emp_delta = (target_y - pred_q).detach()
                td_scale = max(self.config.td_info_scale, 1e-4)
                surprise = torch.clamp(emp_delta.abs() / td_scale, max=1.0).mean().item()
                self.cumulative_info += float(surprise)

        # Continuous Polyak Target Tracking: target = (1 - tau) * target + tau * q_net
        if self.config.tau > 0:
            with torch.no_grad():
                tau = self.config.tau
                for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

        return float(loss.item())

