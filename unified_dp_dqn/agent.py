# Canonical Pure Thompson Sampling DP-DQN Agent
# File: unified_dp_dqn/agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from .config import DPDQNConfig
from .networks import QNetwork, ReplayBuffer
from .base_measures import get_base_measure
from .sampler import VashishthaMaillardSampler, DirichletPosteriorSampler


class DPDQNAgent:
    """Canonical Pure Thompson Sampling DP-DQN Agent.

    Algorithmic Invariants:
    1. ZERO online gradient steps during the episode: policy remains 100% frozen.
    2. Target network updated via Polyak tracking ONLY at the beginning of each episode.
    3. Exactly ONE Dirichlet Process random measure sampled per episode at reset.
    4. Living network q_net takes W warmstart gradient steps against target_net on this single draw.
    """

    def __init__(self, config: DPDQNConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cpu")
        self.rng = np.random.RandomState(config.seed)
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # 1. Living Q-Network and Target Network (with LayerNorm)
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

        # 2. Optimizer & Loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        # 3. Base Measure & Replay Buffer
        bm_name = "haar" if getattr(config, "haar_angle", False) else config.base_measure_type
        self.base_measure = get_base_measure(
            name=bm_name,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
        )

        self.replay = ReplayBuffer(
            capacity=config.buffer_capacity,
            state_dim=config.state_dim,
        )

        if hasattr(self.base_measure, "set_replay"):
            self.base_measure.set_replay(self.replay)

        # 4. DP Posterior Sampler
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
        """Invoked at episode boundaries to update target net and draw DP Thompson hypothesis."""
        self.episodes_completed += 1

        # Check if enough replay transitions exist
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        # 1. Update target network ONLY at the beginning of episode via Polyak
        if self.config.tau > 0:
            with torch.no_grad():
                tau = self.config.tau
                for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

        # 2. Draw exactly ONE DP posterior random measure for this episode
        n_stat = self.cumulative_info if self.config.use_td_info_gain_decay else None
        s, a, r, sn, done, q_weights = self.sampler.sample(
            self.replay,
            device=self.device,
            n_stat_override=n_stat,
        )
        effective_batch = s.shape[0]

        # 3. Fit living q_net for W warmstart steps to this single frozen draw
        self.q_net.train()
        for _ in range(self.config.warmstart_steps):
            with torch.no_grad():
                if getattr(self.config, "use_double_dqn", False):
                    best_a = self.q_net(sn).argmax(dim=1, keepdim=True)
                    next_q = self.target_net(sn).gather(1, best_a).squeeze(1)
                else:
                    next_q = self.target_net(sn).max(dim=1)[0]
                target_y = r + (1.0 - done) * self.config.gamma * next_q

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * effective_batch).mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip)
            self.optimizer.step()

        # 4. Accumulate TD information gain for adaptive prior decay
        if self.config.use_td_info_gain_decay:
            with torch.no_grad():
                emp_delta = (target_y - pred_q).detach()
                td_scale = max(self.config.td_info_scale, 1e-4)
                surprise = torch.clamp(emp_delta.abs() / td_scale, max=1.0).mean().item()
                self.cumulative_info += float(surprise)

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select discrete action greedily w.r.t. frozen q_net."""
        state_t = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            q_vals = self.q_net(state_t)
            return int(q_vals.argmax(dim=1).item())

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Record transition. ZERO online SGD updates during rollout."""
        self.replay.push(state, action, reward, next_state, done)
        self.total_steps += 1
        return None
