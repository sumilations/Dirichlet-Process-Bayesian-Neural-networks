"""
Policy and Value Baseline Neural Network Architectures with Layer Normalization
for DP-BNN REINFORCE.
Reference: Vashishtha PhD Thesis Chapter 4 (Algorithm 3 & Section 4.2)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from typing import Tuple, Optional


class CategoricalPolicyNet(nn.Module):
    """
    Discrete Action Policy Network with optional Layer Normalization.
    Outputs action logits for Categorical distribution.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_layer_norm = use_layer_norm

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.act1 = nn.Tanh()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.act2 = nn.Tanh()

        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.act2(self.ln2(self.fc2(h)))
        logits = self.out(h)
        return logits

    def get_action_and_log_prob(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor]:
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze()

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class GaussianPolicyNet(nn.Module):
    """
    Continuous Action Gaussian Policy Network with Layer Normalization.
    Outputs mean and log-std for continuous actions.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        use_layer_norm: bool = True,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_layer_norm = use_layer_norm
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.act1 = nn.Tanh()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.act2 = nn.Tanh()

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.act2(self.ln2(self.fc2(h)))
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action_and_log_prob(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(), log_prob.squeeze()

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(states)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy


class ValueBaselineNet(nn.Module):
    """
    State-Value Estimator V_phi(s) with Layer Normalization for Baseline Subtraction.
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.state_dim = state_dim
        self.use_layer_norm = use_layer_norm

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.act1 = nn.Tanh()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.act2 = nn.Tanh()

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.act2(self.ln2(self.fc2(h)))
        v = self.out(h)
        return v.squeeze(-1)
