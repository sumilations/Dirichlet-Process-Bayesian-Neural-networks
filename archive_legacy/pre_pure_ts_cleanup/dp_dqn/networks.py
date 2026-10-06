"""Neural Network architectures for DP-DQN with Layer Normalization."""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Deep Q-Network with optional Layer Normalization.

    Layer Normalization stabilizes the network against the scale fluctuations
    inherent to Dirichlet Process stick-breaking importance weights q_k ~ GEM(alpha + B).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_layer_norm = use_layer_norm

        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }
        act_cls = act_map.get(activation.lower(), nn.ReLU)

        layers = []
        in_dim = state_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
