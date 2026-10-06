"""Neural Feature Extractor for Neural-Linear DP-DQN."""

import torch
import torch.nn as nn


class NeuralFeatureNet(nn.Module):
    """Deep representation extractor with Layer Normalization.

    Extracts a d-dimensional feature representation phi(s) from raw state inputs.
    Appends a constant bias dimension 1.0, enabling exact affine Bayesian linear regression.
    """

    def __init__(self, in_dim: int, feature_dim: int = 32, use_layer_norm: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.feature_dim = feature_dim
        self.phi_dim = feature_dim + 1  # includes bias term

        if use_layer_norm:
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
            )
        else:
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, feature_dim),
                nn.ReLU(),
            )

        # Output projection for standard forward pass during representation training
        self.head = nn.Linear(self.phi_dim, 2, bias=False)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features phi(s) and append constant 1.0 for affine bias."""
        h = self.backbone(x)
        ones = torch.ones((h.shape[0], 1), dtype=h.dtype, device=h.device)
        return torch.cat([h, ones], dim=-1)  # (batch, phi_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.extract_features(x)
        return self.head(phi)
