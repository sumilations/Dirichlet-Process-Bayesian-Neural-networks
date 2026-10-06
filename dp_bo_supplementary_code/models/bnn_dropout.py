"""
Standard Bayesian Neural Network Surrogate (MC-Dropout).

Implements the standard parameter-space BNN baseline (Gal & Ghahramani, 2016).
Estimates epistemic uncertainty through S=20 stochastic forward passes with active dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dp_surrogate import FastMLP


class MCDropoutSurrogate:
    """Standard BNN using MC-Dropout with S forward passes."""
    def __init__(self, in_dim=2, hidden_dim=64, dropout_p=0.2, lr=1e-2, num_samples=20):
        self.in_dim = in_dim
        self.num_samples = num_samples
        self.model = FastMLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=1, dropout_p=dropout_p)
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs=50):
        y_mean = y.mean()
        y_std = y.std().clamp(min=1e-4)
        y_norm = (y - y_mean) / y_std

        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = F.mse_loss(pred, y_norm)
            loss.backward()
            self.optimizer.step()

    def predict(self, cand_pool: torch.Tensor):
        self.model.train()  # Keep dropout active for MC sampling
        with torch.no_grad():
            preds = torch.stack([self.model(cand_pool) for _ in range(self.num_samples)], dim=0)
            mu = preds.mean(dim=0).squeeze(-1)
            sigma = preds.std(dim=0).squeeze(-1).clamp(min=1e-4)
        return mu, sigma

    def select_query(self, cand_pool: torch.Tensor, beta=2.2):
        mu, sigma = self.predict(cand_pool)
        lcb = mu - beta * sigma
        best_idx = torch.argmin(lcb)
        return cand_pool[best_idx:best_idx + 1]
