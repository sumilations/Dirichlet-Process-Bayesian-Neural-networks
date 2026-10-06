"""
Data-Space Dirichlet Process Bayesian Neural Networks (DP-BNNs).

Implements:
1. SingleNetworkDPTS: Pure Non-Parametric Thompson Sampling via a single deterministic
   network trained on a random Dirichlet posterior measure draw (Algorithm 1).
2. MultiHeadDPBO: Multi-head parameter ensemble baseline under Dirichlet posterior weights
   with Lower Confidence Bound (LCB) acquisition.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet


class FastMLP(nn.Module):
    """Compact 2-layer MLP surrogate with SiLU activation."""
    def __init__(self, in_dim=2, hidden_dim=64, out_dim=1, dropout_p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SingleNetworkDPTS:
    """
    Single-Network Non-Parametric Dirichlet Process Thompson Sampling (Algorithm 1).
    
    Because the DP posterior is an infinite-dimensional distribution over probability
    distributions, drawing a single realization of the posterior distribution and
    optimizing a single deterministic neural network directly instantiates an exact
    functional sample from the posterior function distribution: f_{\theta_t} ~ Pi(\cdot | D_t).
    
    Greedy query selection x_{t+1} = argmin_{x \in X} f_{\theta_t}(x) executes pure,
    hyperparameter-free Thompson Sampling.
    """
    def __init__(self, in_dim=2, hidden_dim=64, alpha=1.0, T_prior=30, sigma_0=3.0, bounds=(-5.0, 5.0), lr=1e-2):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.T_prior = T_prior
        self.sigma_0 = sigma_0
        self.bounds = bounds
        self.lr = lr
        self.model = FastMLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)

    def sample_posterior_function(self, X: torch.Tensor, y: torch.Tensor, epochs=50):
        """
        Samples a single realization of the DP posterior measure and fits the single network.
        """
        t = X.shape[0]
        T = self.T_prior
        total_N = t + T

        # Base measure P_0 parameters (in standardized space: mu_0 = 0, sigma_0)
        y_mean = y.mean()
        y_std = y.std().clamp(min=1e-4)
        y_norm = (y - y_mean) / y_std

        # Sample synthetic base-measure pseudo-atoms from P_0
        if isinstance(self.bounds, tuple) and len(self.bounds) == 2 and not isinstance(self.bounds[0], (list, tuple, torch.Tensor)):
            low, high = self.bounds
            X_prior = torch.rand(T, self.in_dim, device=X.device) * (high - low) + low
        elif isinstance(self.bounds, torch.Tensor):
            low = self.bounds[:, 0]
            high = self.bounds[:, 1]
            X_prior = torch.rand(T, self.in_dim, device=X.device) * (high - low) + low
        else:
            low = torch.tensor([b[0] for b in self.bounds], device=X.device, dtype=torch.float32)
            high = torch.tensor([b[1] for b in self.bounds], device=X.device, dtype=torch.float32)
            X_prior = torch.rand(T, self.in_dim, device=X.device) * (high - low) + low

        y_prior = torch.randn(T, 1, device=y.device) * self.sigma_0

        X_comb = torch.cat([X, X_prior], dim=0)
        y_comb = torch.cat([y_norm, y_prior], dim=0)

        # Dirichlet random measure weights: conc = [1, ..., 1, alpha/T, ..., alpha/T]
        conc = torch.ones(total_N, dtype=torch.float32, device=X.device)
        conc[t:] = self.alpha / float(T)

        dir_dist = Dirichlet(conc)
        w = (dir_dist.sample().unsqueeze(1) * total_N).to(X.device)

        # Fit single network on weighted risk minimization (Eq. 7 in paper)
        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            pred = self.model(X_comb)
            loss = (w * (pred - y_comb) ** 2).mean()
            loss.backward()
            self.optimizer.step()

    def select_query(self, cand_pool: torch.Tensor):
        """Executes greedy selection under the sampled posterior function."""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(cand_pool).squeeze(-1)
            best_idx = torch.argmin(preds)
            return cand_pool[best_idx:best_idx + 1]


class MultiHeadDPBO:
    """
    Multi-Head DP-BO Ensemble Baseline.
    Trains K independent heads under distinct Dirichlet posterior measure draws.
    Acquires candidates via Lower Confidence Bound (LCB = mu - beta * sigma).
    """
    def __init__(self, in_dim=2, num_heads=4, hidden_dim=64, alpha=1.0, T_prior=30, sigma_0=3.0, bounds=(-5.0, 5.0), lr=1e-2):
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.alpha = alpha
        self.T_prior = T_prior
        self.sigma_0 = sigma_0
        self.bounds = bounds
        self.lr = lr
        self.heads = [FastMLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=1) for _ in range(num_heads)]
        self.X_train = None
        self.y_train = None

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs=50):
        self.X_train = X.clone()
        self.y_train = y.clone()
        t = X.shape[0]
        T = self.T_prior
        total_N = t + T

        y_mean = y.mean()
        y_std = y.std().clamp(min=1e-4)
        y_norm = (y - y_mean) / y_std

        conc = torch.ones(total_N, dtype=torch.float32, device=X.device)
        conc[t:] = self.alpha / float(T)

        for k in range(self.num_heads):
            if isinstance(self.bounds, tuple) and len(self.bounds) == 2 and not isinstance(self.bounds[0], (list, tuple, torch.Tensor)):
                low, high = self.bounds
                X_prior = torch.rand(T, self.in_dim, device=X.device) * (high - low) + low
            elif isinstance(self.bounds, torch.Tensor):
                low = self.bounds[:, 0]
                high = self.bounds[:, 1]
                X_prior = torch.rand(T, self.in_dim, device=X.device) * (high - low) + low
            else:
                low = torch.tensor([b[0] for b in self.bounds], device=X.device, dtype=torch.float32)
                high = torch.tensor([b[1] for b in self.bounds], device=X.device, dtype=torch.float32)
                X_prior = torch.rand(T, self.in_dim, device=X.device) * (high - low) + low

            y_prior = torch.randn(T, 1, device=y.device) * self.sigma_0
            X_comb = torch.cat([X, X_prior], dim=0)
            y_comb = torch.cat([y_norm, y_prior], dim=0)

            dir_dist = Dirichlet(conc)
            w_k = (dir_dist.sample().unsqueeze(1) * total_N).to(X.device)

            model = self.heads[k]
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-3)
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                pred = model(X_comb)
                loss = (w_k * (pred - y_comb) ** 2).mean()
                loss.backward()
                optimizer.step()

    def predict(self, cand_pool: torch.Tensor):
        for model in self.heads:
            model.eval()
        with torch.no_grad():
            preds = torch.stack([model(cand_pool) for model in self.heads], dim=0)
            mu = preds.mean(dim=0).squeeze(-1)
            sigma = preds.std(dim=0).squeeze(-1).clamp(min=1e-4)
        return mu, sigma

    def select_query(self, cand_pool: torch.Tensor, beta=2.2):
        mu, sigma = self.predict(cand_pool)
        lcb = mu - beta * sigma
        best_idx = torch.argmin(lcb)
        return cand_pool[best_idx:best_idx + 1]
