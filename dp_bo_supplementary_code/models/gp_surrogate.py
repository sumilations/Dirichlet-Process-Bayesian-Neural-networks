"""
Exact Gaussian Process Regression Surrogate.

Implements exact GP-BO with Matérn / RBF stationary kernel and LCB acquisition.
"""

import math
import torch


class ExactGPSurrogate:
    """Exact Gaussian Process Regression with Matérn / RBF Kernel."""
    def __init__(self, in_dim=2, lengthscale=None, sigma_f=1.0, noise_var=1e-4):
        self.in_dim = in_dim
        self.lengthscale = lengthscale if lengthscale is not None else math.sqrt(in_dim)
        self.sigma_f = sigma_f
        self.noise_var = noise_var
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def _kernel(self, X1: torch.Tensor, X2: torch.Tensor):
        dists_sq = torch.cdist(X1, X2) ** 2
        return (self.sigma_f ** 2) * torch.exp(-dists_sq / (2.0 * (self.lengthscale ** 2)))

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        self.X_train = X.clone()
        y_mean = y.mean()
        y_std = y.std().clamp(min=1e-4)
        self.y_train = (y - y_mean) / y_std

        t = X.shape[0]
        K = self._kernel(X, X) + self.noise_var * torch.eye(t, device=X.device)
        self.K_inv = torch.linalg.pinv(K)

    def predict(self, cand_pool: torch.Tensor):
        K_star = self._kernel(cand_pool, self.X_train)
        K_star_star = (self.sigma_f ** 2) * torch.ones(cand_pool.shape[0], 1, device=cand_pool.device)

        mu = (K_star @ (self.K_inv @ self.y_train)).squeeze(-1)
        var = (K_star_star - torch.sum(K_star @ self.K_inv * K_star, dim=1, keepdim=True)).squeeze(-1)
        sigma = torch.sqrt(var.clamp(min=1e-4))
        return mu, sigma

    def select_query(self, cand_pool: torch.Tensor, beta=2.2):
        mu, sigma = self.predict(cand_pool)
        lcb = mu - beta * sigma
        best_idx = torch.argmin(lcb)
        return cand_pool[best_idx:best_idx + 1]
