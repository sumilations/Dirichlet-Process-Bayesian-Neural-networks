"""Closed-Form Bayesian Linear Regression for Neural-Linear DP-DQN."""

from typing import Optional, Tuple
import numpy as np
import torch


class BayesianLinearHead:
    """Analytical Closed-Form Bayesian Linear Readout for each action.

    Under a Dirichlet Process posterior sample F ~ DP, computes the exact
    posterior mean and covariance for linear weights w_a without gradient descent.
    Draws exact Thompson samples w_a ~ N(mu_a, Sigma_a).
    """

    def __init__(
        self,
        phi_dim: int,
        action_dim: int = 2,
        prior_precision: float = 1.0,
        noise_variance: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.phi_dim = phi_dim
        self.action_dim = action_dim
        self.prior_precision = float(prior_precision)
        self.noise_variance = float(noise_variance)
        self.rng = np.random.RandomState(seed)

        # Sufficient statistics for each action: Lambda_a and b_a
        self.Lambda = np.zeros((action_dim, phi_dim, phi_dim), dtype=np.float32)
        self.b = np.zeros((action_dim, phi_dim), dtype=np.float32)
        self.mu = np.zeros((action_dim, phi_dim), dtype=np.float32)

        # Current active episodic Thompson sample
        self.sampled_weights = np.zeros((action_dim, phi_dim), dtype=np.float32)
        self.reset_priors()

    def reset_priors(self):
        """Reset precision matrices to prior: Lambda_0 = lambda * I."""
        for a in range(self.action_dim):
            self.Lambda[a] = np.eye(self.phi_dim, dtype=np.float32) * self.prior_precision
            self.b[a] = np.zeros(self.phi_dim, dtype=np.float32)
            self.mu[a] = np.zeros(self.phi_dim, dtype=np.float32)
            self.sampled_weights[a] = np.zeros(self.phi_dim, dtype=np.float32)

    def update_from_dp_sample(
        self,
        phi: np.ndarray,      # (B, phi_dim)
        actions: np.ndarray,  # (B,)
        targets: np.ndarray,  # (B,)
        q_weights: np.ndarray # (B,) Dirichlet stick-breaking weights
    ):
        """Update precision matrix and linear vector in closed-form from a DP sample.

        Lambda_a = lambda * I + sum_{i: a_i=a} q_i * phi_i * phi_i^T
        b_a = sum_{i: a_i=a} q_i * phi_i * y_i
        """
        self.reset_priors()

        for a in range(self.action_dim):
            mask = (actions == a)
            if not np.any(mask):
                continue

            phi_a = phi[mask]             # (N_a, phi_dim)
            y_a = targets[mask]           # (N_a,)
            w_a = q_weights[mask]         # (N_a,)

            # Scale features by sqrt of importance weights: phi_weighted = sqrt(w) * phi
            sqrt_w = np.sqrt(w_a)[:, np.newaxis]
            phi_scaled = phi_a * sqrt_w

            # Weighted outer product: Phi_a^T Q_a Phi_a
            self.Lambda[a] += phi_scaled.T @ phi_scaled
            # Weighted target vector: Phi_a^T Q_a y_a
            self.b[a] += (phi_a * w_a[:, np.newaxis]).T @ y_a

            # Analytical solve for posterior mean: mu_a = Lambda_a^{-1} b_a
            try:
                self.mu[a] = np.linalg.solve(self.Lambda[a], self.b[a])
            except np.linalg.LinAlgError:
                self.mu[a] = np.linalg.pinv(self.Lambda[a]) @ self.b[a]

    def sample_thompson_weights(self) -> np.ndarray:
        """Draw exact Thompson sample w_a ~ N(mu_a, sigma^2 * Lambda_a^{-1}) via Cholesky.

        Returns:
            sampled_weights: array of shape (action_dim, phi_dim)
        """
        for a in range(self.action_dim):
            try:
                # Cholesky of precision: Lambda_a = L @ L^T
                L = np.linalg.cholesky(self.Lambda[a])
                # Draw standard normal vector
                eps = self.rng.normal(0.0, np.sqrt(self.noise_variance), size=self.phi_dim).astype(np.float32)
                # Solve L^T * z = eps => Cov(z) = sigma^2 * Lambda_a^{-1}
                z = np.linalg.solve(L.T, eps)
                self.sampled_weights[a] = self.mu[a] + z
            except np.linalg.LinAlgError:
                # Fallback to posterior mean if singular
                self.sampled_weights[a] = self.mu[a].copy()

        return self.sampled_weights

    def predict(self, phi: np.ndarray, use_thompson: bool = True) -> np.ndarray:
        """Compute Q-values Q(s, a) = phi(s)^T w_a.

        Args:
            phi: feature vector of shape (batch, phi_dim) or (phi_dim,)
            use_thompson: if True, uses sampled w; if False, uses posterior mean mu.

        Returns:
            Q-values of shape (batch, action_dim)
        """
        if phi.ndim == 1:
            phi = phi[np.newaxis, :]

        W = self.sampled_weights if use_thompson else self.mu  # (action_dim, phi_dim)
        return phi @ W.T  # (batch, action_dim)
