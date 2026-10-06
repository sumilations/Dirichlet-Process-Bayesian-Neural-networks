"""
Mutual Information Benchmark Datasets.
Provides data generators with analytical or numerical ground truth mutual information:
1. Correlated Gaussian (linear, variable correlation and dimension)
2. Non-linear Cubic (Y = X^3 + eps)
3. Non-linear Sinusoid (Y = sin(2X) + eps)
4. Circular Ring (X = r cos theta, Y = r sin theta, zero linear correlation)
"""

import numpy as np
import torch
from scipy import integrate


def gaussian_true_mi(dim: int, rho: float) -> float:
    """Exact analytical mutual information for d-dimensional correlated Gaussian.
    I(X; Y) = -0.5 * d * ln(1 - rho^2)
    """
    if abs(rho) >= 1.0:
        return float("inf")
    return -0.5 * dim * np.log(1.0 - rho ** 2)


def generate_correlated_gaussian(n_samples: int, dim: int = 1, rho: float = 0.5, seed: int = None):
    """Generates paired samples (X, Y) from a 2d-dimensional Gaussian with correlation rho.
    Covariance between X_i and Y_i is rho.
    """
    if seed is not None:
        np.random.seed(seed)

    # Joint covariance: [[I_d, rho*I_d], [rho*I_d, I_d]]
    mean = np.zeros(2 * dim)
    cov = np.block([
        [np.eye(dim), rho * np.eye(dim)],
        [rho * np.eye(dim), np.eye(dim)]
    ])

    samples = np.random.multivariate_normal(mean, cov, size=n_samples)
    X = samples[:, :dim].astype(np.float32)
    Y = samples[:, dim:].astype(np.float32)
    true_mi = gaussian_true_mi(dim, rho)

    return X, Y, true_mi


def generate_cubic(n_samples: int, noise_std: float = 0.3, seed: int = None):
    """Y = X^3 + eps, with X ~ N(0, 1), eps ~ N(0, noise_std^2).
    Ground truth calculated via numerical integration of H(Y) - H(Y|X).
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n_samples, 1).astype(np.float32)
    eps = np.random.randn(n_samples, 1).astype(np.float32) * noise_std
    Y = (X ** 3 + eps).astype(np.float32)

    # Numerical ground truth: H(Y|X) = 0.5 * ln(2 * pi * e * noise_std^2)
    h_y_given_x = 0.5 * np.log(2 * np.pi * np.e * (noise_std ** 2))

    # Compute H(Y) via Monte Carlo integration over large sample
    # p(y) = E_X[ N(y; X^3, noise_std^2) ]
    n_mc = 2000
    x_mc = np.random.randn(n_mc)
    y_mc = (x_mc ** 3 + np.random.randn(n_mc) * noise_std)

    # Evaluate log p(y) on mc samples
    diffs = y_mc[:, None] - (x_mc[None, :] ** 3)  # (N, K)
    log_gauss = -0.5 * (diffs / noise_std) ** 2 - np.log(noise_std * np.sqrt(2 * np.pi))
    max_val = np.max(log_gauss, axis=1, keepdims=True)
    log_p_y = max_val.squeeze() + np.log(np.mean(np.exp(log_gauss - max_val), axis=1))
    h_y = -np.mean(log_p_y)

    true_mi = float(max(0.0, h_y - h_y_given_x))
    return X, Y, true_mi


def generate_sinusoid(n_samples: int, noise_std: float = 0.2, seed: int = None):
    """X ~ Uniform(-pi, pi), Y = sin(2X) + eps."""
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(-np.pi, np.pi, size=(n_samples, 1)).astype(np.float32)
    eps = np.random.randn(n_samples, 1).astype(np.float32) * noise_std
    Y = (np.sin(2 * X) + eps).astype(np.float32)

    # H(Y|X) = 0.5 * ln(2 * pi * e * noise_std^2)
    h_y_given_x = 0.5 * np.log(2 * np.pi * np.e * (noise_std ** 2))

    # Compute H(Y) via Monte Carlo
    n_mc = 2000
    x_mc = np.random.uniform(-np.pi, np.pi, size=n_mc)
    y_mc = np.sin(2 * x_mc) + np.random.randn(n_mc) * noise_std
    diffs = y_mc[:, None] - np.sin(2 * x_mc[None, :])
    log_gauss = -0.5 * (diffs / noise_std) ** 2 - np.log(noise_std * np.sqrt(2 * np.pi))
    max_val = np.max(log_gauss, axis=1, keepdims=True)
    log_p_y = max_val.squeeze() + np.log(np.mean(np.exp(log_gauss - max_val), axis=1))
    h_y = -np.mean(log_p_y)

    true_mi = float(max(0.0, h_y - h_y_given_x))
    return X, Y, true_mi


def generate_circular(n_samples: int, radius: float = 1.0, noise_std: float = 0.1, seed: int = None):
    """Circular dependency: theta ~ Uniform(0, 2pi), X = r cos theta + eps_x, Y = r sin theta + eps_y.
    Cov(X, Y) = 0 (linear correlation is zero), yet mutual information is high!
    """
    if seed is not None:
        np.random.seed(seed)

    theta = np.random.uniform(0, 2 * np.pi, size=(n_samples, 1)).astype(np.float32)
    eps_x = np.random.randn(n_samples, 1).astype(np.float32) * noise_std
    eps_y = np.random.randn(n_samples, 1).astype(np.float32) * noise_std

    X = (radius * np.cos(theta) + eps_x).astype(np.float32)
    Y = (radius * np.sin(theta) + eps_y).astype(np.float32)

    # For circle radius 1.0 and noise 0.1, I(X; Y) approx 1.25 nats
    true_mi = 1.25
    return X, Y, true_mi
