"""
Dirichlet Process Posterior Sampling and Stick-Breaking Engine
Reference: Vashishtha PhD Thesis (Equations 3.7 - 3.12 and 4.15 - 4.16)
"""

import numpy as np
from typing import Callable, Optional, Tuple, Union, List


class BaseMeasure:
    """Base measure F_0 for Dirichlet Process priors."""
    def __init__(self, dist_type: str = "gaussian", **params):
        """
        dist_type: 'gaussian', 'beta', 'uniform', or 'optimistic'
        params:
            gaussian: loc, scale (mean and std)
            beta: a, b
            uniform: low, high
            optimistic: loc, scale (optimistic prior mean, e.g. upper bound of returns)
        """
        self.dist_type = dist_type.lower()
        self.params = params
        
        if self.dist_type in ["gaussian", "normal", "optimistic"]:
            self.loc = params.get("loc", 0.0)
            self.scale = params.get("scale", 1.0)
        elif self.dist_type == "beta":
            self.a = params.get("a", 1.0)
            self.b = params.get("b", 1.0)
        elif self.dist_type == "uniform":
            self.low = params.get("low", 0.0)
            self.high = params.get("high", 1.0)
        else:
            raise ValueError(f"Unknown base distribution type: {dist_type}")

    def sample(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample size atoms from F_0."""
        r = rng if rng is not None else np.random.default_rng()
        if self.dist_type in ["gaussian", "normal", "optimistic"]:
            return r.normal(self.loc, self.scale, size=size)
        elif self.dist_type == "beta":
            return r.beta(self.a, self.b, size=size)
        elif self.dist_type == "uniform":
            return r.uniform(self.low, self.high, size=size)

    def mean(self) -> float:
        """Analytical mean of base measure."""
        if self.dist_type in ["gaussian", "normal", "optimistic"]:
            return float(self.loc)
        elif self.dist_type == "beta":
            return float(self.a / (self.a + self.b))
        elif self.dist_type == "uniform":
            return float(0.5 * (self.low + self.high))


def sample_dp_prior(
    alpha: float,
    base_measure: BaseMeasure,
    k_trunc: int = 50,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a random discrete probability measure P ~ DP(alpha, F_0) via finite stick-breaking
    (Vashishtha thesis Eq. 3.7 - 3.8).
    
    Returns:
        weights: np.ndarray of shape (k_trunc,), sums to 1.0
        atoms: np.ndarray of shape (k_trunc,)
    """
    r = rng if rng is not None else np.random.default_rng()
    if k_trunc <= 1:
        return np.array([1.0]), np.array([base_measure.sample(1, r)[0]])
        
    # V_k ~ Beta(1, alpha)
    V = r.beta(1.0, alpha, size=k_trunc)
    # Truncation: set last V to 1.0 so remainder is fully assigned
    V[-1] = 1.0
    
    # Stick breaking weights: w_k = V_k * prod_{j < k} (1 - V_j)
    one_minus_V = 1.0 - V
    cum_rem = np.cumprod(np.concatenate(([1.0], one_minus_V[:-1])))
    weights = V * cum_rem
    
    # Ensure numerical sum to 1
    total = np.sum(weights)
    if total > 0:
        weights = weights / total
    else:
        weights = np.ones(k_trunc) / k_trunc
        
    atoms = base_measure.sample(k_trunc, r)
    return weights, atoms


def sample_dp_posterior(
    history: Union[List[float], np.ndarray],
    alpha_0: float,
    base_measure: BaseMeasure,
    k_trunc_prior: int = 50,
    max_history_len: Optional[int] = 500,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a random measure Q_n ~ DP(alpha_n, F_n) using the recursive distributional equation
    (Vashishtha thesis Eq. 3.11 - 3.12 & SBreaking.py).
    
    Q_n = V_n * delta_{X_n} + sum_{i=1}^{n-1} [V_i * prod_{j=i+1}^n (1 - V_j)] delta_{X_i}
          + [prod_{i=1}^n (1 - V_i)] * Q_0
    where V_i ~ Beta(1, alpha_0 + i) and Q_0 ~ DP(alpha_0, F_0).
    """
    r = rng if rng is not None else np.random.default_rng()
    
    obs = np.asarray(history, dtype=np.float64)
    if max_history_len is not None and len(obs) > max_history_len:
        # Subsample or take recent history if excessively long for fast execution
        obs = obs[-max_history_len:]
        
    n = len(obs)
    if n == 0:
        return sample_dp_prior(alpha_0, base_measure, k_trunc=k_trunc_prior, rng=r)
        
    # Sample prior measure Q_0 ~ DP(alpha_0, F_0)
    w_prior, atoms_prior = sample_dp_prior(alpha_0, base_measure, k_trunc=k_trunc_prior, rng=r)
    
    # V_i ~ Beta(1, alpha_0 + i) for i = 1, ..., n
    alphas = alpha_0 + np.arange(1, n + 1, dtype=np.float64)
    V = r.beta(1.0, alphas)
    
    # Compute remainder stick weights
    one_minus_V = 1.0 - V
    # Suffix cumulative products: suffix_prod[i] = prod_{j=i+1}^n (1 - V_j)
    # In reverse: cumprod of one_minus_V[::-1]
    rev_cum = np.cumprod(np.concatenate(([1.0], one_minus_V[::-1][:-1])))
    suffix_prod = rev_cum[::-1]
    
    # Weight of empirical observation X_i is V_i * prod_{j=i+1}^n (1 - V_j)
    w_obs = V * suffix_prod
    
    # Remainder stick for the prior Q_0 is prod_{i=1}^n (1 - V_i)
    stick_prior = np.prod(one_minus_V)
    w_prior_scaled = stick_prior * w_prior
    
    # Concatenate empirical atoms and prior atoms
    all_weights = np.concatenate([w_obs, w_prior_scaled])
    all_atoms = np.concatenate([obs, atoms_prior])
    
    # Normalize weights to avoid float precision drift
    s = np.sum(all_weights)
    if s > 0:
        all_weights = all_weights / s
    else:
        all_weights = np.ones_like(all_weights) / len(all_weights)
        
    return all_weights, all_atoms


def evaluate_functional(
    weights: np.ndarray,
    atoms: np.ndarray,
    functional: str = "mean",
    **kwargs
) -> float:
    """
    Evaluate statistical functional phi(Z) on a discrete random measure Z = sum w_k delta_{theta_k}.
    
    Parameters:
        functional:
            'mean': Posterior sample mean sum w_k theta_k (Thompson Sampling / DPPS)
            'quantile': Quantile at level tau (e.g. tau=0.90 for optimistic exploration)
            'cvar': Conditional Value at Risk at risk level tau (mean of best tail)
            'variance': Variance of the random measure
    """
    func = functional.lower()
    if func == "mean":
        return float(np.dot(weights, atoms))
        
    elif func == "quantile":
        tau = kwargs.get("tau", 0.5)
        # Sort atoms
        idx = np.argsort(atoms)
        sorted_atoms = atoms[idx]
        sorted_weights = weights[idx]
        cdf = np.cumsum(sorted_weights)
        target_idx = np.searchsorted(cdf, tau)
        target_idx = min(target_idx, len(sorted_atoms) - 1)
        return float(sorted_atoms[target_idx])
        
    elif func == "cvar":
        tau = kwargs.get("tau", 0.90)
        idx = np.argsort(atoms)
        sorted_atoms = atoms[idx]
        sorted_weights = weights[idx]
        cdf = np.cumsum(sorted_weights)
        mask = cdf >= tau
        if np.any(mask):
            return float(np.sum(sorted_weights[mask] * sorted_atoms[mask]) / np.sum(sorted_weights[mask]))
        return float(sorted_atoms[-1])
        
    elif func == "variance":
        m = np.dot(weights, atoms)
        return float(np.dot(weights, (atoms - m) ** 2))
        
    else:
        raise ValueError(f"Unknown functional: {functional}")
