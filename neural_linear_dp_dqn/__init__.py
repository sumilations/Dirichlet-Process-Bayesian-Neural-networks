"""Neural-Linear Dirichlet Process Deep Q-Network (Neural-Linear DP-DQN).

Closed-form analytical Bayesian linear projection combined with
non-parametric data-space Dirichlet Process priors.
"""

from .model import NeuralFeatureNet
from .bayesian_linear import BayesianLinearHead
from .agent import NeuralLinearDPDQNAgent

__all__ = [
    "NeuralFeatureNet",
    "BayesianLinearHead",
    "NeuralLinearDPDQNAgent",
]
