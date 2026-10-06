"""
Bayesian Distributional Reinforcement Learning with Dirichlet Processes
Reference: Vashishtha PhD Thesis, Section 4.5 & Algorithm 5
"""

from .dp_posterior import (
    BaseMeasure,
    sample_dp_prior,
    sample_dp_posterior,
    evaluate_functional
)
from .agent import BayesianDistributionalRL_DP
from .environments import RiverSwimEnv, CliffWalkingEnv, DeepSeaTabularEnv

__all__ = [
    "BaseMeasure",
    "sample_dp_prior",
    "sample_dp_posterior",
    "evaluate_functional",
    "BayesianDistributionalRL_DP",
    "RiverSwimEnv",
    "CliffWalkingEnv",
    "DeepSeaTabularEnv"
]
