"""Unified Dirichlet Process Deep Q-Network (DP-DQN) Framework.

A modular, extensible implementation of Data-Space Dirichlet Process Value Learning:
- F ~ DP(alpha, F_0) prior over MDP experience space
- Sethuraman stick-breaking posterior sampling q ~ GEM(alpha + B)
- Episodic Thompson Sampling via Target Warm-Start
- Modular base measures (Cart-Pole, Deep Sea, Gaussian, Zero, Custom)
- Built-in support for continuous control and discrete deep exploration
"""

from .config import DPDQNConfig
from .agent import DPDQNAgent
from .base_measures import (
    BaseMeasure,
    CartPoleResonantBaseMeasure,
    DeepSeaBaseMeasure,
    UniformGaussianBaseMeasure,
    ZeroBaseMeasure,
    CustomBaseMeasure,
    get_base_measure,
)
from .environments import make_env, EnvWrapper
from .trainer import train

__all__ = [
    "DPDQNConfig",
    "DPDQNAgent",
    "BaseMeasure",
    "CartPoleResonantBaseMeasure",
    "DeepSeaBaseMeasure",
    "UniformGaussianBaseMeasure",
    "ZeroBaseMeasure",
    "CustomBaseMeasure",
    "get_base_measure",
    "make_env",
    "EnvWrapper",
    "train",
]
