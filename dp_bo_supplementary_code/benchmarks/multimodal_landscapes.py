"""
Canonical Multimodal Benchmark Objectives (Ackley 2D, Levy 2D, Rosenbrock 4D, Rastrigin 4D).
"""

import math
import torch


def ackley_2d(X: torch.Tensor) -> torch.Tensor:
    """
    Ackley function in 2D.
    Global minimum f(0, 0) = 0. Domain [-5, 5]^2.
    """
    x = X[:, 0:1]
    y = X[:, 1:2]
    term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(0.5 * (x ** 2 + y ** 2)))
    term2 = -torch.exp(0.5 * (torch.cos(2.0 * math.pi * x) + torch.cos(2.0 * math.pi * y)))
    return term1 + term2 + math.e + 20.0


def levy_2d(X: torch.Tensor) -> torch.Tensor:
    """
    Levy function in 2D.
    Global minimum f(1, 1) = 0. Domain [-5, 5]^2.
    """
    x = X[:, 0:1]
    y = X[:, 1:2]
    w1 = 1.0 + (x - 1.0) / 4.0
    w2 = 1.0 + (y - 1.0) / 4.0
    term1 = torch.sin(math.pi * w1) ** 2
    term2 = (w1 - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(math.pi * w1 + 1.0) ** 2)
    term3 = (w2 - 1.0) ** 2 * (1.0 + torch.sin(2.0 * math.pi * w2) ** 2)
    return term1 + term2 + term3


def rosenbrock_4d(X: torch.Tensor) -> torch.Tensor:
    """
    Rosenbrock function in 4D.
    Global minimum f(1, 1, 1, 1) = 0. Domain [-2, 2]^4.
    """
    total = 0.0
    for i in range(3):
        xi = X[:, i:i+1]
        xnext = X[:, i+1:i+2]
        total = total + 100.0 * (xnext - xi ** 2) ** 2 + (1.0 - xi) ** 2
    return total


def rastrigin_4d(X: torch.Tensor) -> torch.Tensor:
    """
    Rastrigin function in 4D.
    Global minimum f(0, 0, 0, 0) = 0. Domain [-4, 4]^4.
    """
    d = X.shape[1]
    total = 10.0 * d
    for i in range(d):
        xi = X[:, i:i+1]
        total = total + (xi ** 2 - 10.0 * torch.cos(2.0 * math.pi * xi))
    return total


def get_multimodal_benchmark(name: str):
    """Factory function for benchmark functions, domains, and suboptimal initialization boxes."""
    if name.lower() == "ackley":
        return {
            "name": "Ackley 2D",
            "fn": ackley_2d,
            "dim": 2,
            "bounds": (-5.0, 5.0),
            "init_box": (3.0, 4.5),
            "opt_val": 0.0,
        }
    elif name.lower() == "levy":
        return {
            "name": "Levy 2D",
            "fn": levy_2d,
            "dim": 2,
            "bounds": (-5.0, 5.0),
            "init_box": (-4.5, -2.5),
            "opt_val": 0.0,
        }
    elif name.lower() == "rosenbrock":
        return {
            "name": "Rosenbrock 4D",
            "fn": rosenbrock_4d,
            "dim": 4,
            "bounds": (-2.0, 2.0),
            "init_box": (-2.0, -1.0),
            "opt_val": 0.0,
        }
    elif name.lower() == "rastrigin":
        return {
            "name": "Rastrigin 4D",
            "fn": rastrigin_4d,
            "dim": 4,
            "bounds": (-4.0, 4.0),
            "init_box": (2.0, 4.0),
            "opt_val": 0.0,
        }
    else:
        raise ValueError(f"Unknown benchmark: {name}")
