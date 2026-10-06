"""
Rover / Robot Trajectory Planning Benchmark (60 Continuous Dimensions).

As described in Appendix D of the paper:
A mobile robot navigates a 2D terrain from (0, 0) to (1, 1).
The trajectory is parameterized by 30 2D waypoints (d = 60 continuous variables in [0, 1]^60).
Penalizes path length, circular obstacle collisions, and curvature.
"""

import math
import torch


class Rover60DEnvironment:
    """60-dimensional Rover navigation environment with 4 obstacle discs."""
    def __init__(self):
        self.start = torch.tensor([0.0, 0.0])
        self.goal = torch.tensor([1.0, 1.0])
        self.obstacles = [
            (torch.tensor([0.25, 0.25]), 0.14),
            (torch.tensor([0.50, 0.65]), 0.16),
            (torch.tensor([0.75, 0.40]), 0.14),
            (torch.tensor([0.45, 0.45]), 0.12),
        ]

    def evaluate(self, x: torch.Tensor) -> float:
        """
        Evaluates 60D waypoint trajectory.
        x: 1D tensor of shape [60] or 2D tensor [1, 60] with values in [0, 1].
        """
        if x.dim() == 2:
            x = x.squeeze(0)

        waypoints = x.reshape(30, 2)
        full_pts = torch.cat([self.start.unsqueeze(0), waypoints, self.goal.unsqueeze(0)], dim=0)

        # 1. Step lengths (L2 distance between consecutive waypoints)
        diffs = full_pts[1:] - full_pts[:-1]
        length_cost = torch.norm(diffs, p=2, dim=1).sum()

        # 2. Obstacle collision penalty
        collision_penalty = 0.0
        for center, radius in self.obstacles:
            dists = torch.norm(full_pts - center, p=2, dim=1)
            violations = torch.clamp(radius - dists, min=0.0)
            collision_penalty = collision_penalty + torch.sum(violations ** 2) * 80.0

        # 3. Path curvature smoothness penalty
        curv_diffs = full_pts[2:] - 2.0 * full_pts[1:-1] + full_pts[:-2]
        smoothness_penalty = torch.norm(curv_diffs, p=2, dim=1).sum() * 4.0

        total_cost = length_cost + collision_penalty + smoothness_penalty
        return total_cost.item()


def evaluate_rover_trajectory(x: torch.Tensor) -> float:
    env = Rover60DEnvironment()
    return env.evaluate(x)
