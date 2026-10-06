"""Cart-Pole Swing-Up Environment matching Section 7.2.2 of arXiv:1703.07608.

Reference:
"Deep Exploration via Randomized Value Functions"
Ian Osband, Benjamin Van Roy, Daniel Russo, Zheng Wen (2017/2019).
Section 7.2.2: Cartpole swing up.
"""

from typing import Dict, Tuple
import numpy as np


class CartpoleSwingupJMLR:
    """Exact Cart-Pole Swing-Up benchmark from Section 7.2.2 of arXiv:1703.07608.

    System Specifications:
    - Cart mass M = 1.0 kg
    - Pole mass m = 0.1 kg
    - Pole length l = 1.0 m (effective half-length / center of mass)
    - Acceleration due to gravity g = 9.8 m/s^2
    - Actions: horizontal forces F_t in {-10.0, 0.0, 10.0} N
    - Action cost: |F_t| / 1000 per step (0.01 for +-10N, 0.0 for 0N)
    - Timescale dt = 0.01 s
    - Episode horizon: t > 10.0 s (exactly 1,000 timesteps)
    - Rigid immovable rail boundaries at x in [-5.0, 5.0] m
    - Initial state: s_0 = (pi, 0, 0, 0) + Unif([-0.05, 0.05]) i.i.d.
    - Reward: +1.0 received when cos(theta) > 0.95 and |theta_dot| <= 1.0, |x| <= 1.0, |x_dot| <= 1.0
    """

    def __init__(self, seed: int = None):
        self.M = 1.0
        self.m = 0.1
        self.l = 1.0
        self.g = 9.8
        self.dt = 0.01
        self.max_time = 10.0
        self.max_steps = int(self.max_time / self.dt)  # 1000 steps

        # Actions: 0 -> -10N, 1 -> 0N, 2 -> +10N
        self.action_forces = [-10.0, 0.0, 10.0]
        self.action_dim = len(self.action_forces)

        # Observation: [cos(theta), sin(theta), theta_dot, x, x_dot, time_elapsed / 10.0]
        self.state_dim = 6

        self.rng = np.random.RandomState(seed)
        self.theta = np.pi
        self.theta_dot = 0.0
        self.x = 0.0
        self.x_dot = 0.0
        self.time = 0.0
        self.step_count = 0
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state: hanging downwards (theta ~ pi) with noise."""
        w = self.rng.uniform(-0.05, 0.05, size=4)
        self.theta = float(np.pi + w[0])
        self.theta_dot = float(w[1])
        self.x = float(w[2])
        self.x_dot = float(w[3])
        self.time = 0.0
        self.step_count = 0
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        """Returns normalized 6-dimensional observation vector."""
        return np.array([
            float(np.cos(self.theta)),
            float(np.sin(self.theta)),
            float(self.theta_dot),
            float(self.x),
            float(self.x_dot),
            float(self.time / self.max_time)
        ], dtype=np.float32)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one timestep of cartpole physics."""
        force = self.action_forces[action_idx]

        # Continuous equations of motion (Euler-Lagrange for cartpole, Eq 7.2 in arXiv:1703.07608)
        cos_th = np.cos(self.theta)
        sin_th = np.sin(self.theta)
        half_l = self.l / 2.0  # Center-of-mass half-length = 0.5m matching Eq 7.2 and bsuite
        pl = self.m * half_l
        m_total = self.M + self.m

        temp = (force + pl * (self.theta_dot ** 2) * sin_th) / m_total
        theta_acc = (self.g * sin_th - cos_th * temp) / (half_l * (4.0 / 3.0 - self.m * (cos_th ** 2) / m_total))
        x_acc = temp - pl * theta_acc * cos_th / m_total

        # Discrete numerical integration with timescale dt = 0.01
        self.x += self.dt * self.x_dot
        self.x_dot += self.dt * x_acc
        self.theta += self.dt * self.theta_dot
        self.theta_dot += self.dt * theta_acc
        self.time += self.dt
        self.step_count += 1

        # Rigid immovable rail boundaries at x in [-5.0, 5.0]
        if self.x <= -5.0:
            self.x = -5.0
            self.x_dot = 0.0
        elif self.x >= 5.0:
            self.x = 5.0
            self.x_dot = 0.0

        # Precision target reward criteria from Section 7.2.2 Footnote 10:
        # cos(theta) > 0.95, |theta_dot| <= 1.0, |x| <= 1.0, |x_dot| <= 1.0
        is_upright = (
            np.cos(self.theta) > 0.95
            and abs(self.theta_dot) <= 1.0
            and abs(self.x) <= 1.0
            and abs(self.x_dot) <= 1.0
        )
        reward = 1.0 if is_upright else 0.0

        # Action cost: |F_t| / 1000
        cost = abs(force) / 1000.0
        step_reward = reward - cost

        done = (self.step_count >= self.max_steps or self.time >= self.max_time)

        info = {
            "is_upright": bool(is_upright),
            "cos_theta": float(np.cos(self.theta)),
            "cost": float(cost),
            "raw_reward": float(reward),
            "x": float(self.x)
        }

        return self.get_obs(), float(step_reward), bool(done), info
