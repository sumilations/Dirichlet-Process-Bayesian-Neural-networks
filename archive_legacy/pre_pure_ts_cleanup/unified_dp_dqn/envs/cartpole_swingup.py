"""Cart-Pole Swing-Up Environment matching Section 7.2.2 of arXiv:1703.07608."""

from typing import Dict, Tuple
import numpy as np


class CartpoleSwingupEnv:
    """Exact Cart-Pole Swing-Up benchmark from Section 7.2.2 of arXiv:1703.07608.

    System Specifications:
    - Cart mass M = 1.0 kg
    - Pole mass m = 0.1 kg
    - Pole length l = 1.0 m
    - Acceleration due to gravity g = 9.8 m/s^2
    - Actions: horizontal forces in {-10.0, 0.0, 10.0} N
    - Action cost: |F_t| / 1000 per step
    - Timescale dt = 0.01 s
    - Episode horizon: t > 10.0 s (exactly 1,000 timesteps)
    - Rigid immovable rail boundaries at x in [-5.0, 5.0] m
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

        self.action_forces = [-10.0, 0.0, 10.0]
        self.action_dim = 3
        self.state_dim = 6  # [cos(theta), sin(theta), theta_dot, x, x_dot, time/10]

        self.x_threshold = 5.0
        self.rng = np.random.RandomState(seed)

        self.state = None
        self.current_step = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 0
        # s_0 = (pi, 0, 0, 0) + Unif([-0.05, 0.05]) i.i.d.
        theta = np.pi + self.rng.uniform(-0.05, 0.05)
        theta_dot = self.rng.uniform(-0.05, 0.05)
        x = self.rng.uniform(-0.05, 0.05)
        x_dot = self.rng.uniform(-0.05, 0.05)
        self.state = np.array([theta, theta_dot, x, x_dot], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        theta, theta_dot, x, x_dot = self.state
        t_norm = float(self.current_step) / float(self.max_steps)
        return np.array([
            np.cos(theta),
            np.sin(theta),
            theta_dot,
            x,
            x_dot,
            t_norm
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_step += 1
        force = self.action_forces[action]

        theta, theta_dot, x, x_dot = self.state

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        total_mass = self.M + self.m
        pole_mass_len = self.m * self.l

        # Equations of motion for inverted pendulum on a cart
        temp = (force + pole_mass_len * theta_dot**2 * sin_th) / total_mass
        theta_acc = (self.g * sin_th - cos_th * temp) / (
            self.l * (4.0 / 3.0 - self.m * cos_th**2 / total_mass)
        )
        x_acc = temp - pole_mass_len * theta_acc * cos_th / total_mass

        # Euler-Cromer integration
        x_dot = x_dot + self.dt * x_acc
        x = x + self.dt * x_dot

        theta_dot = theta_dot + self.dt * theta_acc
        theta = theta + self.dt * theta_dot
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Inelastic boundary collision at rail edges [-5.0, 5.0]
        if x < -self.x_threshold:
            x = -self.x_threshold
            x_dot = 0.0
        elif x > self.x_threshold:
            x = self.x_threshold
            x_dot = 0.0

        self.state = np.array([theta, theta_dot, x, x_dot], dtype=np.float32)

        # Reward: +1.0 for upright balance
        is_upright = bool(
            np.cos(theta) > 0.95
            and abs(theta_dot) <= 1.0
            and abs(x) <= 1.0
            and abs(x_dot) <= 1.0
        )
        action_cost = abs(force) / 1000.0
        reward = (1.0 if is_upright else 0.0) - action_cost

        done = bool(self.current_step >= self.max_steps)

        info = {
            "is_upright": is_upright,
            "cos_theta": float(np.cos(theta)),
            "x": float(x),
            "step": self.current_step
        }

        return self._get_obs(), float(reward), done, info
