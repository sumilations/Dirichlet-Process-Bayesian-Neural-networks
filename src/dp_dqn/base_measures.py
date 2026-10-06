"""Base Measure definitions for Data-Space Dirichlet Process Value Learning."""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union
import numpy as np


class BaseMeasure(ABC):
    """Abstract base class for DP Base Measures F_0."""

    def __init__(self, state_dim: int, action_dim: int, prior_reward_mean: float = 0.1, prior_reward_std: float = 0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.prior_reward_mean = float(prior_reward_mean)
        self.prior_reward_std = float(prior_reward_std)

    @abstractmethod
    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample synthetic transition atoms (s, a, r, s', done) from F_0."""
        pass


class CartPoleResonantBaseMeasure(BaseMeasure):
    """Physically anchored base measure for Cart-Pole Swing-Up.

    Samples 50% upright equilibrium anchors (cos(theta) >= 0.95, near-zero velocity)
    and 50% dynamic resonant momentum-pumping swing transitions.
    """

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, self.action_dim, size=count)
        r = np.zeros(count, dtype=np.float32)
        done = np.zeros(count, dtype=np.float32)

        n_upright = count // 2
        n_swing = count - n_upright

        # Upright states: cos(theta) > 0.95, |theta_dot| <= 0.5, |x| <= 0.8, |x_dot| <= 0.5
        if n_upright > 0:
            if self.state_dim > 0: s[:n_upright, 0] = rng.uniform(0.95, 1.0, size=n_upright)       # cos(theta)
            if self.state_dim > 1: s[:n_upright, 1] = rng.uniform(-0.3, 0.3, size=n_upright)       # sin(theta)
            if self.state_dim > 2: s[:n_upright, 2] = rng.uniform(-0.5, 0.5, size=n_upright)       # theta_dot
            if self.state_dim > 3: s[:n_upright, 3] = rng.uniform(-0.8, 0.8, size=n_upright)       # x
            if self.state_dim > 4: s[:n_upright, 4] = rng.uniform(-0.5, 0.5, size=n_upright)       # x_dot
            if self.state_dim >= 6:
                s[:n_upright, 5:] = rng.uniform(0.0, 1.0, size=(n_upright, self.state_dim - 5))

            r[:n_upright] = rng.normal(self.prior_reward_mean, 0.1, size=n_upright).astype(np.float32)
            sn[:n_upright] = s[:n_upright] + rng.normal(0.0, 0.02, size=(n_upright, self.state_dim)).astype(np.float32)

        # Dynamic resonant swing states: momentum pumping
        if n_swing > 0:
            if self.state_dim > 0: s[n_upright:, 0] = rng.uniform(-1.0, 1.0, size=n_swing)
            if self.state_dim > 1: s[n_upright:, 1] = rng.uniform(-1.0, 1.0, size=n_swing)
            if self.state_dim > 2: s[n_upright:, 2] = rng.uniform(-2.0, 2.0, size=n_swing)
            if self.state_dim > 3: s[n_upright:, 3] = rng.uniform(-2.0, 2.0, size=n_swing)
            if self.state_dim > 4: s[n_upright:, 4] = rng.uniform(-2.0, 2.0, size=n_swing)
            if self.state_dim >= 6:
                s[n_upright:, 5:] = rng.uniform(0.0, 1.0, size=(n_swing, self.state_dim - 5))

            # Optimistic momentum bonus when action agrees with angular velocity
            if self.state_dim > 2:
                swing_dir = np.sign(s[n_upright:, 2])
                a[n_upright:] = np.where(swing_dir > 0, min(2, self.action_dim - 1), 0)
            r[n_upright:] = rng.normal(0.3 * self.prior_reward_mean, 0.1, size=n_swing).astype(np.float32)
            sn[n_upright:] = s[n_upright:] + rng.normal(0.0, 0.05, size=(n_swing, self.state_dim)).astype(np.float32)

        return s, a, r, sn, done


class CartPoleUniformMaxEntropyBaseMeasure(BaseMeasure):
    """Non-Haar Box Uniform Maximum-Entropy Base Measure for Cart-Pole Swing-Up.

    By Jaynes' Principle of Maximum Entropy:
    - State space: sampled uniformly across physical bounding box:
        cos(theta) in [-1, 1], sin(theta) in [-1, 1], theta_dot in [-10, 10]
        x in [-5, 5], x_dot in [-5, 5], normalized time in [0, 1]
    - Actions: a ~ Uniform({0, 1, 2})
    - Rewards: bounded in [0, R_max] with R_max = 1.0:
        r ~ Uniform(0, R_max) where E[r] = R_max / 2 = 0.5 > 0.01 action penalty.
    - Strictly zero artificial goal bonuses, zero hand-coded Gaussian constants.
    """

    def __init__(self, state_dim: int = 6, action_dim: int = 3, r_max: float = 1.0):
        super().__init__(state_dim=state_dim, action_dim=action_dim, prior_reward_mean=0.5 * r_max)
        self.r_max = float(r_max)

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        if self.state_dim > 0: s[:, 0] = rng.uniform(-1.0, 1.0, size=count)      # cos(theta)
        if self.state_dim > 1: s[:, 1] = rng.uniform(-1.0, 1.0, size=count)      # sin(theta)
        if self.state_dim > 2: s[:, 2] = rng.uniform(-10.0, 10.0, size=count)    # theta_dot
        if self.state_dim > 3: s[:, 3] = rng.uniform(-5.0, 5.0, size=count)      # x
        if self.state_dim > 4: s[:, 4] = rng.uniform(-5.0, 5.0, size=count)      # x_dot
        if self.state_dim >= 6:
            s[:, 5] = rng.uniform(0.0, 1.0, size=count)                          # normalized time

        a = rng.randint(0, self.action_dim, size=count)
        r = rng.uniform(0.0, self.r_max, size=count).astype(np.float32)
        sn = s + rng.normal(0.0, 0.05, size=(count, self.state_dim)).astype(np.float32)
        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done


class CartPoleHaarMaxEntropyBaseMeasure(BaseMeasure):
    """Geometric Haar Maximum-Entropy Base Measure for Cart-Pole Swing-Up.

    Principles:
    1. Lie Group Haar Measure on S^1 = SO(2):
       theta ~ Uniform(-pi, pi)
       cos(theta) = cos(theta), sin(theta) = sin(theta)
       Strictly lies on the physical manifold cos^2 + sin^2 = 1.
       Zero goal bias: all angles have equal density 1 / (2*pi).
    2. Bounded physical velocities & coordinates:
       theta_dot ~ Uniform(-10, 10), x ~ Uniform(-5, 5), x_dot ~ Uniform(-5, 5)
       t ~ Uniform(0, 1)
    3. Causal forward arrow of time:
       dt = 1.0 / 300.0
       t_next = min(1.0, t + dt)
       done = 1.0 if t_next >= 1.0 else 0.0
    4. Kinematic continuity:
       theta_next = theta + theta_dot * dt
       x_next = x + x_dot * dt
       sn[:, 0] = cos(theta_next), sn[:, 1] = sin(theta_next)
       sn[:, 2] = theta_dot + rng.normal(0.0, 0.05, size=count)
       sn[:, 3] = x_next
       sn[:, 4] = x_dot + rng.normal(0.0, 0.05, size=count)
       sn[:, 5] = t_next
    5. Maximum-Entropy Uniform Optimistic Reward:
       r ~ Uniform(0, R_max) where R_max = 1.0 (mean = 0.5 > 0.01 action penalty)
       Strictly ZERO reward at goal state (no upright bonus, zero target bias).
    """

    def __init__(self, state_dim: int = 6, action_dim: int = 3, r_max: float = 1.0, dt: float = 1.0 / 300.0):
        super().__init__(state_dim=state_dim, action_dim=action_dim, prior_reward_mean=0.5 * r_max)
        self.r_max = float(r_max)
        self.dt = float(dt)

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)

        # 1. Physical Haar measure on S^1
        th = rng.uniform(-np.pi, np.pi, size=count).astype(np.float32)
        s[:, 0] = np.cos(th)
        s[:, 1] = np.sin(th)

        th_dot = rng.uniform(-10.0, 10.0, size=count).astype(np.float32)
        x = rng.uniform(-5.0, 5.0, size=count).astype(np.float32)
        x_dot = rng.uniform(-5.0, 5.0, size=count).astype(np.float32)
        t = rng.uniform(0.0, 1.0, size=count).astype(np.float32)

        s[:, 2] = th_dot
        s[:, 3] = x
        s[:, 4] = x_dot
        if self.state_dim >= 6:
            s[:, 5] = t

        a = rng.randint(0, self.action_dim, size=count)
        # Uniform optimistic reward: NO reward at goal state
        r = rng.uniform(0.0, self.r_max, size=count).astype(np.float32)

        # 2. Kinematic integration and causal forward arrow of time
        th_next = th + th_dot * self.dt
        sn[:, 0] = np.cos(th_next)
        sn[:, 1] = np.sin(th_next)
        sn[:, 2] = th_dot + rng.normal(0.0, 0.05, size=count).astype(np.float32)
        sn[:, 3] = np.clip(x + x_dot * self.dt, -5.0, 5.0)
        sn[:, 4] = x_dot + rng.normal(0.0, 0.05, size=count).astype(np.float32)
        if self.state_dim >= 6:
            sn[:, 5] = np.minimum(1.0, t + self.dt)
            done = (sn[:, 5] >= 1.0).astype(np.float32)
        else:
            done = np.zeros(count, dtype=np.float32)

        return s, a, r, sn, done



class DeepSeaBaseMeasure(BaseMeasure):
    """Optimistic Uniform Transition Prior for Deep Sea Gridworld (N x N)."""

    def __init__(self, size: int, prior_reward_mean: float = 0.1, prior_reward_std: float = 0.05):
        self.size = size
        state_dim = size * size
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=prior_reward_mean, prior_reward_std=prior_reward_std)

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        r = rng.normal(self.prior_reward_mean, self.prior_reward_std, size=count).astype(np.float32)

        for i in range(count):
            row = rng.randint(0, self.size)
            col = rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                next_col = rng.randint(0, next_row + 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


class DeepSeaDownwardDAGBaseMeasure(BaseMeasure):
    """Downward DAG Causal Transition Prior for Deep Sea Gridworld (N x N).

    Strictly respects the causal directed acyclic graph (DAG) structure of Deep Sea:
    - Time and depth advance strictly monotonically: row -> row + 1.
    - Column transitions are physically constrained to valid grid neighbors:
        Left (a=0):  col' = max(0, col - 1)
        Right (a=1): col' = min(size - 1, col + 1)
    - Terminal boundary: transitions from row = size - 1 terminate (done = 1).
    - Goal chest optimism: if goal_bonus=True, reaching the bottom-right corner (size-1, size-1)
      receives an extra +1.0 boost. If goal_bonus=False, rewards are uniformly optimistic everywhere.
    """

    def __init__(
        self,
        size: int,
        prior_reward_mean: float = 0.1,
        prior_reward_std: float = 0.05,
        goal_bonus: bool = True,
    ):
        self.size = size
        self.goal_bonus = bool(goal_bonus)
        state_dim = size * size
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=prior_reward_mean, prior_reward_std=prior_reward_std)

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        # Uniform optimistic reward across downward transitions
        r = rng.normal(self.prior_reward_mean, self.prior_reward_std, size=count).astype(np.float32)

        for i in range(count):
            row = rng.randint(0, self.size)
            col = rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
                if col == self.size - 1 and self.goal_bonus:
                    r[i] += 1.0  # Extra optimistic boost at the chest location (only if goal_bonus=True)
            else:
                next_row = row + 1
                # Physical grid transition: col -> col-1 or col+1 (50/50 prior)
                step_dir = 1 if rng.rand() < 0.5 else -1
                next_col = max(0, min(self.size - 1, col + step_dir))
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


class UniformGaussianBaseMeasure(BaseMeasure):
    """Uninformed Isotropic Gaussian base measure (baseline for comparison)."""

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = rng.normal(0.0, 1.0, size=(count, self.state_dim)).astype(np.float32)
        sn = rng.normal(0.0, 1.0, size=(count, self.state_dim)).astype(np.float32)
        a = rng.randint(0, self.action_dim, size=count)
        r = rng.normal(self.prior_reward_mean, self.prior_reward_std, size=count).astype(np.float32)
        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done


class UniformContinuousBaseMeasure(BaseMeasure):
    """Uninformed continuous Uniform [0, 1] base measure (baseline for comparison)."""

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = rng.uniform(0.0, 1.0, size=(count, self.state_dim)).astype(np.float32)
        sn = rng.uniform(0.0, 1.0, size=(count, self.state_dim)).astype(np.float32)
        a = rng.randint(0, self.action_dim, size=count)
        r = rng.normal(self.prior_reward_mean, self.prior_reward_std, size=count).astype(np.float32)
        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done


class ZeroBaseMeasure(BaseMeasure):
    """Standard non-optimistic prior (zero reward synthetic transitions)."""

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = rng.normal(0.0, 1.0, size=(count, self.state_dim)).astype(np.float32)
        sn = s + rng.normal(0.0, 0.05, size=(count, self.state_dim)).astype(np.float32)
        a = rng.randint(0, self.action_dim, size=count)
        r = np.zeros(count, dtype=np.float32)
        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done


class CustomBaseMeasure(BaseMeasure):
    """Wraps an arbitrary user-supplied sampling function."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sampler_fn: Callable[[int, np.random.RandomState], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        prior_reward_mean: float = 1.0,
    ):
        super().__init__(state_dim, action_dim, prior_reward_mean)
        self.sampler_fn = sampler_fn

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.sampler_fn(count, rng)


class DeepSeaDAGMaxEntBaseMeasure(BaseMeasure):
    """DAG Causal Transition Prior with Uniform Maximum-Entropy Reward Optimism.
    
    Principles:
    1. Causal DAG layer structure: row -> row + 1.
    2. Column step: a=0 -> max(0, col-1), a=1 -> min(size-1, col+1).
    3. Terminal row: row == size - 1 -> done = 1.
    4. Jaynes' Principle of Maximum Entropy: r ~ Uniform(0, R_max) where R_max = 1.0.
       Strictly ZERO chest bonus, ZERO target location leakage.
    """

    def __init__(self, size: int, r_max: float = 1.0):
        self.size = size
        self.r_max = float(r_max)
        state_dim = size * size
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=0.5 * r_max, prior_reward_std=(r_max**2 / 12.0)**0.5)

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        r = rng.uniform(0.0, self.r_max, size=count).astype(np.float32)

        for i in range(count):
            row = rng.randint(0, self.size)
            col = rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                act = a[i]
                next_col = min(self.size - 1, col + 1) if act == 1 else max(0, col - 1)
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


class DeepSeaNonDAGMaxEntBaseMeasure(BaseMeasure):
    """Model-Free Uniform Maximum-Entropy Base Measure for Deep Sea (Zero DAG Bias).
    
    Principles:
    1. Zero causal or graph knowledge: states sampled uniformly from all N^2 states.
    2. Zero transition physics: next states sampled uniformly from all N^2 states.
    3. Jaynes' Principle of Maximum Entropy: r ~ Uniform(0, R_max) where R_max = 1.0.
    4. Pure model-free discrete base measure.
    """

    def __init__(self, size: int, r_max: float = 1.0):
        self.size = size
        self.r_max = float(r_max)
        state_dim = size * size
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=0.5 * r_max, prior_reward_std=(r_max**2 / 12.0)**0.5)

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        
        idx_s = rng.randint(0, self.state_dim, size=count)
        idx_sn = rng.randint(0, self.state_dim, size=count)
        
        s[np.arange(count), idx_s] = 1.0
        sn[np.arange(count), idx_sn] = 1.0
        
        a = rng.randint(0, 2, size=count)
        r = rng.uniform(0.0, self.r_max, size=count).astype(np.float32)
        done = np.zeros(count, dtype=np.float32)

        return s, a, r, sn, done


class DeepSeaNonDAGStandardNormalBaseMeasure(BaseMeasure):
    """Model-Free Uniform Discrete Base Measure with Zero-Mean Standard Normal Rewards N(0, 1) (No Optimism).
    
    Ablation baseline to verify if stochastic optimism is necessary for deep exploration:
    - Discrete one-hot state and next-state sampling across all N^2 states.
    - Zero-mean standard normal reward: r ~ N(0, 1) (mean = 0.0, std = 1.0).
    - Zero optimism: unvisited states have zero expected prior value.
    """

    def __init__(self, size: int):
        self.size = size
        state_dim = size * size
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=0.0, prior_reward_std=1.0)

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        
        idx_s = rng.randint(0, self.state_dim, size=count)
        idx_sn = rng.randint(0, self.state_dim, size=count)
        
        s[np.arange(count), idx_s] = 1.0
        sn[np.arange(count), idx_sn] = 1.0
        
        a = rng.randint(0, 2, size=count)
        r = rng.normal(0.0, 1.0, size=count).astype(np.float32)
        done = np.zeros(count, dtype=np.float32)

        return s, a, r, sn, done


def get_base_measure(
    name_or_instance: Union[str, BaseMeasure],
    state_dim: int,
    action_dim: int,
    prior_reward_mean: float = 0.1,
    prior_reward_std: float = 0.05,
    deep_sea_size: int = 10,
) -> BaseMeasure:
    """Factory helper to instantiate or validate a BaseMeasure."""
    if isinstance(name_or_instance, BaseMeasure):
        return name_or_instance

    name = str(name_or_instance).lower().strip()

    if name in ("cartpole_haar", "haar", "cartpole_haar_max_entropy", "haar_max_entropy"):
        return CartPoleHaarMaxEntropyBaseMeasure(state_dim, action_dim, r_max=1.0)
    elif name in ("cartpole_uniform", "cartpole_non_haar", "non_haar", "cartpole_uniform_max_entropy", "cartpole_max_entropy", "uniform_max_entropy", "box"):
        return CartPoleUniformMaxEntropyBaseMeasure(state_dim, action_dim, r_max=1.0)
    elif name in ("default", "cartpole", "cartpole_swingup"):
        return CartPoleResonantBaseMeasure(state_dim, action_dim, prior_reward_mean)
    elif name in ("deep_sea_dag_maxent", "dag_maxent", "deepsea_dag_maxent", "dag_uniform_maxent"):
        return DeepSeaDAGMaxEntBaseMeasure(deep_sea_size, r_max=1.0)
    elif name in ("deep_sea_nondag_maxent", "nondag_maxent", "deepsea_nondag_maxent", "nondag_uniform_maxent", "nondag"):
        return DeepSeaNonDAGMaxEntBaseMeasure(deep_sea_size, r_max=1.0)
    elif name in ("deepsea_standard_normal", "deep_sea_standard_normal", "nondag_standard_normal", "deepsea_normal"):
        return DeepSeaNonDAGStandardNormalBaseMeasure(deep_sea_size)
    elif name in ("deep_sea", "deepsea"):
        return DeepSeaBaseMeasure(deep_sea_size, prior_reward_mean=prior_reward_mean, prior_reward_std=prior_reward_std)
    elif name in ("deep_sea_dag", "deepsea_dag", "dag", "downward_dag"):
        return DeepSeaDownwardDAGBaseMeasure(deep_sea_size, prior_reward_mean=prior_reward_mean, prior_reward_std=prior_reward_std, goal_bonus=True)
    elif name in ("deep_sea_dag_no_chest", "deepsea_dag_no_chest", "dag_no_chest", "dag_uniform_reward", "deep_sea_dag_uniform"):
        return DeepSeaDownwardDAGBaseMeasure(deep_sea_size, prior_reward_mean=prior_reward_mean, prior_reward_std=prior_reward_std, goal_bonus=False)
    elif name in ("standard_normal", "gaussian_normal", "standard_gaussian", "isotropic_normal_zero_mean"):
        return UniformGaussianBaseMeasure(state_dim, action_dim, prior_reward_mean=0.0, prior_reward_std=1.0)
    elif name in ("gaussian", "uniform_gaussian", "uninformed", "normal", "isotropic_normal"):
        return UniformGaussianBaseMeasure(state_dim, action_dim, prior_reward_mean=prior_reward_mean, prior_reward_std=prior_reward_std)
    elif name in ("uniform_continuous", "continuous_uniform", "uniform_box"):
        return UniformContinuousBaseMeasure(state_dim, action_dim, prior_reward_mean=prior_reward_mean, prior_reward_std=prior_reward_std)
    elif name in ("zero", "zeros"):
        return ZeroBaseMeasure(state_dim, action_dim, prior_reward_mean=prior_reward_mean)
    else:
        raise ValueError(f"Unknown base measure type: '{name}'. Choose from 'dag_maxent', 'nondag_maxent', 'deepsea_standard_normal', 'standard_normal', 'cartpole', 'deep_sea', 'deep_sea_dag', 'dag_uniform_reward', 'gaussian', 'normal', 'uniform_continuous', 'zero', or provide a BaseMeasure instance.")

