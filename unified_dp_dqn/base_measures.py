"""Pluggable Base Measures for Unified DP-DQN."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseMeasure(ABC):
    """Abstract base class for Dirichlet Process base measure F_0."""

    @abstractmethod
    def sample(
        self,
        count: int,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample synthetic transition atoms (s, a, r, s', done) from F_0."""
        pass


class UniformBoxBaseMeasure(BaseMeasure):
    """Jaynes' Maximum-Entropy Uniform Base Measure over a Cartesian bounding box."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        low: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
        r_max: float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.low = np.full(state_dim, -1.0, dtype=np.float32) if low is None else np.asarray(low, dtype=np.float32)
        self.high = np.full(state_dim, 1.0, dtype=np.float32) if high is None else np.asarray(high, dtype=np.float32)
        self.r_max = float(r_max)

    def sample(
        self,
        count: int,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = rng.uniform(self.low, self.high, size=(count, self.state_dim)).astype(np.float32)
        a = rng.randint(0, self.action_dim, size=count)
        # Maximum-entropy reward: Uniform(0, R_max) (mean = 0.5 * R_max)
        r = rng.uniform(0.0, self.r_max, size=count).astype(np.float32)
        sn = s + rng.normal(0.0, 0.05, size=(count, self.state_dim)).astype(np.float32)
        sn = np.clip(sn, self.low, self.high)
        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done


class HaarManifoldBaseMeasure(BaseMeasure):
    """Invariant Haar Base Measure over the Lie Group / Circle Manifold S¹."""

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        r_max: float = 1.0,
        dt: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.r_max = float(r_max)
        self.dt = dt

    def sample(
        self,
        count: int,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)

        # 1. Angle strictly on S¹ manifold: theta ~ Uniform(-pi, pi)
        theta = rng.uniform(-np.pi, np.pi, size=count).astype(np.float32)
        s[:, 0] = np.cos(theta)
        s[:, 1] = np.sin(theta)

        # 2. Kinematic state coordinates
        if self.state_dim >= 5:
            theta_dot = rng.uniform(-10.0, 10.0, size=count).astype(np.float32)
            x = rng.uniform(-5.0, 5.0, size=count).astype(np.float32)
            x_dot = rng.uniform(-5.0, 5.0, size=count).astype(np.float32)
            s[:, 2] = theta_dot
            s[:, 3] = x
            s[:, 4] = x_dot

        if self.state_dim >= 6:
            s[:, 5] = rng.uniform(0.0, 1.0, size=count).astype(np.float32)

        a = rng.randint(0, self.action_dim, size=count)
        r = rng.uniform(0.0, self.r_max, size=count).astype(np.float32)

        # 3. Kinematic next state preserving S¹ manifold: theta' = theta + theta_dot * dt
        sn = s.copy()
        if self.state_dim >= 5:
            theta_next = theta + theta_dot * self.dt
            sn[:, 0] = np.cos(theta_next)
            sn[:, 1] = np.sin(theta_next)
            sn[:, 3] = np.clip(x + x_dot * self.dt, -5.0, 5.0)

        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done


class GaussianBaseMeasure(BaseMeasure):
    """Gaussian Base Measure with configurable mean and variance."""

    def __init__(self, state_dim: int, action_dim: int, prior_mean: float = 0.5, prior_std: float = 0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.prior_mean = float(prior_mean)
        self.prior_std = float(prior_std)

    def sample(
        self,
        count: int,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = rng.randn(count, self.state_dim).astype(np.float32)
        a = rng.randint(0, self.action_dim, size=count)
        r = rng.normal(self.prior_mean, self.prior_std, size=count).astype(np.float32)
        sn = s + rng.normal(0.0, 0.1, size=(count, self.state_dim)).astype(np.float32)
        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done


class ZeroBaseMeasure(BaseMeasure):
    """Zero Base Measure (represents the collapsed prior alpha -> 0)."""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def sample(
        self,
        count: int,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, self.action_dim, size=count)
        r = np.zeros(count, dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done


class EmpiricalBaseMeasure(BaseMeasure):
    """Empirical Data-Space Base Measure (Empirical Bayes / UCB Prior).

    Draws transitions (s, a, s') directly from empirical replay buffer experience,
    and assigns an uninformative optimistic prior reward r_0 ~ Uniform(0, R_max).
    Completely model-free: zero physics equations, zero simulator cheats!
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        replay=None,
        r_max: float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay = replay
        self.r_max = float(r_max)

    def set_replay(self, replay):
        self.replay = replay

    def sample(
        self,
        count: int,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.replay is not None and len(self.replay) >= count:
            s, a, _, sn, _ = self.replay.sample(count, rng)
            r = rng.uniform(0.0, self.r_max, size=count).astype(np.float32)
            done = np.zeros(count, dtype=np.float32)
            return s, a, r, sn, done
        elif self.replay is not None and len(self.replay) > 0:
            idx = rng.randint(0, len(self.replay), size=count)
            s = self.replay.states[idx].copy()
            a = self.replay.actions[idx].copy()
            sn = self.replay.next_states[idx].copy()
            r = rng.uniform(0.0, self.r_max, size=count).astype(np.float32)
            done = np.zeros(count, dtype=np.float32)
            return s, a, r, sn, done
        else:
            # Fallback before any experience is gathered: standard uninformative Gaussian
            s = rng.randn(count, self.state_dim).astype(np.float32)
            a = rng.randint(0, self.action_dim, size=count)
            r = rng.uniform(0.0, self.r_max, size=count).astype(np.float32)
            sn = s + rng.normal(0.0, 0.1, size=(count, self.state_dim)).astype(np.float32)
            done = np.zeros(count, dtype=np.float32)
            return s, a, r, sn, done


class DeepSeaBaseMeasure(BaseMeasure):
    """Max-Entropy Uniform Optimistic Transition Prior for Deep Sea without DAG.

    Samples reachable states uniformly without DAG directional constraints.
    Rewards are max-entropy uniform: r ~ Uniform(0, r_max), with strictly ZERO goal bonus.
    """

    def __init__(self, size: int = 10, r_max: float = 1.0):
        self.size = size
        state_dim = size * size
        self.state_dim = state_dim
        self.action_dim = 2
        self.r_max = float(r_max)

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        # Max-entropy uniform optimistic reward: r ~ Uniform(0, R_max)
        r = rng.uniform(0.0, self.r_max, size=count).astype(np.float32)

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


class DeepSeaDAGBaseMeasure(BaseMeasure):
    """Max-Entropy Uniform Optimistic Transition Prior for Deep Sea with DAG.

    Strictly respects the causal directed acyclic graph (DAG) structure:
    Depth row -> row + 1, and column transitions constrained to adjacent neighbors.
    Rewards are max-entropy uniform: r ~ Uniform(0, r_max), with strictly ZERO goal bonus.
    """

    def __init__(self, size: int = 10, r_max: float = 1.0):
        self.size = size
        state_dim = size * size
        self.state_dim = state_dim
        self.action_dim = 2
        self.r_max = float(r_max)

    def sample(
        self, count: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)
        # Max-entropy uniform optimistic reward: r ~ Uniform(0, R_max)
        r = rng.uniform(0.0, self.r_max, size=count).astype(np.float32)

        for i in range(count):
            row = rng.randint(0, self.size)
            col = rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                step_dir = 1 if rng.rand() < 0.5 else -1
                next_col = max(0, min(self.size - 1, col + step_dir))
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


def get_base_measure(
    name: str,
    state_dim: int,
    action_dim: int,
    low: Optional[np.ndarray] = None,
    high: Optional[np.ndarray] = None,
    r_max: float = 1.0,
    size: Optional[int] = None,
) -> BaseMeasure:
    """Factory helper to obtain a base measure instance."""
    name_clean = name.lower().strip()
    if "dag" in name_clean:
        deep_sea_size = size or int(np.sqrt(state_dim))
        return DeepSeaDAGBaseMeasure(size=deep_sea_size)
    elif "deep_sea" in name_clean or "deepsea" in name_clean:
        deep_sea_size = size or int(np.sqrt(state_dim))
        return DeepSeaBaseMeasure(size=deep_sea_size)
    elif "haar" in name_clean:
        return HaarManifoldBaseMeasure(state_dim=state_dim, action_dim=action_dim, r_max=r_max)
    elif "empirical" in name_clean:
        return EmpiricalBaseMeasure(state_dim=state_dim, action_dim=action_dim, r_max=r_max)
    elif "gaussian" in name_clean:
        return GaussianBaseMeasure(state_dim=state_dim, action_dim=action_dim)
    elif "zero" in name_clean:
        return ZeroBaseMeasure(state_dim=state_dim, action_dim=action_dim)
    else:  # default: uniform box
        return UniformBoxBaseMeasure(state_dim=state_dim, action_dim=action_dim, low=low, high=high, r_max=r_max)
