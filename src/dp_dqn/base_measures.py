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

    if name in ("default", "deep_sea", "deepsea"):
        return DeepSeaBaseMeasure(deep_sea_size, prior_reward_mean=prior_reward_mean, prior_reward_std=prior_reward_std)
    elif name in ("deep_sea_dag_maxent", "dag_maxent", "deepsea_dag_maxent", "dag_uniform_maxent"):
        return DeepSeaDAGMaxEntBaseMeasure(deep_sea_size, r_max=1.0)
    elif name in ("deep_sea_nondag_maxent", "nondag_maxent", "deepsea_nondag_maxent", "nondag_uniform_maxent", "nondag"):
        return DeepSeaNonDAGMaxEntBaseMeasure(deep_sea_size, r_max=1.0)
    elif name in ("deepsea_standard_normal", "deep_sea_standard_normal", "nondag_standard_normal", "deepsea_normal"):
        return DeepSeaNonDAGStandardNormalBaseMeasure(deep_sea_size)
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
        raise ValueError(f"Unknown base measure type: '{name}'. Choose from 'deep_sea', 'dag_maxent', 'nondag_maxent', 'deepsea_standard_normal', 'standard_normal', 'gaussian', 'normal', 'uniform_continuous', 'zero', or provide a BaseMeasure instance.")

