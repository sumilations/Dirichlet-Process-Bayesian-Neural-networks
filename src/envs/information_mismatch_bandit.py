import numpy as np
from typing import Tuple, Dict, Any, Optional


class InformationMismatchBandit:
    """Information-Action Mismatch Bandit Benchmark (Russo & Van Roy, 2014; Lattimore & Szepesvari, 2017).
    
    A canonical benchmark proving the failure of Thompson Sampling and the optimality of IDS:
    - Latent state theta in {-1, +1} sampled with probability 0.5 at round 0.
    - Arm 0 (Safe Arm): Expected reward mu_0 = 0.0, noise sigma_0 = 0.1. Reveals zero info about theta.
    - Arm 1 (Risky Arm): Expected reward mu_1 = +Delta if theta = +1, else -Delta.
      High observation noise sigma_1 = 1.0, requiring dozens of pulls to identify theta.
    - Arm 2 (Revealing Information Arm): Expected reward mu_2 = -epsilon (small cost),
      Low observation noise sigma_2 = 0.05, and observation Y_2 = -epsilon + theta * 1.0 + N(0, sigma_2^2).
      One single pull completely reveals theta with near certainty!
      
    Why Thompson Sampling Fails:
    - In every possible state of the world, Arm 2 is strictly suboptimal (0.0 or +Delta > -epsilon).
    - Therefore P(A* = 2) = 0 under the prior and all posteriors!
    - Thompson Sampling NEVER pulls Arm 2! It must repeatedly pull the noisy Arm 1,
      incurring high cumulative regret whenever theta = -1.
    
    Why IDS Succeeds:
    - Delta(Arm 2) = epsilon is tiny.
    - Information Gain I(Arm 2) is massive (low noise, high mutual info with theta).
    - Information Ratio Psi(Arm 2) = epsilon^2 / I(Arm 2) is minimal.
    - IDS pulls Arm 2 on round 1, identifies theta with 99.9% certainty, and exploits optimal arm thereafter.
    """

    def __init__(
        self,
        delta: float = 1.0,
        epsilon: float = 0.05,
        risky_noise: float = 1.0,
        info_noise: float = 0.05,
        safe_noise: float = 0.1,
        context_dim: int = 2,
        seed: Optional[int] = None,
    ):
        self.delta = float(delta)
        self.epsilon = float(epsilon)
        self.risky_noise = float(risky_noise)
        self.info_noise = float(info_noise)
        self.safe_noise = float(safe_noise)
        self.context_dim = context_dim
        self.num_arms = 3

        self.rng = np.random.RandomState(seed)
        self.theta = 1.0
        self.reset()

    def reset(self) -> np.ndarray:
        self.theta = 1.0 if self.rng.rand() < 0.5 else -1.0
        return self.sample_context()

    def sample_context(self) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=self.context_dim).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        if action == 0:
            noise = float(self.rng.normal(0.0, self.safe_noise))
            mean_reward = 0.0
            reward = 0.0 + noise
        elif action == 1:
            noise = float(self.rng.normal(0.0, self.risky_noise))
            mean_reward = self.delta if self.theta > 0 else -self.delta
            reward = mean_reward + noise
        elif action == 2:
            noise = float(self.rng.normal(0.0, self.info_noise))
            mean_reward = -self.epsilon
            # Revealing signal: reward reveals theta directly
            reward = -self.epsilon + self.theta * 1.0 + noise
        else:
            raise ValueError(f'Invalid action {action}')

        optimal_arm = 1 if self.theta > 0 else 0
        optimal_mean = self.delta if self.theta > 0 else 0.0
        regret = optimal_mean - mean_reward

        info = {
            'theta': self.theta,
            'optimal_arm': optimal_arm,
            'optimal_mean': optimal_mean,
            'mean_reward': mean_reward,
            'regret': regret,
        }

        next_context = self.sample_context()
        return next_context, reward, optimal_mean, info
