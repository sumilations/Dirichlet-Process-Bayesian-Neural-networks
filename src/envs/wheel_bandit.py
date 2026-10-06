import numpy as np


class WheelBandit:
    """Wheel Bandit benchmark from Riquelme, Tucker, Snoek (ICLR 2018):

    'Deep Bayesian Bandits Showdown: An Empirical Evaluation of Deep Alternative Linear Models'

    Contexts: x in R^2, uniformly sampled from the unit ball ||x||_2 <= 1.
    Arms: K = 5 (indices 0, 1, 2, 3, 4).
      - Arm 0 (Center/Safe): Returns reward with mean mu_0 = 1.2 everywhere.
      - Arms 1-4:
        - If ||x||_2 <= delta: Mean reward is mu = 1.0 for all arms 1-4.
        - If ||x||_2 > delta: Exactly one arm corresponding to the quadrant of x
          returns a high reward mu_high = 50.0. The other three arms return mu = 1.0.
    """

    def __init__(self, delta=0.7, noise_std=0.01, seed=None):
        self.delta = delta
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)
        self.num_arms = 5
        self.context_dim = 2

    def sample_context(self):
        """Uniformly sample a 2D point from the unit ball ||x|| <= 1."""
        while True:
            pt = self.rng.uniform(-1.0, 1.0, size=2)
            if np.linalg.norm(pt) <= 1.0:
                return pt.astype(np.float32)

    def get_quadrant_arm(self, context):
        """Returns the high-reward arm index (1, 2, 3, 4) for context outside delta."""
        x1, x2 = context[0], context[1]
        if x1 >= 0 and x2 >= 0:
            return 1  # Quadrant 1 (+, +)
        elif x1 < 0 and x2 >= 0:
            return 2  # Quadrant 2 (-, +)
        elif x1 < 0 and x2 < 0:
            return 3  # Quadrant 3 (-, -)
        else:
            return 4  # Quadrant 4 (+, -)

    def get_mean_rewards(self, context):
        """Compute the true expected rewards for all 5 arms given the context."""
        radius = np.linalg.norm(context)
        means = np.ones(self.num_arms, dtype=np.float32)
        means[0] = 1.2  # Arm 0 always has mean 1.2

        if radius > self.delta:
            high_arm = self.get_quadrant_arm(context)
            means[high_arm] = 50.0

        return means

    def step(self, context, action):
        """Pull arm `action` for given `context`.

        Returns:
            reward: stochastic scalar reward
            expected_reward: ground truth mean reward for the chosen action
            optimal_expected_reward: ground truth mean reward for the optimal action
            regret: optimal_expected_reward - expected_reward
        """
        means = self.get_mean_rewards(context)
        expected_reward = means[action]
        optimal_expected_reward = np.max(means)
        regret = optimal_expected_reward - expected_reward

        # Add Gaussian observation noise
        stochastic_reward = expected_reward + self.rng.normal(0.0, self.noise_std)

        return float(stochastic_reward), float(expected_reward), float(optimal_expected_reward), float(regret)
