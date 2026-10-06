import numpy as np


class DeceptiveOasisBandit:
    """The 'Deceptive Oasis' Contextual Bandit Benchmark.

    Exposes the fatal blindspot of BootDQN with Randomized Priors (Global Prior Collapse)
    versus the Data-Space Dirichlet Process Prior of DP-BNN:

    - Context: x in [-1, 1]^2
    - Arm 0 ('Comfort Trap'): Returns mu = 2.0 everywhere.
    - Arm 1 ('The Oasis'): Returns mu = 1.0 in 90% of the space, BUT returns
      a massive jackpot of mu = 50.0 in a localized oasis pocket centered at [0.6, 0.6] with radius 0.35.
    - Arms 2, 3: Distractor arms with mu = 0.5 everywhere.
    - Observation noise: Gaussian sigma = 0.05.

    Why BootDQN-RP Fails (0% Discovery):
    Smooth global MLP priors generalize the 'Arm 1 is worse than Arm 0' penalty globally,
    suppressing Arm 1 across the whole space. BootDQN gets permanently locked in the comfort trap.

    Why DP-BNN Succeeds (>90% Discovery):
    Data-space base measure F_0 continuously samples optimistic synthetic coordinates across the space.
    When a context enters the oasis, the data-space prior forces local exploration, instantly uncovering
    the 50.0 jackpot and locking in exploitation.
    """

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.num_arms = 4
        self.context_dim = 2
        self.oasis_center = np.array([0.6, 0.6], dtype=np.float32)
        self.oasis_radius = 0.35

    def sample_context(self):
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    def is_in_oasis(self, context):
        return np.linalg.norm(context - self.oasis_center) <= self.oasis_radius

    def get_mean_rewards(self, context):
        means = np.array([2.0, 1.0, 0.5, 0.5], dtype=np.float32)
        if self.is_in_oasis(context):
            means[1] = 50.0  # Jackpot!
        return means

    def step(self, context, action):
        means = self.get_mean_rewards(context)
        expected_reward = means[action]
        optimal_expected_reward = np.max(means)
        regret = optimal_expected_reward - expected_reward
        stochastic_reward = expected_reward + self.rng.normal(0.0, 0.05)
        in_oasis = bool(self.is_in_oasis(context))
        is_optimal = bool(action == (1 if in_oasis else 0))

        return float(stochastic_reward), float(expected_reward), float(optimal_expected_reward), float(regret), in_oasis, is_optimal
