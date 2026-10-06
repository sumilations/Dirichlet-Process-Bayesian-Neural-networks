import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.dp_bnn import DPBNNAgent
from src.models.baselines import RandomizedPriorEnsembleAgent, DeepEnsembleAgent


class FineMarginBandit:
    """The Fine-Margin Razor Contextual Bandit.

    Specifically highlights the Asymptotic Contraction failure of BootDQN+RP:
    - Context: x in [-1, 1]
    - Arm 0: mu_0(x) = 5.0
    - Arm 1: mu_1(x) = 5.0 + 0.35 * sin(2 * pi * x)
    - Arms 2, 3: mu = 0.0 (traps)
    - Observation noise: 0.05
    """
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.num_arms = 4
        self.context_dim = 1

    def sample_context(self):
        return self.rng.uniform(-1.0, 1.0, size=1).astype(np.float32)

    def get_mean_rewards(self, context):
        x = context[0]
        means = np.zeros(self.num_arms, dtype=np.float32)
        means[0] = 5.0
        means[1] = 5.0 + 0.35 * np.sin(2 * np.pi * x)
        means[2] = 0.0
        means[3] = 0.0
        return means

    def step(self, context, action):
        means = self.get_mean_rewards(context)
        expected_reward = means[action]
        optimal_expected_reward = np.max(means)
        regret = optimal_expected_reward - expected_reward
        stochastic_reward = expected_reward + self.rng.normal(0.0, 0.05)
        return float(stochastic_reward), float(expected_reward), float(optimal_expected_reward), float(regret)


def test_fine_margin():
    num_steps = 2000
    seeds = [42, 100, 2024]

    dp_regrets = []
    boot_regrets = []

    for seed in seeds:
        env = FineMarginBandit(seed=seed)
        dp = DPBNNAgent(
            context_dim=1, num_arms=4, hidden_dim=64, alpha=10.0,
            truncation_K=100, prior_reward_mean=6.0, lr=0.01,
            steps_per_decision=2, seed=seed
        )
        boot = RandomizedPriorEnsembleAgent(
            context_dim=1, num_arms=4, num_models=5, hidden_dim=64,
            prior_scale=3.0, lr=0.01, steps_per_decision=2, seed=seed
        )

        dp_total = 0.0
        boot_total = 0.0

        for t in range(num_steps):
            ctx = env.sample_context()

            # DP step
            a_dp = dp.select_action(ctx)
            r_dp, _, _, reg_dp = env.step(ctx, a_dp)
            dp.update(ctx, a_dp, r_dp)
            dp_total += reg_dp

            # Boot step
            a_boot = boot.select_action(ctx)
            r_boot, _, _, reg_boot = env.step(ctx, a_boot)
            boot.update(ctx, a_boot, r_boot)
            boot_total += reg_boot

        dp_regrets.append(dp_total)
        boot_regrets.append(boot_total)
        print(f"[Seed {seed}] DP Regret: {dp_total:.2f} | BootDQN-RP Regret: {boot_total:.2f}")

    print(f"--> Final DP Mean: {np.mean(dp_regrets):.2f} vs BootDQN-RP: {np.mean(boot_regrets):.2f}")


if __name__ == "__main__":
    test_fine_margin()
