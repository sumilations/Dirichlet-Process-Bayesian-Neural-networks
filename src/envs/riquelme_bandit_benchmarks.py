"""Riquelme et al. (ICLR 2018) 'Deep Bayesian Bandits Showdown' Environments.

Implements benchmark environments from:
'Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling'
(Carlos Riquelme, George Tucker, Jasper Snoek, ICLR 2018)

Benchmarks:
1. MushroomBandit (d=117, K=2)
2. ShuttleBandit (d=9, K=7)
3. AdultBandit (d=93, K=14)
4. WheelBandit (d=2, K=5)
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd


class MushroomBandit:
    """Mushroom Bandit benchmark from Blundell et al. (2015) & Riquelme et al. (2018).
    
    Context: 22 categorical attributes one-hot encoded into d=117 binary features.
    Actions: K=2:
      - Action 0: Don't eat -> Reward = 0.
      - Action 1: Eat ->
          - If edible: Reward = +5.0
          - If poisonous: Reward = +5.0 (prob 0.5), -35.0 (prob 0.5) [Expected = -15.0]
    
    Optimal Policy:
      - Edible: Eat (E[r] = 5.0)
      - Poisonous: Don't eat (E[r] = 0.0)
    """

    def __init__(self, data_path: str = "data/agaricus-lepiota.data", seed: int = 42):
        self.rng = np.random.RandomState(seed)
        df = pd.read_csv(data_path, header=None)
        self.labels = (df[0] == "e").astype(int).values  # 1: edible, 0: poisonous
        self.features = pd.get_dummies(df.drop(columns=[0])).values.astype(np.float32)
        self.n_samples = len(self.labels)
        self.context_dim = self.features.shape[1]
        self.num_arms = 2
        self.current_idx = 0
        self.order = self.rng.permutation(self.n_samples)

    def sample_context(self) -> np.ndarray:
        idx = self.order[self.current_idx % self.n_samples]
        self.current_idx += 1
        return self.features[idx]

    def step(self, context: np.ndarray, action: int) -> Tuple[float, float, float, float]:
        # Identify true label: recover index from previous sample
        idx = self.order[(self.current_idx - 1) % self.n_samples]
        is_edible = bool(self.labels[idx] == 1)

        if is_edible:
            mean_rewards = np.array([0.0, 5.0], dtype=np.float32)
            if action == 1:
                stochastic_reward = 5.0
            else:
                stochastic_reward = 0.0
        else:
            mean_rewards = np.array([0.0, -15.0], dtype=np.float32)
            if action == 1:
                stochastic_reward = 5.0 if self.rng.rand() < 0.5 else -35.0
            else:
                stochastic_reward = 0.0

        expected_reward = float(mean_rewards[action])
        optimal_expected_reward = float(np.max(mean_rewards))
        regret = optimal_expected_reward - expected_reward
        return stochastic_reward, expected_reward, optimal_expected_reward, regret


class ShuttleBandit:
    """Statlog Shuttle Bandit benchmark from Riquelme et al. (2018).
    
    Context: 9 numerical indicators of radiator subsystem.
    Actions: K=7 subsystem states (classes 0 through 6).
    Class 0 is dominant (~78.4% frequency).
    Reward: 1 if chosen action == true state, 0 otherwise.
    """

    def __init__(self, data_path: str = "data/shuttle.trn", seed: int = 42):
        self.rng = np.random.RandomState(seed)
        df = pd.read_csv(data_path, sep=" ", header=None)
        raw_x = df.iloc[:, :9].values.astype(np.float32)
        # Normalize features to zero mean, unit variance
        self.features = (raw_x - np.mean(raw_x, axis=0)) / (np.std(raw_x, axis=0) + 1e-6)
        self.labels = (df.iloc[:, 9].values - 1).astype(int)  # 0 to 6
        self.n_samples = len(self.labels)
        self.context_dim = 9
        self.num_arms = 7
        self.current_idx = 0
        self.order = self.rng.permutation(self.n_samples)

    def sample_context(self) -> np.ndarray:
        idx = self.order[self.current_idx % self.n_samples]
        self.current_idx += 1
        return self.features[idx]

    def step(self, context: np.ndarray, action: int) -> Tuple[float, float, float, float]:
        idx = self.order[(self.current_idx - 1) % self.n_samples]
        true_label = self.labels[idx]
        reward = 1.0 if action == true_label else 0.0
        expected_reward = reward
        optimal_expected_reward = 1.0
        regret = optimal_expected_reward - expected_reward
        return reward, expected_reward, optimal_expected_reward, regret


class AdultBandit:
    """Adult Census Bandit benchmark from Riquelme et al. (2018).
    
    Predicts 1 of 14 occupations from demographic covariates.
    Context: d=93 binarized features.
    Actions: K=14 occupations.
    Reward: 1 if chosen occupation == true occupation, 0 otherwise.
    """

    def __init__(self, data_path: str = "data/adult.data", seed: int = 42):
        self.rng = np.random.RandomState(seed)
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        df = pd.read_csv(data_path, names=columns, sep=', ', engine='python')
        df = df[df['occupation'] != '?'].reset_index(drop=True)
        unique_occs = sorted(df['occupation'].unique())
        occ_to_idx = {occ: i for i, occ in enumerate(unique_occs)}
        self.labels = df['occupation'].map(occ_to_idx).values.astype(int)

        X_df = df.drop(columns=['occupation'])
        X_encoded = pd.get_dummies(X_df).values.astype(np.float32)
        # Normalize continuous columns
        self.features = (X_encoded - np.mean(X_encoded, axis=0)) / (np.std(X_encoded, axis=0) + 1e-6)
        self.n_samples = len(self.labels)
        self.context_dim = self.features.shape[1]
        self.num_arms = len(unique_occs)
        self.current_idx = 0
        self.order = self.rng.permutation(self.n_samples)

    def sample_context(self) -> np.ndarray:
        idx = self.order[self.current_idx % self.n_samples]
        self.current_idx += 1
        return self.features[idx]

    def step(self, context: np.ndarray, action: int) -> Tuple[float, float, float, float]:
        idx = self.order[(self.current_idx - 1) % self.n_samples]
        true_label = self.labels[idx]
        reward = 1.0 if action == true_label else 0.0
        expected_reward = reward
        optimal_expected_reward = 1.0
        regret = optimal_expected_reward - expected_reward
        return reward, expected_reward, optimal_expected_reward, regret
