import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .dp_bnn import ContextualMLP


class EpsilonGreedyAgent:
    """Standard Neural Network with Epsilon-Greedy Exploration."""

    def __init__(
        self,
        context_dim=2,
        num_arms=5,
        hidden_dim=64,
        epsilon=0.05,
        lr=0.01,
        weight_decay=1e-4,
        batch_size=64,
        steps_per_decision=5,
        seed=None
    ):
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.steps_per_decision = steps_per_decision

        self.rng = np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.model = ContextualMLP(context_dim, num_arms, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.contexts = []
        self.actions = []
        self.rewards = []

    def select_action(self, context):
        if len(self.rewards) < self.num_arms:
            return len(self.rewards)

        if self.rng.rand() < self.epsilon:
            return self.rng.randint(0, self.num_arms)

        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(context).unsqueeze(0)
            preds = self.model(x_t).squeeze(0).numpy()
            return int(np.argmax(preds))

    def update(self, context, action, reward):
        self.contexts.append(context)
        self.actions.append(action)
        self.rewards.append(reward)

        n = len(self.rewards)
        if n < self.num_arms:
            return

        self.model.train()
        for _ in range(self.steps_per_decision):
            batch_indices = self.rng.choice(n, size=min(n, self.batch_size), replace=True)
            batch_x = torch.from_numpy(np.array([self.contexts[i] for i in batch_indices], dtype=np.float32))
            batch_a = torch.from_numpy(np.array([self.actions[i] for i in batch_indices], dtype=np.int64))
            batch_r = torch.from_numpy(np.array([self.rewards[i] for i in batch_indices], dtype=np.float32))

            self.optimizer.zero_grad()
            preds = self.model(batch_x)
            chosen_preds = preds.gather(1, batch_a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(chosen_preds, batch_r)
            loss.backward()
            self.optimizer.step()


class DeepEnsembleAgent:
    """Deep Ensemble / Bootstrapped Neural Network Agent.

    Maintains M independent neural networks with randomized initializations.
    At decision time, samples one network m ~ Uniform(1..M) for Thompson sampling.
    """

    def __init__(
        self,
        context_dim=2,
        num_arms=5,
        num_models=5,
        hidden_dim=64,
        lr=0.01,
        weight_decay=1e-4,
        batch_size=64,
        steps_per_decision=5,
        seed=None
    ):
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.num_models = num_models
        self.batch_size = batch_size
        self.steps_per_decision = steps_per_decision
        self.rng = np.random.RandomState(seed)

        self.models = []
        self.optimizers = []
        for i in range(num_models):
            if seed is not None:
                torch.manual_seed(seed + i * 1000)
            m = ContextualMLP(context_dim, num_arms, hidden_dim)
            opt = optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
            self.models.append(m)
            self.optimizers.append(opt)

        self.contexts = []
        self.actions = []
        self.rewards = []

    def select_action(self, context):
        if len(self.rewards) < self.num_arms:
            return len(self.rewards)

        # Thompson Sampling: sample one ensemble member at random
        head_idx = self.rng.randint(0, self.num_models)
        active_model = self.models[head_idx]

        active_model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(context).unsqueeze(0)
            preds = active_model(x_t).squeeze(0).numpy()
            return int(np.argmax(preds))

    def update(self, context, action, reward):
        self.contexts.append(context)
        self.actions.append(action)
        self.rewards.append(reward)

        n = len(self.rewards)
        if n < self.num_arms:
            return

        # Train each ensemble member on bootstrapped resamples
        for m, opt in zip(self.models, self.optimizers):
            m.train()
            for _ in range(self.steps_per_decision):
                batch_indices = self.rng.choice(n, size=min(n, self.batch_size), replace=True)
                batch_x = torch.from_numpy(np.array([self.contexts[i] for i in batch_indices], dtype=np.float32))
                batch_a = torch.from_numpy(np.array([self.actions[i] for i in batch_indices], dtype=np.int64))
                batch_r = torch.from_numpy(np.array([self.rewards[i] for i in batch_indices], dtype=np.float32))

                opt.zero_grad()
                preds = m(batch_x)
                chosen_preds = preds.gather(1, batch_a.unsqueeze(1)).squeeze(1)
                loss = nn.functional.mse_loss(chosen_preds, batch_r)
                loss.backward()
                opt.step()


class FeatureExtractor(nn.Module):
    """Latent feature extractor for Neural-Linear Bayesian Bandit."""

    def __init__(self, context_dim=2, hidden_dim=64, feature_dim=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.body(x)


class NeuralLinearAgent:
    """Neural Linear Bayesian Bandit:

    Uses a neural network for representation learning and maintains an analytical
    Bayesian Linear Regression posterior over the top layer weights for each arm.
    Action selection is performed via exact Linear Thompson Sampling.
    """

    def __init__(
        self,
        context_dim=2,
        num_arms=5,
        hidden_dim=64,
        feature_dim=32,
        prior_variance=1.0,
        noise_variance=0.01,
        lr=0.01,
        weight_decay=1e-4,
        steps_per_decision=5,
        seed=None
    ):
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.feature_dim = feature_dim
        self.prior_var = prior_variance
        self.noise_var = noise_variance
        self.steps_per_decision = steps_per_decision
        self.rng = np.random.RandomState(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.feature_net = FeatureExtractor(context_dim, hidden_dim, feature_dim)
        # Prediction head for representation training
        self.head = nn.Linear(feature_dim, num_arms)
        self.optimizer = optim.Adam(
            list(self.feature_net.parameters()) + list(self.head.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        # Precision matrices (Lambda_a) and b_a vectors for each arm
        # Prior: w_a ~ N(0, prior_var * I) => Lambda_0 = (1 / prior_var) * I
        self.lambda_matrices = [
            (1.0 / self.prior_var) * np.eye(feature_dim + 1, dtype=np.float32)
            for _ in range(num_arms)
        ]
        self.b_vectors = [
            np.zeros(feature_dim + 1, dtype=np.float32)
            for _ in range(num_arms)
        ]

        self.contexts = []
        self.actions = []
        self.rewards = []

    def _get_phi(self, context):
        self.feature_net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(context).unsqueeze(0)
            feat = self.feature_net(x_t).squeeze(0).numpy()
            # Include bias term
            return np.append(feat, 1.0).astype(np.float32)

    def select_action(self, context):
        if len(self.rewards) < self.num_arms:
            return len(self.rewards)

        phi = self._get_phi(context)
        sampled_rewards = np.zeros(self.num_arms, dtype=np.float32)

        # Thompson sampling: sample weight vector from posterior for each arm
        for a in range(self.num_arms):
            cov = np.linalg.inv(self.lambda_matrices[a])
            mu = cov @ self.b_vectors[a]
            # Symmetrize covariance for numerical stability
            cov = 0.5 * (cov + cov.T)
            w_sample = self.rng.multivariate_normal(mu, cov)
            sampled_rewards[a] = np.dot(phi, w_sample)

        return int(np.argmax(sampled_rewards))

    def update(self, context, action, reward):
        self.contexts.append(context)
        self.actions.append(action)
        self.rewards.append(reward)

        phi = self._get_phi(context)
        # Update exact Bayesian Linear Regression statistics
        self.lambda_matrices[action] += (1.0 / self.noise_var) * np.outer(phi, phi)
        self.b_vectors[action] += (reward / self.noise_var) * phi

        n = len(self.rewards)
        if n < self.num_arms:
            return

        # Periodically refine feature representations
        self.feature_net.train()
        self.head.train()
        for _ in range(self.steps_per_decision):
            batch_indices = self.rng.choice(n, size=min(n, 64), replace=True)
            batch_x = torch.from_numpy(np.array([self.contexts[i] for i in batch_indices], dtype=np.float32))
            batch_a = torch.from_numpy(np.array([self.actions[i] for i in batch_indices], dtype=np.int64))
            batch_r = torch.from_numpy(np.array([self.rewards[i] for i in batch_indices], dtype=np.float32))

            self.optimizer.zero_grad()
            feats = self.feature_net(batch_x)
            preds = self.head(feats)
            chosen_preds = preds.gather(1, batch_a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(chosen_preds, batch_r)
            loss.backward()
            self.optimizer.step()


class RandomizedPriorEnsembleAgent:
    """BootDQN with Randomized Prior Functions (Osband et al., NeurIPS 2018).

    Each ensemble member m computes:
        Q_m(x, a) = f_{theta_m}(x)_a + beta * p_m(x)_a
    where p_m(x) is a fixed, randomly initialized prior network with frozen weights.
    Only f_{theta_m} is trained to fit the residual.
    """

    def __init__(
        self,
        context_dim=2,
        num_arms=5,
        num_models=5,
        hidden_dim=64,
        prior_scale=3.0,
        lr=0.01,
        weight_decay=1e-4,
        batch_size=64,
        steps_per_decision=5,
        seed=None
    ):
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.num_models = num_models
        self.prior_scale = float(prior_scale)
        self.batch_size = batch_size
        self.steps_per_decision = steps_per_decision
        self.rng = np.random.RandomState(seed)

        self.trainable_models = []
        self.prior_models = []
        self.optimizers = []

        for i in range(num_models):
            # Trainable network
            if seed is not None:
                torch.manual_seed(seed + i * 2000)
            trainable_m = ContextualMLP(context_dim, num_arms, hidden_dim)
            opt = optim.Adam(trainable_m.parameters(), lr=lr, weight_decay=weight_decay)

            # Frozen randomized prior network
            if seed is not None:
                torch.manual_seed(seed + i * 2000 + 777)
            prior_m = ContextualMLP(context_dim, num_arms, hidden_dim)
            for param in prior_m.parameters():
                param.requires_grad = False
            prior_m.eval()

            self.trainable_models.append(trainable_m)
            self.prior_models.append(prior_m)
            self.optimizers.append(opt)

        self.contexts = []
        self.actions = []
        self.rewards = []

    def select_action(self, context):
        if len(self.rewards) < self.num_arms:
            return len(self.rewards)

        # Thompson Sampling: select random ensemble head
        head_idx = self.rng.randint(0, self.num_models)
        trainable_m = self.trainable_models[head_idx]
        prior_m = self.prior_models[head_idx]

        trainable_m.eval()
        prior_m.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(context).unsqueeze(0)
            f_val = trainable_m(x_t).squeeze(0)
            p_val = prior_m(x_t).squeeze(0)
            q_val = (f_val + self.prior_scale * p_val).numpy()
            return int(np.argmax(q_val))

    def update(self, context, action, reward):
        self.contexts.append(context)
        self.actions.append(action)
        self.rewards.append(reward)

        n = len(self.rewards)
        if n < self.num_arms:
            return

        # Train each head on bootstrapped data
        for trainable_m, prior_m, opt in zip(self.trainable_models, self.prior_models, self.optimizers):
            trainable_m.train()
            prior_m.eval()

            for _ in range(self.steps_per_decision):
                batch_indices = self.rng.choice(n, size=min(n, self.batch_size), replace=True)
                batch_x = torch.from_numpy(np.array([self.contexts[i] for i in batch_indices], dtype=np.float32))
                batch_a = torch.from_numpy(np.array([self.actions[i] for i in batch_indices], dtype=np.int64))
                batch_r = torch.from_numpy(np.array([self.rewards[i] for i in batch_indices], dtype=np.float32))

                opt.zero_grad()
                f_preds = trainable_m(batch_x)
                with torch.no_grad():
                    p_preds = prior_m(batch_x)
                total_preds = f_preds + self.prior_scale * p_preds

                chosen_preds = total_preds.gather(1, batch_a.unsqueeze(1)).squeeze(1)
                loss = nn.functional.mse_loss(chosen_preds, batch_r)
                loss.backward()
                opt.step()

