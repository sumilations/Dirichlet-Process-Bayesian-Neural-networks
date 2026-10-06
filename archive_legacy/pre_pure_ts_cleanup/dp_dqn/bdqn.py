"""Bayesian Deep Q-Network (BDQN) for Deep Exploration.

Reference:
K. Azizzadenesheli, E. Brunskill, A. Anandkumar (2018).
"Efficient Exploration through Bayesian Deep Q-Networks."
NeurIPS / arXiv:1802.04412.

Architecture:
- Shared feature representation: 2-layer MLP with LayerNorm (hidden_dim=20).
- Top layer: Exact Bayesian Linear Regression posterior for each discrete action:
    w_a ~ N(mu_a, Sigma_a) where Sigma_a = Lambda_a^{-1}, mu_a = Sigma_a b_a.
- Episode-Horizon Thompson Sampling:
    At episode start, sample w_a ~ N(mu_a, Sigma_a) for each action.
    Act greedily w.r.t. sampled Q_sample(s, a) = phi(s)^T w_a throughout the episode.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .sampler import ReplayBuffer


class BDQNFeatureExtractor(nn.Module):
    """Deep feature representation network matching MLP-20 with LayerNorm."""

    def __init__(self, state_dim: int, hidden_dim: int = 20, use_layer_norm: bool = True):
        super().__init__()
        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BayesianDeepQNetworkAgent:
    """Bayesian Deep Q-Network (BDQN) Agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 20,
        prior_variance: float = 1.0,
        noise_variance: float = 0.1,
        gamma: float = 0.99,
        tau: float = 0.05,
        lr: float = 1e-3,
        sgd_period: int = 2,
        batch_size: int = 128,
        capacity: int = 30000,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = hidden_dim + 1  # include bias term
        self.prior_var = float(prior_variance)
        self.noise_var = float(noise_variance)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.sgd_period = int(sgd_period)
        self.batch_size = int(batch_size)
        self.device = device or torch.device("cpu")
        self.rng = np.random.RandomState(seed)

        if seed is not None:
            torch.manual_seed(seed)

        # Feature representations
        self.feature_net = BDQNFeatureExtractor(state_dim, hidden_dim).to(self.device)
        self.target_feature_net = BDQNFeatureExtractor(state_dim, hidden_dim).to(self.device)
        self.target_feature_net.load_state_dict(self.feature_net.state_dict())
        for p in self.target_feature_net.parameters():
            p.requires_grad = False

        # Head weights for representation learning: output dim = action_dim
        self.q_head = nn.Linear(hidden_dim, action_dim, bias=True).to(self.device)
        self.optimizer = optim.Adam(
            list(self.feature_net.parameters()) + list(self.q_head.parameters()),
            lr=lr,
        )

        # Exact Bayesian Linear Regression statistics for each action
        # Lambda_a = (1 / prior_var) * I
        self.lambda_matrices = [
            (1.0 / self.prior_var) * np.eye(self.feature_dim, dtype=np.float32)
            for _ in range(action_dim)
        ]
        self.b_vectors = [
            np.zeros(self.feature_dim, dtype=np.float32)
            for _ in range(action_dim)
        ]

        # Active Thompson sampled weights for current episode
        self.sampled_weights = [
            np.zeros(self.feature_dim, dtype=np.float32)
            for _ in range(action_dim)
        ]

        self.replay = ReplayBuffer(capacity=capacity, state_dim=state_dim)
        self.total_steps = 0
        self.episodes_completed = 0
        self.reset_episode()

    def reset_episode(self):
        """Thompson sampling: sample weight vector for each action from BLR posterior."""
        self.episodes_completed += 1
        for a in range(self.action_dim):
            try:
                cov = np.linalg.inv(self.lambda_matrices[a])
                cov = 0.5 * (cov + cov.T)  # Symmetrize
                mu = cov @ self.b_vectors[a]
                min_eig = np.min(np.real(np.linalg.eigvals(cov)))
                if min_eig < 1e-6:
                    cov += (1e-6 - min_eig) * np.eye(self.feature_dim, dtype=np.float32)
                w = self.rng.multivariate_normal(mu, cov).astype(np.float32)
            except Exception:
                w = self.rng.normal(0.0, float(np.sqrt(self.prior_var)), size=self.feature_dim).astype(np.float32)
            self.sampled_weights[a] = w

    def _get_phi(self, s: np.ndarray, use_target: bool = False) -> np.ndarray:
        net = self.target_feature_net if use_target else self.feature_net
        net.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(np.ascontiguousarray(s, dtype=np.float32)).unsqueeze(0).to(self.device)
            feat = net(s_t).squeeze(0).cpu().numpy()
            return np.append(feat, 1.0).astype(np.float32)

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action. In exploration, uses Thompson sampled weights."""
        phi = self._get_phi(state)
        q_vals = np.zeros(self.action_dim, dtype=np.float32)

        for a in range(self.action_dim):
            if eval_mode:
                cov = np.linalg.inv(self.lambda_matrices[a])
                mu = cov @ self.b_vectors[a]
                q_vals[a] = float(np.dot(phi, mu))
            else:
                q_vals[a] = float(np.dot(phi, self.sampled_weights[a]))

        return int(np.argmax(q_vals))

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        self.replay.push(state, action, reward, next_state, done)
        self.total_steps += 1

        # Online update of Bayesian Linear Regression statistics
        phi = self._get_phi(state)
        with torch.no_grad():
            phi_next = self._get_phi(next_state, use_target=True)
            q_next_max = max(np.dot(phi_next, self.sampled_weights[a]) for a in range(self.action_dim))
            target_val = reward + (1.0 - float(done)) * self.gamma * q_next_max

        inv_noise = 1.0 / self.noise_var
        self.lambda_matrices[action] += inv_noise * np.outer(phi, phi)
        self.b_vectors[action] += inv_noise * phi * target_val

        # Periodic SGD update on feature representation
        if len(self.replay) >= self.batch_size and self.total_steps % self.sgd_period == 0:
            return self._update()
        return None

    def _update(self) -> float:
        s, a, r, sn, done = self.replay.sample(self.batch_size, self.rng)
        s_t = torch.from_numpy(s).to(self.device)
        a_t = torch.from_numpy(a).to(self.device)
        r_t = torch.from_numpy(r).to(self.device)
        sn_t = torch.from_numpy(sn).to(self.device)
        d_t = torch.from_numpy(done).to(self.device)

        with torch.no_grad():
            next_feats = self.target_feature_net(sn_t)
            next_q = self.q_head(next_feats)
            max_next_q, _ = torch.max(next_q, dim=1)
            targets = r_t + (1.0 - d_t) * self.gamma * max_next_q

        self.feature_net.train()
        feats = self.feature_net(s_t)
        q_vals = self.q_head(feats)
        pred_q = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)

        loss = nn.functional.smooth_l1_loss(pred_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.feature_net.parameters()) + list(self.q_head.parameters()), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            for tp, p in zip(self.target_feature_net.parameters(), self.feature_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return float(loss.item())
