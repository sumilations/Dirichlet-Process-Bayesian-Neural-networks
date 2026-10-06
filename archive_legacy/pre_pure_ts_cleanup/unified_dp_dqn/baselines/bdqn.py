"""Bayesian Deep Q-Networks (BDQN) with Exact Posterior Linear Regression (Azizzadenesheli et al., 2018)."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..networks import ReplayBuffer


class BDQNFeatureExtractor(nn.Module):
    """Deep feature representation network."""

    def __init__(self, state_dim: int, hidden_dim: int = 50, use_layer_norm: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BayesianDeepQNetworkAgent:
    """Bayesian Deep Q-Network (BDQN) Agent.

    Top layer uses exact Bayesian Linear Regression per action:
        w_a ~ N(mu_a, Sigma_a).
    At episode start, samples w_a from Gaussian posterior (Thompson sampling).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 50,
        prior_variance: float = 1.0,
        noise_variance: float = 0.1,
        gamma: float = 0.99,
        tau: float = 0.05,
        lr: float = 1e-3,
        sgd_period: int = 2,
        batch_size: int = 64,
        capacity: int = 1000000,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = hidden_dim + 1
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

        self.feature_net = BDQNFeatureExtractor(state_dim, hidden_dim).to(self.device)
        self.target_feature_net = BDQNFeatureExtractor(state_dim, hidden_dim).to(self.device)
        self.target_feature_net.load_state_dict(self.feature_net.state_dict())
        for p in self.target_feature_net.parameters():
            p.requires_grad = False

        self.q_head = nn.Linear(hidden_dim, action_dim, bias=True).to(self.device)
        self.optimizer = optim.Adam(
            list(self.feature_net.parameters()) + list(self.q_head.parameters()),
            lr=lr,
        )

        self.lambda_matrices = [
            (1.0 / self.prior_var) * np.eye(self.feature_dim, dtype=np.float32)
            for _ in range(action_dim)
        ]
        self.b_vectors = [
            np.zeros(self.feature_dim, dtype=np.float32)
            for _ in range(action_dim)
        ]
        self.sampled_weights = [
            self.rng.normal(0.0, float(np.sqrt(self.prior_var)), size=self.feature_dim).astype(np.float32)
            for _ in range(action_dim)
        ]

        self.replay = ReplayBuffer(capacity=capacity, state_dim=state_dim)
        self.total_steps = 0
        self.episodes_completed = 0
        self.reset_episode()

    def reset_episode(self) -> None:
        """Thompson sampling: sample weight vector for each action from BLR posterior."""
        self.episodes_completed += 1
        for a in range(self.action_dim):
            try:
                cov = np.linalg.inv(self.lambda_matrices[a])
                cov = 0.5 * (cov + cov.T)
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

        # Online BLR statistics update
        phi = self._get_phi(state)
        self.lambda_matrices[action] += (1.0 / self.noise_var) * np.outer(phi, phi)
        with torch.no_grad():
            phi_next = self._get_phi(next_state, use_target=True)
            next_q = np.array([float(np.dot(phi_next, self.sampled_weights[a])) for a in range(self.action_dim)])
            target_val = reward + (1.0 - float(done)) * self.gamma * np.max(next_q)
        self.b_vectors[action] += (1.0 / self.noise_var) * phi * target_val

        if len(self.replay) >= self.batch_size and self.total_steps % self.sgd_period == 0:
            return self._train_step()
        return None

    def _train_step(self) -> float:
        s_b, a_b, r_b, sn_b, d_b = self.replay.sample(self.batch_size, self.rng)
        s_t = torch.from_numpy(s_b).to(self.device)
        a_t = torch.from_numpy(a_b).to(self.device)
        r_t = torch.from_numpy(r_b).to(self.device)
        sn_t = torch.from_numpy(sn_b).to(self.device)
        d_t = torch.from_numpy(d_b).to(self.device)

        self.feature_net.eval()
        with torch.no_grad():
            next_feat = self.target_feature_net(sn_t)
            next_q = self.q_head(next_feat)
            max_next_q, _ = torch.max(next_q, dim=1)
            targets = r_t + (1.0 - d_t) * self.gamma * max_next_q

        self.feature_net.train()
        self.optimizer.zero_grad()
        feat = self.feature_net(s_t)
        q_pred = self.q_head(feat)
        chosen_q = q_pred.gather(1, a_t.unsqueeze(1)).squeeze(1)

        loss = nn.functional.mse_loss(chosen_q, targets)
        loss.backward()
        self.optimizer.step()

        # Polyak target tracking
        with torch.no_grad():
            for tgt_p, p in zip(self.target_feature_net.parameters(), self.feature_net.parameters()):
                tgt_p.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return float(loss.item())
