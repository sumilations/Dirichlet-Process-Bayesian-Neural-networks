"""
Mutual Information Estimators:
1. DP-BNN (Predictive MI): Nonparametric Dirichlet Process BNN predictive density ratio estimator.
2. DP-MINE: Bayesian Donsker-Varadhan with Dirichlet Process prior.
3. MINE: Mutual Information Neural Estimation (Belghazi et al., 2018).
4. NWJ: Nguyen, Wainwright, Jordan (2010) f-divergence lower bound.
5. InfoNCE: Contrastive Predictive Coding estimator (van den Oord et al., 2018).
6. KSG: Kraskov-Stögbauer-Grassberger (2004) k-nearest neighbor nonparametric estimator.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.special import digamma


# ==============================================================================
# 1. Classical Baseline: KSG Estimator (Kraskov et al., 2004)
# ==============================================================================

class KSG_Estimator:
    """Kraskov-Stögbauer-Grassberger (KSG-1) non-parametric mutual information estimator.
    Uses Chebyshev (L_infinity) distance with k-nearest neighbors.
    """
    def __init__(self, k: int = 3):
        self.k = k

    def estimate(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Estimates I(X; Y) using the KSG-1 formula.
        X: (N, d_x), Y: (N, d_y)
        """
        N = X.shape[0]
        if N <= self.k + 1:
            return 0.0

        # Add tiny jitter to break ties
        rng = np.random.RandomState(42)
        X_jitter = X + 1e-10 * rng.randn(*X.shape)
        Y_jitter = Y + 1e-10 * rng.randn(*Y.shape)
        Z = np.hstack([X_jitter, Y_jitter])

        # Build KD-trees with Chebyshev (L_inf) metric
        tree_z = cKDTree(Z)
        tree_x = cKDTree(X_jitter)
        tree_y = cKDTree(Y_jitter)

        # Query (k+1)-th neighbor distance (k-th excluding self)
        # distances is (N, k+1)
        distances, _ = tree_z.query(Z, k=self.k + 1, p=np.inf)
        eps = distances[:, self.k]  # distance to k-th neighbor (0-indexed k is (k+1)-th point)

        # Count strictly within eps in marginal spaces
        # Using eps - 1e-15 to count strictly less than eps
        n_x = np.zeros(N)
        n_y = np.zeros(N)

        for i in range(N):
            r = eps[i]
            # query_ball_point with strictly smaller radius
            n_x[i] = len(tree_x.query_ball_point(X_jitter[i], r=r - 1e-14, p=np.inf)) - 1
            n_y[i] = len(tree_y.query_ball_point(Y_jitter[i], r=r - 1e-14, p=np.inf)) - 1

        # Clip counts to at least 0
        n_x = np.maximum(n_x, 0)
        n_y = np.maximum(n_y, 0)

        # KSG-1 formula: I = psi(k) - (1/N) * sum(psi(n_x + 1) + psi(n_y + 1)) + psi(N)
        mi = digamma(self.k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(N)
        return float(max(0.0, mi))


# ==============================================================================
# 2. Neural Critic Architecture for MINE, NWJ, and InfoNCE
# ==============================================================================

class CriticMLP(nn.Module):
    """MLP critic network mapping (x, y) -> scalar score."""
    def __init__(self, dim_x: int, dim_y: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_y, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.cat([x, y], dim=-1)
        return self.net(xy)


# ==============================================================================
# 3. MINE: Mutual Information Neural Estimation (Belghazi et al., 2018)
# ==============================================================================

class MINE_Estimator:
    """Donsker-Varadhan variational mutual information estimator."""
    def __init__(self, dim_x: int, dim_y: int, hidden_dim: int = 64, lr: float = 1e-3, ema_decay: float = 0.99):
        self.critic = CriticMLP(dim_x, dim_y, hidden_dim)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.ema_decay = ema_decay
        self.ema_exp_t = 1.0

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """One training step with a batch (x, y). Returns current MI estimate."""
        batch_size = x.size(0)
        perm = torch.randperm(batch_size)
        y_marginal = y[perm]

        t_joint = self.critic(x, y)
        t_marginal = self.critic(x, y_marginal)

        exp_t_marg = torch.exp(torch.clamp(t_marginal, -20.0, 20.0))
        mean_exp = torch.mean(exp_t_marg)

        # Unbiased gradient via moving average denominator
        self.ema_exp_t = self.ema_decay * self.ema_exp_t + (1 - self.ema_decay) * mean_exp.item()

        # Donsker-Varadhan objective (maximize)
        loss = -(torch.mean(t_joint) - torch.log(mean_exp + 1e-8))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optimizer.step()

        return float(max(0.0, -loss.item()))

    def estimate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Evaluates MI on a dataset."""
        with torch.no_grad():
            batch_size = x.size(0)
            perm = torch.randperm(batch_size)
            y_marginal = y[perm]
            t_joint = self.critic(x, y)
            t_marginal = self.critic(x, y_marginal)
            mi = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(torch.clamp(t_marginal, -20.0, 20.0))) + 1e-8)
            return float(max(0.0, mi.item()))


# ==============================================================================
# 4. NWJ Estimator (Nguyen et al., 2010)
# ==============================================================================

class NWJ_Estimator:
    """f-divergence lower bound mutual information estimator."""
    def __init__(self, dim_x: int, dim_y: int, hidden_dim: int = 64, lr: float = 1e-3):
        self.critic = CriticMLP(dim_x, dim_y, hidden_dim)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        batch_size = x.size(0)
        perm = torch.randperm(batch_size)
        y_marginal = y[perm]

        t_joint = self.critic(x, y)
        t_marginal = self.critic(x, y_marginal)

        # NWJ loss: - (E_joint[T] - (1/e) * E_marginal[e^T])
        exp_t_marg = torch.exp(torch.clamp(t_marginal, -20.0, 20.0))
        nwj_mi = torch.mean(t_joint) - (1.0 / np.e) * torch.mean(exp_t_marg)
        loss = -nwj_mi

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optimizer.step()

        return float(max(0.0, nwj_mi.item()))

    def estimate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            batch_size = x.size(0)
            perm = torch.randperm(batch_size)
            y_marginal = y[perm]
            t_joint = self.critic(x, y)
            t_marginal = self.critic(x, y_marginal)
            exp_t_marg = torch.exp(torch.clamp(t_marginal, -20.0, 20.0))
            mi = torch.mean(t_joint) - (1.0 / np.e) * torch.mean(exp_t_marg)
            return float(max(0.0, mi.item()))


# ==============================================================================
# 5. InfoNCE Estimator (van den Oord et al., 2018)
# ==============================================================================

class InfoNCE_Estimator:
    """Contrastive Predictive Coding mutual information lower bound."""
    def __init__(self, dim_x: int, dim_y: int, hidden_dim: int = 64, lr: float = 1e-3):
        self.proj_x = nn.Sequential(nn.Linear(dim_x, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.proj_y = nn.Sequential(nn.Linear(dim_y, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.optimizer = torch.optim.Adam(list(self.proj_x.parameters()) + list(self.proj_y.parameters()), lr=lr)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        z_x = F.normalize(self.proj_x(x), dim=-1)  # (B, H)
        z_y = F.normalize(self.proj_y(y), dim=-1)  # (B, H)

        scores = torch.matmul(z_x, z_y.T) / 0.1  # (B, B)
        labels = torch.arange(x.size(0), device=x.device)

        loss = F.cross_entropy(scores, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # I_NCE = log(B) - cross_entropy_loss
        mi_bound = np.log(x.size(0)) - loss.item()
        return float(max(0.0, mi_bound))

    def estimate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            z_x = F.normalize(self.proj_x(x), dim=-1)
            z_y = F.normalize(self.proj_y(y), dim=-1)
            scores = torch.matmul(z_x, z_y.T) / 0.1
            labels = torch.arange(x.size(0), device=x.device)
            loss = F.cross_entropy(scores, labels)
            mi_bound = np.log(x.size(0)) - loss.item()
            return float(max(0.0, mi_bound))


# ==============================================================================
# 6. DP-MINE: Bayesian Donsker-Varadhan with Dirichlet Process Prior
# ==============================================================================

class DPMINE_Estimator:
    """Bayesian MINE regularized by a Dirichlet Process prior on critic space.
    Draws M critic particles via stick-breaking weights to smooth the partition function.
    """
    def __init__(self, dim_x: int, dim_y: int, num_particles: int = 5,
                 hidden_dim: int = 64, alpha: float = 1.0, lr: float = 1e-3):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.M = num_particles
        self.alpha = alpha

        # Ensemble of critic networks with diverse priors
        self.critics = nn.ModuleList([
            CriticMLP(dim_x, dim_y, hidden_dim) for _ in range(num_particles)
        ])
        self.priors = [
            CriticMLP(dim_x, dim_y, hidden_dim) for _ in range(num_particles)
        ]
        for p in self.priors:
            for param in p.parameters():
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.critics.parameters(), lr=lr)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        B = x.size(0)
        perm = torch.randperm(B)
        y_marginal = y[perm]

        total_loss = 0.0
        # Stick-breaking weights
        v = np.random.beta(1.0, self.alpha, size=self.M)
        weights = np.zeros(self.M)
        rem = 1.0
        for m in range(self.M):
            weights[m] = v[m] * rem
            rem *= (1.0 - v[m])
        weights /= weights.sum()

        mi_estimates = []
        for m in range(self.M):
            critic = self.critics[m]
            prior = self.priors[m]

            t_joint = critic(x, y) + 0.1 * prior(x, y)
            t_marginal = critic(x, y_marginal) + 0.1 * prior(x, y_marginal)

            exp_t = torch.exp(torch.clamp(t_marginal, -15.0, 15.0))
            loss_m = -(torch.mean(t_joint) - torch.log(torch.mean(exp_t) + 1e-8))

            # Prior regularization
            prior_reg = 0.0
            for cp, pp in zip(critic.parameters(), prior.parameters()):
                prior_reg += torch.sum((cp - pp) ** 2)

            total_loss += weights[m] * (loss_m + (self.alpha / (self.alpha + B)) * 0.01 * prior_reg)
            mi_estimates.append(max(0.0, -loss_m.item()))

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 1.0)
        self.optimizer.step()

        return float(np.mean(mi_estimates))

    def estimate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            B = x.size(0)
            perm = torch.randperm(B)
            y_marginal = y[perm]
            mis = []
            for m in range(self.M):
                t_joint = self.critics[m](x, y) + 0.1 * self.priors[m](x, y)
                t_marginal = self.critics[m](x, y_marginal) + 0.1 * self.priors[m](x, y_marginal)
                exp_t = torch.exp(torch.clamp(t_marginal, -15.0, 15.0))
                mi = torch.mean(t_joint) - torch.log(torch.mean(exp_t) + 1e-8)
                mis.append(max(0.0, mi.item()))
            return float(np.mean(mis))


# ==============================================================================
# 7. DP-BNN (Predictive MI Estimator): Nonparametric Conditional Density Ratio
# ==============================================================================

class BatchedGaussianMLP(nn.Module):
    """Vectorized M-particle MLP predicting mean and log-variance:
    x (B, d_x) -> mu (M, B, d_y), log_var (M, B, d_y).
    """
    def __init__(self, M: int, dim_in: int, dim_out: int, hidden_dim: int = 64):
        super().__init__()
        self.M = M
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Layer 1: (M, dim_in, hidden_dim)
        self.w1 = nn.Parameter(torch.randn(M, dim_in, hidden_dim) * np.sqrt(2.0 / dim_in))
        self.b1 = nn.Parameter(torch.zeros(M, 1, hidden_dim))

        # Layer 2: (M, hidden_dim, hidden_dim)
        self.w2 = nn.Parameter(torch.randn(M, hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(M, 1, hidden_dim))

        # Output Mean: (M, hidden_dim, dim_out)
        self.w_mu = nn.Parameter(torch.randn(M, hidden_dim, dim_out) * 0.1)
        self.b_mu = nn.Parameter(torch.zeros(M, 1, dim_out))

        # Output LogVar: (M, hidden_dim, dim_out)
        self.w_lv = nn.Parameter(torch.randn(M, hidden_dim, dim_out) * 0.05)
        self.b_lv = nn.Parameter(torch.zeros(M, 1, dim_out) - 1.0)  # Initial log_var ~ -1.0

    def forward(self, x: torch.Tensor):
        # x is (B, dim_in) -> expand to (M, B, dim_in)
        B = x.size(0)
        x_exp = x.unsqueeze(0).expand(self.M, B, self.dim_in)

        h1 = F.relu(torch.bmm(x_exp, self.w1) + self.b1)  # (M, B, hidden_dim)
        h2 = F.relu(torch.bmm(h1, self.w2) + self.b2)      # (M, B, hidden_dim)

        mu = torch.bmm(h2, self.w_mu) + self.b_mu          # (M, B, dim_out)
        log_var = torch.bmm(h2, self.w_lv) + self.b_lv    # (M, B, dim_out)
        log_var = torch.clamp(log_var, -6.0, 3.0)          # Prevent degenerate variance

        return mu, log_var


class DPBNN_MIEstimator:
    """Dirichlet Process Bayesian Neural Network for Mutual Information Estimation.
    Models p(y|x) as a DP mixture of neural predictive Gaussians:
        p_DP(y|x) = (1/M) sum_{m=1}^M N(y; mu^(m)(x), diag(sigma_m^2(x)))
    Computes mutual information via exact closed-form predictive log-density ratios:
        I(X; Y) = (1/N) sum_i [ log p_DP(y_i|x_i) - log ( (1/N) sum_j p_DP(y_i|x_j) ) ]
    Optionally evaluates bidirectional symmetry: 0.5 * (I(X->Y) + I(Y->X)).
    """
    def __init__(self, dim_x: int, dim_y: int, num_particles: int = 10,
                 hidden_dim: int = 64, alpha: float = 1.0, lr: float = 2e-3,
                 symmetric: bool = True):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.M = num_particles
        self.alpha = alpha
        self.symmetric = symmetric

        # Model X -> Y
        self.model_xy = BatchedGaussianMLP(num_particles, dim_x, dim_y, hidden_dim)
        self.prior_xy = BatchedGaussianMLP(num_particles, dim_x, dim_y, hidden_dim)
        for p in self.prior_xy.parameters():
            p.requires_grad = False

        self.opt_xy = torch.optim.Adam(self.model_xy.parameters(), lr=lr)

        # Model Y -> X for symmetric estimation
        if symmetric:
            self.model_yx = BatchedGaussianMLP(num_particles, dim_y, dim_x, hidden_dim)
            self.prior_yx = BatchedGaussianMLP(num_particles, dim_y, dim_x, hidden_dim)
            for p in self.prior_yx.parameters():
                p.requires_grad = False
            self.opt_yx = torch.optim.Adam(self.model_yx.parameters(), lr=lr)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Trains DP-BNN models on a batch (x, y) under stick-breaking weights."""
        B = x.size(0)

        # Draw stick-breaking weights for particles
        v = np.random.beta(1.0, self.alpha, size=self.M)
        weights = np.zeros(self.M, dtype=np.float32)
        rem = 1.0
        for m in range(self.M):
            weights[m] = v[m] * rem
            rem *= (1.0 - v[m])
        weights /= weights.sum()
        w_tensor = torch.tensor(weights, device=x.device).unsqueeze(1).unsqueeze(2)  # (M, 1, 1)

        # Train X -> Y
        mu, log_var = self.model_xy(x)  # (M, B, d_y)
        with torch.no_grad():
            mu_0, _ = self.prior_xy(x)

        # Negative log-likelihood: 0.5 * ( (y - mu)^2 / var + log_var + log(2pi) )
        y_exp = y.unsqueeze(0).expand(self.M, B, self.dim_y)
        inv_var = torch.exp(-log_var)
        nll = 0.5 * torch.sum(inv_var * ((y_exp - mu) ** 2) + log_var, dim=-1, keepdim=True)  # (M, B, 1)

        # Base measure penalty (pulls toward prior to avoid overfitting)
        prior_pen = 0.5 * torch.sum(((mu - mu_0) ** 2), dim=-1, keepdim=True)
        reg_coef = float(self.alpha / (self.alpha + B))

        loss_xy = torch.mean(torch.sum(w_tensor * (nll + reg_coef * 0.05 * prior_pen), dim=0))

        self.opt_xy.zero_grad()
        loss_xy.backward()
        torch.nn.utils.clip_grad_norm_(self.model_xy.parameters(), 1.0)
        self.opt_xy.step()

        # Train Y -> X if symmetric
        if self.symmetric:
            mu_yx, log_var_yx = self.model_yx(y)
            with torch.no_grad():
                mu_0_yx, _ = self.prior_yx(y)
            x_exp = x.unsqueeze(0).expand(self.M, B, self.dim_x)
            inv_var_yx = torch.exp(-log_var_yx)
            nll_yx = 0.5 * torch.sum(inv_var_yx * ((x_exp - mu_yx) ** 2) + log_var_yx, dim=-1, keepdim=True)
            prior_pen_yx = 0.5 * torch.sum(((mu_yx - mu_0_yx) ** 2), dim=-1, keepdim=True)
            loss_yx = torch.mean(torch.sum(w_tensor * (nll_yx + reg_coef * 0.05 * prior_pen_yx), dim=0))

            self.opt_yx.zero_grad()
            loss_yx.backward()
            torch.nn.utils.clip_grad_norm_(self.model_yx.parameters(), 1.0)
            self.opt_yx.step()

        return float(loss_xy.item())

    def _compute_directional_mi(self, x: torch.Tensor, y: torch.Tensor, model: BatchedGaussianMLP) -> float:
        """Computes I(X; Y) = (1/N) sum_i [ log p(y_i|x_i) - log p(y_i) ]
        x: (N, d_x), y: (N, d_y)
        """
        N = x.size(0)
        d_out = y.size(1)
        mu, log_var = model(x)  # mu: (M, N, d_out), log_var: (M, N, d_out)
        inv_var = torch.exp(-log_var)

        # 1. Conditional Log-Likelihood log p(y_i | x_i):
        # For each point i: mixture over M models
        # log N(y_i; mu^(m)(x_i), var^(m)(x_i))
        y_exp = y.unsqueeze(0).expand(self.M, N, d_out)
        diff_diag = y_exp - mu  # (M, N, d_out)
        log_prob_diag = -0.5 * (torch.sum(inv_var * (diff_diag ** 2) + log_var, dim=-1) + d_out * np.log(2 * np.pi))  # (M, N)
        # Mixture over M particles: log ( (1/M) sum_m exp(log_prob) )
        log_p_cond = torch.logsumexp(log_prob_diag, dim=0) - np.log(self.M)  # (N,)

        # 2. Marginal Log-Likelihood log p(y_i):
        # p(y_i) = (1 / (N * M)) sum_j sum_m N(y_i; mu^(m)(x_j), var^(m)(x_j))
        # Compute in chunks if N is large
        chunk_size = min(N, 500)
        log_p_marg_list = []

        for i_start in range(0, N, chunk_size):
            i_end = min(N, i_start + chunk_size)
            y_chunk = y[i_start:i_end]  # (B_chunk, d_out)
            B_chunk = y_chunk.size(0)

            # mu is (M, N, d_out), inv_var is (M, N, d_out)
            # y_chunk is (B_chunk, d_out)
            # Reshape for broadcasting:
            # y_c: (1, 1, B_chunk, d_out)
            # mu: (M, N, 1, d_out)
            y_c = y_chunk.view(1, 1, B_chunk, d_out)
            mu_exp = mu.unsqueeze(2)           # (M, N, 1, d_out)
            inv_var_exp = inv_var.unsqueeze(2) # (M, N, 1, d_out)
            lv_exp = log_var.unsqueeze(2)      # (M, N, 1, d_out)

            diff = y_c - mu_exp  # (M, N, B_chunk, d_out)
            log_prob_all = -0.5 * (torch.sum(inv_var_exp * (diff ** 2) + lv_exp, dim=-1) + d_out * np.log(2 * np.pi))  # (M, N, B_chunk)

            # Sum over M and N: logsumexp over dim 0 and 1
            # Flatten (M, N) into (M * N, B_chunk)
            log_prob_flat = log_prob_all.view(self.M * N, B_chunk)
            log_p_marg_chunk = torch.logsumexp(log_prob_flat, dim=0) - np.log(self.M * N)  # (B_chunk,)
            log_p_marg_list.append(log_p_marg_chunk)

        log_p_marg = torch.cat(log_p_marg_list, dim=0)  # (N,)

        # Mutual information: mean of (log_p_cond - log_p_marg)
        mi_samples = log_p_cond - log_p_marg
        mi = torch.mean(mi_samples).item()
        return float(max(0.0, mi))

    def estimate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Evaluates mutual information on dataset (x, y)."""
        with torch.no_grad():
            mi_xy = self._compute_directional_mi(x, y, self.model_xy)
            if self.symmetric:
                mi_yx = self._compute_directional_mi(y, x, self.model_yx)
                return float(0.5 * (mi_xy + mi_yx))
            return float(mi_xy)
