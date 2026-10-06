import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict, Any


class BatchedMLP(nn.Module):
    """Batched ensemble of M neural networks computed simultaneously in parallel.
    
    Weights have shape [M, in_dim, out_dim], enabling all M models to evaluate
    and train in a single vectorized tensor operation without Python loops.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, num_models: int = 20):
        super().__init__()
        self.num_models = num_models
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # Layer 1: [M, in_dim, hidden_dim]
        self.w1 = nn.Parameter(torch.empty(num_models, in_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_models, 1, hidden_dim))

        # Layer 2: [M, hidden_dim, hidden_dim]
        self.w2 = nn.Parameter(torch.empty(num_models, hidden_dim, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_models, 1, hidden_dim))

        # Layer 3: [M, hidden_dim, out_dim]
        self.w3 = nn.Parameter(torch.empty(num_models, hidden_dim, out_dim))
        self.b3 = nn.Parameter(torch.zeros(num_models, 1, out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.w1, self.w2, self.w3]:
            nn.init.kaiming_uniform_(w, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_1d = (x.ndim == 1)
        if is_1d:
            x_b = x.unsqueeze(0).unsqueeze(0).expand(self.num_models, 1, self.in_dim)
        elif x.ndim == 2:
            x_b = x.unsqueeze(0).expand(self.num_models, x.shape[0], self.in_dim)
        elif x.ndim == 3:
            x_b = x
        else:
            raise ValueError(f'Unsupported input dimension: {x.ndim}')

        h1 = torch.relu(torch.bmm(x_b, self.w1) + self.b1)
        h2 = torch.relu(torch.bmm(h1, self.w2) + self.b2)
        out = torch.bmm(h2, self.w3) + self.b3

        if is_1d:
            return out.squeeze(1)  # [num_models, out_dim]
        return out  # [num_models, batch_size, out_dim]


class DPBNN_IDSAgent:
    """Dirichlet Process Bayesian Neural Network with Information Directed Sampling (DP-BNN IDS).
    
    Implements nonparametric Information Directed Sampling (Russo & Van Roy 2014, 2018):
    1. Nonparametric Posterior: F ~ DP(alpha, F_0).
    2. Batched stick-breaking draws: Draws M independent posterior measures P^{(1)}, ..., P^{(M)}.
    3. Parallel particle adaptation: Fits M networks to the drawn measures in a single vectorized pass.
    4. Closed-form Information Gain: Computes I_t(A*; Y_a) (Mutual Information) or Var(f(x, a)).
    5. Information Ratio Optimization: Minimizes Psi_t(p) = (p^T Delta)^2 / (p^T I) in closed form.
    """

    def __init__(
        self,
        context_dim: int = 2,
        num_arms: int = 3,
        num_models: int = 20,
        hidden_dim: int = 64,
        alpha: float = 10.0,
        truncation_K: int = 50,
        prior_reward_mean: float = 0.5,
        prior_reward_std: float = 0.5,
        noise_std: float = 0.5,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        steps_per_decision: int = 8,
        info_type: str = 'mutual_information',  # 'mutual_information' or 'variance'
        action_selection: str = 'randomized',     # 'randomized' (optimal 2-action distribution) or 'deterministic'
        min_exploration_prob: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.num_models = num_models
        self.hidden_dim = hidden_dim
        self.alpha = float(alpha)
        self.truncation_K = int(truncation_K)
        self.prior_reward_mean = float(prior_reward_mean)
        self.prior_reward_std = float(prior_reward_std)
        self.noise_std = float(noise_std)
        self.lr = lr
        self.weight_decay = weight_decay
        self.steps_per_decision = steps_per_decision
        self.info_type = info_type
        self.action_selection = action_selection
        self.min_exploration_prob = min_exploration_prob

        self.rng = np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.models = BatchedMLP(context_dim, num_arms, hidden_dim, num_models)
        self.optimizer = optim.Adam(self.models.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.contexts = []
        self.actions = []
        self.rewards = []

        self.last_regret_vec = None
        self.last_info_vec = None
        self.last_sampling_dist = None

    def sample_base_measure_atoms(self, count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.rng.uniform(-1.0, 1.0, size=(count, self.context_dim)).astype(np.float32)
        a = self.rng.randint(0, self.num_arms, size=count)
        r = self.rng.normal(self.prior_reward_mean, self.prior_reward_std, size=count).astype(np.float32)
        return x, a, r

    def sample_posterior_measures_batched(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(self.rewards)
        M = self.num_models
        K = self.truncation_K
        total_mass = self.alpha + n

        V = self.rng.beta(1.0, total_mass, size=(M, K)).astype(np.float32)
        V[:, -1] = 1.0

        cum_prod = np.cumprod(1.0 - V, axis=1)
        Q = np.empty_like(V)
        Q[:, 0] = V[:, 0]
        Q[:, 1:] = V[:, 1:] * cum_prod[:, :-1]
        Q_sum = Q.sum(axis=1, keepdims=True)
        Q = Q / (Q_sum + 1e-12)

        prob_emp = n / total_mass if total_mass > 0 else 0.0
        is_emp = (self.rng.rand(M, K) < prob_emp) if n > 0 else np.zeros((M, K), dtype=bool)
        k_emp = int(np.sum(is_emp))
        k_syn = (M * K) - k_emp

        atom_x = np.zeros((M, K, self.context_dim), dtype=np.float32)
        atom_a = np.zeros((M, K), dtype=np.int64)
        atom_r = np.zeros((M, K), dtype=np.float32)

        if k_emp > 0:
            rand_idx = self.rng.randint(0, n, size=k_emp)
            emp_x = np.array([self.contexts[i] for i in rand_idx], dtype=np.float32)
            emp_a = np.array([self.actions[i] for i in rand_idx], dtype=np.int64)
            emp_r = np.array([self.rewards[i] for i in rand_idx], dtype=np.float32)

            atom_x[is_emp] = emp_x
            atom_a[is_emp] = emp_a
            atom_r[is_emp] = emp_r

        if k_syn > 0:
            syn_x, syn_a, syn_r = self.sample_base_measure_atoms(k_syn)
            atom_x[~is_emp] = syn_x
            atom_a[~is_emp] = syn_a
            atom_r[~is_emp] = syn_r

        return (
            torch.from_numpy(atom_x),
            torch.from_numpy(atom_a),
            torch.from_numpy(atom_r),
            torch.from_numpy(Q),
        )

    def train_on_drawn_measures(self):
        atom_x, atom_a, atom_r, q_weights = self.sample_posterior_measures_batched()

        self.models.train()
        for _ in range(self.steps_per_decision):
            self.optimizer.zero_grad()
            preds = self.models(atom_x)
            chosen_preds = preds.gather(2, atom_a.unsqueeze(2)).squeeze(2)
            loss = torch.sum(q_weights * (chosen_preds - atom_r) ** 2)
            loss.backward()
            self.optimizer.step()

    def compute_regret_and_information(
        self, context: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.models.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(context.astype(np.float32))
            preds = self.models(x_t).numpy()  # [M, num_arms]

        mu = preds.mean(axis=0)  # [num_arms]
        opt_arms = preds.argmax(axis=1)  # [M]
        opt_vals = preds.max(axis=1)    # [M]
        mu_star = opt_vals.mean()

        delta = np.maximum(0.0, mu_star - mu)
        best_arm = int(np.argmax(mu))
        delta[best_arm] = 0.0

        alpha_opt = np.zeros(self.num_arms, dtype=np.float32)
        for j in range(self.num_arms):
            alpha_opt[j] = np.mean(opt_arms == j)

        info_gain = np.zeros(self.num_arms, dtype=np.float32)
        var_obs = self.noise_std ** 2

        if self.info_type == 'mutual_information':
            for j in range(self.num_arms):
                if alpha_opt[j] > 0.0:
                    mask_j = (opt_arms == j)
                    cond_mean_j = preds[mask_j].mean(axis=0)
                    info_gain += alpha_opt[j] * ((cond_mean_j - mu) ** 2) / (2.0 * var_obs)
        else:
            info_gain = preds.var(axis=0, ddof=1) / (2.0 * var_obs)

        info_gain = np.maximum(1e-8, info_gain)
        return mu, delta, info_gain, alpha_opt

    def solve_optimal_sampling_distribution(
        self, delta: np.ndarray, info_gain: np.ndarray
    ) -> np.ndarray:
        K = self.num_arms
        best_arm = int(np.argmin(delta))
        I_best = info_gain[best_arm]

        best_psi = float('inf')
        best_p = np.zeros(K, dtype=np.float32)
        best_p[best_arm] = 1.0

        if I_best > 1e-4:
            return best_p

        q_grid = np.linspace(0.0, 1.0, 101)

        for a in range(K):
            if a == best_arm:
                continue

            del_a = delta[a]
            I_a = info_gain[a]

            num = (q_grid * del_a) ** 2
            den = (1.0 - q_grid) * I_best + q_grid * I_a + 1e-12
            psi_curve = num / den

            min_idx = np.argmin(psi_curve[1:]) + 1
            min_psi = psi_curve[min_idx]
            opt_q = q_grid[min_idx]

            if min_psi < best_psi:
                best_psi = min_psi
                best_p = np.zeros(K, dtype=np.float32)
                best_p[best_arm] = 1.0 - opt_q
                best_p[a] = opt_q

        if self.min_exploration_prob > 0.0:
            best_p = (1.0 - self.min_exploration_prob * K) * best_p + self.min_exploration_prob
            best_p = best_p / best_p.sum()

        return best_p

    def select_action(self, context: np.ndarray) -> int:
        if len(self.rewards) < self.num_arms:
            return len(self.rewards)

        self.train_on_drawn_measures()
        mu, delta, info_gain, alpha_opt = self.compute_regret_and_information(context)
        self.last_regret_vec = delta
        self.last_info_vec = info_gain

        if self.action_selection == 'deterministic':
            ratio = (delta ** 2) / (info_gain + 1e-12)
            action = int(np.argmin(ratio))
            p = np.zeros(self.num_arms, dtype=np.float32)
            p[action] = 1.0
            self.last_sampling_dist = p
            return action
        else:
            p = self.solve_optimal_sampling_distribution(delta, info_gain)
            self.last_sampling_dist = p
            action = int(self.rng.choice(self.num_arms, p=p))
            return action

    def update(self, context: np.ndarray, action: int, reward: float):
        self.contexts.append(np.array(context, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
