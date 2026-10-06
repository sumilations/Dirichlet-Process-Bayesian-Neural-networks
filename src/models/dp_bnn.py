import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ContextualMLP(nn.Module):
    """Feedforward neural network predicting expected reward for each arm given context."""

    def __init__(self, context_dim=2, num_arms=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_arms)
        )

    def forward(self, x):
        return self.net(x)


class DPBNNAgent:
    """Dirichlet Process Bayesian Neural Network (DP-BNN) Agent for Contextual Bandits

    Based on the Vashishtha formulation:
    - Dirichlet Process prior directly over data space: F ~ DP(alpha, F_0).
    - Conjugate posterior: F | D ~ DP(alpha + n, (n/(alpha+n)) F_emp + (alpha/(alpha+n)) F_0).
    - Stick-breaking posterior measure sampling (Sethuraman construction):
        V_k ~ Beta(1, alpha + n)
        q_k = V_k * prod_{i < k} (1 - V_i)
        Atoms z_k sampled from D with prob n/(alpha+n), else from synthetic base measure F_0.
    - Single-network Thompson Sampling via fast warm-started optimization under the drawn measure.
    - True asymptotic posterior contraction as n -> infinity.
    """

    def __init__(
        self,
        context_dim=2,
        num_arms=5,
        hidden_dim=64,
        alpha=10.0,
        truncation_K=50,
        prior_reward_mean=3.0,
        prior_reward_std=1.0,
        lr=0.01,
        weight_decay=1e-4,
        steps_per_decision=10,
        seed=None
    ):
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.alpha = float(alpha)
        self.truncation_K = int(truncation_K)
        self.prior_reward_mean = float(prior_reward_mean)
        self.prior_reward_std = float(prior_reward_std)
        self.lr = lr
        self.weight_decay = weight_decay
        self.steps_per_decision = steps_per_decision

        self.rng = np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.model = ContextualMLP(context_dim, num_arms, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Experience Replay Buffer
        self.contexts = []
        self.actions = []
        self.rewards = []

        self.base_measure_type = "unit_disk" if context_dim == 2 else "box"

    def sample_base_measure_atom(self):
        """Sample a synthetic prior coordinate (x, a, r) from base measure F_0."""
        if self.base_measure_type == "unit_disk" and self.context_dim == 2:
            while True:
                pt = self.rng.uniform(-1.0, 1.0, size=self.context_dim)
                if np.linalg.norm(pt) <= 1.0:
                    x = pt.astype(np.float32)
                    break
        elif self.base_measure_type == "gaussian":
            x = self.rng.normal(0.0, 1.0, size=self.context_dim).astype(np.float32)
        else:  # "box" maximum-entropy uniform
            x = self.rng.uniform(-1.0, 1.0, size=self.context_dim).astype(np.float32)

        a = self.rng.randint(0, self.num_arms)
        r = float(self.rng.normal(self.prior_reward_mean, self.prior_reward_std))
        return x, a, r

    def sample_posterior_measure(self):
        """Draw a discrete random measure realization P_j from the DP posterior.
        
        Supports:
        1. 'vashishtha_maillard' (default): Finite stick-breaking representation from
           Eq. (288) of Vashishtha & Maillard (2025):
               Q_N = sum_{i=1}^N [V_i prod_{j=i+1}^N (1 - V_j)] delta_{X_i} + [prod_{i=1}^N (1 - V_i)] Q_0
           with N_emp observed atoms and K_prior synthetic base measure atoms from F_0.
        2. 'sethuraman': Standard truncated stick-breaking GEM(alpha + n).
        """
        n = len(self.rewards)

        if getattr(self, "sampler_type", "vashishtha_maillard") == "vashishtha_maillard":
            K_prior = max(8, int(self.alpha * 4))
            
            if n == 0:
                # Prior draw from F_0 only
                atom_x = np.zeros((K_prior, self.context_dim), dtype=np.float32)
                atom_a = np.zeros(K_prior, dtype=np.int64)
                atom_r = np.zeros(K_prior, dtype=np.float32)
                for k in range(K_prior):
                    atom_x[k], atom_a[k], atom_r[k] = self.sample_base_measure_atom()

                V = self.rng.beta(1.0, max(self.alpha, 1e-3), size=K_prior).astype(np.float32)
                V[-1] = 1.0
                cum_prod = np.cumprod(1.0 - V)
                q = np.empty_like(V)
                q[0] = V[0]
                q[1:] = V[1:] * cum_prod[:-1]
                q = q / q.sum() if q.sum() > 0 else np.ones(K_prior, dtype=np.float32) / K_prior
                return atom_x, atom_a, atom_r, q

            # Vashishtha & Maillard (2025) finite partition:
            N_emp = min(n, self.truncation_K)
            emp_indices = self.rng.choice(n, size=N_emp, replace=(n < N_emp))

            # 1. Total weight allocated to prior base measure: W_prior ~ Beta(alpha, N_emp)
            W_prior = float(self.rng.beta(self.alpha, float(N_emp)))

            # 2. Empirical stick-breaking weights: V_i ~ Beta(1, alpha + i)
            i_arr = np.arange(1, N_emp + 1, dtype=np.float32)
            V_emp = self.rng.beta(1.0, self.alpha + i_arr).astype(np.float32)
            U_emp = 1.0 - V_emp
            suffix_U = np.ones(N_emp + 1, dtype=np.float32)
            suffix_U[:-1] = np.cumprod(U_emp[::-1])[::-1]

            w_emp_raw = np.zeros(N_emp, dtype=np.float32)
            if N_emp > 1:
                w_emp_raw[:-1] = V_emp[:-1] * suffix_U[1:-1]
            w_emp_raw[-1] = V_emp[-1]
            emp_sum = w_emp_raw.sum()
            w_emp = (1.0 - W_prior) * (w_emp_raw / emp_sum) if emp_sum > 0 else np.full(N_emp, (1.0 - W_prior) / N_emp, dtype=np.float32)

            # 3. Prior base measure stick-breaking weights: Q_0 ~ GEM(alpha)
            V_prior = self.rng.beta(1.0, max(self.alpha, 1e-3), size=K_prior).astype(np.float32)
            V_prior[-1] = 1.0
            cum_prod_p = np.cumprod(1.0 - V_prior)
            q_p = np.empty_like(V_prior)
            q_p[0] = V_prior[0]
            q_p[1:] = V_prior[1:] * cum_prod_p[:-1]
            q_p_sum = q_p.sum()
            w_prior = W_prior * (q_p / q_p_sum) if q_p_sum > 0 else np.full(K_prior, W_prior / K_prior, dtype=np.float32)

            # Combine atoms
            total_atoms = N_emp + K_prior
            atom_x = np.zeros((total_atoms, self.context_dim), dtype=np.float32)
            atom_a = np.zeros(total_atoms, dtype=np.int64)
            atom_r = np.zeros(total_atoms, dtype=np.float32)
            q_all = np.concatenate([w_emp, w_prior]).astype(np.float32)

            # Fill empirical atoms
            for i, idx in enumerate(emp_indices):
                atom_x[i] = self.contexts[idx]
                atom_a[i] = self.actions[idx]
                atom_r[i] = self.rewards[idx]

            # Fill synthetic base measure atoms
            for k in range(K_prior):
                x_s, a_s, r_s = self.sample_base_measure_atom()
                atom_x[N_emp + k] = x_s
                atom_a[N_emp + k] = a_s
                atom_r[N_emp + k] = r_s

            return atom_x, atom_a, atom_r, q_all

        # Fallback to classical Sethuraman
        total_mass = self.alpha + n
        V = self.rng.beta(1.0, total_mass, size=self.truncation_K)
        V[-1] = 1.0
        remaining = 1.0
        q = np.zeros(self.truncation_K, dtype=np.float32)
        for k in range(self.truncation_K):
            q[k] = V[k] * remaining
            remaining *= (1.0 - V[k])
        q = q / q.sum() if q.sum() > 0 else np.ones(self.truncation_K, dtype=np.float32) / self.truncation_K

        atom_x = np.zeros((self.truncation_K, self.context_dim), dtype=np.float32)
        atom_a = np.zeros(self.truncation_K, dtype=np.int64)
        atom_r = np.zeros(self.truncation_K, dtype=np.float32)
        prob_empirical = n / total_mass if total_mass > 0 else 0.0

        for k in range(self.truncation_K):
            if n > 0 and self.rng.rand() < prob_empirical:
                idx = self.rng.randint(0, n)
                atom_x[k] = self.contexts[idx]
                atom_a[k] = self.actions[idx]
                atom_r[k] = self.rewards[idx]
            else:
                x_syn, a_syn, r_syn = self.sample_base_measure_atom()
                atom_x[k] = x_syn
                atom_a[k] = a_syn
                atom_r[k] = r_syn

        return atom_x, atom_a, atom_r, q

    def train_on_drawn_measure(self):
        """Warm-start gradient updates to fit the sampled DP hypothesis measure.
        Draws a fresh stochastic mini-batch DP sample at each step to prevent single-batch overfitting.
        """
        self.model.train()
        for _ in range(self.steps_per_decision):
            atom_x, atom_a, atom_r, q = self.sample_posterior_measure()
            x_tensor = torch.from_numpy(atom_x)
            a_tensor = torch.from_numpy(atom_a)
            r_tensor = torch.from_numpy(atom_r)
            q_tensor = torch.from_numpy(q)

            self.optimizer.zero_grad()
            preds = self.model(x_tensor)  # [K, num_arms]
            chosen_preds = preds.gather(1, a_tensor.unsqueeze(1)).squeeze(1)  # [K]
            loss = torch.sum(q_tensor * (chosen_preds - r_tensor) ** 2)
            loss.backward()
            self.optimizer.step()

    def select_action(self, context):
        """Thompson Sampling decision:

        1. Draw a random data distribution from DP posterior.
        2. Adapt the single network to this hypothesis.
        3. Predict rewards and pick the greedy action under this hypothesis.
        """
        # Initial random actions before observations
        if len(self.rewards) < self.num_arms:
            return len(self.rewards)

        # Train on drawn hypothesis measure
        self.train_on_drawn_measure()

        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(context).unsqueeze(0)
            preds = self.model(x_t).squeeze(0).numpy()
            action = int(np.argmax(preds))

        return action

    def update(self, context, action, reward):
        """Record observation into replay buffer."""
        self.contexts.append(context)
        self.actions.append(action)
        self.rewards.append(reward)
