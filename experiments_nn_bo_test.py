import gym, time, math, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

DIM = 36
BOUNDS = (-2.0, 2.0)

def evaluate_policy(w_np, seeds=[101, 102, 103]):
    W1 = w_np[:24].reshape(4, 6)
    W2 = w_np[24:].reshape(6, 2)
    env = gym.make("CartPole-v1")
    total_ret = 0.0
    for seed in seeds:
        s = env.reset(seed=seed)
        if isinstance(s, tuple): s = s[0]
        done = False
        ret = 0
        while not done:
            h = np.tanh(s @ W1)
            a = int(np.argmax(h @ W2))
            step_res = env.step(a)
            s = step_res[0]
            ret += 1
            done = step_res[2] or step_res[3]
            if ret >= 500: break
        total_ret += ret
    avg_ret = total_ret / len(seeds)
    return 500.0 - avg_ret  # Regret: 0 is optimal (500 steps)

# -------------------------------------------------------------
# Surrogate 1: DP-BNN (True Data-Space Dirichlet Process with Base Measure)
# -------------------------------------------------------------
class DPBNNSurrogate(nn.Module):
    def __init__(self, in_dim=36, hidden_dim=64, num_heads=4, alpha=1.0, T_prior=20):
        super().__init__()
        self.in_dim = in_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_heads)
        ])
        self.num_heads = num_heads
        self.alpha = alpha
        self.T_prior = T_prior

    def fit(self, X_emp, y_emp, steps=40, lr=1e-3):
        t_emp = X_emp.shape[0]
        # Generate T_prior synthetic pseudo-observations from base measure F_0
        # Base measure: uniform random weights, prior regret ~ mean of empirical + noise
        X_prior = torch.rand(self.T_prior, self.in_dim) * (BOUNDS[1] - BOUNDS[0]) + BOUNDS[0]
        y_prior = torch.full((self.T_prior, 1), 250.0) + torch.randn(self.T_prior, 1) * 50.0
        
        X_combined = torch.cat([X_emp, X_prior], dim=0)
        y_combined = torch.cat([y_emp, y_prior], dim=0)
        total_N = t_emp + self.T_prior
        
        # Dirichlet parameter: 1 for empirical, alpha/T for prior
        dir_conc = torch.ones(total_N)
        dir_conc[t_emp:] = self.alpha / float(self.T_prior)
        
        # Sample weights for each head
        for k in range(self.num_heads):
            w = Dirichlet(dir_conc).sample().unsqueeze(1) * total_N
            opt = torch.optim.AdamW(self.heads[k].parameters(), lr=lr, weight_decay=1e-4)
            for _ in range(steps):
                opt.zero_grad()
                pred = self.heads[k](X_combined)
                loss = (w * (pred - y_combined)**2).mean()
                loss.backward()
                opt.step()

    def predict(self, X_cand):
        with torch.no_grad():
            preds = torch.stack([head(X_cand) for head in self.heads], dim=0) # [K, N, 1]
            mu = preds.mean(dim=0).squeeze(-1)
            sigma = preds.std(dim=0).squeeze(-1)
        return mu, sigma

# -------------------------------------------------------------
# Surrogate 2: Standard BNN (MC-Dropout)
# -------------------------------------------------------------
class MCDropoutSurrogate(nn.Module):
    def __init__(self, in_dim=36, hidden_dim=64, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, 1)
        )
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)

    def fit(self, X_emp, y_emp, steps=40):
        self.net.train()
        for _ in range(steps):
            self.opt.zero_grad()
            pred = self.net(X_emp)
            loss = F.mse_loss(pred, y_emp)
            loss.backward()
            self.opt.step()

    def predict(self, X_cand, S=20):
        self.net.train() # Keep dropout active
        with torch.no_grad():
            preds = torch.stack([self.net(X_cand) for _ in range(S)], dim=0)
            mu = preds.mean(dim=0).squeeze(-1)
            sigma = preds.std(dim=0).squeeze(-1).clamp(min=1e-4)
        return mu, sigma

# -------------------------------------------------------------
# Surrogate 3: Gaussian Process (Exact GP with RBF)
# -------------------------------------------------------------
class GPSurrogate:
    def __init__(self, lengthscale=math.sqrt(DIM), noise_var=1e-2):
        self.l = lengthscale
        self.noise = noise_var
        self.X = None
        self.y = None
        self.K_inv = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        n = X.shape[0]
        dists_sq = torch.cdist(X, X)**2
        K = torch.exp(-dists_sq / (2.0 * self.l**2)) + self.noise * torch.eye(n)
        self.K_inv = torch.linalg.pinv(K)

    def predict(self, X_cand):
        with torch.no_grad():
            K_star = torch.exp(-torch.cdist(X_cand, self.X)**2 / (2.0 * self.l**2))
            mu = (K_star @ (self.K_inv @ self.y)).squeeze(-1)
            var = 1.0 - torch.sum(K_star @ self.K_inv * K_star, dim=1)
            sigma = torch.sqrt(var.clamp(min=1e-4))
        return mu, sigma

print("Surrogates initialized successfully!")

# -------------------------------------------------------------
# Run 36D Neural Policy Search BO Benchmark
# -------------------------------------------------------------
NUM_INIT = 6
NUM_ITERS = 30
CAND_POOL = 1500

# Common initial seeds
X_init = torch.rand(NUM_INIT, DIM) * (BOUNDS[1] - BOUNDS[0]) + BOUNDS[0]
y_init = torch.tensor([[evaluate_policy(X_init[i].numpy())] for i in range(NUM_INIT)], dtype=torch.float32)

print(f"\n[*] Starting 36D Neural Network Policy Search (Init Best: {y_init.min().item():.1f} steps regret)")
results = {}

for method in ["Random_Search", "Standard_BNN", "GP_BO", "DP_BO"]:
    print(f"\n---> Running {method}...")
    X_hist = X_init.clone()
    y_hist = y_init.clone()
    best_curve = [y_hist.min().item()]
    step_times = []
    
    if method == "Standard_BNN":
        surr = MCDropoutSurrogate(in_dim=DIM)
    elif method == "GP_BO":
        surr = GPSurrogate()
    elif method == "DP_BO":
        surr = DPBNNSurrogate(in_dim=DIM, alpha=1.0, T_prior=25)

    t0_start = time.perf_counter()
    for it in range(NUM_ITERS):
        t0_step = time.perf_counter()
        
        # Candidate pool: mix of uniform Sobol + localized perturbations around top-3 best points
        cand_uniform = torch.rand(CAND_POOL - 300, DIM) * (BOUNDS[1] - BOUNDS[0]) + BOUNDS[0]
        top_indices = torch.topk(-y_hist.squeeze(), k=min(3, y_hist.shape[0])).indices
        top_pts = X_hist[top_indices]
        perturbs = top_pts.repeat(100, 1) + torch.randn(300, DIM) * 0.15
        perturbs = perturbs.clamp(BOUNDS[0], BOUNDS[1])
        cand_pool = torch.cat([cand_uniform, perturbs], dim=0)

        if method == "Random_Search":
            next_idx = np.random.randint(cand_pool.shape[0])
            next_x = cand_pool[next_idx:next_idx+1]
        else:
            # Fit surrogate
            y_mean, y_std = y_hist.mean(), y_hist.std().clamp(min=1.0)
            y_norm = (y_hist - y_mean) / y_std
            
            if method == "GP_BO":
                surr.fit(X_hist, y_norm)
                mu, sig = surr.predict(cand_pool)
            elif method == "Standard_BNN":
                surr.fit(X_hist, y_norm, steps=30)
                mu, sig = surr.predict(cand_pool)
            elif method == "DP_BO":
                surr.fit(X_hist, y_norm, steps=30)
                mu, sig = surr.predict(cand_pool)
            
            # Lower Confidence Bound: LCB = mu - beta * sig
            lcb = mu - 2.0 * sig
            best_idx = torch.argmin(lcb)
            next_x = cand_pool[best_idx:best_idx+1]

        # Evaluate real policy in environment
        new_val = evaluate_policy(next_x[0].numpy())
        X_hist = torch.cat([X_hist, next_x], dim=0)
        y_hist = torch.cat([y_hist, torch.tensor([[new_val]], dtype=torch.float32)], dim=0)
        
        best_so_far = min(best_curve[-1], new_val)
        best_curve.append(best_so_far)
        step_times.append((time.perf_counter() - t0_step) * 1000.0)

    elapsed = time.perf_counter() - t0_start
    print(f"[{method}] Final Regret: {best_curve[-1]:.1f} (Solved Return: {500.0 - best_curve[-1]:.1f}/500) | Time: {elapsed:.2f}s | Avg Step: {np.mean(step_times):.1f}ms")
    results[method] = {
        "final_regret": best_curve[-1],
        "solved_return": 500.0 - best_curve[-1],
        "curve": best_curve,
        "avg_step_ms": float(np.mean(step_times)),
        "total_time_s": elapsed
    }

with open("/Users/sumitvashishtha/Desktop/DP-BNNs/results_nn_policy_bo.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n[+] Benchmark finished and saved to results_nn_policy_bo.json!")
