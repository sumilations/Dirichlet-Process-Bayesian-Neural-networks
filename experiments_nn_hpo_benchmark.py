"""
Neural Network Hyperparameter Optimization (AutoML) Benchmark.
Compares DP-BO against GP-BO, Standard BNN (MC-Dropout), and Random Search
on tuning a deep classification neural network across 5 continuous hyperparameters.
"""

import time, math, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

# -------------------------------------------------------------
# 1. Dataset Generation: Multi-Class Non-Linear Manifold
# -------------------------------------------------------------
N = 1500
D = 12
K = 5
centers = torch.randn(K, D) * 1.5
X_list, y_list = [], []
for k in range(K):
    pts = centers[k] + torch.randn(N // K, D) * 1.3
    pts[:, :6] = pts[:, :6] + 0.4 * torch.sin(pts[:, 6:] * 2.0)
    X_list.append(pts)
    y_list.append(torch.full((N // K,), k, dtype=torch.long))

X_all = torch.cat(X_list, dim=0)
y_all = torch.cat(y_list, dim=0)
perm = torch.randperm(X_all.shape[0])
X_all, y_all = X_all[perm], y_all[perm]

X_tr, y_tr = X_all[:1000], y_all[:1000]
X_val, y_val = X_all[1000:], y_all[1000:]

# Hyperparameter search bounds (5 dimensions)
# 0: log10(lr) in [-4.0, 0.0]
# 1: log10(wd) in [-5.0, -1.0]
# 2: dropout in [0.0, 0.5]
# 3: width in [16, 96]
# 4: beta1 in [0.80, 0.99]
BOUNDS = torch.tensor([
    [-4.0, 0.0],
    [-5.0, -1.0],
    [0.0, 0.5],
    [16.0, 96.0],
    [0.80, 0.99]
])

def evaluate_nn_hparams(hparams):
    """Evaluates validation classification error (%) of a neural network."""
    lr = float(10.0 ** hparams[0])
    wd = float(10.0 ** hparams[1])
    p_drop = float(np.clip(hparams[2], 0.0, 0.5))
    width = int(np.clip(hparams[3], 16, 96))
    beta1 = float(np.clip(hparams[4], 0.80, 0.99))
    
    # Sharp non-stationary divergence cliff: if lr > 0.20, training diverges
    if lr > 0.20:
        return 80.0 + float(np.random.rand() * 2.0)
        
    net = nn.Sequential(
        nn.Linear(D, width),
        nn.SiLU(),
        nn.Dropout(p_drop),
        nn.Linear(width, width),
        nn.SiLU(),
        nn.Dropout(p_drop),
        nn.Linear(width, K)
    )
    
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd, betas=(beta1, 0.999))
    crit = nn.CrossEntropyLoss()
    
    batch_size = 64
    for ep in range(25):
        perm_idx = torch.randperm(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], batch_size):
            idx = perm_idx[i:i+batch_size]
            opt.zero_grad()
            out = net(X_tr[idx])
            loss = crit(out, y_tr[idx])
            if torch.isnan(loss) or torch.isinf(loss):
                return 80.0
            loss.backward()
            opt.step()
            
    net.eval()
    with torch.no_grad():
        preds = torch.argmax(net(X_val), dim=1)
        val_err = (preds != y_val).float().mean().item() * 100.0
    return val_err

# -------------------------------------------------------------
# 2. Surrogates
# -------------------------------------------------------------
class DPBNNSurrogate(nn.Module):
    """DP-BNN with exact Ferguson DP random measure sampling and conservative prior."""
    def __init__(self, in_dim=5, hidden_dim=48, num_heads=4, alpha=1.0, T_prior=25):
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
        self.alpha = alpha
        self.T_prior = T_prior

    def fit(self, X_emp, y_emp, steps=35, lr=2e-3):
        t_emp = X_emp.shape[0]
        T = self.T_prior
        total_N = t_emp + T
        conc = torch.ones(total_N)
        conc[t_emp:] = self.alpha / float(T)
        
        low = BOUNDS[:, 0]
        high = BOUNDS[:, 1]
        
        for k in range(len(self.heads)):
            X_prior = torch.rand(T, self.in_dim) * (high - low) + low
            # Conservative prior: mean +1.0 in normalized space (mediocre/suboptimal), std 1.0
            y_prior = 1.0 + torch.randn(T, 1) * 1.0
            
            X_comb = torch.cat([X_emp, X_prior], dim=0)
            y_comb = torch.cat([y_emp, y_prior], dim=0)
            
            w = Dirichlet(conc).sample().unsqueeze(1) * total_N
            opt = torch.optim.AdamW(self.heads[k].parameters(), lr=lr, weight_decay=1e-3)
            for _ in range(steps):
                opt.zero_grad()
                pred = self.heads[k](X_comb)
                loss = (w * (pred - y_comb)**2).mean()
                loss.backward()
                opt.step()

    def predict(self, cand):
        with torch.no_grad():
            preds = torch.stack([h(cand) for h in self.heads], dim=0)
            return preds.mean(0).squeeze(-1), preds.std(0).squeeze(-1).clamp(min=1e-4)

class MCBNNSurrogate(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=48, p=0.2):
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

    def fit(self, X_emp, y_emp, steps=35, lr=2e-3):
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-3)
        self.net.train()
        for _ in range(steps):
            opt.zero_grad()
            loss = F.mse_loss(self.net(X_emp), y_emp)
            loss.backward()
            opt.step()

    def predict(self, cand):
        self.net.train()
        with torch.no_grad():
            preds = torch.stack([self.net(cand) for _ in range(20)], dim=0)
            return preds.mean(0).squeeze(-1), preds.std(0).squeeze(-1).clamp(min=1e-4)

class GPSurrogate:
    def __init__(self, in_dim=5):
        self.l = math.sqrt(in_dim)
        self.X = None
        self.y = None
        self.K_inv = None

    def fit(self, X, y):
        self.X, self.y = X, y
        scale = BOUNDS[:, 1] - BOUNDS[:, 0]
        X_s = X / scale
        dists_sq = torch.cdist(X_s, X_s)**2
        K = torch.exp(-dists_sq / (2.0 * self.l**2)) + 1e-3 * torch.eye(X.shape[0])
        self.K_inv = torch.linalg.pinv(K)

    def predict(self, cand):
        scale = BOUNDS[:, 1] - BOUNDS[:, 0]
        cand_s = cand / scale
        X_s = self.X / scale
        with torch.no_grad():
            Ks = torch.exp(-torch.cdist(cand_s, X_s)**2 / (2.0 * self.l**2))
            mu = (Ks @ (self.K_inv @ self.y)).squeeze(-1)
            var = 1.0 - torch.sum(Ks @ self.K_inv * Ks, dim=1)
            return mu, torch.sqrt(var.clamp(min=1e-4))

# -------------------------------------------------------------
# 3. Benchmark Execution Loop
# -------------------------------------------------------------
NUM_INIT = 6
NUM_ITERS = 28
CAND_POOL = 1000

# Suboptimal initial points (default conservative hyperparams)
# Low learning rates: 1e-4 to 5e-4, high weight decay, conservative widths
X_init = torch.tensor([
    [-4.0, -2.0, 0.40, 24.0, 0.85],
    [-3.8, -1.5, 0.35, 20.0, 0.88],
    [-3.5, -2.5, 0.45, 18.0, 0.82],
    [-3.7, -2.0, 0.30, 28.0, 0.90],
    [-3.9, -1.8, 0.40, 22.0, 0.84],
    [-3.6, -2.2, 0.35, 24.0, 0.86]
], dtype=torch.float32)

y_init = torch.tensor([[evaluate_nn_hparams(X_init[i].numpy())] for i in range(NUM_INIT)], dtype=torch.float32)

print(f"\n[*] Starting Neural Network HPO Benchmark (Initial Best Validation Error: {y_init.min().item():.2f}%)")
results = {}

methods = ["Random_Search", "Standard_BNN", "GP_BO", "DP_BO"]

for method in methods:
    print(f"\n---> Running {method}...")
    X_hist = X_init.clone()
    y_hist = y_init.clone()
    best_curve = [y_hist.min().item()]
    step_times = []

    if method == "Standard_BNN":
        surr = MCBNNSurrogate(in_dim=5)
    elif method == "GP_BO":
        surr = GPSurrogate(in_dim=5)
    elif method == "DP_BO":
        surr = DPBNNSurrogate(in_dim=5, alpha=1.0, T_prior=25)

    t0_start = time.perf_counter()
    for it in range(NUM_ITERS):
        t0_step = time.perf_counter()
        
        # Candidate pool: 60% local perturbation around top-2 best, 40% global uniform
        low, high = BOUNDS[:, 0], BOUNDS[:, 1]
        cand_unif = torch.rand(400, 5) * (high - low) + low
        top_idx = torch.topk(-y_hist.squeeze(), k=min(2, y_hist.shape[0])).indices
        best_pts = X_hist[top_idx]
        scale_vec = (high - low) * 0.18
        perturbs = best_pts.repeat(300, 1) + torch.randn(600, 5) * scale_vec
        cand_pool = torch.cat([perturbs.clamp(low, high), cand_unif], dim=0)

        if method == "Random_Search":
            next_idx = np.random.randint(cand_pool.shape[0])
            next_x = cand_pool[next_idx:next_idx+1]
        else:
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
                
            # Lower Confidence Bound (minimizing validation error)
            lcb = (mu * y_std + y_mean) - 1.0 * (sig * y_std)
            best_idx = torch.argmin(lcb)
            next_x = cand_pool[best_idx:best_idx+1]

        # Evaluate real neural network
        new_val = evaluate_nn_hparams(next_x[0].numpy())
        X_hist = torch.cat([X_hist, next_x], dim=0)
        y_hist = torch.cat([y_hist, torch.tensor([[new_val]], dtype=torch.float32)], dim=0)
        
        best_so_far = min(best_curve[-1], new_val)
        best_curve.append(best_so_far)
        step_times.append((time.perf_counter() - t0_step) * 1000.0)

    elapsed = time.perf_counter() - t0_start
    print(f"[{method}] Final Validation Error: {best_curve[-1]:.2f}% | Total Time: {elapsed:.2f}s | Avg Step: {np.mean(step_times):.1f}ms")
    results[method] = {
        "final_val_error": best_curve[-1],
        "curve": best_curve,
        "avg_step_ms": float(np.mean(step_times)),
        "total_time_s": elapsed
    }

with open("/Users/sumitvashishtha/Desktop/DP-BNNs/results_nn_hpo_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n[+] Calibrated HPO Benchmark finished and saved to results_nn_hpo_benchmark.json!")
