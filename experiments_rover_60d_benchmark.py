"""
Rover / Robot Trajectory Planning Benchmark (60 Continuous Dimensions).
Widely used in high-dimensional Bayesian Optimization (Wang et al., 2018; TuRBO, NeurIPS 2019).
A rover navigates a 2D terrain with hazardous obstacles from (0, 0) to (1, 1).
The trajectory is parameterized by 30 2D waypoints (d = 60 continuous variables in [0, 1]^60).
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

DIM = 60
BOUNDS = torch.tensor([[0.0, 1.0] for _ in range(DIM)])

# -------------------------------------------------------------
# 1. 60D Rover Trajectory Objective Function
# -------------------------------------------------------------
def evaluate_rover_trajectory(traj_params):
    """
    Evaluates a 60D trajectory: 30 waypoints in [0, 1]^2.
    Objective = Path Length + Collision Penalty + Smoothness Penalty.
    Lower is better.
    """
    if isinstance(traj_params, torch.Tensor):
        pts = traj_params.detach().cpu().numpy().reshape(30, 2)
    else:
        pts = np.array(traj_params).reshape(30, 2)
        
    start = np.array([0.0, 0.0])
    goal = np.array([1.0, 1.0])
    full_path = np.vstack([start, pts, goal])  # [32, 2]
    
    # 1. Trajectory length
    diffs = np.diff(full_path, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1) + 1e-8)
    length_cost = float(np.sum(segment_lengths))
    
    # 2. Obstacle collision penalty
    # 4 circular hazardous obstacles blocking direct line of sight
    obstacles = [
        (np.array([0.25, 0.25]), 0.14),
        (np.array([0.50, 0.65]), 0.16),
        (np.array([0.75, 0.40]), 0.14),
        (np.array([0.45, 0.45]), 0.12)
    ]
    collision_cost = 0.0
    for obs_center, obs_radius in obstacles:
        dists = np.sqrt(np.sum((full_path - obs_center)**2, axis=1) + 1e-8)
        penetration = np.maximum(0.0, obs_radius - dists)
        collision_cost += float(80.0 * np.sum(penetration**2))
        
    # 3. Curvature / Smoothness cost
    second_diffs = np.diff(diffs, axis=0)
    smoothness_cost = float(4.0 * np.sum(second_diffs**2))
    
    return length_cost + collision_cost + smoothness_cost

# -------------------------------------------------------------
# 2. Surrogates for 60-Dimensional Space
# -------------------------------------------------------------
class DPBNNSurrogate(nn.Module):
    """Data-Space Dirichlet Process BNN for 60-dimensional optimization."""
    def __init__(self, in_dim=60, hidden_dim=64, num_heads=4, alpha=1.0, T_prior=30):
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
        
        for k in range(len(self.heads)):
            X_prior = torch.rand(T, self.in_dim)
            # Conservative prior: mean +1.0 in normalized space, std 1.0
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
    def __init__(self, in_dim=60, hidden_dim=64, p=0.2):
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
    def __init__(self, in_dim=60):
        self.l = math.sqrt(in_dim)
        self.X = None
        self.y = None
        self.K_inv = None

    def fit(self, X, y):
        self.X, self.y = X, y
        dists_sq = torch.cdist(X, X)**2
        K = torch.exp(-dists_sq / (2.0 * self.l**2)) + 1e-3 * torch.eye(X.shape[0])
        self.K_inv = torch.linalg.pinv(K)

    def predict(self, cand):
        with torch.no_grad():
            Ks = torch.exp(-torch.cdist(cand, self.X)**2 / (2.0 * self.l**2))
            mu = (Ks @ (self.K_inv @ self.y)).squeeze(-1)
            var = 1.0 - torch.sum(Ks @ self.K_inv * Ks, dim=1)
            return mu, torch.sqrt(var.clamp(min=1e-4))

# -------------------------------------------------------------
# 3. 60D Benchmark Loop
# -------------------------------------------------------------
NUM_INIT = 8
NUM_ITERS = 40
CAND_POOL = 1200

# Suboptimal initial trajectories: straight lines + noisy random zigzags that hit obstacles
X_init = []
for i in range(NUM_INIT):
    t_interp = np.linspace(0, 1, 32)[1:-1]
    # Add random zigzag perturbation
    noise = np.random.randn(30, 2) * (0.05 + 0.05 * i)
    traj_i = np.column_stack([t_interp, t_interp]) + noise
    X_init.append(torch.tensor(np.clip(traj_i.flatten(), 0.0, 1.0), dtype=torch.float32))

X_init = torch.stack(X_init, dim=0)  # [8, 60]
y_init = torch.tensor([[evaluate_rover_trajectory(X_init[i])] for i in range(NUM_INIT)], dtype=torch.float32)

print(f"\n[*] Starting 60D Rover Trajectory Planning BO (Initial Best Cost: {y_init.min().item():.2f})")
results = {}

methods = ["Random_Search", "Standard_BNN", "GP_BO", "DP_BO"]

for method in methods:
    print(f"\n---> Running {method}...")
    X_hist = X_init.clone()
    y_hist = y_init.clone()
    best_curve = [y_hist.min().item()]
    step_times = []

    if method == "Standard_BNN":
        surr = MCBNNSurrogate(in_dim=DIM)
    elif method == "GP_BO":
        surr = GPSurrogate(in_dim=DIM)
    elif method == "DP_BO":
        surr = DPBNNSurrogate(in_dim=DIM, alpha=1.0, T_prior=30)

    t0_start = time.perf_counter()
    for it in range(NUM_ITERS):
        t0_step = time.perf_counter()
        
        # Candidate generation in 60D:
        # 60% trust-region perturbations around top-2 best paths, 40% uniform exploration
        top_idx = torch.topk(-y_hist.squeeze(), k=min(2, y_hist.shape[0])).indices
        best_pts = X_hist[top_idx]
        perturbs = best_pts.repeat(360, 1) + torch.randn(720, DIM) * 0.08
        cand_unif = torch.rand(480, DIM)
        cand_pool = torch.cat([perturbs.clamp(0.0, 1.0), cand_unif], dim=0)

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
                
            # Lower Confidence Bound
            lcb = (mu * y_std + y_mean) - 1.0 * (sig * y_std)
            best_idx = torch.argmin(lcb)
            next_x = cand_pool[best_idx:best_idx+1]

        # Evaluate real 60D rover trajectory
        new_val = evaluate_rover_trajectory(next_x[0])
        X_hist = torch.cat([X_hist, next_x], dim=0)
        y_hist = torch.cat([y_hist, torch.tensor([[new_val]], dtype=torch.float32)], dim=0)
        
        best_so_far = min(best_curve[-1], new_val)
        best_curve.append(best_so_far)
        step_times.append((time.perf_counter() - t0_step) * 1000.0)

    elapsed = time.perf_counter() - t0_start
    print(f"[{method}] Final Trajectory Cost: {best_curve[-1]:.3f} | Total Time: {elapsed:.2f}s | Avg Step: {np.mean(step_times):.1f}ms")
    results[method] = {
        "final_cost": best_curve[-1],
        "curve": best_curve,
        "avg_step_ms": float(np.mean(step_times)),
        "total_time_s": elapsed
    }

with open("/Users/sumitvashishtha/Desktop/DP-BNNs/results_rover_60d_bo.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n[+] 60D Rover Trajectory Benchmark finished and saved to results_rover_60d_bo.json!")
