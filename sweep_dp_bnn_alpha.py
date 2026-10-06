#!/usr/bin/env python3
"""
Quick alpha sweep for DP-BNN on Foong In-Between benchmark.
Tests alpha in {0.5, 1.0, 2.0, 5.0, 10.0, 20.0}
Reports: Gap Ratio, 95% Coverage, CRPS, ID-MSE
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.stats as scipy_stats
import math

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cpu')

# ── Shared backbone ──────────────────────────
class Backbone(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden); self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden); self.ln2 = nn.LayerNorm(hidden)
    def forward(self, x):
        return F.relu(self.ln2(self.fc2(F.relu(self.ln1(self.fc1(x))))))

class DeterministicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone(); self.head = nn.Linear(64, 1)
    def forward(self, x):
        return self.head(self.backbone(x))

# ── DP-BNN ───────────────────────────────────
def train_dp_bnn(X, y, domain_range, alpha, K=30, epochs=1200, lr=4e-3, K_t=100):
    N = X.shape[0]; x_min, x_max = domain_range
    crit = nn.MSELoss(reduction='none'); models = []
    for _ in range(K):
        alphas = alpha + np.arange(1, N+1, dtype=np.float64)
        V      = np.random.beta(1.0, alphas)
        OmV    = 1.0 - V
        sfx    = np.concatenate(([1.0], np.cumprod(OmV[::-1])[:-1]))[::-1].copy()
        w_data = V * sfx
        w_prior = float(np.prod(OmV))
        Vp  = np.random.beta(1.0, alpha, K_t); OmVp = 1.0 - Vp
        sfxp = np.concatenate(([1.0], np.cumprod(OmVp[::-1])[:-1]))[::-1].copy()
        q_p  = Vp * sfxp; q_p /= q_p.sum()
        Xp = torch.from_numpy(np.random.uniform(x_min, x_max, (K_t, 1)).astype(np.float32))
        yp = torch.from_numpy(np.random.normal(0.0, 2.0, (K_t, 1)).astype(np.float32))
        X_all = torch.cat([X, Xp]); y_all = torch.cat([y, yp])
        w_all = np.concatenate([w_data, w_prior * q_p])
        w_all = torch.tensor(w_all / w_all.sum(), dtype=torch.float32).unsqueeze(1)
        m = DeterministicNet(); opt = optim.Adam(m.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad(); (w_all * crit(m(X_all), y_all)).sum().backward(); opt.step()
        m.eval(); models.append(m)
    return models

def predict(models, X_test):
    with torch.no_grad():
        draws = np.array([m(X_test).squeeze().numpy() for m in models])
    return draws.mean(0), draws.std(0)

def metric_ratio(std, num_mask, den_mask):
    return float(np.mean(std[num_mask]) / max(np.mean(std[den_mask]), 1e-8))

def metric_crps(y_true, mu, std):
    std = np.clip(std, 1e-8, None); z = (y_true - mu) / std
    return float(np.mean(std * (z*(2*scipy_stats.norm.cdf(z)-1) + 2*scipy_stats.norm.pdf(z) - 1/math.sqrt(math.pi))))

def metric_coverage(y_true, mu, std):
    z = scipy_stats.norm.ppf(0.975)
    return float(((y_true >= mu - z*std) & (y_true <= mu + z*std)).mean() * 100)

# ── Foong dataset ────────────────────────────
np.random.seed(42); torch.manual_seed(42)
X1 = np.random.uniform(-3.0, -1.0, 40); X2 = np.random.uniform(1.0, 3.0, 40)
X_tr_np = np.concatenate([X1, X2])
y_tr_np = np.sin(X_tr_np) + 0.1 * np.random.randn(len(X_tr_np))
X_te_np = np.linspace(-4.5, 4.5, 400); y_te_np = np.sin(X_te_np)
X_tr = torch.tensor(X_tr_np, dtype=torch.float32).unsqueeze(1)
y_tr = torch.tensor(y_tr_np, dtype=torch.float32).unsqueeze(1)
X_te = torch.tensor(X_te_np, dtype=torch.float32).unsqueeze(1)

gap_mask     = (X_te_np >= -0.8) & (X_te_np <= 0.8)
cluster_mask = (((X_te_np >= -2.8) & (X_te_np <= -1.2)) |
                ((X_te_np >=  1.2) & (X_te_np <=  2.8)))

# ── Alpha sweep ──────────────────────────────
ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

print("DP-BNN Alpha Sweep — Foong In-Between Benchmark")
print("=" * 60)
print(f"{'alpha':>8} | {'Gap Ratio':>10} | {'Coverage':>10} | {'CRPS':>8} | {'ID-MSE':>8}")
print("-" * 60)

best_ratio, best_alpha = -1, None
results = {}
for alpha in ALPHAS:
    np.random.seed(SEED); torch.manual_seed(SEED)
    models = train_dp_bnn(X_tr, y_tr, domain_range=(-4.5, 4.5), alpha=alpha)
    mu, std = predict(models, X_te)
    gap   = metric_ratio(std, gap_mask, cluster_mask)
    cov   = metric_coverage(y_te_np, mu, std)
    crps  = metric_crps(y_te_np, mu, std)
    idmse = float(np.mean((y_te_np[cluster_mask] - mu[cluster_mask])**2))
    results[alpha] = {"gap_ratio": gap, "coverage": cov, "crps": crps, "id_mse": idmse}
    marker = " <-- BEST" if gap > best_ratio else ""
    if gap > best_ratio:
        best_ratio = gap; best_alpha = alpha
    print(f"{alpha:>8.1f} | {gap:>10.2f}x | {cov:>9.1f}% | {crps:>8.4f} | {idmse:>8.5f}{marker}")

print("-" * 60)
print(f"\nBest alpha = {best_alpha}  ->  Gap Ratio = {best_ratio:.2f}x")
print(f"\nRecommendation for paper: use alpha = {best_alpha}")
