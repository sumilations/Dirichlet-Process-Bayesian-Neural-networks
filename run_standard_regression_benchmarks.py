#!/usr/bin/env python3
"""
Standard Regression Benchmarks for DP-BNN (with LayerNorm):
1. Osband et al. (NeurIPS 2018, Section 3 / Figure 2) Benchmark
2. Foong et al. (ICML 2019 / UAI 2020) 'In-Between' Uncertainty Benchmark

Compares:
- Deep Ensembles (with LayerNorm)
- Vanilla BootDQN / Bootstrap (with LayerNorm)
- BootDQN + Randomized Priors (with LayerNorm)
- DP-BNN (Vashishtha Formulation, with LayerNorm)
- DP-BNN (Without LayerNorm Ablation)

Runs on Apple Silicon MPS GPU if available, with CPU fallback.
"""

import os
import json
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Device setup (MPS GPU if available)
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using acceleration device: {DEVICE}')

# Global seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------------------------------------
# Core Network Architecture (Shared Across All Methods)
# -------------------------------------------------------------
class MLPRegression(nn.Module):
    def __init__(self, hidden_dim=64, use_layer_norm=True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        self.to(DEVICE)
            
    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------
# Baselines & DP-BNN Algorithms
# -------------------------------------------------------------

# 1. Standard Deep Ensembles (Lakshminarayanan et al., 2017)
def train_deep_ensemble(X, y, num_models=10, epochs=800, lr=0.005, use_layer_norm=True):
    models = []
    criterion = nn.MSELoss()
    for _ in range(num_models):
        model = MLPRegression(hidden_dim=64, use_layer_norm=use_layer_norm)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        models.append(model)
    return models

# 2. Vanilla BootDQN / Bootstrap
def train_vanilla_bootstrap(X, y, num_models=10, epochs=800, lr=0.005, use_layer_norm=True):
    models = []
    n = X.shape[0]
    criterion = nn.MSELoss(reduction='none')
    for _ in range(num_models):
        model = MLPRegression(hidden_dim=64, use_layer_norm=use_layer_norm)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        w = torch.from_numpy(np.random.poisson(1.0, n)).float().to(DEVICE).unsqueeze(1)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.mean(criterion(model(X), y) * w)
            loss.backward()
            optimizer.step()
        models.append(model)
    return models

# 3. BootDQN + Randomized Priors (Osband et al., 2018)
class NetworkWithPrior(nn.Module):
    def __init__(self, beta=1.0, hidden_dim=64, use_layer_norm=True):
        super().__init__()
        self.model = MLPRegression(hidden_dim=hidden_dim, use_layer_norm=use_layer_norm)
        self.prior = MLPRegression(hidden_dim=hidden_dim, use_layer_norm=use_layer_norm)
        for p in self.prior.parameters():
            p.requires_grad = False
        self.beta = beta
        self.to(DEVICE)
        
    def forward(self, x):
        return self.model(x) + self.beta * self.prior(x)

def train_boot_prior(X, y, num_models=10, epochs=800, lr=0.005, beta=1.0, use_layer_norm=True):
    models = []
    n = X.shape[0]
    criterion = nn.MSELoss(reduction='none')
    for _ in range(num_models):
        model = NetworkWithPrior(beta=beta, hidden_dim=64, use_layer_norm=use_layer_norm)
        optimizer = optim.Adam(model.model.parameters(), lr=lr)
        w = torch.from_numpy(np.random.poisson(1.0, n)).float().to(DEVICE).unsqueeze(1)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.mean(criterion(model(X), y) * w)
            loss.backward()
            optimizer.step()
        models.append(model)
    return models

# 4. DP-BNN (Vashishtha Formulation, Eq. 288 + Stick Breaking)
def train_dp_bnn(X, y, domain_range, num_models=10, epochs=800, lr=0.005, alpha=2.0, K_t=100, use_layer_norm=True):
    models = []
    N = X.shape[0]
    x_min, x_max = domain_range
    criterion = nn.MSELoss(reduction='none')
    
    for _ in range(num_models):
        # Exact Vashishtha & Maillard stick breaking weights (Eq. 288)
        alphas = alpha + np.arange(1, N + 1)
        V = np.random.beta(1.0, alphas)
        OmV = 1.0 - V
        k_prod = np.cumprod(OmV[::-1])
        w_data = np.zeros(N)
        w_data[-1] = V[-1]
        w_data[:-1] = V[:-1] * (k_prod[::-1][1:])
        w_prior_total = k_prod[-1] # prod_{i=1}^N (1 - V_i)
        
        # Prior Q_0 ~ DP(alpha, F_0) via stick-breaking
        V_prior = np.random.beta(1.0, alpha, K_t)
        OmV_p = 1.0 - V_prior
        k_prior = np.cumprod(OmV_p)
        q_prior = np.zeros(K_t)
        q_prior[0] = V_prior[0]
        q_prior[1:-1] = V_prior[1:-1] * k_prior[:-2]
        q_prior[-1] = k_prior[-2]
        q_prior = q_prior / np.sum(q_prior)
        
        # Prior base atoms Z_k ~ Uniform(domain_range), Y_k ~ Normal(0, sigma_prior^2)
        X_p = torch.from_numpy(np.random.uniform(x_min, x_max, (K_t, 1))).float().to(DEVICE)
        y_p = torch.from_numpy(np.random.normal(0.0, 2.0, (K_t, 1))).float().to(DEVICE)
        
        X_all = torch.cat([X, X_p], dim=0)
        y_all = torch.cat([y, y_p], dim=0)
        
        # Combined weights normalized to 1.0
        w_all = np.concatenate([w_data, w_prior_total * q_prior])
        w_all = torch.tensor(w_all, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        w_all = w_all / w_all.sum()
        
        model = MLPRegression(hidden_dim=64, use_layer_norm=use_layer_norm)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.sum(w_all * criterion(model(X_all), y_all))
            loss.backward()
            optimizer.step()
        models.append(model)
    return models

def predict_ensemble(models, X_test):
    preds = []
    with torch.no_grad():
        for m in models:
            m.eval()
            p = m(X_test).squeeze().cpu().numpy()
            preds.append(p)
    preds = np.array(preds)
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0)
    return mean, std, preds

# -------------------------------------------------------------
# BENCHMARK 1: Osband et al. (NeurIPS 2018 Figure 2)
# -------------------------------------------------------------
def run_osband_benchmark():
    print("\n" + "="*70)
    print("BENCHMARK 1: Osband et al. (NeurIPS 2018 Figure 2) Regression")
    print("="*70)
    
    # 10 data points on [-2, 2], y = x + 1.5 * N(0, 1)
    np.random.seed(10)
    n_data = 10
    X_train_np = np.linspace(-2.0, 2.0, n_data)
    y_train_np = X_train_np + 1.5 * np.random.randn(n_data)
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    
    X_test_np = np.linspace(-3.5, 3.5, 300)
    y_test_true = X_test_np # Underlying linear trend
    X_test = torch.tensor(X_test_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    
    print("Training models on Osband Figure 2 dataset...")
    m_deep = train_deep_ensemble(X_train, y_train, use_layer_norm=True)
    m_boot = train_vanilla_bootstrap(X_train, y_train, use_layer_norm=True)
    m_prior = train_boot_prior(X_train, y_train, beta=1.0, use_layer_norm=True)
    m_dp_ln = train_dp_bnn(X_train, y_train, domain_range=(-3.5, 3.5), alpha=2.0, use_layer_norm=True)
    m_dp_no_ln = train_dp_bnn(X_train, y_train, domain_range=(-3.5, 3.5), alpha=2.0, use_layer_norm=False)
    
    methods = {
        "Deep Ensembles (LN)": m_deep,
        "Vanilla BootDQN (LN)": m_boot,
        "BootDQN + Priors (LN)": m_prior,
        "DP-BNN (Ours, LN)": m_dp_ln,
        "DP-BNN (No LayerNorm)": m_dp_no_ln
    }
    
    results = {}
    preds_dict = {}
    
    # Extrapolation bounds (|x| > 2.0) vs In-Distribution (|x| <= 2.0)
    id_idx = np.where(np.abs(X_test_np) <= 2.0)[0]
    ood_idx = np.where(np.abs(X_test_np) > 2.0)[0]
    
    print(f"{'Method':<25} | {'ID Std':<10} | {'OOD Std':<10} | {'OOD/ID Ratio':<14}")
    print("-"*65)
    for name, models in methods.items():
        mu, sig, raw = predict_ensemble(models, X_test)
        preds_dict[name] = (mu, sig, raw)
        id_std = float(np.mean(sig[id_idx]))
        ood_std = float(np.mean(sig[ood_idx]))
        ratio = ood_std / max(id_std, 1e-6)
        results[name] = {"id_std": id_std, "ood_std": ood_std, "ratio": ratio}
        print(f"{name:<25} | {id_std:<10.3f} | {ood_std:<10.3f} | {ratio:<14.2f}x")
    print("-"*65)
    
    # Generate 4-panel figure matching Osband Figure 2 layout
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axs = axs.flatten()
    showcase = ["Deep Ensembles (LN)", "Vanilla BootDQN (LN)", "BootDQN + Priors (LN)", "DP-BNN (Ours, LN)"]
    titles = ["(a) Deep Ensembles (LayerNorm)", "(b) Vanilla BootDQN (LayerNorm)", "(c) BootDQN + Randomized Priors (LN)", "(d) DP-BNN (Vashishtha Formulation, LN)"]
    
    for idx, (k, title) in enumerate(zip(showcase, titles)):
        ax = axs[idx]
        mu, sig, raw = preds_dict[k]
        for h in range(min(5, raw.shape[0])):
            ax.plot(X_test_np, raw[h], color='lightblue', alpha=0.5, linestyle='--', lw=1.2)
        ax.plot(X_test_np, mu, color='blue', lw=2.2, label='Mean Prediction')
        ax.fill_between(X_test_np, mu - 1.96*sig, mu + 1.96*sig, color='blue', alpha=0.18, label='95% Credible Interval')
        ax.scatter(X_train_np, y_train_np, color='black', s=24, zorder=5, label='Training Data (N=10)')
        ax.axvspan(-2.0, 2.0, color='lightgreen', alpha=0.06, label='In-Distribution Region')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(-8.0, 8.0)
        ax.set_xlim(-3.5, 3.5)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
            
    plt.suptitle("Benchmark 1: Osband et al. (NeurIPS 2018 Figure 2) Extrapolation Showdown", fontsize=13, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/benchmark_osband_figure2.png", dpi=300)
    plt.close()
    print("Saved figure to results/benchmark_osband_figure2.png")
    return results

# -------------------------------------------------------------
# BENCHMARK 2: Foong et al. (ICML 2019) In-Between Uncertainty
# -------------------------------------------------------------
def run_foong_inbetween_benchmark():
    print("\n" + "="*70)
    print("BENCHMARK 2: Foong et al. (ICML 2019) In-Between Uncertainty Benchmark")
    print("="*70)
    
    # 2 clusters: [-3, -1] and [1, 3], gap (-1, 1), y = sin(x) + noise
    np.random.seed(42)
    n_cluster = 40
    X1 = np.random.uniform(-3.0, -1.0, n_cluster)
    X2 = np.random.uniform(1.0, 3.0, n_cluster)
    X_train_np = np.concatenate([X1, X2])
    y_train_np = np.sin(X_train_np) + 0.1 * np.random.randn(len(X_train_np))
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    
    X_test_np = np.linspace(-4.5, 4.5, 400)
    y_test_true = np.sin(X_test_np)
    X_test = torch.tensor(X_test_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    
    print("Training models on Foong In-Between dataset...")
    m_deep = train_deep_ensemble(X_train, y_train, use_layer_norm=True)
    m_boot = train_vanilla_bootstrap(X_train, y_train, use_layer_norm=True)
    m_prior = train_boot_prior(X_train, y_train, beta=0.5, use_layer_norm=True)
    m_dp_ln = train_dp_bnn(X_train, y_train, domain_range=(-4.5, 4.5), alpha=2.0, use_layer_norm=True)
    m_dp_no_ln = train_dp_bnn(X_train, y_train, domain_range=(-4.5, 4.5), alpha=2.0, use_layer_norm=False)
    
    methods = {
        "Deep Ensembles (LN)": m_deep,
        "Vanilla BootDQN (LN)": m_boot,
        "BootDQN + Priors (LN)": m_prior,
        "DP-BNN (Ours, LN)": m_dp_ln,
        "DP-BNN (No LayerNorm)": m_dp_no_ln
    }
    
    results = {}
    preds_dict = {}
    
    # In-Between Gap: x in [-0.8, 0.8] vs Clusters: |x| in [1.2, 2.8]
    gap_idx = np.where((X_test_np >= -0.8) & (X_test_np <= 0.8))[0]
    cluster_idx = np.where(((X_test_np >= -2.8) & (X_test_np <= -1.2)) | ((X_test_np >= 1.2) & (X_test_np <= 2.8)))[0]
    
    print(f"{'Method':<25} | {'Cluster Std':<12} | {'Gap Std':<12} | {'In-Between Ratio':<16}")
    print("-"*70)
    for name, models in methods.items():
        mu, sig, raw = predict_ensemble(models, X_test)
        preds_dict[name] = (mu, sig, raw)
        cluster_std = float(np.mean(sig[cluster_idx]))
        gap_std = float(np.mean(sig[gap_idx]))
        ratio = gap_std / max(cluster_std, 1e-6)
        results[name] = {"cluster_std": cluster_std, "gap_std": gap_std, "inbetween_ratio": ratio}
        print(f"{name:<25} | {cluster_std:<12.3f} | {gap_std:<12.3f} | {ratio:<16.2f}x")
    print("-"*70)
    
    # Generate 4-panel figure matching Foong et al.
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axs = axs.flatten()
    showcase = ["Deep Ensembles (LN)", "Vanilla BootDQN (LN)", "BootDQN + Priors (LN)", "DP-BNN (Ours, LN)"]
    titles = ["(a) Deep Ensembles (Collapsed Gap)", "(b) Vanilla BootDQN (Collapsed Gap)", "(c) BootDQN + Priors (Partial Dome)", "(d) DP-BNN (Vashishtha Formulation, Epistemic Dome)"]
    
    for idx, (k, title) in enumerate(zip(showcase, titles)):
        ax = axs[idx]
        mu, sig, raw = preds_dict[k]
        for h in range(min(5, raw.shape[0])):
            ax.plot(X_test_np, raw[h], color='lightblue', alpha=0.5, linestyle='--', lw=1.2)
        ax.plot(X_test_np, mu, color='blue', lw=2.2, label='Mean Prediction')
        ax.fill_between(X_test_np, mu - 1.96*sig, mu + 1.96*sig, color='mediumpurple', alpha=0.22, label='95% Credible Interval')
        ax.scatter(X_train_np, y_train_np, color='crimson', s=18, zorder=5, label='Observed Clusters')
        ax.axvspan(-1.0, 1.0, color='gray', alpha=0.10, label='Unobserved In-Between Void')
        ax.plot(X_test_np, y_test_true, color='darkgreen', linestyle=':', lw=1.2, label='Ground Truth Sin(x)')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xlim(-4.5, 4.5)
        if idx == 0:
            ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
            
    plt.suptitle("Benchmark 2: Foong et al. (ICML 2019) 'In-Between' Uncertainty Showdown", fontsize=13, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/benchmark_foong_inbetween.png", dpi=300)
    plt.close()
    print("Saved figure to results/benchmark_foong_inbetween.png")
    return results

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    res_osband = run_osband_benchmark()
    res_foong = run_foong_inbetween_benchmark()
    
    all_res = {
        "osband_neurips2018": res_osband,
        "foong_icml2019": res_foong
    }
    with open('results/standard_benchmarks_summary.json', 'w') as f:
        json.dump(all_res, f, indent=2)
    print("\nAll standard benchmarks successfully executed and saved to results/!")
