#!/usr/bin/env python3
"""
Benchmark: 1D Regression with DP-BNN (with LayerNorm) vs. Baselines
Target: Disjoint domain splits with sudden jump inside epistemic void gap.
f(x) = 2.0 * sin(x) + (1.5 if x > 0 else -1.5)
Training flanks: [-4.0, -1.5] and [1.5, 4.0]
Void gap: [-1.5, 1.5] with jump at x = 0.
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

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def true_function(x):
    return 2.0 * np.sin(x) + np.where(x > 0, 1.5, -1.5)

def generate_data(n_samples=200, noise_std=0.1):
    n_half = n_samples // 2
    X_left = np.random.uniform(-4.0, -1.5, n_half)
    X_right = np.random.uniform(1.5, 4.0, n_half)
    X_train_np = np.concatenate([X_left, X_right])
    y_train_np = true_function(X_train_np) + np.random.normal(0, noise_std, len(X_train_np))
    
    # Sort for cleaner plotting / handling
    sort_idx = np.argsort(X_train_np)
    X_train_np = X_train_np[sort_idx]
    y_train_np = y_train_np[sort_idx]
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
    
    # Test grid across [-5.0, 5.0]
    X_test_np = np.linspace(-5.0, 5.0, 400)
    y_test_np = true_function(X_test_np)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)
    
    return X_train, y_train, X_test, y_test, X_train_np, y_train_np, X_test_np, y_test_np

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
            
    def forward(self, x):
        return self.net(x)

# 1. Standard Deep Ensembles
def train_deep_ensemble(X, y, num_models=10, epochs=500, lr=0.005, use_layer_norm=True):
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
def train_vanilla_bootstrap(X, y, num_models=10, epochs=500, lr=0.005, use_layer_norm=True):
    models = []
    n = X.shape[0]
    criterion = nn.MSELoss(reduction='none')
    for _ in range(num_models):
        model = MLPRegression(hidden_dim=64, use_layer_norm=use_layer_norm)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        w = torch.from_numpy(np.random.poisson(1.0, n)).float().unsqueeze(1)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.mean(criterion(model(X), y) * w)
            loss.backward()
            optimizer.step()
        models.append(model)
    return models

# 3. BootDQN + Randomized Priors
class NetworkWithPrior(nn.Module):
    def __init__(self, beta=0.4, hidden_dim=64, use_layer_norm=True):
        super().__init__()
        self.model = MLPRegression(hidden_dim=hidden_dim, use_layer_norm=use_layer_norm)
        self.prior = MLPRegression(hidden_dim=hidden_dim, use_layer_norm=use_layer_norm)
        for p in self.prior.parameters():
            p.requires_grad = False
        self.beta = beta
        
    def forward(self, x):
        return self.model(x) + self.beta * self.prior(x)

def train_boot_prior(X, y, num_models=10, epochs=500, lr=0.005, beta=0.4, use_layer_norm=True):
    models = []
    n = X.shape[0]
    criterion = nn.MSELoss(reduction='none')
    for _ in range(num_models):
        model = NetworkWithPrior(beta=beta, hidden_dim=64, use_layer_norm=use_layer_norm)
        optimizer = optim.Adam(model.model.parameters(), lr=lr)
        w = torch.from_numpy(np.random.poisson(1.0, n)).float().unsqueeze(1)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.mean(criterion(model(X), y) * w)
            loss.backward()
            optimizer.step()
        models.append(model)
    return models

# 4. DP-BNN (Vashishtha Formulation)
def sample_base_measure_f0(size=1):
    X_p = torch.from_numpy(np.random.uniform(-5.0, 5.0, (size, 1))).float()
    y_p = torch.from_numpy(np.random.normal(0.0, 2.0, (size, 1))).float()
    return X_p, y_p

def train_dp_bnn(X, y, num_models=10, epochs=500, lr=0.005, alpha=2.0, K_trunc=100, use_layer_norm=True):
    models = []
    n = X.shape[0]
    alpha_n = alpha + n
    
    for _ in range(num_models):
        # Sethuraman stick-breaking weights from Beta(1, alpha + n)
        beta_dist = torch.distributions.Beta(1.0, alpha_n)
        v = beta_dist.sample((K_trunc - 1,))
        q = torch.zeros(K_trunc)
        rem = 1.0
        for k in range(K_trunc - 1):
            q[k] = v[k] * rem
            rem *= (1.0 - v[k])
        q[K_trunc - 1] = rem
        
        # Conjugate atom selection
        p_emp = n / alpha_n
        X_atoms, y_atoms = [], []
        for k in range(K_trunc):
            if np.random.rand() < p_emp:
                idx = np.random.randint(0, n)
                X_atoms.append(X[idx])
                y_atoms.append(y[idx])
            else:
                X_p, y_p = sample_base_measure_f0(1)
                X_atoms.append(X_p.squeeze(0))
                y_atoms.append(y_p.squeeze(0))
        X_atoms = torch.stack(X_atoms)
        y_atoms = torch.stack(y_atoms)
        
        model = MLPRegression(hidden_dim=64, use_layer_norm=use_layer_norm)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction='none')
        
        for _ in range(epochs):
            optimizer.zero_grad()
            raw_loss = criterion(model(X_atoms), y_atoms)
            loss = torch.sum(q.unsqueeze(1) * raw_loss)
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
    preds = np.array(preds) # [K, N_test]
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0)
    return mean, std, preds

# Metrics
def calculate_crps_gaussian(y_true, mu, sigma):
    sigma = np.clip(sigma, 1e-6, None)
    z = (y_true - mu) / sigma
    pdf = stats.norm.pdf(z)
    cdf = stats.norm.cdf(z)
    crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))

def calculate_empirical_coverage(y_true, mu, sigma, confidence=0.95):
    tail = (1.0 - confidence) / 2.0
    z_score = stats.norm.ppf(1.0 - tail)
    lower = mu - z_score * sigma
    upper = mu + z_score * sigma
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside) * 100.0)

def calculate_variance_gain(X_test_np, std):
    gap_idx = np.where((X_test_np >= -1.0) & (X_test_np <= 1.0))[0]
    flank_idx = np.where((X_test_np >= 2.0) | (X_test_np <= -2.0))[0]
    var_gap = np.mean(std[gap_idx] ** 2)
    var_flank = max(np.mean(std[flank_idx] ** 2), 1e-6)
    return float(var_gap / var_flank)

def calculate_indist_mse(X_test_np, y_test_np, mu):
    flank_idx = np.where(((X_test_np >= -4.0) & (X_test_np <= -1.5)) | ((X_test_np >= 1.5) & (X_test_np <= 4.0)))[0]
    return float(np.mean((y_test_np[flank_idx] - mu[flank_idx]) ** 2))

def run_experiment():
    print('Generating synthetic 1D regression dataset with central epistemic void...')
    X_train, y_train, X_test, y_test, X_train_np, y_train_np, X_test_np, y_test_np = generate_data(200, 0.1)
    
    print('1. Training Deep Ensembles (with LayerNorm)...')
    m_deep = train_deep_ensemble(X_train, y_train, use_layer_norm=True)
    
    print('2. Training Vanilla Bootstrap (with LayerNorm)...')
    m_boot = train_vanilla_bootstrap(X_train, y_train, use_layer_norm=True)
    
    print('3. Training BootDQN + Randomized Priors (with LayerNorm)...')
    m_prior = train_boot_prior(X_train, y_train, use_layer_norm=True)
    
    print('4. Training DP-BNN (Vashishtha Formulation, with LayerNorm)...')
    m_dp_ln = train_dp_bnn(X_train, y_train, use_layer_norm=True)
    
    print('5. Training DP-BNN (No LayerNorm Ablation)...')
    m_dp_no_ln = train_dp_bnn(X_train, y_train, use_layer_norm=False)
    
    methods = {
        'Deep Ensembles (LN)': m_deep,
        'Vanilla BootDQN (LN)': m_boot,
        'BootDQN + Priors (LN)': m_prior,
        'DP-BNN (Vashishtha, LN)': m_dp_ln,
        'DP-BNN (No LayerNorm)': m_dp_no_ln
    }
    
    results = {}
    predictions = {}
    
    print("\n" + "="*85)
    print(f"{'Method':<28} | {'95% Cov (%)':<12} | {'CRPS Score':<12} | {'OOD Var Gain':<14} | {'Flank MSE':<10}")
    print("="*85)
    
    for name, models in methods.items():
        mu, std, raws = predict_ensemble(models, X_test)
        predictions[name] = {'mu': mu, 'std': std, 'raws': raws}
        cov = calculate_empirical_coverage(y_test_np, mu, std)
        crps = calculate_crps_gaussian(y_test_np, mu, std)
        gain = calculate_variance_gain(X_test_np, std)
        mse = calculate_indist_mse(X_test_np, y_test_np, mu)
        results[name] = {
            'coverage_95': cov,
            'crps': crps,
            'variance_gain': gain,
            'flank_mse': mse,
            'peak_gap_std': float(np.max(std[np.where((X_test_np >= -1.5) & (X_test_np <= 1.5))[0]]))
        }
        print(f"{name:<28} | {cov:<12.2f} | {crps:<12.4f} | {gain:<14.2f}x | {mse:<10.4f}")
    print('='*85)
    
    os.makedirs('results', exist_ok=True)
    
    with open('results/dp_bnn_regression_layernorm_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Generate 4-Panel Benchmark Showdown Figure (Matching Slide 5)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    
    showdown_methods = [
        'Deep Ensembles (LN)',
        'Vanilla BootDQN (LN)',
        'BootDQN + Priors (LN)',
        'DP-BNN (Vashishtha, LN)'
    ]
    display_titles = [
        'Deep Ensembles (LayerNorm)',
        'Vanilla BootDQN (LayerNorm)',
        'BootDQN + Randomized Priors (LayerNorm)',
        'DP-BNN (Vashishtha Formulation, LayerNorm)'
    ]
    
    for idx, (m_key, title) in enumerate(zip(showdown_methods, display_titles)):
        ax = axs[idx]
        pred = predictions[m_key]
        mu = pred['mu']
        sig = pred['std']
        raw = pred['raws']
        
        for h in range(min(5, raw.shape[0])):
            ax.plot(X_test_np, raw[h], color='lightblue', alpha=0.5, linestyle='--', lw=1.2)
            
        ax.plot(X_test_np, mu, color='blue', lw=2.2, label='Ensemble Mean')
        ax.fill_between(X_test_np, mu - 1.96 * sig, mu + 1.96 * sig, color='blue', alpha=0.18, label='95% Credible Interval')
        ax.scatter(X_train_np, y_train_np, color='red', s=16, zorder=5, label='Observed Data', alpha=0.75)
        ax.axvspan(-1.5, 1.5, color='gray', alpha=0.08, label='Data Void Gap')
        ax.plot(X_test_np, y_test_np, color='darkgreen', linestyle=':', lw=1.2, alpha=0.7, label='True Function')
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(-5.5, 5.5)
        ax.set_xlim(-5.2, 5.2)
        if idx == 0:
            ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
            
    plt.suptitle('Uncertainty Landscape Under Disjoint Domain Splits (With Layer Normalization)', fontsize=14, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/uncertainty_landscape_epistemic_dome_layernorm.png', dpi=300)
    plt.close()
    print("\nSaved figure to results/uncertainty_landscape_epistemic_dome_layernorm.png")
    
    # Generate Side-by-Side LayerNorm Ablation Figure for DP-BNN
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for ax, m_key, title in [
        (ax1, 'DP-BNN (No LayerNorm)', 'DP-BNN Without LayerNorm (Standard MLP)'),
        (ax2, 'DP-BNN (Vashishtha, LN)', 'DP-BNN With LayerNorm (Stabilized)')
    ]:
        pred = predictions[m_key]
        mu = pred['mu']
        sig = pred['std']
        raw = pred['raws']
        
        for h in range(min(5, raw.shape[0])):
            ax.plot(X_test_np, raw[h], color='lightblue', alpha=0.5, linestyle='--', lw=1.2)
            
        ax.plot(X_test_np, mu, color='blue', lw=2.2, label='Ensemble Mean')
        ax.fill_between(X_test_np, mu - 1.96 * sig, mu + 1.96 * sig, color='mediumpurple', alpha=0.25, label='95% Credible Interval')
        ax.scatter(X_train_np, y_train_np, color='red', s=16, zorder=5, label='Observed Data', alpha=0.75)
        ax.axvspan(-1.5, 1.5, color='gray', alpha=0.08, label='Data Void Gap')
        ax.plot(X_test_np, y_test_np, color='darkgreen', linestyle=':', lw=1.2, alpha=0.7, label='True Function')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(-5.5, 5.5)
        ax.set_xlim(-5.2, 5.2)
        ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
        
    plt.suptitle('Impact of Layer Normalization on DP-BNN Epistemic Uncertainty & Manifold Regularization', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/dp_bnn_layernorm_ablation.png', dpi=300)
    plt.close()
    print('Saved figure to results/dp_bnn_layernorm_ablation.png')

if __name__ == '__main__':
    run_experiment()
