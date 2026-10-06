#!/usr/bin/env python3
"""
Benchmark: Dirichlet Process Physics-Informed Neural Networks (DP-PINNs) vs
Bayesian PINNs (B-PINNs) and Standard PINNs on Canonical Nonlinear Inverse Problem:
1D Nonlinear Poisson / Diffusion-Reaction System from Yang, Meng, Karniadakis (JCP 2021 Section 3.2.2 / 3.3.1):
0.01 * u_xx + k * tanh(u) = f(x),  x in [-0.7, 0.7]

OBJECTIVE:
Simultaneously reconstruct the unknown field u(x) and discover the unknown physical
reaction rate parameter k (True: k = 0.70) from sparse, highly noisy observations (10% noise).

ARCHITECTURES EVALUATED:
1. Standard PINN (Raissi et al. 2019, deterministic, lambda=1.0, No LayerNorm)
2. Standard PINN + LayerNorm (deterministic, lambda=1.0)
3. B-PINN (Yang et al. 2021, Stochastic Gradient Langevin Dynamics / SGLD, lambda=1.0)
4. DP-PINN (No LayerNorm, Stick-Breaking, strictly NO lambda)
5. DP-PINN (With LayerNorm, Stick-Breaking, strictly NO lambda, 5-draw ensemble)
"""

import os
import json
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Setup & Device
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)
DEVICE = torch.device("cpu")

RESULTS_DIR = "/Users/sumitvashishtha/Desktop/DP-BNNs/results"
ARTIFACTS_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Physical System Definition: Yang, Meng, Karniadakis (JCP 2021)
# 0.01 * u_xx + k * tanh(u) = f(x),  x in [-0.7, 0.7]
# ---------------------------------------------------------------------------
DIFFUSION = 0.01
K_TRUE = 0.70

def exact_u(x):
    """Manufactured exact solution: u(x) = sin^3(6x)"""
    return torch.sin(6.0 * x) ** 3

def exact_f(x):
    """Exact source term: f(x) = 0.01 * u''(x) + 0.7 * tanh(u(x))"""
    s = torch.sin(6.0 * x)
    c = torch.cos(6.0 * x)
    u_xx = 108.0 * (2.0 * s * (c ** 2) - (s ** 3))
    u = s ** 3
    return DIFFUSION * u_xx + K_TRUE * torch.tanh(u)

# Evaluation Grid (dense)
X_EVAL = torch.linspace(-0.7, 0.7, 250).unsqueeze(1).to(DEVICE)
U_TRUE = exact_u(X_EVAL)
F_EVAL = exact_f(X_EVAL)

# Sparse, noisy observations (10 sensors total: 2 boundary + 8 interior)
x_boundary = torch.tensor([[-0.7], [0.7]], device=DEVICE)
x_interior = torch.linspace(-0.55, 0.55, 8).unsqueeze(1).to(DEVICE)
X_OBS = torch.cat([x_boundary, x_interior], dim=0)

torch.manual_seed(42)
NOISE_STD = 0.10  # 10% relative Gaussian noise
Y_OBS = exact_u(X_OBS) + NOISE_STD * torch.randn_like(X_OBS)
N_DATA = len(X_OBS)

# ---------------------------------------------------------------------------
# Neural Architectures
# ---------------------------------------------------------------------------
class InversePINN_MLP(nn.Module):
    """Multi-Layer Perceptron supporting optional Layer Normalization and unknown parameter k."""
    def __init__(self, in_dim=1, out_dim=1, hidden_dim=64, num_layers=3, use_layer_norm=True, init_k=0.0):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        layers = []
        curr_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.k_param = nn.Parameter(torch.tensor([init_k]))
        
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# Dirichlet Process Stick-Breaking Sampler (STRICTLY NO LAMBDA)
# ---------------------------------------------------------------------------
def sample_stick_breaking(alpha: float, n_data: int, K_t: int, device=DEVICE):
    """Sample Sethuraman stick-breaking weights q_k ~ GEM(alpha + N)."""
    V = torch.distributions.Beta(1.0, alpha + n_data).sample((K_t,)).to(device)
    remaining = torch.cumprod(1.0 - V, dim=0)
    q = torch.zeros(K_t, device=device)
    q[0] = V[0]
    q[1:] = V[1:] * remaining[:-1]
    return q / q.sum()

# ---------------------------------------------------------------------------
# Benchmark Execution
# ---------------------------------------------------------------------------
def run_inverse_benchmark(epochs=3000, n_col=150):
    print("=" * 80)
    print("CHALLENGING INVERSE BENCHMARK: DP-PINN VS B-PINN VS STANDARD PINN")
    print(f"Physical Equation: 0.01 * u_xx + k * tanh(u) = f(x),  x in [-0.7, 0.7]")
    print(f"Benchmark Source: Yang, Meng, Karniadakis (J. Comput. Phys. 2021 Section 3.2.2 / 3.3.1)")
    print(f"True Parameter: k* = {K_TRUE:.4f} | Observations: {N_DATA} points | Noise Scale: {NOISE_STD*100:.1f}%")
    print("=" * 80)
    
    results = {}
    predictions = {}
    k_distributions = {}
    
    # -----------------------------------------------------------------------
    # 1. Standard PINN (Raissi et al. 2019, No LayerNorm, lambda=1.0)
    # -----------------------------------------------------------------------
    print("\n[1/5] Training Standard PINN (Raissi 2019, No LayerNorm, lambda=1.0)...")
    torch.manual_seed(42)
    m_std = InversePINN_MLP(use_layer_norm=False, init_k=0.0).to(DEVICE)
    opt_std = torch.optim.Adam(m_std.parameters(), lr=2e-3)
    sched_std = torch.optim.lr_scheduler.CosineAnnealingLR(opt_std, T_max=epochs, eta_min=1e-5)
    
    t0 = time.time()
    for ep in range(epochs):
        opt_std.zero_grad()
        pred_obs = m_std(X_OBS)
        loss_data = torch.mean((pred_obs - Y_OBS) ** 2)
        
        x_col = torch.rand(n_col, 1, device=DEVICE) * 1.4 - 0.7
        x_col.requires_grad_(True)
        u_c = m_std(x_col)
        u_x = torch.autograd.grad(u_c, x_col, grad_outputs=torch.ones_like(u_c), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f_c = exact_f(x_col)
        res = DIFFUSION * u_xx + m_std.k_param * torch.tanh(u_c) - f_c
        loss_pde = torch.mean(res ** 2)
        
        loss = loss_data + 1.0 * loss_pde  # lambda = 1.0
        loss.backward()
        opt_std.step()
        sched_std.step()
        
    t_std = time.time() - t0
    with torch.no_grad():
        pred_std = m_std(X_EVAL)
        rel_l2_std = (torch.norm(pred_std - U_TRUE) / torch.norm(U_TRUE)).item()
        l_inf_std = torch.max(torch.abs(pred_std - U_TRUE)).item()
        mse_std = torch.mean((pred_std - U_TRUE) ** 2).item()
        k_val_std = m_std.k_param.item()
        err_k_std = abs(k_val_std - K_TRUE)
        
    print(f"   -> Standard PINN: Inferred k = {k_val_std:.4f} (Error: {err_k_std:.4f}) | Field Rel L2 = {rel_l2_std*100:.2f}% | Time: {t_std:.2f}s")
    predictions["Standard PINN (Raissi 2019)"] = {"mean": pred_std.cpu().numpy().flatten(), "std": np.zeros(len(X_EVAL))}
    k_distributions["Standard PINN (Raissi 2019)"] = {"mean": k_val_std, "std": 0.0, "samples": [k_val_std]}
    results["Standard PINN (Raissi 2019)"] = {"k_mean": k_val_std, "k_std": 0.0, "k_err": err_k_std, "rel_l2": rel_l2_std, "l_inf": l_inf_std, "mse": mse_std, "time": t_std}

    # -----------------------------------------------------------------------
    # 2. Standard PINN + LayerNorm (lambda=1.0)
    # -----------------------------------------------------------------------
    print("\n[2/5] Training Standard PINN + LayerNorm (lambda=1.0)...")
    torch.manual_seed(42)
    m_ln = InversePINN_MLP(use_layer_norm=True, init_k=0.0).to(DEVICE)
    opt_ln = torch.optim.Adam(m_ln.parameters(), lr=2e-3)
    sched_ln = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ln, T_max=epochs, eta_min=1e-5)
    
    t0 = time.time()
    for ep in range(epochs):
        opt_ln.zero_grad()
        pred_obs = m_ln(X_OBS)
        loss_data = torch.mean((pred_obs - Y_OBS) ** 2)
        
        x_col = torch.rand(n_col, 1, device=DEVICE) * 1.4 - 0.7
        x_col.requires_grad_(True)
        u_c = m_ln(x_col)
        u_x = torch.autograd.grad(u_c, x_col, grad_outputs=torch.ones_like(u_c), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f_c = exact_f(x_col)
        res = DIFFUSION * u_xx + m_ln.k_param * torch.tanh(u_c) - f_c
        loss_pde = torch.mean(res ** 2)
        
        loss = loss_data + 1.0 * loss_pde
        loss.backward()
        opt_ln.step()
        sched_ln.step()
        
    t_ln = time.time() - t0
    with torch.no_grad():
        pred_ln = m_ln(X_EVAL)
        rel_l2_ln = (torch.norm(pred_ln - U_TRUE) / torch.norm(U_TRUE)).item()
        l_inf_ln = torch.max(torch.abs(pred_ln - U_TRUE)).item()
        mse_ln = torch.mean((pred_ln - U_TRUE) ** 2).item()
        k_val_ln = m_ln.k_param.item()
        err_k_ln = abs(k_val_ln - K_TRUE)
        
    print(f"   -> Standard PINN + LN: Inferred k = {k_val_ln:.4f} (Error: {err_k_ln:.4f}) | Field Rel L2 = {rel_l2_ln*100:.2f}% | Time: {t_std:.2f}s")
    predictions["Standard PINN + LayerNorm"] = {"mean": pred_ln.cpu().numpy().flatten(), "std": np.zeros(len(X_EVAL))}
    k_distributions["Standard PINN + LayerNorm"] = {"mean": k_val_ln, "std": 0.0, "samples": [k_val_ln]}
    results["Standard PINN + LayerNorm"] = {"k_mean": k_val_ln, "k_std": 0.0, "k_err": err_k_ln, "rel_l2": rel_l2_ln, "l_inf": l_inf_ln, "mse": mse_ln, "time": t_ln}

    # -----------------------------------------------------------------------
    # 3. B-PINN (Yang et al. 2021, SGLD with Gaussian Prior, lambda=1.0)
    # -----------------------------------------------------------------------
    print("\n[3/5] Training B-PINN (Yang et al. 2021, SGLD with Gaussian Prior)...")
    torch.manual_seed(42)
    m_b = InversePINN_MLP(use_layer_norm=False, init_k=0.0).to(DEVICE)
    opt_b = torch.optim.Adam(m_b.parameters(), lr=2e-3)
    
    t0 = time.time()
    # Phase 1: Warm-up optimization (1500 epochs)
    for ep in range(1500):
        opt_b.zero_grad()
        pred_obs = m_b(X_OBS)
        loss_data = torch.mean((pred_obs - Y_OBS) ** 2)
        x_col = torch.rand(n_col, 1, device=DEVICE) * 1.4 - 0.7
        x_col.requires_grad_(True)
        u_c = m_b(x_col)
        u_x = torch.autograd.grad(u_c, x_col, grad_outputs=torch.ones_like(u_c), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f_c = exact_f(x_col)
        res = DIFFUSION * u_xx + m_b.k_param * torch.tanh(u_c) - f_c
        loss = loss_data + 1.0 * torch.mean(res ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_b.parameters(), max_norm=1.0)
        opt_b.step()
        
    # Phase 2: SGLD posterior sampling (1200 epochs)
    b_k_samples = []
    b_u_samples = []
    sgld_lr = 1e-4
    for ep in range(1200):
        opt_b.zero_grad()
        pred_obs = m_b(X_OBS)
        loss_data = torch.mean((pred_obs - Y_OBS) ** 2)
        
        x_col = torch.rand(n_col, 1, device=DEVICE) * 1.4 - 0.7
        x_col.requires_grad_(True)
        u_c = m_b(x_col)
        u_x = torch.autograd.grad(u_c, x_col, grad_outputs=torch.ones_like(u_c), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f_c = exact_f(x_col)
        res = DIFFUSION * u_xx + m_b.k_param * torch.tanh(u_c) - f_c
        loss_pde = torch.mean(res ** 2)
        prior_reg = 0.01 * sum(torch.sum(p ** 2) for p in m_b.parameters())
        
        loss = loss_data + 1.0 * loss_pde + prior_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_b.parameters(), max_norm=1.0)
        
        with torch.no_grad():
            for p in m_b.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p) * np.sqrt(2.0 * sgld_lr * 1e-3)
                    p.data -= sgld_lr * p.grad + noise
                    
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                b_k_samples.append(m_b.k_param.item())
                b_u_samples.append(m_b(X_EVAL).cpu().numpy().flatten())
                
    t_b = time.time() - t0
    b_k_arr = np.array(b_k_samples)
    b_k_mean = float(np.mean(b_k_arr))
    b_k_std = float(np.std(b_k_arr))
    err_k_b = abs(b_k_mean - K_TRUE)
    
    b_u_stack = np.stack(b_u_samples, axis=0)
    b_u_mean = np.mean(b_u_stack, axis=0)
    b_u_std = np.std(b_u_stack, axis=0)
    u_true_np = U_TRUE.cpu().numpy().flatten()
    rel_l2_b = float(np.linalg.norm(b_u_mean - u_true_np) / np.linalg.norm(u_true_np))
    l_inf_b = float(np.max(np.abs(b_u_mean - u_true_np)))
    mse_b = float(np.mean((b_u_mean - u_true_np) ** 2))
    
    print(f"   -> B-PINN (SGLD): Inferred k = {b_k_mean:.4f} +/- {b_k_std:.4f} (Error: {err_k_b:.4f}) | Field Rel L2 = {rel_l2_b*100:.2f}% | Time: {t_b:.2f}s")
    predictions["B-PINN (Yang et al. 2021)"] = {"mean": b_u_mean, "std": b_u_std}
    k_distributions["B-PINN (Yang et al. 2021)"] = {"mean": b_k_mean, "std": b_k_std, "samples": b_k_samples}
    results["B-PINN (Yang et al. 2021)"] = {"k_mean": b_k_mean, "k_std": b_k_std, "k_err": err_k_b, "rel_l2": rel_l2_b, "l_inf": l_inf_b, "mse": mse_b, "time": t_b}

    # -----------------------------------------------------------------------
    # 4. DP-PINN (No LayerNorm, Stick-Breaking, STRICTLY NO LAMBDA)
    # -----------------------------------------------------------------------
    print("\n[4/5] Training DP-PINN (No LayerNorm, Stick-Breaking, Strictly NO Lambda)...")
    alpha = 25.0
    K_phys = 140
    K_total = N_DATA + K_phys
    
    dp_noln_k_samples = []
    dp_noln_u_samples = []
    t0 = time.time()
    for m_idx in range(5):
        torch.manual_seed(150 + m_idx)
        m_dp_noln = InversePINN_MLP(use_layer_norm=False, init_k=0.0).to(DEVICE)
        opt_dp_noln = torch.optim.Adam(m_dp_noln.parameters(), lr=2e-3)
        sched_dp_noln = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dp_noln, T_max=epochs, eta_min=1e-5)
        
        for ep in range(epochs):
            opt_dp_noln.zero_grad()
            q = sample_stick_breaking(alpha, N_DATA, K_total)
            q_data = q[:N_DATA]
            q_phys = q[N_DATA:]
            
            pred_obs = m_dp_noln(X_OBS)
            loss_data = torch.sum(q_data.unsqueeze(1) * (pred_obs - Y_OBS) ** 2)
            
            x_col = torch.rand(K_phys, 1, device=DEVICE) * 1.4 - 0.7
            x_col.requires_grad_(True)
            u_c = m_dp_noln(x_col)
            u_x = torch.autograd.grad(u_c, x_col, grad_outputs=torch.ones_like(u_c), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            f_c = exact_f(x_col)
            res = DIFFUSION * u_xx + m_dp_noln.k_param * torch.tanh(u_c) - f_c
            loss_pde = torch.sum(q_phys.unsqueeze(1) * (res ** 2))
            
            loss = loss_data + loss_pde  # STRICTLY NO LAMBDA!
            loss.backward()
            opt_dp_noln.step()
            sched_dp_noln.step()
            
        with torch.no_grad():
            dp_noln_k_samples.append(m_dp_noln.k_param.item())
            dp_noln_u_samples.append(m_dp_noln(X_EVAL).cpu().numpy().flatten())
            
    t_dp_noln = time.time() - t0
    dp_noln_k_arr = np.array(dp_noln_k_samples)
    dp_noln_k_mean = float(np.mean(dp_noln_k_arr))
    dp_noln_k_std = float(np.std(dp_noln_k_arr))
    err_k_dp_noln = abs(dp_noln_k_mean - K_TRUE)
    
    dp_noln_u_stack = np.stack(dp_noln_u_samples, axis=0)
    dp_noln_u_mean = np.mean(dp_noln_u_stack, axis=0)
    dp_noln_u_std = np.std(dp_noln_u_stack, axis=0)
    rel_l2_dp_noln = float(np.linalg.norm(dp_noln_u_mean - u_true_np) / np.linalg.norm(u_true_np))
    l_inf_dp_noln = float(np.max(np.abs(dp_noln_u_mean - u_true_np)))
    mse_dp_noln = float(np.mean((dp_noln_u_mean - u_true_np) ** 2))
    
    print(f"   -> DP-PINN (No LN): Inferred k = {dp_noln_k_mean:.4f} +/- {dp_noln_k_std:.4f} (Error: {err_k_dp_noln:.4f}) | Field Rel L2 = {rel_l2_dp_noln*100:.2f}% | Time: {t_dp_noln:.2f}s")
    predictions["DP-PINN (No LayerNorm)"] = {"mean": dp_noln_u_mean, "std": dp_noln_u_std}
    k_distributions["DP-PINN (No LayerNorm)"] = {"mean": dp_noln_k_mean, "std": dp_noln_k_std, "samples": dp_noln_k_samples}
    results["DP-PINN (No LayerNorm)"] = {"k_mean": dp_noln_k_mean, "k_std": dp_noln_k_std, "k_err": err_k_dp_noln, "rel_l2": rel_l2_dp_noln, "l_inf": l_inf_dp_noln, "mse": mse_dp_noln, "time": t_dp_noln}

    # -----------------------------------------------------------------------
    # 5. DP-PINN (With LayerNorm, Stick-Breaking, STRICTLY NO LAMBDA, 5 Draws)
    # -----------------------------------------------------------------------
    print("\n[5/5] Training DP-PINN (With LayerNorm, Stick-Breaking, Strictly NO Lambda, 5 Draws)...")
    dp_ln_k_samples = []
    dp_ln_u_samples = []
    t0 = time.time()
    for m_idx in range(5):
        torch.manual_seed(200 + m_idx)
        m_dp_ln = InversePINN_MLP(use_layer_norm=True, init_k=0.0).to(DEVICE)
        opt_dp_ln = torch.optim.Adam(m_dp_ln.parameters(), lr=2e-3)
        sched_dp_ln = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dp_ln, T_max=epochs, eta_min=1e-5)
        
        for ep in range(epochs):
            opt_dp_ln.zero_grad()
            q = sample_stick_breaking(alpha, N_DATA, K_total)
            q_data = q[:N_DATA]
            q_phys = q[N_DATA:]
            
            pred_obs = m_dp_ln(X_OBS)
            loss_data = torch.sum(q_data.unsqueeze(1) * (pred_obs - Y_OBS) ** 2)
            
            x_col = torch.rand(K_phys, 1, device=DEVICE) * 1.4 - 0.7
            x_col.requires_grad_(True)
            u_c = m_dp_ln(x_col)
            u_x = torch.autograd.grad(u_c, x_col, grad_outputs=torch.ones_like(u_c), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            f_c = exact_f(x_col)
            res = DIFFUSION * u_xx + m_dp_ln.k_param * torch.tanh(u_c) - f_c
            loss_pde = torch.sum(q_phys.unsqueeze(1) * (res ** 2))
            
            loss = loss_data + loss_pde  # STRICTLY NO LAMBDA!
            loss.backward()
            opt_dp_ln.step()
            sched_dp_ln.step()
            
        with torch.no_grad():
            dp_ln_k_samples.append(m_dp_ln.k_param.item())
            dp_ln_u_samples.append(m_dp_ln(X_EVAL).cpu().numpy().flatten())
            
    t_dp_ln = time.time() - t0
    dp_ln_k_arr = np.array(dp_ln_k_samples)
    dp_ln_k_mean = float(np.mean(dp_ln_k_arr))
    dp_ln_k_std = float(np.std(dp_ln_k_arr))
    err_k_dp_ln = abs(dp_ln_k_mean - K_TRUE)
    
    dp_ln_u_stack = np.stack(dp_ln_u_samples, axis=0)
    dp_ln_u_mean = np.mean(dp_ln_u_stack, axis=0)
    dp_ln_u_std = np.std(dp_ln_u_stack, axis=0)
    rel_l2_dp_ln = float(np.linalg.norm(dp_ln_u_mean - u_true_np) / np.linalg.norm(u_true_np))
    l_inf_dp_ln = float(np.max(np.abs(dp_ln_u_mean - u_true_np)))
    mse_dp_ln = float(np.mean((dp_ln_u_mean - u_true_np) ** 2))
    
    print(f"   -> DP-PINN (With LN): Inferred k = {dp_ln_k_mean:.4f} +/- {dp_ln_k_std:.4f} (Error: {err_k_dp_ln:.4f}) | Field Rel L2 = {rel_l2_dp_ln*100:.2f}% | Time: {t_dp_ln:.2f}s")
    predictions["DP-PINN (With LayerNorm)"] = {"mean": dp_ln_u_mean, "std": dp_ln_u_std}
    k_distributions["DP-PINN (With LayerNorm)"] = {"mean": dp_ln_k_mean, "std": dp_ln_k_std, "samples": dp_ln_k_samples}
    results["DP-PINN (With LayerNorm)"] = {"k_mean": dp_ln_k_mean, "k_std": dp_ln_k_std, "k_err": err_k_dp_ln, "rel_l2": rel_l2_dp_ln, "l_inf": l_inf_dp_ln, "mse": mse_dp_ln, "time": t_dp_ln}

    # -----------------------------------------------------------------------
    # Comparative Summary Matrix
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL COMPARISON MATRIX: NONLINEAR INVERSE PROBLEM (True k = 0.7000, 10% Noise)")
    print("=" * 80)
    print(f"{'Model Architecture':<30} | {'Inferred k (Mean +/- Std)':<24} | {'k Error':<10} | {'Field Rel L2':<12}")
    print("-" * 80)
    for name, res in results.items():
        if res["k_std"] > 0:
            k_str = f"{res['k_mean']:.4f} +/- {res['k_std']:.4f}"
        else:
            k_str = f"{res['k_mean']:.4f} (point)"
        print(f"{name:<30} | {k_str:<24} | {res['k_err']:>8.4f}   | {res['rel_l2']*100:>9.2f}%")
    print("=" * 80)
    
    # -----------------------------------------------------------------------
    # Save Results JSON
    # -----------------------------------------------------------------------
    summary_path = os.path.join(RESULTS_DIR, "inverse_bpinns_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "problem": {
                "name": "1D Nonlinear Poisson Inverse Problem (Yang et al. JCP 2021)",
                "equation": "0.01 * u_xx + k * tanh(u) = f(x)",
                "k_true": K_TRUE,
                "n_observations": N_DATA,
                "noise_scale": NOISE_STD,
                "domain": "[-0.7, 0.7]"
            },
            "results": results
        }, f, indent=2)
    shutil.copy(summary_path, os.path.join(ARTIFACTS_DIR, "inverse_bpinns_summary.json"))
    print(f"Saved summary JSON to {summary_path}")

    # -----------------------------------------------------------------------
    # Generate Publication Figure
    # -----------------------------------------------------------------------
    x_np = X_EVAL.cpu().numpy().flatten()
    x_obs_np = X_OBS.cpu().numpy().flatten()
    y_obs_np = Y_OBS.cpu().numpy().flatten()
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    
    # Palette
    c_true = '#111111'
    c_std = '#D62728'    # Red
    c_ln = '#FF7F0E'     # Orange
    c_bpinn = '#2CA02C'  # Green (B-PINN)
    c_dp_noln = '#8C564B'# Brown
    c_dp_ln = '#1F77B4'  # Deep Blue (DP-PINN)
    
    # Subplot A: Reconstructed State Field u(x)
    ax = axs[0, 0]
    ax.plot(x_np, u_true_np, 'k--', linewidth=2.5, label="Exact Analytical $u(x) = \\sin^3(6x)$", zorder=5)
    ax.scatter(x_obs_np, y_obs_np, color='red', marker='x', s=90, linewidth=2.5, zorder=6,
               label=f"Noisy Sensors ($N={N_DATA}$, $\\sigma={NOISE_STD*100:.0f}\\%$)")
    
    ax.plot(x_np, predictions["Standard PINN (Raissi 2019)"]["mean"], color=c_std, linewidth=1.8, linestyle=':',
            label=f"Standard PINN [Rel $L_2$={results['Standard PINN (Raissi 2019)']['rel_l2']*100:.1f}%]")
    
    bp_m = predictions["B-PINN (Yang et al. 2021)"]["mean"]
    bp_s = predictions["B-PINN (Yang et al. 2021)"]["std"]
    ax.plot(x_np, bp_m, color=c_bpinn, linewidth=2.0, linestyle='-.',
            label=f"B-PINN (Yang 2021) [Rel $L_2$={results['B-PINN (Yang et al. 2021)']['rel_l2']*100:.1f}%]")
    ax.fill_between(x_np, bp_m - 1.96 * bp_s, bp_m + 1.96 * bp_s, color=c_bpinn, alpha=0.15, label="B-PINN 95% Credible Interval")
    
    dp_m = predictions["DP-PINN (With LayerNorm)"]["mean"]
    dp_s = predictions["DP-PINN (With LayerNorm)"]["std"]
    ax.plot(x_np, dp_m, color=c_dp_ln, linewidth=2.8,
            label=f"DP-PINN (With LN, NO $\\lambda$) [Rel $L_2$={results['DP-PINN (With LayerNorm)']['rel_l2']*100:.1f}%]")
    ax.fill_between(x_np, dp_m - 1.96 * dp_s, dp_m + 1.96 * dp_s, color=c_dp_ln, alpha=0.25, label="DP-PINN 95% Credible Interval")
    
    ax.set_title("A. Latent Field Reconstruction $u(x)$ from Noisy Sparse Sensors", fontsize=13, fontweight='bold')
    ax.set_xlabel("Spatial coordinate $x$", fontsize=11)
    ax.set_ylabel("Field $u(x)$", fontsize=11)
    ax.set_ylim(-1.4, 1.4)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot B: Posterior Distribution of Parameter k
    ax = axs[0, 1]
    ax.axvline(K_TRUE, color='black', linestyle='--', linewidth=2.5, label=f"True Parameter $k^* = {K_TRUE:.4f}$", zorder=6)
    ax.axvline(results["Standard PINN (Raissi 2019)"]["k_mean"], color=c_std, linestyle=':', linewidth=2.0,
               label=f"Standard PINN: $\\hat{{k}} = {results['Standard PINN (Raissi 2019)']['k_mean']:.4f}$")
    ax.axvline(results["Standard PINN + LayerNorm"]["k_mean"], color=c_ln, linestyle='-.', linewidth=2.0,
               label=f"Standard PINN + LN: $\\hat{{k}} = {results['Standard PINN + LayerNorm']['k_mean']:.4f}$")
    
    # Plot B-PINN Gaussian posterior density
    k_grid = np.linspace(0.60, 0.80, 300)
    p_bpinn = norm.pdf(k_grid, results["B-PINN (Yang et al. 2021)"]["k_mean"], results["B-PINN (Yang et al. 2021)"]["k_std"])
    ax.plot(k_grid, p_bpinn, color=c_bpinn, linewidth=2.2,
            label=f"B-PINN Posterior: ${results['B-PINN (Yang et al. 2021)']['k_mean']:.4f} \\pm {results['B-PINN (Yang et al. 2021)']['k_std']:.4f}$")
    ax.fill_between(k_grid, 0, p_bpinn, color=c_bpinn, alpha=0.15)
    
    # Plot DP-PINN Gaussian posterior density
    p_dp = norm.pdf(k_grid, results["DP-PINN (With LayerNorm)"]["k_mean"], results["DP-PINN (With LayerNorm)"]["k_std"])
    ax.plot(k_grid, p_dp, color=c_dp_ln, linewidth=2.8,
            label=f"DP-PINN Posterior: ${results['DP-PINN (With LayerNorm)']['k_mean']:.4f} \\pm {results['DP-PINN (With LayerNorm)']['k_std']:.4f}$")
    ax.fill_between(k_grid, 0, p_dp, color=c_dp_ln, alpha=0.25)
    
    ax.set_title("B. Inferred Reaction Rate Parameter Posterior $p(k \\mid \\mathcal{D})$", fontsize=13, fontweight='bold')
    ax.set_xlabel("Parameter $k$", fontsize=11)
    ax.set_ylabel("Posterior Probability Density", fontsize=11)
    ax.set_xlim(0.60, 0.80)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot C: Epistemic Uncertainty Quantification Comparison
    ax = axs[1, 0]
    ax.plot(x_np, bp_s, color=c_bpinn, linewidth=2.2, linestyle='-.', label="B-PINN (Yang 2021) Posterior $\\sigma_u(x)$")
    ax.fill_between(x_np, 0, bp_s, color=c_bpinn, alpha=0.15)
    ax.plot(x_np, dp_s, color=c_dp_ln, linewidth=2.8, label="DP-PINN (With LayerNorm) Posterior $\\sigma_u(x)$")
    ax.fill_between(x_np, 0, dp_s, color=c_dp_ln, alpha=0.25)
    
    # Sensor locations on x-axis
    ax.scatter(x_obs_np, np.zeros_like(x_obs_np), color='red', marker='^', s=70, zorder=6, label="Sensor Locations")
    ax.set_title("C. Pointwise Epistemic Uncertainty Profile $\\sigma_u(x)$", fontsize=13, fontweight='bold')
    ax.set_xlabel("Spatial coordinate $x$", fontsize=11)
    ax.set_ylabel("Posterior Standard Deviation $\\sigma_u(x)$", fontsize=11)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot D: Error Metrics Comparison Bar Chart
    ax = axs[1, 1]
    model_labels = ["Standard\nPINN", "Standard\n+ LN", "B-PINN\n(SGLD)", "DP-PINN\n(No LN)", "DP-PINN\n(With LN)"]
    k_errors = [results[m]["k_err"] * 100 for m in results.keys()]
    field_errors = [results[m]["rel_l2"] * 100 for m in results.keys()]
    
    x_indices = np.arange(len(model_labels))
    width = 0.35
    
    rects1 = ax.bar(x_indices - width/2, k_errors, width, label="Parameter Error $|k - k^*| \\times 100$", color='#E24A33', edgecolor='black', linewidth=1.1)
    rects2 = ax.bar(x_indices + width/2, field_errors, width, label="Field Rel $L_2$ Error (%)", color='#348ABD', edgecolor='black', linewidth=1.1)
    
    for r in rects1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 0.2, f"{h:.2f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 0.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    ax.set_title("D. Inverse Problem Benchmark Error Metrics (Lower is Better)", fontsize=13, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylabel("Error Metric Value", fontsize=11)
    ax.set_ylim(0, max(max(k_errors), max(field_errors)) + 3.0)
    ax.legend(loc="upper right", fontsize=9.0, framealpha=0.95)
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    
    plt.suptitle("Dirichlet Process PINNs (DP-PINNs) vs B-PINNs vs Standard PINNs on Nonlinear Inverse Problem\n"
                 "Benchmark: 1D Nonlinear Poisson $0.01 u_{xx} + k \\tanh(u) = f(x)$ with 10% Sensor Noise [Yang et al. JCP 2021]",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "inverse_bpinns_showdown.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    shutil.copy(fig_path, os.path.join(ARTIFACTS_DIR, "inverse_bpinns_showdown.png"))
    print(f"Saved publication figure to {fig_path}")

if __name__ == "__main__":
    run_inverse_benchmark(epochs=3000, n_col=150)
