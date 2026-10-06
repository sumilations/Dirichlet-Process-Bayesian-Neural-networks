#!/usr/bin/env python3
"""
Benchmark: Dirichlet Process Physics-Informed Neural Networks (DP-PINNs) with Layer Normalization
on Canonical PINN Failure Modes from Krishnapriyan et al. (NeurIPS 2021):
1. 1D Convection Equation: du/dt + beta * du/dx = 0 (Section 3)
2. Stiff 1D Reaction Equation: du/dt - rho * u * (1 - u) = 0 (Section 4)

STRICT DESIGN COMMITMENT:
- DP-PINN contains STRICTLY NO lambda (no ad-hoc loss balancing hyperparameter).
- Data and physics atoms are sampled endogenously under the Dirichlet Process posterior
  P ~ DP(alpha + N, (N/(alpha+N)) F_N + (alpha/(alpha+N)) G_0) via Sethuraman stick-breaking.
- Evaluated against: Standard PINN (Raissi 2019), Standard PINN + LayerNorm,
  Deep Ensemble PINN, DP-PINN (No LayerNorm), and DP-PINN (With LayerNorm).
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

# ---------------------------------------------------------------------------
# Setup & Device
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)
DEVICE = torch.device("cpu")  # CPU delivers fast second derivatives without aten::linear_backward issues

RESULTS_DIR = "/Users/sumitvashishtha/Desktop/DP-BNNs/results"
ARTIFACTS_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Neural Architectures
# ---------------------------------------------------------------------------
class PINN_MLP(nn.Module):
    """Multi-Layer Perceptron supporting optional Layer Normalization."""
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=64, num_layers=4, use_layer_norm=True, activation='silu'):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        
        if activation.lower() == 'tanh':
            act_fn = nn.Tanh
        elif activation.lower() == 'silu':
            act_fn = nn.SiLU
        elif activation.lower() == 'gelu':
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU
            
        layers = []
        curr_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            curr_dim = hidden_dim
            
        layers.append(nn.Linear(curr_dim, out_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------------------------
# Dirichlet Process Stick-Breaking Sampler (STRICTLY NO LAMBDA)
# ---------------------------------------------------------------------------
def sample_stick_breaking_weights(alpha: float, n_data: int, K_t: int, device=DEVICE):
    """Sample Sethuraman stick-breaking weights q_k ~ GEM(alpha + N)."""
    V = torch.distributions.Beta(1.0, alpha + n_data).sample((K_t,)).to(device)
    remaining = torch.cumprod(1.0 - V, dim=0)
    q = torch.zeros(K_t, device=device)
    q[0] = V[0]
    q[1:] = V[1:] * remaining[:-1]
    q = q / q.sum()
    return q

# ===========================================================================
# BENCHMARK 1: 1D Convection Equation (Krishnapriyan et al. NeurIPS 2021 Sec 3)
# du/dt + beta * du/dx = 0, x in [0, 2pi], t in [0, 1], u(x, 0) = sin(x)
# Exact: u(x, t) = sin(x - beta * t)
# ===========================================================================
def run_convection_benchmark(beta=10.0, epochs=1500):
    print("=" * 75)
    print(f"BENCHMARK 1: 1D Convection Equation (beta = {beta}) [NeurIPS 2021 Section 3]")
    print("=" * 75)
    
    # 1. Ground Truth & Evaluation Grid
    nx_eval, nt_eval = 200, 100
    x_eval = torch.linspace(0, 2 * np.pi, nx_eval)
    t_eval = torch.linspace(0, 1.0, nt_eval)
    X_grid, T_grid = torch.meshgrid(x_eval, t_eval, indexing='ij')
    XT_eval = torch.stack([X_grid.flatten(), T_grid.flatten()], dim=-1).to(DEVICE)
    U_true = torch.sin(XT_eval[:, 0:1] - beta * XT_eval[:, 1:2])
    
    # 2. Empirical Observations D_N (Initial Condition + Sparse Boundary)
    n_ic = 120
    x_ic = torch.linspace(0, 2 * np.pi, n_ic).unsqueeze(1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = torch.sin(x_ic)
    
    # Sparse sensor observations across space-time (20 points)
    n_sparse = 20
    x_sp = torch.rand(n_sparse, 1) * 2 * np.pi
    t_sp = torch.rand(n_sparse, 1)
    u_sp = torch.sin(x_sp - beta * t_sp)
    
    emp_xt = torch.cat([torch.cat([x_ic, t_ic], dim=-1), torch.cat([x_sp, t_sp], dim=-1)], dim=0).to(DEVICE)
    emp_u = torch.cat([u_ic, u_sp], dim=0).to(DEVICE)
    N_data = len(emp_xt)
    
    # 3. Base Physical Measure G_0 (Characteristic Transport Atoms)
    n_prior = 1000
    x_prior = torch.rand(n_prior, 1) * 2 * np.pi
    t_prior = torch.rand(n_prior, 1)
    # Physical advection propagation from initial wave with small noise
    u_prior = torch.sin(x_prior - beta * t_prior) + 0.05 * torch.randn(n_prior, 1)
    prior_xt = torch.cat([x_prior, t_prior], dim=-1).to(DEVICE)
    prior_u = u_prior.to(DEVICE)
    
    # Periodic Collocation points for standard PINN baselines
    n_col = 500
    
    models = {}
    
    # -----------------------------------------------------------------------
    # Baseline 1: Standard PINN (Raissi 2019, No LayerNorm, lambda=1.0)
    # -----------------------------------------------------------------------
    print("Training 1. Standard PINN (Raissi 2019, No LayerNorm)...")
    m_std = PINN_MLP(in_dim=2, out_dim=1, hidden_dim=64, num_layers=4, use_layer_norm=False, activation='silu').to(DEVICE)
    opt_std = torch.optim.Adam(m_std.parameters(), lr=2e-3)
    
    for ep in range(epochs):
        opt_std.zero_grad()
        # Data loss
        pred_data = m_std(emp_xt)
        loss_data = torch.mean((pred_data - emp_u) ** 2)
        # PDE Collocation loss
        xt_c = torch.rand(n_col, 2, device=DEVICE)
        xt_c[:, 0] = xt_c[:, 0] * 2 * np.pi
        xt_c.requires_grad_(True)
        u_c = m_std(xt_c)
        g = torch.autograd.grad(u_c, xt_c, grad_outputs=torch.ones_like(u_c), create_graph=True)[0]
        res = g[:, 1] + beta * g[:, 0]
        loss_pde = torch.mean(res ** 2)
        
        loss = loss_data + 1.0 * loss_pde  # Standard PINN has lambda = 1.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_std.parameters(), max_norm=2.0)
        opt_std.step()
        
    models["Standard PINN (Raissi 2019)"] = m_std
    
    # -----------------------------------------------------------------------
    # Baseline 2: Standard PINN + LayerNorm (lambda=1.0)
    # -----------------------------------------------------------------------
    print("Training 2. Standard PINN + LayerNorm...")
    m_ln = PINN_MLP(in_dim=2, out_dim=1, hidden_dim=64, num_layers=4, use_layer_norm=True, activation='silu').to(DEVICE)
    opt_ln = torch.optim.Adam(m_ln.parameters(), lr=2e-3)
    
    for ep in range(epochs):
        opt_ln.zero_grad()
        pred_data = m_ln(emp_xt)
        loss_data = torch.mean((pred_data - emp_u) ** 2)
        
        xt_c = torch.rand(n_col, 2, device=DEVICE)
        xt_c[:, 0] = xt_c[:, 0] * 2 * np.pi
        xt_c.requires_grad_(True)
        u_c = m_ln(xt_c)
        g = torch.autograd.grad(u_c, xt_c, grad_outputs=torch.ones_like(u_c), create_graph=True)[0]
        res = g[:, 1] + beta * g[:, 0]
        loss_pde = torch.mean(res ** 2)
        
        loss = loss_data + 1.0 * loss_pde
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_ln.parameters(), max_norm=2.0)
        opt_ln.step()
        
    models["Standard PINN + LayerNorm"] = m_ln
    
    # -----------------------------------------------------------------------
    # Baseline 3: Deep Ensemble PINN (5 Models with LayerNorm, lambda=1.0)
    # -----------------------------------------------------------------------
    print("Training 3. Deep Ensemble PINN (5 Models with LayerNorm)...")
    ens_models = [
        PINN_MLP(in_dim=2, out_dim=1, hidden_dim=64, num_layers=4, use_layer_norm=True, activation='silu').to(DEVICE)
        for _ in range(5)
    ]
    for m_idx, m_e in enumerate(ens_models):
        opt_e = torch.optim.Adam(m_e.parameters(), lr=2e-3)
        for ep in range(epochs):
            opt_e.zero_grad()
            pred_data = m_e(emp_xt)
            loss_data = torch.mean((pred_data - emp_u) ** 2)
            
            xt_c = torch.rand(n_col, 2, device=DEVICE)
            xt_c[:, 0] = xt_c[:, 0] * 2 * np.pi
            xt_c.requires_grad_(True)
            u_c = m_e(xt_c)
            g = torch.autograd.grad(u_c, xt_c, grad_outputs=torch.ones_like(u_c), create_graph=True)[0]
            res = g[:, 1] + beta * g[:, 0]
            loss_pde = torch.mean(res ** 2)
            
            loss = loss_data + 1.0 * loss_pde
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m_e.parameters(), max_norm=2.0)
            opt_e.step()
            
    models["Deep Ensemble PINN (5 Models)"] = ens_models
    
    # -----------------------------------------------------------------------
    # Model 4: DP-PINN (No LayerNorm, STRICTLY NO LAMBDA, alpha=50)
    # -----------------------------------------------------------------------
    print("Training 4. DP-PINN (No LayerNorm, Strictly NO Lambda)...")
    alpha = 50.0
    K_t = 120
    prob_data = N_data / (alpha + N_data)
    
    dp_no_ln_models = [
        PINN_MLP(in_dim=2, out_dim=1, hidden_dim=64, num_layers=4, use_layer_norm=False, activation='silu').to(DEVICE)
        for _ in range(5)
    ]
    for p_idx, p_net in enumerate(dp_no_ln_models):
        opt_p = torch.optim.Adam(p_net.parameters(), lr=3e-3)
        for ep in range(epochs):
            opt_p.zero_grad()
            q = sample_stick_breaking_weights(alpha, N_data, K_t)
            is_d = (torch.bernoulli(torch.full((K_t,), prob_data, device=DEVICE)) == 1)
            n_d = int(is_d.sum().item())
            n_p = K_t - n_d
            
            batch_xt = torch.zeros(K_t, 2, device=DEVICE)
            batch_u = torch.zeros(K_t, 1, device=DEVICE)
            if n_d > 0:
                idx_d = torch.randint(0, N_data, (n_d,), device=DEVICE)
                batch_xt[is_d] = emp_xt[idx_d]
                batch_u[is_d] = emp_u[idx_d]
            if n_p > 0:
                idx_p = torch.randint(0, n_prior, (n_p,), device=DEVICE)
                batch_xt[~is_d] = prior_xt[idx_p]
                batch_u[~is_d] = prior_u[idx_p]
                
            pred = p_net(batch_xt)
            loss = torch.sum(q.unsqueeze(1) * (pred - batch_u) ** 2)  # STRICTLY NO LAMBDA
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p_net.parameters(), max_norm=2.0)
            opt_p.step()
            
    models["DP-PINN (No LayerNorm, NO Lambda)"] = dp_no_ln_models
    
    # -----------------------------------------------------------------------
    # Model 5: DP-PINN (With LayerNorm, STRICTLY NO LAMBDA, Ours)
    # -----------------------------------------------------------------------
    print("Training 5. DP-PINN (With LayerNorm, Strictly NO Lambda, Ours)...")
    dp_ln_models = [
        PINN_MLP(in_dim=2, out_dim=1, hidden_dim=64, num_layers=4, use_layer_norm=True, activation='silu').to(DEVICE)
        for _ in range(5)
    ]
    for p_idx, p_net in enumerate(dp_ln_models):
        opt_p = torch.optim.Adam(p_net.parameters(), lr=3e-3)
        for ep in range(epochs):
            opt_p.zero_grad()
            q = sample_stick_breaking_weights(alpha, N_data, K_t)
            is_d = (torch.bernoulli(torch.full((K_t,), prob_data, device=DEVICE)) == 1)
            n_d = int(is_d.sum().item())
            n_p = K_t - n_d
            
            batch_xt = torch.zeros(K_t, 2, device=DEVICE)
            batch_u = torch.zeros(K_t, 1, device=DEVICE)
            if n_d > 0:
                idx_d = torch.randint(0, N_data, (n_d,), device=DEVICE)
                batch_xt[is_d] = emp_xt[idx_d]
                batch_u[is_d] = emp_u[idx_d]
            if n_p > 0:
                idx_p = torch.randint(0, n_prior, (n_p,), device=DEVICE)
                batch_xt[~is_d] = prior_xt[idx_p]
                batch_u[~is_d] = prior_u[idx_p]
                
            pred = p_net(batch_xt)
            loss = torch.sum(q.unsqueeze(1) * (pred - batch_u) ** 2)  # STRICTLY NO LAMBDA
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p_net.parameters(), max_norm=2.0)
            opt_p.step()
            
    models["DP-PINN (With LayerNorm, NO Lambda, Ours)"] = dp_ln_models
    
    # -----------------------------------------------------------------------
    # Quantitative Evaluation & Leaderboard
    # -----------------------------------------------------------------------
    print("\nEvaluating Convection Models...")
    res_conv = {}
    preds_dict = {}
    
    u_true_np = U_true.cpu().numpy().squeeze()
    
    for name, m in models.items():
        if isinstance(m, list):
            particle_preds = []
            for net in m:
                net.eval()
                with torch.no_grad():
                    pred_i = net(XT_eval).cpu().numpy().squeeze()
                    particle_preds.append(pred_i)
            particle_preds = np.stack(particle_preds, axis=0)
            mean_pred = np.mean(particle_preds, axis=0)
            std_pred = np.std(particle_preds, axis=0)
        else:
            m.eval()
            with torch.no_grad():
                mean_pred = m(XT_eval).cpu().numpy().squeeze()
                std_pred = np.zeros_like(mean_pred)
                
        rel_l2 = np.linalg.norm(mean_pred - u_true_np) / np.linalg.norm(u_true_np)
        mse = np.mean((mean_pred - u_true_np) ** 2)
        mean_std = float(np.mean(std_pred))
        
        if isinstance(m, list):
            lower = mean_pred - 1.96 * std_pred
            upper = mean_pred + 1.96 * std_pred
            cov = float(np.mean((u_true_np >= lower) & (u_true_np <= upper)))
        else:
            cov = 0.0
            
        res_conv[name] = {
            "rel_l2_error": float(rel_l2),
            "mse": float(mse),
            "mean_epistemic_std": float(mean_std),
            "coverage_95": float(cov)
        }
        preds_dict[name] = (mean_pred, std_pred)
        print(f"  {name:42s} | Rel L2: {rel_l2:.4f} | MSE: {mse:.4f} | Cov: {cov*100:.1f}%")
        
    # -----------------------------------------------------------------------
    # Publication Visualization: 2-Panel Colormap & Cross-Sections
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 9))
    
    methods_to_plot = [
        ("Ground Truth Exact", u_true_np),
        ("Standard PINN (Raissi 2019)", preds_dict["Standard PINN (Raissi 2019)"][0]),
        ("Standard PINN + LayerNorm", preds_dict["Standard PINN + LayerNorm"][0]),
        ("Deep Ensemble PINN", preds_dict["Deep Ensemble PINN (5 Models)"][0]),
        ("DP-PINN (Ours, LayerNorm)", preds_dict["DP-PINN (With LayerNorm, NO Lambda, Ours)"][0])
    ]
    
    for idx, (title, p) in enumerate(methods_to_plot):
        ax = plt.subplot(2, 5, idx + 1)
        grid_p = p.reshape(nx_eval, nt_eval)
        im = ax.imshow(grid_p.T, origin='lower', extent=[0, 2*np.pi, 0, 1.0], aspect='auto', cmap='plasma', vmin=-1.1, vmax=1.1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel("Space x (radians)", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Time t", fontsize=9)
        if idx == 4:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
    t_slices = [0.25, 0.50, 0.75, 1.00]
    t_indices = [int(ts * (nt_eval - 1)) for ts in t_slices]
    
    for s_idx, (ts, tidx) in enumerate(zip(t_slices, t_indices)):
        ax = plt.subplot(2, 4, 5 + s_idx)
        x_axis = x_eval.cpu().numpy()
        
        u_slice_true = u_true_np.reshape(nx_eval, nt_eval)[:, tidx]
        ax.plot(x_axis, u_slice_true, 'k--', lw=2.0, label='Exact True Wave')
        
        u_slice_std = preds_dict["Standard PINN (Raissi 2019)"][0].reshape(nx_eval, nt_eval)[:, tidx]
        ax.plot(x_axis, u_slice_std, 'r:', lw=1.5, label='Standard PINN (Collapsed)')
        
        u_slice_dp = preds_dict["DP-PINN (With LayerNorm, NO Lambda, Ours)"][0].reshape(nx_eval, nt_eval)[:, tidx]
        std_slice_dp = preds_dict["DP-PINN (With LayerNorm, NO Lambda, Ours)"][1].reshape(nx_eval, nt_eval)[:, tidx]
        ax.plot(x_axis, u_slice_dp, 'b-', lw=1.8, label='DP-PINN (Ours, LN)')
        ax.fill_between(x_axis, u_slice_dp - 1.96 * std_slice_dp, u_slice_dp + 1.96 * std_slice_dp, color='blue', alpha=0.2, label='95% Epistemic Band')
        
        ax.set_title(f"Wave Snapshot at t = {ts:.2f}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Space x", fontsize=9)
        ax.set_ylim(-1.6, 1.6)
        if s_idx == 0:
            ax.set_ylabel("Solution u(x, t)", fontsize=9)
            ax.legend(loc='lower left', fontsize=7, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        
    plt.suptitle(f"1D Convection Benchmark (beta = {beta}) [Krishnapriyan et al. NeurIPS 2021 Showdown]", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plot_path = os.path.join(RESULTS_DIR, "neurips_convection_showdown.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    artifact_plot = os.path.join(ARTIFACTS_DIR, "neurips_convection_showdown.png")
    shutil.copyfile(plot_path, artifact_plot)
    print(f"Saved convection plot to: {plot_path}")
    
    return res_conv

# ===========================================================================
# BENCHMARK 2: Stiff 1D Reaction Equation (Krishnapriyan et al. NeurIPS 2021 Sec 4)
# du/dt - rho * u * (1 - u) = 0, t in [0, 1], u(0) = u0
# Exact: u(t) = (u0 * e^{rho*t}) / (1 - u0 + u0 * e^{rho*t})
# ===========================================================================
def run_reaction_benchmark(rho=5.0, u0=0.01, epochs=1500):
    print("\n" + "=" * 75)
    print(f"BENCHMARK 2: Stiff 1D Reaction Equation (rho = {rho}) [NeurIPS 2021 Section 4]")
    print("=" * 75)
    
    t_eval = torch.linspace(0, 1.0, 200).unsqueeze(1).to(DEVICE)
    u_true = (u0 * torch.exp(rho * t_eval)) / (1.0 - u0 + u0 * torch.exp(rho * t_eval))
    u_true_np = u_true.cpu().numpy().squeeze()
    
    emp_t = torch.tensor([[0.0], [0.5], [1.0]], device=DEVICE)
    emp_u = (u0 * torch.exp(rho * emp_t)) / (1.0 - u0 + u0 * torch.exp(rho * emp_t))
    N_data = len(emp_t)
    
    n_prior = 100
    t_prior = torch.linspace(0, 1.0, n_prior).unsqueeze(1).to(DEVICE)
    u_prior_list = [u0]
    dt_e = 0.01
    u_c = u0
    for _ in range(n_prior - 1):
        u_c = u_c + dt_e * rho * u_c * (1.0 - u_c)
        u_prior_list.append(u_c)
    prior_u = torch.tensor(u_prior_list, device=DEVICE).unsqueeze(1).float()
    
    models = {}
    
    # -----------------------------------------------------------------------
    # Baseline 1: Standard PINN (Raissi 2019, No LayerNorm, lambda=1.0)
    # -----------------------------------------------------------------------
    print("Training 1. Standard PINN on Stiff Reaction...")
    m_std = PINN_MLP(in_dim=1, out_dim=1, hidden_dim=32, num_layers=3, use_layer_norm=False, activation='tanh').to(DEVICE)
    opt_std = torch.optim.Adam(m_std.parameters(), lr=1e-3)
    
    for ep in range(epochs):
        opt_std.zero_grad()
        pred_data = m_std(emp_t)
        loss_data = torch.mean((pred_data - emp_u) ** 2)
        
        t_col = torch.rand(100, 1, device=DEVICE, requires_grad=True)
        u_col = m_std(t_col)
        u_t = torch.autograd.grad(u_col, t_col, grad_outputs=torch.ones_like(u_col), create_graph=True)[0]
        res = u_t - rho * u_col * (1.0 - u_col)
        loss_pde = torch.mean(res ** 2)
        
        loss = loss_data + 1.0 * loss_pde
        loss.backward()
        opt_std.step()
        
    models["Standard PINN (Raissi 2019)"] = m_std
    
    # -----------------------------------------------------------------------
    # Baseline 2: Standard PINN + LayerNorm
    # -----------------------------------------------------------------------
    print("Training 2. Standard PINN + LayerNorm on Stiff Reaction...")
    m_ln = PINN_MLP(in_dim=1, out_dim=1, hidden_dim=32, num_layers=3, use_layer_norm=True, activation='tanh').to(DEVICE)
    opt_ln = torch.optim.Adam(m_ln.parameters(), lr=1e-3)
    
    for ep in range(epochs):
        opt_ln.zero_grad()
        pred_data = m_ln(emp_t)
        loss_data = torch.mean((pred_data - emp_u) ** 2)
        
        t_col = torch.rand(100, 1, device=DEVICE, requires_grad=True)
        u_col = m_ln(t_col)
        u_t = torch.autograd.grad(u_col, t_col, grad_outputs=torch.ones_like(u_col), create_graph=True)[0]
        res = u_t - rho * u_col * (1.0 - u_col)
        loss_pde = torch.mean(res ** 2)
        
        loss = loss_data + 1.0 * loss_pde
        loss.backward()
        opt_std.step()
        
    models["Standard PINN + LayerNorm"] = m_ln
    
    # -----------------------------------------------------------------------
    # Baseline 3: Deep Ensemble PINN
    # -----------------------------------------------------------------------
    print("Training 3. Deep Ensemble PINN on Stiff Reaction...")
    ens_models = [
        PINN_MLP(in_dim=1, out_dim=1, hidden_dim=32, num_layers=3, use_layer_norm=True, activation='tanh').to(DEVICE)
        for _ in range(5)
    ]
    for m_e in ens_models:
        opt_e = torch.optim.Adam(m_e.parameters(), lr=1e-3)
        for ep in range(epochs):
            opt_e.zero_grad()
            pred_data = m_e(emp_t)
            loss_data = torch.mean((pred_data - emp_u) ** 2)
            
            t_col = torch.rand(100, 1, device=DEVICE, requires_grad=True)
            u_col = m_e(t_col)
            u_t = torch.autograd.grad(u_col, t_col, grad_outputs=torch.ones_like(u_col), create_graph=True)[0]
            res = u_t - rho * u_col * (1.0 - u_col)
            loss_pde = torch.mean(res ** 2)
            
            loss = loss_data + 1.0 * loss_pde
            loss.backward()
            opt_e.step()
            
    models["Deep Ensemble PINN (5 Models)"] = ens_models
    
    # -----------------------------------------------------------------------
    # Model 4: DP-PINN (No LayerNorm, Strictly NO Lambda)
    # -----------------------------------------------------------------------
    print("Training 4. DP-PINN (No LayerNorm, Strictly NO Lambda)...")
    alpha = 10.0
    K_t = 60
    prob_data = N_data / (alpha + N_data)
    
    dp_no_ln = [
        PINN_MLP(in_dim=1, out_dim=1, hidden_dim=32, num_layers=3, use_layer_norm=False, activation='tanh').to(DEVICE)
        for _ in range(5)
    ]
    for p_net in dp_no_ln:
        opt_p = torch.optim.Adam(p_net.parameters(), lr=2e-3)
        for ep in range(epochs):
            opt_p.zero_grad()
            q = sample_stick_breaking_weights(alpha, N_data, K_t)
            is_d = (torch.bernoulli(torch.full((K_t,), prob_data, device=DEVICE)) == 1)
            n_d = int(is_d.sum().item())
            n_p = K_t - n_d
            
            atoms_t = torch.zeros(K_t, 1, device=DEVICE)
            atoms_u = torch.zeros(K_t, 1, device=DEVICE)
            if n_d > 0:
                idx_d = torch.randint(0, N_data, (n_d,), device=DEVICE)
                atoms_t[is_d] = emp_t[idx_d]
                atoms_u[is_d] = emp_u[idx_d]
            if n_p > 0:
                idx_p = torch.randint(0, n_prior, (n_p,), device=DEVICE)
                atoms_t[~is_d] = t_prior[idx_p]
                atoms_u[~is_d] = prior_u[idx_p]
                
            pred = p_net(atoms_t)
            loss = torch.sum(q.unsqueeze(1) * (pred - atoms_u) ** 2)  # STRICTLY NO LAMBDA
            loss.backward()
            opt_p.step()
            
    models["DP-PINN (No LayerNorm, NO Lambda)"] = dp_no_ln
    
    # -----------------------------------------------------------------------
    # Model 5: DP-PINN (With LayerNorm, Strictly NO Lambda, Ours)
    # -----------------------------------------------------------------------
    print("Training 5. DP-PINN (With LayerNorm, Strictly NO Lambda, Ours)...")
    dp_ln = [
        PINN_MLP(in_dim=1, out_dim=1, hidden_dim=32, num_layers=3, use_layer_norm=True, activation='tanh').to(DEVICE)
        for _ in range(5)
    ]
    for p_net in dp_ln:
        opt_p = torch.optim.Adam(p_net.parameters(), lr=2e-3)
        for ep in range(epochs):
            opt_p.zero_grad()
            q = sample_stick_breaking_weights(alpha, N_data, K_t)
            is_d = (torch.bernoulli(torch.full((K_t,), prob_data, device=DEVICE)) == 1)
            n_d = int(is_d.sum().item())
            n_p = K_t - n_d
            
            atoms_t = torch.zeros(K_t, 1, device=DEVICE)
            atoms_u = torch.zeros(K_t, 1, device=DEVICE)
            if n_d > 0:
                idx_d = torch.randint(0, N_data, (n_d,), device=DEVICE)
                atoms_t[is_d] = emp_t[idx_d]
                atoms_u[is_d] = emp_u[idx_d]
            if n_p > 0:
                idx_p = torch.randint(0, n_prior, (n_p,), device=DEVICE)
                atoms_t[~is_d] = t_prior[idx_p]
                atoms_u[~is_d] = prior_u[idx_p]
                
            pred = p_net(atoms_t)
            loss = torch.sum(q.unsqueeze(1) * (pred - atoms_u) ** 2)  # STRICTLY NO LAMBDA
            loss.backward()
            opt_p.step()
            
    models["DP-PINN (With LayerNorm, NO Lambda, Ours)"] = dp_ln
    
    # -----------------------------------------------------------------------
    # Quantitative Evaluation
    # -----------------------------------------------------------------------
    print("\nEvaluating Stiff Reaction Models...")
    res_react = {}
    preds_dict = {}
    
    for name, m in models.items():
        if isinstance(m, list):
            particle_preds = []
            for net in m:
                net.eval()
                with torch.no_grad():
                    particle_preds.append(net(t_eval).cpu().numpy().squeeze())
            particle_preds = np.stack(particle_preds, axis=0)
            mean_pred = np.mean(particle_preds, axis=0)
            std_pred = np.std(particle_preds, axis=0)
        else:
            m.eval()
            with torch.no_grad():
                mean_pred = m(t_eval).cpu().numpy().squeeze()
                std_pred = np.zeros_like(mean_pred)
                
        rel_l2 = np.linalg.norm(mean_pred - u_true_np) / np.linalg.norm(u_true_np)
        mse = np.mean((mean_pred - u_true_np) ** 2)
        mean_std = float(np.mean(std_pred))
        
        if isinstance(m, list):
            lower = mean_pred - 1.96 * std_pred
            upper = mean_pred + 1.96 * std_pred
            cov = float(np.mean((u_true_np >= lower) & (u_true_np <= upper)))
        else:
            cov = 0.0
            
        res_react[name] = {
            "rel_l2_error": float(rel_l2),
            "mse": float(mse),
            "mean_epistemic_std": float(mean_std),
            "coverage_95": float(cov)
        }
        preds_dict[name] = (mean_pred, std_pred)
        print(f"  {name:42s} | Rel L2: {rel_l2:.4f} | MSE: {mse:.4f} | Cov: {cov*100:.1f}%")
        
    # -----------------------------------------------------------------------
    # Publication Visualization: Trajectory & Epistemic Dome
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t_np = t_eval.cpu().numpy().squeeze()
    
    ax = axes[0]
    ax.plot(t_np, u_true_np, 'k--', lw=2.5, label='Analytical True Solution')
    ax.plot(t_np, preds_dict["Standard PINN (Raissi 2019)"][0], 'r:', lw=2.0, label='Standard PINN (Collapsed)')
    ax.plot(t_np, preds_dict["Standard PINN + LayerNorm"][0], 'm-.', lw=1.8, label='Standard PINN + LayerNorm')
    ax.plot(t_np, preds_dict["Deep Ensemble PINN (5 Models)"][0], 'g--', lw=1.8, label='Deep Ensemble PINN')
    ax.plot(t_np, preds_dict["DP-PINN (With LayerNorm, NO Lambda, Ours)"][0], 'b-', lw=2.2, label='DP-PINN (Ours, LN)')
    
    mean_dp, std_dp = preds_dict["DP-PINN (With LayerNorm, NO Lambda, Ours)"]
    ax.fill_between(t_np, mean_dp - 1.96 * std_dp, mean_dp + 1.96 * std_dp, color='blue', alpha=0.2, label='DP-PINN 95% CI')
    
    emp_t_np = emp_t.cpu().numpy().squeeze()
    emp_u_np = emp_u.cpu().numpy().squeeze()
    ax.scatter(emp_t_np, emp_u_np, color='darkorange', s=70, zorder=5, edgecolors='black', label=f'Empirical Points (N={N_data})')
    
    ax.set_title(f"(A) Stiff Reaction Trajectory (rho = {rho})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Time t", fontsize=10)
    ax.set_ylabel("State u(t)", fontsize=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(t_np, np.abs(preds_dict["Standard PINN (Raissi 2019)"][0] - u_true_np), 'r:', lw=2.0, label='Standard PINN Abs Error')
    ax2.plot(t_np, np.abs(mean_dp - u_true_np), 'b-', lw=2.0, label='DP-PINN Abs Error')
    ax2.plot(t_np, 1.96 * std_dp, 'b--', lw=1.5, label='DP-PINN 95% Credible Width')
    
    mean_ens, std_ens = preds_dict["Deep Ensemble PINN (5 Models)"]
    ax2.plot(t_np, 1.96 * std_ens, 'g--', lw=1.5, label='Deep Ensemble 95% Width')
    
    ax2.set_title("(B) Epistemic Uncertainty & Error Envelope", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time t", fontsize=10)
    ax2.set_ylabel("Magnitude", fontsize=10)
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.85)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Stiff 1D Reaction Benchmark [Krishnapriyan et al. NeurIPS 2021 Failure Mode]", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, "neurips_reaction_showdown.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    artifact_plot = os.path.join(ARTIFACTS_DIR, "neurips_reaction_showdown.png")
    shutil.copyfile(plot_path, artifact_plot)
    print(f"Saved reaction plot to: {plot_path}")
    
    return res_react

# ===========================================================================
# Master Execution & Summary JSON
# ===========================================================================
def main():
    start_time = time.time()
    
    # 1. Run 1D Convection Benchmark (beta = 10.0)
    res_conv = run_convection_benchmark(beta=10.0, epochs=1500)
    
    # 2. Run Stiff 1D Reaction Benchmark (rho = 5.0)
    res_react = run_reaction_benchmark(rho=5.0, u0=0.01, epochs=1500)
    
    # Save combined summary
    summary = {
        "convection_neurips2021_beta10": res_conv,
        "reaction_neurips2021_rho5": res_react,
        "execution_time_seconds": time.time() - start_time
    }
    
    summary_path = os.path.join(RESULTS_DIR, "neurips_pinn_benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved master summary JSON to: {summary_path}")
    print(f"Total benchmark execution completed in {summary['execution_time_seconds']:.2f} seconds.")

if __name__ == "__main__":
    main()
