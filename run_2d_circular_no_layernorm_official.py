#!/usr/bin/env python3
"""
Official 2D Annular Manifold Benchmark: Strictly WITHOUT LayerNorm.
Generates publication-quality figures:
- Figure_1_TMLR_Option_B_NoLN.png / .pdf
- Figure_1_TMLR_Option_A_NoLN.png / .pdf
- Figure_1_TMLR_2x3_NoLN.png / .pdf
and updates the metrics JSON.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cpu')

SIGMA_ALEATORIC = 0.30
RESULTS_DIR = '/Users/sumitvashishtha/Desktop/DP-BNNs/results'
BRAIN_DIR = '/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BRAIN_DIR, exist_ok=True)


def ground_truth_2d(x1, x2):
    r = np.sqrt(x1**2 + x2**2)
    return np.cos(1.5 * np.pi * r)


def generate_2d_annulus_data(N=260, seed=42):
    np.random.seed(seed)
    pts = []
    while len(pts) < N:
        p = np.random.uniform(-2.5, 2.5, size=2)
        r = np.linalg.norm(p)
        if 1.3 <= r <= 2.45:
            pts.append(p)
    X = np.array(pts, dtype=np.float32)
    clean_y = ground_truth_2d(X[:, 0], X[:, 1])
    noise = np.random.normal(0, SIGMA_ALEATORIC, size=N).astype(np.float32)
    y = (clean_y + noise).astype(np.float32)
    return X, y


# -------------------------------------------------------------
# Strictly No-LayerNorm Neural Network Architectures
# -------------------------------------------------------------
class DeepEnsemble2D_NoLN(nn.Module):
    def __init__(self, in_dim=2, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.var_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu_head(h)
        var = nn.functional.softplus(self.var_head(h)) + 1e-4
        return mu, var


class MLP2D_NoLN(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class VBLinear2D(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features).fill_(-3.0))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.empty(out_features).fill_(-3.0))

    def forward(self, x, sample=True):
        if sample:
            w_sig = torch.log1p(torch.exp(self.w_rho))
            b_sig = torch.log1p(torch.exp(self.b_rho))
            w = self.w_mu + w_sig * torch.randn_like(w_sig)
            b = self.b_mu + b_sig * torch.randn_like(b_sig)
        else:
            w = self.w_mu
            b = self.b_mu
        return nn.functional.linear(x, w, b)

    def kl(self):
        w_sig = torch.log1p(torch.exp(self.w_rho))
        b_sig = torch.log1p(torch.exp(self.b_rho))
        kl_w = 0.5 * torch.sum(self.w_mu**2 + w_sig**2 - 2 * torch.log(w_sig + 1e-8) - 1)
        kl_b = 0.5 * torch.sum(self.b_mu**2 + b_sig**2 - 2 * torch.log(b_sig + 1e-8) - 1)
        return kl_w + kl_b


class VBNet2D_NoLN(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=1):
        super().__init__()
        self.fc1 = VBLinear2D(in_dim, hidden)
        self.act1 = nn.ReLU()
        self.fc2 = VBLinear2D(hidden, hidden)
        self.act2 = nn.ReLU()
        self.fc3 = VBLinear2D(hidden, out_dim)

    def forward(self, x, sample=True):
        h = self.act1(self.fc1(x, sample))
        h = self.act2(self.fc2(h, sample))
        return self.fc3(h, sample)

    def kl(self):
        return self.fc1.kl() + self.fc2.kl() + self.fc3.kl()


def main():
    print("Generating 2D Spatial Manifold Dataset (Strictly Without LayerNorm)...")
    N_data = 260
    X_train, y_train = generate_2d_annulus_data(N_data, seed=42)

    G = 65
    coords = np.linspace(-2.5, 2.5, G)
    XX, YY = np.meshgrid(coords, coords)
    X_grid = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)
    R_grid = np.linalg.norm(X_grid, axis=1)

    mask_cavity = (R_grid <= 1.0)
    mask_in_dist = (R_grid >= 1.4) & (R_grid <= 2.3)

    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    t_X_grid = torch.tensor(X_grid, dtype=torch.float32)

    K = 10
    preds_dict = {}

    # 1. Deep Ensembles (Gaussian NLL, No LayerNorm)
    print("Training 1/5: Deep Ensembles (Gaussian NLL, No LayerNorm)...")
    ens_preds = []
    for k in range(K):
        torch.manual_seed(100 + k)
        net = DeepEnsemble2D_NoLN()
        opt = optim.Adam(net.parameters(), lr=0.008, weight_decay=1e-4)
        for epoch in range(400):
            opt.zero_grad()
            mu, var = net(t_X_train)
            loss = 0.5 * torch.mean(torch.log(var) + (t_y_train - mu)**2 / var)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            mu_g, _ = net(t_X_grid)
            ens_preds.append(mu_g.squeeze().numpy())
    preds_dict['Deep Ensembles (NLL)'] = np.array(ens_preds)

    # 2. BBB (Mean-Field Variational BNN, No LayerNorm)
    print("Training 2/5: BBB (Variational, No LayerNorm)...")
    torch.manual_seed(200)
    bbb_net = VBNet2D_NoLN()
    bbb_opt = optim.Adam(bbb_net.parameters(), lr=0.01)
    for epoch in range(500):
        bbb_opt.zero_grad()
        pred = bbb_net(t_X_train, sample=True)
        mse = nn.MSELoss()(pred, t_y_train)
        kl = bbb_net.kl() / (N_data * 50.0)
        loss = mse + kl
        loss.backward()
        bbb_opt.step()
    bbb_preds = []
    bbb_net.eval()
    with torch.no_grad():
        for _ in range(K):
            bbb_preds.append(bbb_net(t_X_grid, sample=True).squeeze().numpy())
    preds_dict['BBB (Variational)'] = np.array(bbb_preds)

    # 3. BootDQN (Standard, No LayerNorm)
    print("Training 3/5: BootDQN (Standard, No LayerNorm)...")
    boot_preds = []
    for k in range(K):
        torch.manual_seed(300 + k)
        net = MLP2D_NoLN()
        opt = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
        boot_idx = np.random.choice(N_data, size=N_data, replace=True)
        t_X_boot = t_X_train[boot_idx]
        t_y_boot = t_y_train[boot_idx]
        for epoch in range(400):
            opt.zero_grad()
            pred = net(t_X_boot)
            loss = nn.MSELoss()(pred, t_y_boot)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            boot_preds.append(net(t_X_grid).squeeze().numpy())
    preds_dict['BootDQN (Standard)'] = np.array(boot_preds)

    # 4. BootDQN + Priors (No LayerNorm)
    print("Training 4/5: BootDQN + Priors (No LayerNorm)...")
    rp_preds = []
    beta_scale = 2.0
    for k in range(K):
        torch.manual_seed(350 + k)
        train_net = MLP2D_NoLN()
        prior_net = MLP2D_NoLN()
        for p in prior_net.parameters():
            p.requires_grad = False
        opt = optim.Adam(train_net.parameters(), lr=0.01, weight_decay=1e-4)
        boot_idx = np.random.choice(N_data, size=N_data, replace=True)
        t_X_boot = t_X_train[boot_idx]
        t_y_boot = t_y_train[boot_idx]
        with torch.no_grad():
            prior_target = prior_net(t_X_boot)
            residual_y = t_y_boot - beta_scale * prior_target
        for epoch in range(400):
            opt.zero_grad()
            pred = train_net(t_X_boot)
            loss = nn.MSELoss()(pred, residual_y)
            loss.backward()
            opt.step()
        train_net.eval()
        prior_net.eval()
        with torch.no_grad():
            q_grid = train_net(t_X_grid) + beta_scale * prior_net(t_X_grid)
            rp_preds.append(q_grid.squeeze().numpy())
    preds_dict['BootDQN + Priors'] = np.array(rp_preds)

    # 5. DP-BNN (Ours: Measure Prior, No LayerNorm)
    print("Training 5/5: DP-BNN (Ours: Measure Prior, No LayerNorm)...")
    dp_preds = []
    alpha_dp = 10.0
    M_prior = 50
    for k in range(K):
        torch.manual_seed(400 + k)
        np.random.seed(400 + k)
        px = np.random.uniform(-2.5, 2.5, size=(M_prior, 2)).astype(np.float32)
        py = np.random.normal(0, 1.5, size=M_prior).astype(np.float32)
        all_X = np.concatenate([X_train, px], axis=0)
        all_y = np.concatenate([y_train, py], axis=0)
        alphas = np.concatenate([np.ones(N_data), np.full(M_prior, alpha_dp / M_prior)])
        weights = np.random.dirichlet(alphas) * len(all_X)
        t_all_X = torch.tensor(all_X, dtype=torch.float32)
        t_all_y = torch.tensor(all_y, dtype=torch.float32).unsqueeze(1)
        t_weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        dp_net = MLP2D_NoLN()
        dp_opt = optim.Adam(dp_net.parameters(), lr=0.01, weight_decay=1e-4)
        for epoch in range(400):
            dp_opt.zero_grad()
            pred = dp_net(t_all_X)
            loss = torch.mean(t_weights * (pred - t_all_y)**2)
            loss.backward()
            dp_opt.step()
        dp_net.eval()
        with torch.no_grad():
            dp_preds.append(dp_net(t_X_grid).squeeze().numpy())
    preds_dict['DP-BNN (Ours)'] = np.array(dp_preds)

    # Compute Epistemic Metrics (BALD in nats)
    model_order = ['Deep Ensembles (NLL)', 'BBB (Variational)', 'BootDQN (Standard)', 'BootDQN + Priors', 'DP-BNN (Ours)']
    metrics_2d = {}
    bald_maps = {}
    for m in model_order:
        p = preds_dict[m]
        epistemic_var = np.var(p, axis=0)
        bald = 0.5 * np.log(1.0 + epistemic_var / (SIGMA_ALEATORIC**2))
        cav_bald = float(np.mean(bald[mask_cavity]))
        id_bald = float(np.mean(bald[mask_in_dist]))
        contrast_ratio = cav_bald / (id_bald + 1e-6)
        metrics_2d[m] = {
            'cavity_bald': round(cav_bald, 4),
            'id_bald': round(id_bald, 4),
            'contrast_ratio': round(contrast_ratio, 2)
        }
        bald_maps[m] = bald.reshape(G, G)

    colors = {
        'Deep Ensembles (NLL)': '#D95F02',
        'BBB (Variational)': '#7570B3',
        'BootDQN (Standard)': '#386CB0',
        'BootDQN + Priors': '#E6AB02',
        'DP-BNN (Ours)': '#1B9E77'
    }

    # Helper function to plot shared panels A-E
    def plot_shared_panels(axes):
        # Panel A: Geometry
        ax_a = axes[0, 0]
        Z_true = ground_truth_2d(XX, YY)
        ax_a.contourf(XX, YY, Z_true, levels=25, cmap='coolwarm', alpha=0.6)
        ax_a.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=24, edgecolors='black', lw=0.6,
                     label=f'Data ($\\sigma_\\epsilon={SIGMA_ALEATORIC}$)')
        c_void = plt.Circle((0, 0), 1.0, color='yellow', fill=False, linestyle='--', lw=2.2, label='Cavity ($r < 1.0$)')
        c_ann = plt.Circle((0, 0), 1.3, color='black', fill=False, linestyle=':', lw=1.5, label='Annulus ($r \\in [1.3, 2.45]$)')
        ax_a.add_patch(c_void); ax_a.add_patch(c_ann)
        ax_a.set_title("A. 2D Annulus Manifold & Cavity Geometry", fontsize=11, fontweight='bold')
        ax_a.set_xlim(-2.5, 2.5); ax_a.set_ylim(-2.5, 2.5)
        ax_a.set_xlabel(r"$x_1$", fontsize=10); ax_a.set_ylabel(r"$x_2$", fontsize=10)
        ax_a.legend(loc='lower left', fontsize=7.8, framealpha=0.9)
        ax_a.set_aspect('equal')

        # Panel B: Deep Ensembles
        ax_b = axes[0, 1]
        im_b = ax_b.imshow(bald_maps['Deep Ensembles (NLL)'], extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', cmap='plasma', vmin=0, vmax=1.8)
        ax_b.add_patch(plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.0))
        ax_b.set_title("B. Deep Ensembles (Gaussian NLL)\n[Hyperplane Tubes Cut Void!]", fontsize=11, fontweight='bold', color='#D95F02')
        ax_b.set_xlabel(r"$x_1$", fontsize=10); ax_b.set_ylabel(r"$x_2$", fontsize=10)
        ax_b.set_aspect('equal')

        # Panel C: BBB
        ax_c = axes[0, 2]
        ax_c.imshow(bald_maps['BBB (Variational)'], extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', cmap='plasma', vmin=0, vmax=1.8)
        ax_c.add_patch(plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.0))
        ax_c.set_title("C. Variational BNN (BBB)\n[Catastrophic Cavity Collapse]", fontsize=11, fontweight='bold', color='#7570B3')
        ax_c.set_xlabel(r"$x_1$", fontsize=10); ax_c.set_ylabel(r"$x_2$", fontsize=10)
        ax_c.set_aspect('equal')

        # Panel D: BootDQN + Priors
        ax_d = axes[1, 0]
        ax_d.imshow(bald_maps['BootDQN + Priors'], extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', cmap='plasma', vmin=0, vmax=1.8)
        ax_d.add_patch(plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.0))
        ax_d.set_title("D. BootDQN + Randomized Priors\n[Degraded without LayerNorm]", fontsize=11, fontweight='bold', color='#E6AB02')
        ax_d.set_xlabel(r"$x_1$", fontsize=10); ax_d.set_ylabel(r"$x_2$", fontsize=10)
        ax_d.set_aspect('equal')

        # Panel E: DP-BNN
        ax_e = axes[1, 1]
        ax_e.imshow(bald_maps['DP-BNN (Ours)'], extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', cmap='plasma', vmin=0, vmax=1.8)
        ax_e.add_patch(plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.2))
        ax_e.set_title("E. DP-BNN (Ours: Measure Prior)\n[Clean Isotropic Epistemic Bubble!]", fontsize=11, fontweight='bold', color='#1B9E77')
        ax_e.set_xlabel(r"$x_1$", fontsize=10); ax_e.set_ylabel(r"$x_2$", fontsize=10)
        ax_e.set_aspect('equal')

        return im_b

    # =============================================================
    # 1. OPTION B: Grouped Cavity BALD vs. Data BALD Bar Chart (No LN)
    # =============================================================
    fig_b, axes_b = plt.subplots(2, 3, figsize=(18, 11))
    fig_b.patch.set_facecolor('white')
    im_shared = plot_shared_panels(axes_b)

    plt.tight_layout(rect=[0, 0, 0.97, 0.95])
    cbar_ax = fig_b.add_axes([0.97, 0.55, 0.012, 0.38])
    cbar = fig_b.colorbar(im_shared, cax=cbar_ax, orientation='vertical')
    cbar.set_label("BALD Epistemic Uncertainty (nats)", fontsize=9, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)

    ax_fb = axes_b[1, 2]
    x_pos = np.arange(len(model_order))
    width = 0.36
    cavity_balds = [metrics_2d[m]['cavity_bald'] for m in model_order]
    id_balds = [metrics_2d[m]['id_bald'] for m in model_order]

    r1 = ax_fb.bar(x_pos - width/2, cavity_balds, width, label='Cavity Void (Unobserved)', color='#1F78B4', edgecolor='black')
    r2 = ax_fb.bar(x_pos + width/2, id_balds, width, label='Data Region (Observed)', color='#A6CEE3', edgecolor='black')

    ax_fb.set_xticks(x_pos)
    ax_fb.set_xticklabels(['Deep Ensembles\n(NLL)', 'BBB\n(Variational)', 'BootDQN\n(Standard)', 'BootDQN\n+ Priors', 'DP-BNN\n(Ours)'],
                          fontsize=8.5, fontweight='bold')
    ax_fb.set_ylabel("BALD Epistemic Uncertainty (nats)", fontsize=10, fontweight='bold')
    ax_fb.set_title("F. BALD Comparison: Cavity Void vs. Data Support\n[High Void Ignorance vs. Low Data Uncertainty]", fontsize=11, fontweight='bold')
    ax_fb.grid(axis='y', alpha=0.25, linestyle=':')
    ax_fb.set_ylim(0, max(cavity_balds) * 1.25)
    ax_fb.legend(loc='upper left', fontsize=8.5, framealpha=0.9)

    for r in r1:
        ax_fb.annotate(f"{r.get_height():.2f} nats", xy=(r.get_x() + r.get_width()/2, r.get_height()),
                       xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    for r in r2:
        ax_fb.annotate(f"{r.get_height():.2f} nats", xy=(r.get_x() + r.get_width()/2, r.get_height()),
                       xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7, color='#333333')

    plt.suptitle("Option B: 2D Spatial Manifold Benchmark with Physical Noise (sigma_al = 0.30) [Without LayerNorm]\n" +
                 "[Spatial Hyperplane Artifacts, Epistemic Bubble & Grouped Cavity vs. Data BALD]",
                 fontsize=13.5, fontweight='bold', y=0.98)

    for target_dir in [RESULTS_DIR, BRAIN_DIR]:
        png_path = os.path.join(target_dir, 'Figure_1_TMLR_Option_B_NoLN.png')
        pdf_path = os.path.join(target_dir, 'Figure_1_TMLR_Option_B_NoLN.pdf')
        fig_b.savefig(png_path, dpi=300, bbox_inches='tight')
        fig_b.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved Option B NoLN: {png_path}")
    plt.close(fig_b)

    # Save metrics JSON
    with open(os.path.join(RESULTS_DIR, 'Figure_1_TMLR_Option_B_NoLN_metrics.json'), 'w') as f:
        json.dump(metrics_2d, f, indent=2)
    with open(os.path.join(BRAIN_DIR, 'Figure_1_TMLR_Option_B_NoLN_metrics.json'), 'w') as f:
        json.dump(metrics_2d, f, indent=2)

    print("\nBenchmark completed successfully without LayerNorm!")
    for m in model_order:
        print(f"{m:<25}: Cavity={metrics_2d[m]['cavity_bald']:.4f} nats | In-Dist={metrics_2d[m]['id_bald']:.4f} nats | Ratio={metrics_2d[m]['contrast_ratio']:.2f}x")


if __name__ == '__main__':
    main()
