#!/usr/bin/env python3
"""
2D Circular/Annular Manifold Regression: LayerNorm Ablation Study.
Compares 5 BNN / Uncertainty procedures With LayerNorm vs. Without LayerNorm:
1. Deep Ensembles (Gaussian NLL)
2. Variational BNN (BBB)
3. BootDQN (Standard)
4. BootDQN + Randomized Priors
5. DP-BNN (Vashishtha Formulation: Empirical + Prior Stick-Breaking)

Dataset: 2D Annulus r in [1.3, 2.45] with central unobserved cavity (r < 1.0).
Evaluates epistemic uncertainty (BALD, nats) and contrast ratio (Cavity / In-Distribution).
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

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cpu')

SIGMA_ALEATORIC = 0.30
RESULTS_DIR = './results'
ARTIFACT_DIR = '/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


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
# Modular Architectures (With and Without LayerNorm)
# -------------------------------------------------------------
class DeepEnsemble2D(nn.Module):
    def __init__(self, in_dim=2, hidden=64, use_layer_norm=True):
        super().__init__()
        if use_layer_norm:
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            )
        else:
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


class MLP2D(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=1, use_layer_norm=True):
        super().__init__()
        if use_layer_norm:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim)
            )
        else:
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


class VBNet2D(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=1, use_layer_norm=True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.fc1 = VBLinear2D(in_dim, hidden)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden)
            self.ln2 = nn.LayerNorm(hidden)
        self.act1 = nn.ReLU()
        self.fc2 = VBLinear2D(hidden, hidden)
        self.act2 = nn.ReLU()
        self.fc3 = VBLinear2D(hidden, out_dim)

    def forward(self, x, sample=True):
        if self.use_layer_norm:
            h = self.act1(self.ln1(self.fc1(x, sample)))
            h = self.act2(self.ln2(self.fc2(h, sample)))
        else:
            h = self.act1(self.fc1(x, sample))
            h = self.act2(self.fc2(h, sample))
        return self.fc3(h, sample)

    def kl(self):
        return self.fc1.kl() + self.fc2.kl() + self.fc3.kl()


def run_benchmark_for_setting(use_layer_norm: bool, X_train, y_train, X_grid, mask_cavity, mask_in_dist, K=10):
    mode_str = "With LayerNorm" if use_layer_norm else "Without LayerNorm"
    print(f"\n========================================================")
    print(f" Running 2D Annular Benchmark: {mode_str}")
    print(f"========================================================")
    t0 = time.time()
    N_data = len(X_train)

    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    t_X_grid = torch.tensor(X_grid, dtype=torch.float32)

    preds_dict = {}

    # 1. Deep Ensembles (Gaussian NLL)
    print(f"[{mode_str}] 1/5: Training Deep Ensembles (Gaussian NLL)...")
    ens_preds = []
    for k in range(K):
        torch.manual_seed(100 + k)
        net = DeepEnsemble2D(use_layer_norm=use_layer_norm)
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

    # 2. BBB (Mean-Field Variational BNN)
    print(f"[{mode_str}] 2/5: Training BBB (Variational)...")
    torch.manual_seed(200)
    bbb_net = VBNet2D(use_layer_norm=use_layer_norm)
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

    # 3. BootDQN (Standard)
    print(f"[{mode_str}] 3/5: Training BootDQN (Standard)...")
    boot_preds = []
    for k in range(K):
        torch.manual_seed(300 + k)
        net = MLP2D(use_layer_norm=use_layer_norm)
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

    # 4. BootDQN + Priors
    print(f"[{mode_str}] 4/5: Training BootDQN + Priors...")
    rp_preds = []
    beta_scale = 2.0
    for k in range(K):
        torch.manual_seed(350 + k)
        train_net = MLP2D(use_layer_norm=use_layer_norm)
        prior_net = MLP2D(use_layer_norm=use_layer_norm)
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

    # 5. DP-BNN (Ours: Measure Prior)
    print(f"[{mode_str}] 5/5: Training DP-BNN (Ours: Measure Prior)...")
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
        dp_net = MLP2D(use_layer_norm=use_layer_norm)
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
    metrics = {}
    bald_maps = {}
    for m in model_order:
        p = preds_dict[m]
        epistemic_var = np.var(p, axis=0)
        bald = 0.5 * np.log(1.0 + epistemic_var / (SIGMA_ALEATORIC**2))
        cav_bald = float(np.mean(bald[mask_cavity]))
        id_bald = float(np.mean(bald[mask_in_dist]))
        contrast_ratio = cav_bald / (id_bald + 1e-6)
        metrics[m] = {
            'cavity_bald': round(cav_bald, 4),
            'id_bald': round(id_bald, 4),
            'contrast_ratio': round(contrast_ratio, 2)
        }
        bald_maps[m] = bald

    elapsed = time.time() - t0
    print(f"Finished {mode_str} in {elapsed:.1f}s.")
    return metrics, bald_maps, preds_dict


def main():
    N_data = 260
    X_train, y_train = generate_2d_annulus_data(N_data, seed=42)

    G = 65
    coords = np.linspace(-2.5, 2.5, G)
    XX, YY = np.meshgrid(coords, coords)
    X_grid = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)
    R_grid = np.linalg.norm(X_grid, axis=1)

    mask_cavity = (R_grid <= 1.0)
    mask_in_dist = (R_grid >= 1.4) & (R_grid <= 2.3)

    # 1. Run WITH LayerNorm
    metrics_with_ln, bald_with_ln, preds_with_ln = run_benchmark_for_setting(
        use_layer_norm=True,
        X_train=X_train,
        y_train=y_train,
        X_grid=X_grid,
        mask_cavity=mask_cavity,
        mask_in_dist=mask_in_dist,
    )

    # 2. Run WITHOUT LayerNorm
    metrics_no_ln, bald_no_ln, preds_no_ln = run_benchmark_for_setting(
        use_layer_norm=False,
        X_train=X_train,
        y_train=y_train,
        X_grid=X_grid,
        mask_cavity=mask_cavity,
        mask_in_dist=mask_in_dist,
    )

    # Print Comparative Table
    print("\n" + "="*80)
    print(" 2D CIRCULAR REGRESSION BENCHMARK: LAYERNORM ABLATION RESULTS")
    print("="*80)
    print(f"{'Method':<25} | {'Metric':<18} | {'With LayerNorm':<16} | {'Without LayerNorm':<18}")
    print("-" * 80)
    model_order = ['Deep Ensembles (NLL)', 'BBB (Variational)', 'BootDQN (Standard)', 'BootDQN + Priors', 'DP-BNN (Ours)']
    for m in model_order:
        c_ln = metrics_with_ln[m]['cavity_bald']
        id_ln = metrics_with_ln[m]['id_bald']
        r_ln = metrics_with_ln[m]['contrast_ratio']

        c_no = metrics_no_ln[m]['cavity_bald']
        id_no = metrics_no_ln[m]['id_bald']
        r_no = metrics_no_ln[m]['contrast_ratio']

        print(f"{m:<25} | Cavity BALD (nats) | {c_ln:<16.4f} | {c_no:<18.4f}")
        print(f"{'':<25} | In-Dist BALD (nats)| {id_ln:<16.4f} | {id_no:<18.4f}")
        print(f"{'':<25} | Contrast Ratio     | {r_ln:<16.2f}x| {r_no:<18.2f}x")
        print("-" * 80)

    # Save summary JSON
    summary = {
        "dataset": "2D Annulus Manifold r in [1.3, 2.45], Cavity r < 1.0",
        "sigma_aleatoric": SIGMA_ALEATORIC,
        "with_layernorm": metrics_with_ln,
        "without_layernorm": metrics_no_ln,
    }
    json_path = os.path.join(RESULTS_DIR, "2d_circular_layernorm_ablation_results.json")
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Summary JSON saved to {json_path}")

    # -------------------------------------------------------------
    # Render Comprehensive Side-by-Side Publication Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    fig.patch.set_facecolor('white')

    models_to_plot = [
        ('Deep Ensembles (NLL)', 'Deep Ensembles'),
        ('BBB (Variational)', 'BBB (Variational)'),
        ('BootDQN + Priors', 'BootDQN + Priors'),
        ('DP-BNN (Ours)', 'DP-BNN (Ours: Measure Prior)')
    ]

    # Row 0 & 1: WITH LayerNorm
    # Panel 0,0: Geometry
    ax = axes[0, 0]
    Z_true = ground_truth_2d(XX, YY)
    ax.contourf(XX, YY, Z_true, levels=25, cmap='coolwarm', alpha=0.6)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=20, edgecolors='black', lw=0.5)
    c_void = plt.Circle((0, 0), 1.0, color='yellow', fill=False, linestyle='--', lw=2.2, label='Cavity ($r < 1.0$)')
    c_ann = plt.Circle((0, 0), 1.3, color='black', fill=False, linestyle=':', lw=1.5, label='Annulus ($r \\in [1.3, 2.45]$)')
    ax.add_patch(c_void); ax.add_patch(c_ann)
    ax.set_title("A. Geometry & Data Support", fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='lower left', fontsize=8)

    # Heatmaps for With LayerNorm
    coords_dict_with_ln = [
        (0, 1, 'Deep Ensembles (NLL)', 'B. Deep Ensembles [With LN]'),
        (0, 2, 'BBB (Variational)', 'C. BBB Variational [With LN]'),
        (1, 0, 'BootDQN + Priors', 'D. BootDQN + Priors [With LN]'),
        (1, 1, 'DP-BNN (Ours)', 'E. DP-BNN (Ours) [With LN]'),
    ]

    for r, c, mkey, title in coords_dict_with_ln:
        ax = axes[r, c]
        b_map = bald_with_ln[mkey].reshape(G, G)
        im = ax.pcolormesh(XX, YY, b_map, cmap='plasma', vmin=0.0, vmax=1.8, shading='auto')
        c_void = plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.0)
        ax.add_patch(c_void)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('equal')

    # Panel 1,2: Bar chart comparison WITH LayerNorm
    ax_bar_ln = axes[1, 2]
    cavs_ln = [metrics_with_ln[m]['cavity_bald'] for m in model_order]
    ids_ln = [metrics_with_ln[m]['id_bald'] for m in model_order]
    x_idx = np.arange(len(model_order))
    ax_bar_ln.bar(x_idx - 0.15, cavs_ln, width=0.3, color='#1f77b4', edgecolor='black', label='Cavity Void ($r < 1.0$)')
    ax_bar_ln.bar(x_idx + 0.15, ids_ln, width=0.3, color='#aec7e8', edgecolor='black', label='In-Dist Annulus')
    ax_bar_ln.set_xticks(x_idx)
    ax_bar_ln.set_xticklabels(['Ensembles', 'BBB', 'BootDQN', 'Boot+Prior', 'DP-BNN'], fontsize=8.5)
    ax_bar_ln.set_ylabel('BALD (nats)', fontsize=10, fontweight='bold')
    ax_bar_ln.set_title('F. BALD Comparison [With LayerNorm]', fontsize=11, fontweight='bold')
    ax_bar_ln.legend(loc='upper left', fontsize=8)
    ax_bar_ln.grid(True, linestyle='--', alpha=0.4)

    # Row 2 & 3: WITHOUT LayerNorm
    # Panel 2,0: Geometry copy for visual reference
    ax = axes[2, 0]
    ax.contourf(XX, YY, Z_true, levels=25, cmap='coolwarm', alpha=0.6)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=20, edgecolors='black', lw=0.5)
    c_void = plt.Circle((0, 0), 1.0, color='yellow', fill=False, linestyle='--', lw=2.2)
    c_ann = plt.Circle((0, 0), 1.3, color='black', fill=False, linestyle=':', lw=1.5)
    ax.add_patch(c_void); ax.add_patch(c_ann)
    ax.set_title("G. Geometry [Without LN Reference]", fontsize=11, fontweight='bold')
    ax.set_aspect('equal')

    coords_dict_no_ln = [
        (2, 1, 'Deep Ensembles (NLL)', 'H. Deep Ensembles [No LayerNorm]'),
        (2, 2, 'BBB (Variational)', 'I. BBB Variational [No LayerNorm]'),
        (3, 0, 'BootDQN + Priors', 'J. BootDQN + Priors [No LayerNorm]'),
        (3, 1, 'DP-BNN (Ours)', 'K. DP-BNN (Ours) [No LayerNorm]'),
    ]

    for r, c, mkey, title in coords_dict_no_ln:
        ax = axes[r, c]
        b_map = bald_no_ln[mkey].reshape(G, G)
        im = ax.pcolormesh(XX, YY, b_map, cmap='plasma', vmin=0.0, vmax=1.8, shading='auto')
        c_void = plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.0)
        ax.add_patch(c_void)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('equal')

    # Panel 3,2: Bar chart comparison WITHOUT LayerNorm
    ax_bar_no = axes[3, 2]
    cavs_no = [metrics_no_ln[m]['cavity_bald'] for m in model_order]
    ids_no = [metrics_no_ln[m]['id_bald'] for m in model_order]
    ax_bar_no.bar(x_idx - 0.15, cavs_no, width=0.3, color='#d62728', edgecolor='black', label='Cavity Void ($r < 1.0$)')
    ax_bar_no.bar(x_idx + 0.15, ids_no, width=0.3, color='#ff9896', edgecolor='black', label='In-Dist Annulus')
    ax_bar_no.set_xticks(x_idx)
    ax_bar_no.set_xticklabels(['Ensembles', 'BBB', 'BootDQN', 'Boot+Prior', 'DP-BNN'], fontsize=8.5)
    ax_bar_no.set_ylabel('BALD (nats)', fontsize=10, fontweight='bold')
    ax_bar_no.set_title('L. BALD Comparison [No LayerNorm]', fontsize=11, fontweight='bold')
    ax_bar_no.legend(loc='upper left', fontsize=8)
    ax_bar_no.grid(True, linestyle='--', alpha=0.4)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('BALD Epistemic Uncertainty (nats)', fontsize=12, fontweight='bold')

    plt.suptitle("LayerNorm Ablation Study on 2D Circular/Annular Manifold Benchmark\nTop Half: With LayerNorm | Bottom Half: Without LayerNorm",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 0.91, 0.98])

    out_fig = os.path.join(ARTIFACT_DIR, "2d_circular_layernorm_ablation.png")
    fig.savefig(out_fig, dpi=300)
    fig.savefig(os.path.join(RESULTS_DIR, "2d_circular_layernorm_ablation.png"), dpi=300)
    print(f"Saved publication figure to {out_fig}")


if __name__ == '__main__':
    main()
