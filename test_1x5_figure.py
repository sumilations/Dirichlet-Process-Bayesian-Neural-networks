import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SIGMA_ALEATORIC = 0.30
RESULTS_DIR = '/Users/sumitvashishtha/Desktop/RLC_2026-3'
BRAIN_DIR = '/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86'

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

class MCDropout2D_NoLN(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=1, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def main():
    print("Training models (No BBB, Strictly Without LayerNorm)...")
    N_data = 260
    X_train, y_train = generate_2d_annulus_data(N_data, seed=42)

    G = 65
    coords = np.linspace(-2.5, 2.5, G)
    XX, YY = np.meshgrid(coords, coords)
    X_grid = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)
    R_grid = np.linalg.norm(X_grid, axis=1)

    mask_cavity = (R_grid <= 1.0)
    mask_in_dist = (R_grid >= 1.3) & (R_grid <= 2.45)

    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    t_X_grid = torch.tensor(X_grid, dtype=torch.float32)

    K = 10
    preds_dict = {}

    # 1. Deep Ensembles
    print("1/4: Deep Ensembles...")
    ens_preds = []
    for k in range(K):
        torch.manual_seed(100 + k)
        model = DeepEnsemble2D_NoLN()
        opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(400):
            opt.zero_grad()
            mu, var = model(t_X_train)
            loss = torch.mean(0.5 * torch.log(var) + 0.5 * (t_y_train - mu)**2 / var)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mu_grid, _ = model(t_X_grid)
            ens_preds.append(mu_grid.squeeze().numpy())
    preds_dict['Deep Ensembles (Gaussian NLL)'] = np.array(ens_preds)

    # 2. BootDQN + Priors
    print("2/4: BootDQN + Randomized Priors...")
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
            residual_y = t_y_boot - beta_scale * prior_net(t_X_boot)
        for _ in range(400):
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
    preds_dict['BootDQN + Rand Priors'] = np.array(rp_preds)

    # 3. MC Dropout (p=0.10) for Bar Chart
    print("3/4: MC Dropout (p=0.10)...")
    torch.manual_seed(500)
    mc_net = MCDropout2D_NoLN(dropout_p=0.10)
    mc_opt = optim.Adam(mc_net.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(400):
        mc_opt.zero_grad()
        pred = mc_net(t_X_train)
        loss = nn.MSELoss()(pred, t_y_train)
        loss.backward()
        mc_opt.step()
    mc_preds = []
    mc_net.train()
    with torch.no_grad():
        for _ in range(K):
            mc_preds.append(mc_net(t_X_grid).squeeze().numpy())
    preds_dict['MC Dropout (p=0.10)'] = np.array(mc_preds)

    # 4. DP-BNN (Ours)
    print("4/4: DP-BNN (Ours)...")
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
        for _ in range(400):
            dp_opt.zero_grad()
            pred = dp_net(t_all_X)
            loss = torch.mean(t_weights * (pred - t_all_y)**2)
            loss.backward()
            dp_opt.step()
        dp_net.eval()
        with torch.no_grad():
            dp_preds.append(dp_net(t_X_grid).squeeze().numpy())
    preds_dict['DP-BNN (Ours)'] = np.array(dp_preds)

    # Compute BALD maps & metrics
    all_models = [
        'Deep Ensembles (Gaussian NLL)',
        'BootDQN + Rand Priors',
        'MC Dropout (p=0.10)',
        'DP-BNN (Ours)'
    ]
    bald_maps = {}
    metrics_2d = {}
    for m in all_models:
        p = preds_dict[m]
        epistemic_var = np.var(p, axis=0)
        bald = 0.5 * np.log(1.0 + epistemic_var / (SIGMA_ALEATORIC**2))
        bald_maps[m] = bald.reshape(G, G)
        cav_b = float(np.mean(bald[mask_cavity]))
        id_b = float(np.mean(bald[mask_in_dist]))
        contrast = cav_b / (id_b + 1e-6)
        metrics_2d[m] = {'cavity_bald': cav_b, 'id_bald': id_b, 'contrast_ratio': contrast}
        print(f"{m:<30}: Cavity={cav_b:.2f} nats | In-Dist={id_b:.2f} nats | Ratio={contrast:.1f}x")

    # ------------------------------------------------------------------
    # Render 1x5 Horizontal Publication Figure with Colorbar under heatmaps
    # A: Data & Geometry
    # B: Deep Ensembles Heatmap
    # C: BootDQN + Priors Heatmap
    # D: DP-BNN (Ours) Heatmap
    # E: Quantitative Bar Chart (with MC Dropout)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18.8, 4.3), dpi=300)
    fig.patch.set_facecolor('white')

    # 2 rows: row 0 for main plots, row 1 for slim horizontal colorbar under B,C,D
    gs = GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1.3], height_ratios=[1, 0.05],
                  wspace=0.32, hspace=0.28, left=0.04, right=0.97, top=0.90, bottom=0.12)

    # Panel A: Data & Geometry
    ax_a = fig.add_subplot(gs[0, 0])
    Z_true = ground_truth_2d(XX, YY)
    ax_a.contourf(XX, YY, Z_true, levels=25, cmap='coolwarm', alpha=0.55)
    ax_a.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=16, edgecolors='black', lw=0.5,
                 label=r'Data ($\sigma_\epsilon=0.30$)')
    ax_a.add_patch(plt.Circle((0, 0), 1.0, color='yellow', fill=False, linestyle='--', lw=2.0, label=r'Cavity ($r \leq 1.0$)'))
    ax_a.add_patch(plt.Circle((0, 0), 1.3, color='black', fill=False, linestyle=':', lw=1.4, label=r'Annulus ($1.3 \leq r \leq 2.45$)'))
    ax_a.add_patch(plt.Circle((0, 0), 2.45, color='black', fill=False, linestyle=':', lw=1.4))
    ax_a.set_title("A. Annulus & Cavity Geometry", fontsize=10.5, fontweight='bold', pad=8)
    ax_a.set_xlim(-2.5, 2.5); ax_a.set_ylim(-2.5, 2.5)
    ax_a.set_xlabel(r"$x_1$", fontsize=9.5); ax_a.set_ylabel(r"$x_2$", fontsize=9.5)
    ax_a.legend(loc='lower left', fontsize=7.2, framealpha=0.92)
    ax_a.set_aspect('equal')
    ax_a.tick_params(labelsize=8.5)

    # Panels B, C, D: Heatmaps
    heatmap_configs = [
        ("B. Deep Ensembles (Gaussian NLL)", 'Deep Ensembles (Gaussian NLL)', gs[0, 1], '#E66101'),
        ("C. BootDQN + Rand Priors", 'BootDQN + Rand Priors', gs[0, 2], '#B27B00'),
        ("D. DP-BNN (Ours)", 'DP-BNN (Ours)', gs[0, 3], '#B2182B')
    ]

    im_shared = None
    for title, mkey, loc, col in heatmap_configs:
        ax = fig.add_subplot(loc)
        im_shared = ax.imshow(
            bald_maps[mkey],
            extent=[-2.5, 2.5, -2.5, 2.5],
            origin='lower',
            cmap='plasma',
            vmin=0.0,
            vmax=1.8,
            interpolation='bicubic'
        )
        ax.add_patch(plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.0))
        ax.add_patch(plt.Circle((0, 0), 1.3, color='white', fill=False, linestyle=':', lw=1.3, alpha=0.8))
        ax.add_patch(plt.Circle((0, 0), 2.45, color='white', fill=False, linestyle=':', lw=1.3, alpha=0.8))
        ax.set_title(title, fontsize=10.5, fontweight='bold', color=col, pad=8)
        ax.set_xlabel(r"$x_1$", fontsize=9.5); ax.set_ylabel(r"$x_2$", fontsize=9.5)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8.5)

    # Horizontal Colorbar spanning under B, C, D: gs[1, 1:4]
    cbar_ax = fig.add_subplot(gs[1, 1:4])
    cbar = fig.colorbar(im_shared, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("BALD Epistemic Uncertainty (nats)", fontsize=9.0, fontweight='bold')
    cbar.ax.tick_params(labelsize=8.0)

    # Panel E: Quantitative Bar Chart in gs[0, 4]
    ax_bar = fig.add_subplot(gs[0, 4])
    methods_bar = ['Deep Ensembles', 'BootDQN\n+ Priors', 'MC Dropout', 'DP-BNN\n(Ours)']
    x_pos = np.arange(len(methods_bar))
    w = 0.35

    bar_model_keys = ['Deep Ensembles (Gaussian NLL)', 'BootDQN + Rand Priors', 'MC Dropout (p=0.10)', 'DP-BNN (Ours)']
    cav_vals = [metrics_2d[m]['cavity_bald'] for m in bar_model_keys]
    id_vals = [metrics_2d[m]['id_bald'] for m in bar_model_keys]

    r1 = ax_bar.bar(x_pos - w/2, cav_vals, w, label=r'Cavity Void ($r \leq 1.0$)', color='#1b7837', edgecolor='black', alpha=0.9, zorder=3)
    r2 = ax_bar.bar(x_pos + w/2, id_vals, w, label=r'Data Annulus ($1.3 \leq r \leq 2.45$)', color='#a6dba0', edgecolor='black', alpha=0.9, zorder=3)

    # Highlight DP-BNN
    r1[3].set_color('#b2182b')
    r2[3].set_color('#fddbc7')

    # Annotate exact numbers for BOTH bars
    for rect in r1:
        h = rect.get_height()
        ax_bar.annotate(f'{h:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.0, fontweight='bold',
                        color='#990000' if rect == r1[3] else '#1b7837')

    for rect in r2:
        h = rect.get_height()
        ax_bar.annotate(f'{h:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.0, fontweight='bold',
                        color='#b2182b' if rect == r2[3] else '#2b83ba')

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(methods_bar, fontsize=8.0, fontweight='bold')
    ax_bar.set_ylabel("BALD Epistemic (nats)", fontsize=9.0, fontweight='bold')
    ax_bar.set_title("E. Cavity vs. Data Support", fontsize=10.5, fontweight='bold', pad=8)
    ax_bar.set_ylim(0, 1.70)
    ax_bar.grid(axis='y', alpha=0.25, linestyle=':')
    ax_bar.legend(loc='upper left', fontsize=7.5, framealpha=0.95)

    out_paths = [
        os.path.join(RESULTS_DIR, 'Figure_1_2D_regression_heatmaps_NoLN.pdf'),
        os.path.join(RESULTS_DIR, 'Figure_1_2D_regression_heatmaps_NoLN.png'),
        os.path.join(BRAIN_DIR, 'Figure_1_2D_regression_heatmaps_NoLN.pdf'),
        os.path.join(BRAIN_DIR, 'Figure_1_2D_regression_heatmaps_NoLN.png'),
    ]
    for p in out_paths:
        fig.savefig(p, bbox_inches='tight')
        print(f"Saved: {p}")
    plt.close(fig)

if __name__ == '__main__':
    main()
