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
    print("Training 5 models on 2D annular manifold without LayerNorm...")
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

    # 1. Deep Ensembles (Gaussian NLL)
    print("1/5: Training Deep Ensembles (Gaussian NLL)...")
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

    # 2. BBB (Variational)
    print("2/5: Training Variational BNN (BBB)...")
    torch.manual_seed(200)
    bbb_net = VBNet2D_NoLN()
    bbb_opt = optim.Adam(bbb_net.parameters(), lr=0.01)
    for epoch in range(500):
        bbb_opt.zero_grad()
        out = bbb_net(t_X_train, sample=True)
        nll = nn.MSELoss()(out, t_y_train) / (2 * (SIGMA_ALEATORIC**2))
        kl = bbb_net.kl() / N_data
        loss = nll + 0.01 * kl
        loss.backward()
        bbb_opt.step()
    bbb_preds = []
    with torch.no_grad():
        for _ in range(K):
            bbb_preds.append(bbb_net(t_X_grid, sample=True).squeeze().numpy())
    preds_dict['BBB (Variational)'] = np.array(bbb_preds)

    # 3. BootDQN + Priors
    print("3/5: Training BootDQN + Randomized Priors...")
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

    # 4. MC Dropout (p=0.10)
    print("4/5: Training MC Dropout (p=0.10)...")
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
    mc_net.train() # Keep dropout active during evaluation
    with torch.no_grad():
        for _ in range(K):
            mc_preds.append(mc_net(t_X_grid).squeeze().numpy())
    preds_dict['MC Dropout (p=0.10)'] = np.array(mc_preds)

    # 5. DP-BNN (Ours: Measure Prior)
    print("5/5: Training DP-BNN (Ours: Measure Prior)...")
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

    # Compute BALD maps
    model_order = [
        'Deep Ensembles (Gaussian NLL)',
        'BBB (Variational)',
        'BootDQN + Rand Priors',
        'MC Dropout (p=0.10)',
        'DP-BNN (Ours)'
    ]
    bald_maps = {}
    metrics_2d = {}
    for m in model_order:
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
    # Render Master 2x3 Figure with Clean Colorbar Placement
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16.8, 9.8), dpi=300)
    fig.patch.set_facecolor('white')

    # GridSpec with 2 rows, 3 columns for subplots, and 1 dedicated column for colorbar
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.28, hspace=0.28,
                  left=0.05, right=0.96, top=0.92, bottom=0.06)

    axes_list = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1])
    ]

    panel_titles = [
        ("A. Deep Ensembles (Gaussian NLL)", '#E66101'),
        ("B. Variational BNN (BBB)", '#5E3C99'),
        ("C. BootDQN + Randomized Priors", '#B27B00'),
        ("D. MC Dropout (p=0.10)", '#2B83BA'),
        ("E. DP-BNN (Ours: Measure Prior)", '#B2182B')
    ]

    im_shared = None
    for idx, (m, ax) in enumerate(zip(model_order, axes_list)):
        im_shared = ax.imshow(
            bald_maps[m],
            extent=[-2.5, 2.5, -2.5, 2.5],
            origin='lower',
            cmap='plasma',
            vmin=0.0,
            vmax=1.8,
            interpolation='bicubic'
        )
        # Overlay Cavity (cyan dashed circle) and Annulus (white dotted)
        ax.add_patch(plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.2))
        ax.add_patch(plt.Circle((0, 0), 1.3, color='white', fill=False, linestyle=':', lw=1.5, alpha=0.8))
        ax.add_patch(plt.Circle((0, 0), 2.45, color='white', fill=False, linestyle=':', lw=1.5, alpha=0.8))

        title_text, title_col = panel_titles[idx]
        ax.set_title(title_text, fontsize=11, fontweight='bold', color=title_col, pad=8)
        ax.set_xlabel(r"$x_1$", fontsize=10)
        ax.set_ylabel(r"$x_2$", fontsize=10)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=9)

    # Dedicated Colorbar Axis spanning top & bottom or top row
    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im_shared, cax=cbar_ax, orientation='vertical')
    cbar.set_label("BALD Epistemic Uncertainty (nats)", fontsize=9.5, fontweight='bold')
    cbar.ax.tick_params(labelsize=8.5)

    # Panel F: Bar Chart in gs[1, 2]
    ax_bar = fig.add_subplot(gs[1, 2])
    methods_bar = ['Deep Ensembles', 'BBB', 'BootDQN\n+ Priors', 'MC Dropout', 'DP-BNN\n(Ours)']
    x_pos = np.arange(len(methods_bar))
    w = 0.35

    cav_vals = [metrics_2d[m]['cavity_bald'] for m in model_order]
    id_vals = [metrics_2d[m]['id_bald'] for m in model_order]

    r1 = ax_bar.bar(x_pos - w/2, cav_vals, w, label=r'Cavity Void ($r \leq 1.0$)', color='#1b7837', edgecolor='black', alpha=0.9, zorder=3)
    r2 = ax_bar.bar(x_pos + w/2, id_vals, w, label=r'Observed Annulus ($1.3 \leq r \leq 2.45$)', color='#a6dba0', edgecolor='black', alpha=0.9, zorder=3)

    # Highlight DP-BNN
    r1[4].set_color('#b2182b')
    r2[4].set_color('#fddbc7')

    # Annotate exact numbers for BOTH bars
    for rect in r1:
        h = rect.get_height()
        ax_bar.annotate(f'{h:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                        color='#990000' if rect == r1[4] else '#1b7837')

    for rect in r2:
        h = rect.get_height()
        ax_bar.annotate(f'{h:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                        color='#b2182b' if rect == r2[4] else '#2b83ba')

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(methods_bar, fontsize=8.5, fontweight='bold')
    ax_bar.set_ylabel("BALD Epistemic Uncertainty (nats)", fontsize=10, fontweight='bold')
    ax_bar.set_title("F. BALD Comparison: Cavity Void vs. Data Support", fontsize=11, fontweight='bold', pad=8)
    ax_bar.set_ylim(0, 1.70)
    ax_bar.grid(axis='y', alpha=0.25, linestyle=':')
    ax_bar.legend(loc='upper left', fontsize=8.5, framealpha=0.95)

    plt.suptitle("2D Annular Manifold Regression: BALD Epistemic Uncertainty Heatmaps & Quantitative Contrast",
                 fontsize=13, fontweight='bold', y=0.98)

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
