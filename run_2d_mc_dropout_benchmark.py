#!/usr/bin/env python3
"""
MC Dropout for 2D Annular Manifold Regression:
Evaluates Monte Carlo Dropout (Gal & Ghahramani, ICML 2016) on the 2D circular regression task.
Tests dropout rates p in [0.05, 0.10, 0.20], both With LayerNorm and Without LayerNorm.
Computes BALD (nats) in the cavity void (r <= 1.0) vs. in-distribution annulus (r in [1.4, 2.3]).
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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cpu')

SIGMA_ALEATORIC = 0.30
RESULTS_DIR = './results'
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


class MCDropout2D(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=1, dropout_p=0.1, use_layer_norm=True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.dropout_p = dropout_p
        if use_layer_norm:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden, out_dim)
            )
        else:
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

    def sample_predictions(self, x, num_samples=50):
        self.train()  # Keep dropout active during evaluation
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(self.forward(x).squeeze().numpy())
        return np.array(preds)


def run_mc_dropout(X_train, y_train, X_grid, mask_cavity, mask_in_dist, dropout_p=0.1, use_layer_norm=True, epochs=400, lr=0.01, K_samples=50):
    t_X = torch.tensor(X_train, dtype=torch.float32)
    t_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    t_grid = torch.tensor(X_grid, dtype=torch.float32)

    model = MCDropout2D(in_dim=2, hidden=64, out_dim=1, dropout_p=dropout_p, use_layer_norm=use_layer_norm)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(t_X), t_y)
        loss.backward()
        optimizer.step()

    # Draw K stochastic forward passes
    preds = model.sample_predictions(t_grid, num_samples=K_samples)
    epistemic_var = np.var(preds, axis=0)
    bald = 0.5 * np.log(1.0 + epistemic_var / (SIGMA_ALEATORIC**2))

    cavity_bald = float(np.mean(bald[mask_cavity]))
    id_bald = float(np.mean(bald[mask_in_dist]))
    contrast_ratio = cavity_bald / (id_bald + 1e-6)

    return {
        'dropout_p': dropout_p,
        'use_layer_norm': use_layer_norm,
        'cavity_bald': round(cavity_bald, 4),
        'id_bald': round(id_bald, 4),
        'contrast_ratio': round(contrast_ratio, 2),
        'bald_map': bald,
        'preds': preds
    }


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

    results = {}
    dropout_rates = [0.05, 0.10, 0.20]

    print("================================================================")
    print(" Running MC Dropout on 2D Annular Manifold Regression")
    print("================================================================")

    for ln in [True, False]:
        ln_str = "With LN" if ln else "No LN"
        results[ln_str] = {}
        for p in dropout_rates:
            key = f"MC Dropout (p={p})"
            res = run_mc_dropout(X_train, y_train, X_grid, mask_cavity, mask_in_dist, dropout_p=p, use_layer_norm=ln)
            results[ln_str][key] = res
            print(f"[{ln_str}] p={p:4.2f} -> Cavity BALD: {res['cavity_bald']:.4f} nats | ID BALD: {res['id_bald']:.4f} nats | Contrast Ratio: {res['contrast_ratio']:.2f}x")

    # Load existing baseline metrics from results/2d_circular_layernorm_ablation_results.json
    with open("results/2d_circular_layernorm_ablation_results.json", "r") as fp:
        baseline_data = json.load(fp)

    print("\n================================================================")
    print(" Complete Multi-Model Comparison Table (With LayerNorm)")
    print("================================================================")
    print(f"{'Method':<28} | {'Cavity BALD':<12} | {'ID BALD':<10} | {'Contrast Ratio':<14}")
    print("-" * 72)
    for m, vals in baseline_data['with_layernorm'].items():
        print(f"{m:<28} | {vals['cavity_bald']:<12.4f} | {vals['id_bald']:<10.4f} | {vals['contrast_ratio']:<14.2f}")
    for p in dropout_rates:
        k = f"MC Dropout (p={p})"
        vals = results["With LN"][k]
        print(f"{k:<28} | {vals['cavity_bald']:<12.4f} | {vals['id_bald']:<10.4f} | {vals['contrast_ratio']:<14.2f}")

    print("\n================================================================")
    print(" Complete Multi-Model Comparison Table (Without LayerNorm)")
    print("================================================================")
    print(f"{'Method':<28} | {'Cavity BALD':<12} | {'ID BALD':<10} | {'Contrast Ratio':<14}")
    print("-" * 72)
    for m, vals in baseline_data['without_layernorm'].items():
        print(f"{m:<28} | {vals['cavity_bald']:<12.4f} | {vals['id_bald']:<10.4f} | {vals['contrast_ratio']:<14.2f}")
    for p in dropout_rates:
        k = f"MC Dropout (p={p})"
        vals = results["No LN"][k]
        print(f"{k:<28} | {vals['cavity_bald']:<12.4f} | {vals['id_bald']:<10.4f} | {vals['contrast_ratio']:<14.2f}")

    # Plot MC Dropout Spatial Uncertainty Heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#ffffff')

    col_idx = 0
    for p in dropout_rates:
        k = f"MC Dropout (p={p})"
        # Row 0: With LayerNorm
        ax = axes[0, col_idx]
        b_map = results["With LN"][k]['bald_map'].reshape(G, G)
        im = ax.pcolormesh(XX, YY, b_map, cmap='plasma', vmin=0.0, vmax=1.8, shading='auto')
        c_void = plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.0)
        c_ann = plt.Circle((0, 0), 1.3, color='black', fill=False, linestyle=':', lw=1.5)
        ax.add_patch(c_void); ax.add_patch(c_ann)
        ax.set_title(f"MC Dropout (p={p}) [With LN]\nContrast: {results['With LN'][k]['contrast_ratio']}x", fontsize=11, fontweight='bold')
        ax.set_aspect('equal')

        # Row 1: No LayerNorm
        ax = axes[1, col_idx]
        b_map_no = results["No LN"][k]['bald_map'].reshape(G, G)
        im = ax.pcolormesh(XX, YY, b_map_no, cmap='plasma', vmin=0.0, vmax=1.8, shading='auto')
        c_void = plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.0)
        c_ann = plt.Circle((0, 0), 1.3, color='black', fill=False, linestyle=':', lw=1.5)
        ax.add_patch(c_void); ax.add_patch(c_ann)
        ax.set_title(f"MC Dropout (p={p}) [No LN]\nContrast: {results['No LN'][k]['contrast_ratio']}x", fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        col_idx += 1

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('BALD Epistemic Uncertainty (nats)', fontsize=12, fontweight='bold')

    plt.suptitle("Monte Carlo Dropout (Gal & Ghahramani 2016) on 2D Annular Manifold\nRow 1: With LayerNorm | Row 2: Without LayerNorm",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 0.91, 0.96])

    out_png = os.path.join(BRAIN_DIR, "mc_dropout_2d_regression_heatmap.png")
    fig.savefig(out_png, dpi=200)
    fig.savefig(os.path.join(RESULTS_DIR, "mc_dropout_2d_regression_heatmap.png"), dpi=200)
    print(f"\nSaved MC Dropout heatmap plot to: {out_png}")

    # Save summary JSON
    summary_save = {
        "with_layernorm": {k: {"cavity_bald": v["cavity_bald"], "id_bald": v["id_bald"], "contrast_ratio": v["contrast_ratio"]} for k, v in results["With LN"].items()},
        "without_layernorm": {k: {"cavity_bald": v["cavity_bald"], "id_bald": v["id_bald"], "contrast_ratio": v["contrast_ratio"]} for k, v in results["No LN"].items()},
    }
    with open(os.path.join(RESULTS_DIR, "mc_dropout_2d_metrics.json"), "w") as fp:
        json.dump(summary_save, fp, indent=2)
    with open(os.path.join(BRAIN_DIR, "mc_dropout_2d_metrics.json"), "w") as fp:
        json.dump(summary_save, fp, indent=2)


if __name__ == '__main__':
    main()
