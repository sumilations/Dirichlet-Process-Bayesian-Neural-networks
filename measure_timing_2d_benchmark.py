#!/usr/bin/env python3
"""Measure exact execution and inference time for each method in the 2D benchmark
with and without LayerNorm.
"""

import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from run_2d_circular_regression_layernorm_ablation import (
    generate_2d_annulus_data, DeepEnsemble2D, VBNet2D, MLP2D, SIGMA_ALEATORIC
)

N_data = 260
X_train, y_train = generate_2d_annulus_data(N_data, seed=42)

G = 65
coords = np.linspace(-2.5, 2.5, G)
XX, YY = np.meshgrid(coords, coords)
X_grid = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)

t_X_train = torch.tensor(X_train, dtype=torch.float32)
t_y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
t_X_grid = torch.tensor(X_grid, dtype=torch.float32)

K = 10
timings = {"with_ln": {}, "no_ln": {}}

for use_ln in [True, False]:
    key = "with_ln" if use_ln else "no_ln"
    
    # 1. Deep Ensembles
    t0 = time.time()
    for k in range(K):
        torch.manual_seed(100 + k)
        net = DeepEnsemble2D(use_layer_norm=use_ln)
        opt = optim.Adam(net.parameters(), lr=0.008, weight_decay=1e-4)
        for _ in range(400):
            opt.zero_grad()
            mu, var = net(t_X_train)
            loss = 0.5 * torch.mean(torch.log(var) + (t_y_train - mu)**2 / var)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            _ = net(t_X_grid)
    timings[key]["Deep Ensembles (NLL)"] = time.time() - t0

    # 2. BBB
    t0 = time.time()
    torch.manual_seed(200)
    bbb_net = VBNet2D(use_layer_norm=use_ln)
    bbb_opt = optim.Adam(bbb_net.parameters(), lr=0.01)
    for _ in range(500):
        bbb_opt.zero_grad()
        pred = bbb_net(t_X_train, sample=True)
        mse = nn.MSELoss()(pred, t_y_train)
        kl = bbb_net.kl() / (N_data * 50.0)
        (mse + kl).backward()
        bbb_opt.step()
    bbb_net.eval()
    with torch.no_grad():
        for _ in range(K):
            _ = bbb_net(t_X_grid, sample=True)
    timings[key]["BBB (Variational)"] = time.time() - t0

    # 3. BootDQN
    t0 = time.time()
    for k in range(K):
        torch.manual_seed(300 + k)
        net = MLP2D(use_layer_norm=use_ln)
        opt = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
        boot_idx = np.random.choice(N_data, size=N_data, replace=True)
        t_X_boot = t_X_train[boot_idx]
        t_y_boot = t_y_train[boot_idx]
        for _ in range(400):
            opt.zero_grad()
            pred = net(t_X_boot)
            loss = nn.MSELoss()(pred, t_y_boot)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            _ = net(t_X_grid)
    timings[key]["BootDQN (Standard)"] = time.time() - t0

    # 4. BootDQN + Priors
    t0 = time.time()
    beta_scale = 2.0
    for k in range(K):
        torch.manual_seed(350 + k)
        train_net = MLP2D(use_layer_norm=use_ln)
        prior_net = MLP2D(use_layer_norm=use_ln)
        for p in prior_net.parameters():
            p.requires_grad = False
        opt = optim.Adam(train_net.parameters(), lr=0.01, weight_decay=1e-4)
        boot_idx = np.random.choice(N_data, size=N_data, replace=True)
        t_X_boot = t_X_train[boot_idx]
        t_y_boot = t_y_train[boot_idx]
        with torch.no_grad():
            prior_target = prior_net(t_X_boot)
            residual_y = t_y_boot - beta_scale * prior_target
        for _ in range(400):
            opt.zero_grad()
            pred = train_net(t_X_boot)
            loss = nn.MSELoss()(pred, residual_y)
            loss.backward()
            opt.step()
        train_net.eval()
        prior_net.eval()
        with torch.no_grad():
            _ = train_net(t_X_grid) + beta_scale * prior_net(t_X_grid)
    timings[key]["BootDQN + Priors"] = time.time() - t0

    # 5. DP-BNN
    t0 = time.time()
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
        dp_net = MLP2D(use_layer_norm=use_ln)
        dp_opt = optim.Adam(dp_net.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(400):
            dp_opt.zero_grad()
            pred = dp_net(t_all_X)
            loss = torch.mean(t_weights * (pred - t_all_y)**2)
            loss.backward()
            dp_opt.step()
        dp_net.eval()
        with torch.no_grad():
            _ = dp_net(t_X_grid)
    timings[key]["DP-BNN (Ours)"] = time.time() - t0

print(json.dumps(timings, indent=2))
with open("results/timings_2d_benchmark.json", "w") as fp:
    json.dump(timings, fp, indent=2)
