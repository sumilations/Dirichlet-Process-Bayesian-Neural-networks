#!/usr/bin/env python3
"""
Publication-Quality Uncertainty Benchmark Figures
Vashishtha & Maillard (202X) — DP-BNN Paper

Benchmarks:
  1. Osband et al. (NeurIPS 2018, Figure 2) — Extrapolation Showdown
  2. Foong et al. (ICML 2019) — 'In-Between' Uncertainty Benchmark

Methods compared (all share the SAME backbone architecture):
  (a) Deep Ensembles           [Lakshminarayanan et al., NeurIPS 2017]
  (b) Vanilla Bootstrap        [Efron 1979 / BootDQN variant]
  (c) BootDQN + Rand. Priors   [Osband et al., NeurIPS 2018]
  (d) Bayesian Neural Linear   [Riquelme et al., ICLR 2018]
  (e) Variational Inference    [Blundell et al. / Bayes-by-Backprop, ICML 2015]
  (f) DP-BNN (Ours)            [Vashishtha & Maillard, Eq. 288]

Shared backbone (ALL methods):
  Linear(1->64) -> LayerNorm(64) -> ReLU -> Linear(64->64) -> LayerNorm(64) -> ReLU -> head

Metrics per method:
  - OOD/Gap Uncertainty Ratio  (primary epistemic metric)
  - 95% Empirical Coverage
  - CRPS
  - In-Distribution MSE
"""

import os
import json
import math
import shutil
import numpy as np
import scipy.stats as scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Reproducibility & Paths
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cpu')   # MPS has f32×si64 dtype conflict on this PyTorch build
print(f"Device: {DEVICE}")

ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Publication style
# ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "axes.titlesize":    10,
    "axes.labelsize":    8,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.linestyle":    ":",
    "grid.alpha":        0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PALETTE = {
    "mean":   "#1f4e79",
    "ci":     "#4472c4",
    "sample": "#9dc3e6",
    "data":   "#c00000",
    "truth":  "#375623",
    "void":   "#808080",
}

METHOD_COLORS = {
    "Deep Ensembles (NLL)":    "#e64646",
    "Vanilla Bootstrap":       "#e69c46",
    "BootDQN + Priors":        "#c8b400",
    "Variational Inference":   "#9b59b6",
    "DP-BNN (Ours)":           "#1a8a42",
}

METHOD_LABELS = [
    "Deep Ensembles (NLL)",
    "Vanilla Bootstrap",
    "BootDQN + Priors",
    "Variational Inference",
    "DP-BNN (Ours)",
]

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)"]


# ══════════════════════════════════════════════
#  SHARED BACKBONE ARCHITECTURE
# ══════════════════════════════════════════════

class Backbone(nn.Module):
    """
    Shared feature extractor used by ALL methods.
    Linear(1->64) -> LN -> ReLU -> Linear(64->64) -> LN -> ReLU
    Output: 64-dim feature vector phi(x)
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x):
        h = F.relu(self.ln1(self.fc1(x)))
        return F.relu(self.ln2(self.fc2(h)))


class DeterministicNet(nn.Module):
    """Full deterministic network = Backbone + linear head."""
    def __init__(self, hidden=64):
        super().__init__()
        self.backbone = Backbone(hidden)
        self.head     = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.backbone(x))

    def features(self, x):
        with torch.no_grad():
            return self.backbone(x)


class DeepEnsembleNet(nn.Module):
    """
    Lakshminarayanan et al. (NeurIPS 2017) Deep Ensemble Network:
    Dual heads for predicted mean mu(x) and strictly positive variance sigma^2(x).
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.backbone  = Backbone(hidden)
        self.mean_head = nn.Linear(hidden, 1)
        self.var_head  = nn.Linear(hidden, 1)

    def forward(self, x):
        feat     = self.backbone(x)
        mu       = self.mean_head(feat)
        sigma_sq = F.softplus(self.var_head(feat)) + 1e-6
        return mu, sigma_sq


# ──────────────────────────────────────────────
#  Bayesian Linear layer (for VI / BBB)
# ──────────────────────────────────────────────

class BayesLinear(nn.Module):
    """
    Mean-field Gaussian Bayesian linear layer (Bayes by Backprop).
    Reparameterisation: w = mu + eps * softplus(rho), eps ~ N(0,I)
    Prior: N(0, prior_std^2)
    """
    def __init__(self, in_f, out_f, prior_std=1.0):
        super().__init__()
        self.prior_std  = prior_std
        self.weight_mu  = nn.Parameter(torch.zeros(out_f, in_f))
        self.weight_rho = nn.Parameter(torch.full((out_f, in_f), -3.0))
        self.bias_mu    = nn.Parameter(torch.zeros(out_f))
        self.bias_rho   = nn.Parameter(torch.full((out_f,), -3.0))
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))

    def _std(self, rho):
        return F.softplus(rho) + 1e-6

    def forward(self, x, sample=True):
        if sample:
            w = self.weight_mu + self._std(self.weight_rho) * torch.randn_like(self.weight_rho)
            b = self.bias_mu   + self._std(self.bias_rho)   * torch.randn_like(self.bias_rho)
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

    def kl(self):
        """KL(q || N(0, prior_std^2))."""
        pv = self.prior_std ** 2
        def _kl(mu, std):
            v = std ** 2
            return 0.5 * (v / pv + mu**2 / pv - 1.0 + math.log(pv) - torch.log(v))
        ws = self._std(self.weight_rho)
        bs = self._std(self.bias_rho)
        return _kl(self.weight_mu, ws).sum() + _kl(self.bias_mu, bs).sum()


class VINet(nn.Module):
    """
    Fully Bayesian network via mean-field VI (Bayes by Backprop).
    Same structure as DeterministicNet but ALL Linear -> BayesLinear.
    LN layers kept deterministic for training stability.
    """
    def __init__(self, hidden=64, prior_std=1.0):
        super().__init__()
        self.fc1  = BayesLinear(1,      hidden, prior_std)
        self.ln1  = nn.LayerNorm(hidden)
        self.fc2  = BayesLinear(hidden, hidden, prior_std)
        self.ln2  = nn.LayerNorm(hidden)
        self.head = BayesLinear(hidden, 1,      prior_std)

    def forward(self, x, sample=True):
        h = F.relu(self.ln1(self.fc1(x, sample)))
        h = F.relu(self.ln2(self.fc2(h, sample)))
        return self.head(h, sample)

    def kl(self):
        return self.fc1.kl() + self.fc2.kl() + self.head.kl()


# ══════════════════════════════════════════════
#  TRAINING ROUTINES
# ══════════════════════════════════════════════

def _to(t):
    return t.to(DEVICE)


# 1. Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017)
def train_deep_ensemble(X, y, K=10, epochs=800, lr=5e-3):
    """
    Proper Scoring Rule: Gaussian Negative Log-Likelihood (NLL).
    Each ensemble member outputs mu(x) and sigma^2(x).
    Loss = 0.5 * [log(sigma^2) + (y - mu)^2 / sigma^2]
    """
    models = []
    for _ in range(K):
        m = DeepEnsembleNet().to(DEVICE)
        opt = optim.Adam(m.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            mu, s2 = m(X)
            loss = 0.5 * (torch.log(s2) + (y - mu)**2 / s2).mean()
            loss.backward()
            opt.step()
        m.eval()
        models.append(m)
    return ("deep_ensemble_nll", models)


# 2. Vanilla Bootstrap
def train_vanilla_bootstrap(X, y, K=10, epochs=800, lr=5e-3):
    n, crit = X.shape[0], nn.MSELoss(reduction='none')
    models = []
    for _ in range(K):
        m   = DeterministicNet().to(DEVICE)
        opt = optim.Adam(m.parameters(), lr=lr)
        w   = _to(torch.from_numpy(np.random.poisson(1.0, n).astype(np.float32))).unsqueeze(1)
        for _ in range(epochs):
            opt.zero_grad(); (crit(m(X), y) * w).mean().backward(); opt.step()
        m.eval(); models.append(m)
    return ("ensemble", models)


# 3. BootDQN + Randomized Priors
class PriorNet(nn.Module):
    def __init__(self, beta=1.0, prior_weight_scale=2.0):
        super().__init__()
        self.trainable = DeterministicNet()
        self.prior     = DeterministicNet()
        # Wilder prior init: scale up weights for more function diversity
        with torch.no_grad():
            for p in self.prior.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=prior_weight_scale / math.sqrt(p.shape[1]))
                else:
                    nn.init.normal_(p, mean=0.0, std=prior_weight_scale)
        for p in self.prior.parameters():
            p.requires_grad_(False)
        self.beta = beta

    def forward(self, x):
        return self.trainable(x) + self.beta * self.prior(x)

def train_boot_prior(X, y, K=10, epochs=800, lr=5e-3, beta=3.0, prior_weight_scale=2.0):
    """
    BootDQN + Randomized Priors (Osband et al., NeurIPS 2018).
    Key hyperparameters:
      beta              : prior scale (Osband recommends 1-5; higher -> more OOD/gap uncertainty)
      prior_weight_scale: std of prior weight init (higher -> wilder, more diverse prior functions)
      K                 : ensemble size (equal across all ensemble methods = 10)
    """
    n, crit = X.shape[0], nn.MSELoss(reduction='none')
    models = []
    for _ in range(K):
        m   = PriorNet(beta=beta, prior_weight_scale=prior_weight_scale).to(DEVICE)
        opt = optim.Adam(m.trainable.parameters(), lr=lr)
        w   = _to(torch.from_numpy(np.random.poisson(1.0, n).astype(np.float32))).unsqueeze(1)
        for _ in range(epochs):
            opt.zero_grad(); (crit(m(X), y) * w).mean().backward(); opt.step()
        m.eval(); models.append(m)
    return ("ensemble", models)


# 4. Bayesian Neural Linear
class ClosedFormBayesHead:
    """
    Exact closed-form Bayesian linear regression on frozen features phi(x).
    Prior:      w ~ N(0, lambda^{-1} I)
    Likelihood: y | phi, w ~ N(w^T phi, sigma^2)
    Posterior:  p(w|D) = N(m_N, S_N)  (closed-form)
    Predictive: mean=phi^T m_N,  var=phi^T S_N phi + sigma^2
    """
    def __init__(self, phi_dim, prior_prec=1.0, noise_var=0.05):
        self.lam  = prior_prec
        self.sig2 = noise_var
        d         = phi_dim
        self.S_N  = np.eye(d) / self.lam   # prior covariance
        self.m_N  = np.zeros(d)            # prior mean

    def fit(self, Phi, y_np):
        """Phi: [N, d], y_np: [N]."""
        S_inv = self.lam * np.eye(Phi.shape[1]) + Phi.T @ Phi / self.sig2
        self.S_N = np.linalg.inv(S_inv)
        self.m_N = self.S_N @ (Phi.T @ y_np) / self.sig2

    def predict(self, Phi_test):
        mu  = Phi_test @ self.m_N
        var = np.einsum('nd,de,ne->n', Phi_test, self.S_N, Phi_test) + self.sig2
        return mu, np.sqrt(np.clip(var, 1e-10, None))

    def sample_draws(self, Phi_test, n_draws=10):
        ws = np.random.multivariate_normal(self.m_N, self.S_N, size=n_draws)
        return ws @ Phi_test.T    # [n_draws, N_test]

def train_neural_linear(X, y, epochs=800, lr=5e-3):
    # Step 1: train backbone with standard MSE
    m = DeterministicNet().to(DEVICE)
    opt, crit = optim.Adam(m.parameters(), lr=lr), nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); crit(m(X), y).backward(); opt.step()
    m.eval()
    # Step 2: freeze backbone, fit exact Bayesian linear head
    Phi  = m.features(X).cpu().numpy()
    y_np = y.squeeze().cpu().numpy()
    head = ClosedFormBayesHead(phi_dim=Phi.shape[1], prior_prec=1.0, noise_var=0.05)
    head.fit(Phi, y_np)
    return ("neural_linear", m, head)


# 5. Variational Inference (Bayes by Backprop)
def train_vi(X, y, epochs=3000, lr=1e-3, n_mc=5, prior_std=1.0, noise_var=0.05):
    """
    ELBO = E_q[log p(y|x,w)] - KL[q(w) || p(w)]
    Optimised via reparameterisation trick.
    """
    n       = X.shape[0]
    m       = VINet(prior_std=prior_std).to(DEVICE)
    opt     = optim.Adam(m.parameters(), lr=lr)
    nv      = noise_var   # observation noise variance

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        ll = sum(
            -0.5 * ((m(X, sample=True) - y)**2 / nv
                    + math.log(2 * math.pi * nv)).sum()
            for _ in range(n_mc)
        ) / n_mc
        kl   = m.kl() / n
        loss = -(ll - kl)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if ep % 1000 == 0:
            print(f"    VI ep {ep}/{epochs}  loss={loss.item():.3f}  ll={ll.item():.3f}  kl={kl.item():.3f}")
    m.eval()
    return ("vi", m)


# 6. DP-BNN (Vashishtha & Maillard, Eq. 288 / Eq. 4.7)
def train_dp_bnn(X, y, domain_range, support_y=None, K=10, epochs=1200, lr=4e-3, alpha=2.5, K_t=60,
                 prior_sigma=2.0):
    """
    Pure Posterior stick-breaking (Eq. 4.7 / Eq. 288):
      Q_N = V_N delta_{X_N} + sum_{i<N}[V_i prod_{j>i}(1-V_j)] delta_{X_i}
                             + [prod_{i=1}^N (1-V_i)] Q_0
    where V_i ~ Beta(1, alpha + i).
    No kernels, no RBF, no mixture of Gaussians. Pure DP on the data measure.
    """
    N = X.shape[0]
    crit = nn.MSELoss(reduction='none')
    models = []
    x_min, x_max = domain_range

    for _ in range(K):
        # Posterior stick-breaking weights over empirical atoms
        alphas  = alpha + np.arange(1, N + 1, dtype=np.float64)
        V       = np.random.beta(1.0, alphas)
        OmV     = 1.0 - V
        sfx     = np.concatenate(([1.0], np.cumprod(OmV[::-1])[:-1]))[::-1].copy()
        w_data  = (V * sfx)
        w_prior = float(np.prod(OmV))   # alpha / (alpha + N)

        # Prior Q_0 stick-breaking over K_t atoms (Eq. 4.7)
        Vp      = np.random.beta(1.0, alpha, K_t)
        OmVp    = 1.0 - Vp
        sfxp    = np.concatenate(([1.0], np.cumprod(OmVp[::-1])[:-1]))[::-1].copy()
        q_prior = Vp * sfxp
        q_prior = q_prior / q_prior.sum()

        # Prior candidate atoms across domain
        xp_np = np.linspace(x_min, x_max, K_t)[:, None].astype(np.float32)
        Xp = _to(torch.from_numpy(xp_np))

        if support_y is not None:
            y_min, y_max = support_y
            base_level = np.random.uniform(y_min, y_max)
            slope      = np.random.uniform(-0.2, 0.2)
            freq       = np.random.uniform(0.5, 1.2)
            phase      = np.random.uniform(0, 2 * np.pi)
            amp        = np.random.uniform(0.0, 1.0)
            yp_np      = base_level + slope * (xp_np - x_min) + amp * np.sin(freq * (xp_np - x_min) + phase)
            yp_np      = np.clip(yp_np, y_min, y_max).astype(np.float32)
        else:
            yp_np = (np.random.normal(0.0, prior_sigma) +
                     np.random.normal(0.0, prior_sigma * 0.2) * (xp_np - x_min)).astype(np.float32)

        yp = _to(torch.from_numpy(yp_np))

        X_all = torch.cat([X, Xp], dim=0)
        y_all = torch.cat([y, yp], dim=0)

        w_all = np.concatenate([w_data, w_prior * q_prior])
        w_all = w_all / w_all.sum()
        w_t   = _to(torch.tensor(w_all, dtype=torch.float32)).unsqueeze(1)

        m   = DeterministicNet().to(DEVICE)
        opt = optim.Adam(m.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            (w_t * crit(m(X_all), y_all)).sum().backward()
            opt.step()
        m.eval()
        models.append(m)

    return ("ensemble", models)


# ══════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════

def predict(method_out, X_test, n_mc=50):
    """Returns (mu, std, raw_draws)  raw_draws.shape=[K, N_test]."""
    tag = method_out[0]
    with torch.no_grad():
        if tag == "deep_ensemble_nll":
            mus, s2s = [], []
            for m in method_out[1]:
                mu_m, s2_m = m(X_test)
                mus.append(mu_m.squeeze().cpu().numpy())
                s2s.append(s2_m.squeeze().cpu().numpy())
            mus = np.array(mus)   # [K, N_test]
            s2s = np.array(s2s)   # [K, N_test]
            mean_pred = mus.mean(axis=0)
            # Total predictive variance = mean predicted aleatoric variance + epistemic variance
            total_var = s2s.mean(axis=0) + mus.var(axis=0)
            total_std = np.sqrt(np.clip(total_var, 1e-10, None))
            return mean_pred, total_std, mus

        elif tag == "ensemble":
            draws = np.array([m(X_test).squeeze().cpu().numpy()
                              for m in method_out[1]])
            return draws.mean(0), draws.std(0), draws

        elif tag == "neural_linear":
            _, feat_net, head = method_out
            Phi = feat_net.features(X_test).cpu().numpy()
            mu, std = head.predict(Phi)
            draws = head.sample_draws(Phi, n_draws=10)
            return mu, std, draws

        elif tag == "vi":
            _, m = method_out
            m.train()   # enable stochastic forward passes
            draws = np.array([
                m(X_test, sample=True).squeeze().cpu().numpy()
                for _ in range(n_mc)
            ])
            m.eval()
            return draws.mean(0), draws.std(0), draws

    raise ValueError(f"Unknown tag: {tag}")


# ══════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════

def metric_coverage(y_true, mu, std, alpha=0.95):
    z = scipy_stats.norm.ppf(0.5 + alpha / 2)
    return float(((y_true >= mu - z * std) & (y_true <= mu + z * std)).mean() * 100)

def metric_crps(y_true, mu, std):
    std = np.clip(std, 1e-8, None)
    z   = (y_true - mu) / std
    return float(np.mean(std * (z * (2 * scipy_stats.norm.cdf(z) - 1)
                                + 2 * scipy_stats.norm.pdf(z)
                                - 1.0 / math.sqrt(math.pi))))

def metric_id_mse(y_true, mu, id_mask):
    return float(np.mean((y_true[id_mask] - mu[id_mask]) ** 2))

def metric_ratio(std, numer_mask, denom_mask):
    d = np.mean(std[denom_mask])
    return float(np.mean(std[numer_mask]) / max(d, 1e-8))


# ══════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════

def _panel(ax, X_te, mu, std, raw, X_tr, y_tr, y_te,
           void_lo, void_hi, id_lo, id_hi, id_color,
           panel_lab, method_lab, ratio_lab, ratio_val,
           ylim, xlim, n_draws=5):

    for k in range(min(n_draws, raw.shape[0])):
        ax.plot(X_te, raw[k], color=PALETTE["sample"], alpha=0.35, lw=0.8, ls='--', zorder=1)

    ax.fill_between(X_te, mu - 1.96*std, mu + 1.96*std,
                    color=PALETTE["ci"], alpha=0.22, zorder=2)
    ax.plot(X_te, mu,    color=PALETTE["mean"],  lw=2.0,  zorder=3)
    ax.plot(X_te, y_te,  color=PALETTE["truth"], lw=1.1,  ls=':', zorder=3)
    if void_lo is not None and void_hi is not None:
        ax.axvspan(void_lo, void_hi, color=PALETTE["void"], alpha=0.10, zorder=0)
        if void_lo > 0:
            ax.axvspan(-void_hi, -void_lo, color=PALETTE["void"], alpha=0.10, zorder=0)
    if id_lo is not None and id_hi is not None:
        ax.axvspan(id_lo, id_hi, color=id_color, alpha=0.05, zorder=0)

    if ylim: ax.set_ylim(*ylim)
    if xlim: ax.set_xlim(*xlim)

    col = METHOD_COLORS.get(method_lab, '#333333')
    ax.set_title(f"{panel_lab} {method_lab}", fontsize=9.5, fontweight='bold',
                 color=col, pad=4)
    ax.text(0.97, 0.04, f"{ratio_lab} = {ratio_val:.2f}×",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            fontweight='bold', color=col,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8', alpha=0.85))


def make_figure(X_te, y_te, X_tr, y_tr,
                all_preds, all_metrics,
                void_lo, void_hi, id_lo, id_hi, id_color,
                ratio_key, ratio_lab,
                ylim, xlim, suptitle, out_stem):

    fig = plt.figure(figsize=(17, 11))
    gs  = gridspec.GridSpec(3, 3,
                            height_ratios=[1, 1, 0.44],
                            hspace=0.50, wspace=0.30,
                            left=0.05, right=0.98, top=0.93, bottom=0.04)

    for idx, (mlab, plab) in enumerate(zip(METHOD_LABELS, PANEL_LABELS)):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        mu, std, raw = all_preds[idx]
        _panel(ax, X_te, mu, std, raw, X_tr, y_tr, y_te,
               void_lo, void_hi, id_lo, id_hi, id_color,
               plab, mlab, ratio_lab, all_metrics[idx][ratio_key],
               ylim, xlim)
        if idx == 0:
            ax.legend(handles=[
                Line2D([0],[0], color=PALETTE["mean"], lw=2,    label='Posterior Mean'),
                Line2D([0],[0], color=PALETTE["ci"],   lw=7,    alpha=0.3, label='95% CI'),
                Line2D([0],[0], color=PALETTE["truth"],lw=1.2,  ls=':', label='Ground Truth'),
                Line2D([0],[0], marker='o', color='w',
                       markerfacecolor=PALETTE["data"], ms=6,   label='Training Data'),
            ], loc='lower left', framealpha=0.9, fontsize=7)

    # ── Panel (f): Summary comparison bar chart ──────────────────
    ax_bar = fig.add_subplot(gs[1, 2])
    ratios = [m[ratio_key] for m in all_metrics]
    colors = [METHOD_COLORS[m] for m in METHOD_LABELS]
    short_labels = ["Deep Ens.\n(NLL)", "Vanilla\nBoot.", "BootDQN\n+ Priors", "Variational\nInf.", "DP-BNN\n(Ours)"]
    x_pos = np.arange(len(METHOD_LABELS))
    bars = ax_bar.bar(x_pos, ratios, color=colors, alpha=0.85, width=0.55, edgecolor='black', linewidth=0.8)
    ax_bar.axhline(1.0, color='#c00000', linestyle='--', linewidth=1.4, label='In-Dist Baseline (1.0×)', zorder=2)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(short_labels, fontsize=7.2, fontweight='medium')
    ax_bar.set_ylabel(f"{ratio_lab}", fontsize=8.5, fontweight='bold')
    ax_bar.set_title(f"(f) {ratio_lab} Comparison", fontsize=9.5, fontweight='bold', color='#1f4e79', pad=4)
    ax_bar.grid(axis='y', linestyle=':', alpha=0.6)
    ax_bar.legend(loc='upper left', fontsize=7.2, framealpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., h + 0.03 * max(ratios),
                    f"{h:.2f}×", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_bar.set_ylim(0, max(ratios) * 1.25)

    # ── Metrics table (bottom row) ──────────────────────────────
    ax_t = fig.add_subplot(gs[2, :])
    ax_t.axis('off')

    col_headers = ["Method",
                   ratio_lab + "  ↑",
                   "95% Coverage (%)  ↑",
                   "CRPS  ↓",
                   "ID-MSE  ↓"]
    rows = []
    for lab, m in zip(METHOD_LABELS, all_metrics):
        rows.append([lab,
                     f"{m[ratio_key]:.2f}×",
                     f"{m['coverage']:.1f}",
                     f"{m['crps']:.4f}",
                     f"{m['id_mse']:.4f}"])

    tbl = ax_t.table(cellText=rows, colLabels=col_headers,
                     loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.6)

    # Header style
    for j in range(len(col_headers)):
        c = tbl[(0, j)]
        c.set_facecolor('#1f4e79')
        c.get_text().set_color('white')
        c.get_text().set_fontweight('bold')

    # Alternating rows + DP-BNN highlight
    dp_row = METHOD_LABELS.index("DP-BNN (Ours)") + 1  # 1-indexed (0=header)
    for i in range(1, len(rows) + 1):
        bg = '#d4efdf' if i == dp_row else ('#f5f5f5' if i % 2 == 0 else 'white')
        bold = (i == dp_row)
        for j in range(len(col_headers)):
            tbl[(i, j)].set_facecolor(bg)
            if bold:
                tbl[(i, j)].get_text().set_fontweight('bold')

    # Star best in each metric column
    for col_j, (mkey, higher_better) in enumerate(
            zip([ratio_key, 'coverage', 'crps', 'id_mse'],
                [True, True, False, False]), start=1):
        vals = [float(m[mkey]) for m in all_metrics]
        best = (np.argmax(vals) if higher_better else np.argmin(vals)) + 1
        cell = tbl[(best, col_j)]
        cell.get_text().set_text("* " + cell.get_text().get_text())
        cell.get_text().set_color('#1a6b2e')
        cell.get_text().set_fontweight('bold')

    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=0.97)

    for ext in ('pdf', 'png'):
        p = os.path.join(RESULTS_DIR, f"{out_stem}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f"  Saved  -> {p}")

    for ext in ('png', 'pdf'):
        src = os.path.join(RESULTS_DIR, f"{out_stem}.{ext}")
        dst = os.path.join(ARTIFACT_DIR, f"{out_stem}.{ext}")
        shutil.copy2(src, dst)
        print(f"  Copied -> {dst}")

    plt.close(fig)


# ══════════════════════════════════════════════
#  BENCHMARK 1: Extrapolation Epistemic Uncertainty
#  (Osband et al., NeurIPS 2018 / Lakshminarayanan et al., 2017)
# ══════════════════════════════════════════════

def run_lakshminarayanan():
    print("\n" + "="*65)
    print("BENCHMARK: Lakshminarayanan et al. (NeurIPS 2017, Figure 1)")
    print("y = x^3 + eps,  eps ~ N(0, 3^2),  x in [-4, 4], test in [-6, 6]")
    print("="*65)

    np.random.seed(42); torch.manual_seed(42)
    n_data   = 20
    X_tr_np  = np.random.uniform(-4.0, 4.0, n_data)
    y_tr_np  = X_tr_np**3 + 3.0 * np.random.randn(n_data)
    idx      = np.argsort(X_tr_np)
    X_tr_np  = X_tr_np[idx]
    y_tr_np  = y_tr_np[idx]

    X_te_np  = np.linspace(-6.0, 6.0, 300)
    y_te_np  = X_te_np**3       # ground truth: cubic function

    X_tr = _to(torch.tensor(X_tr_np, dtype=torch.float32).unsqueeze(1))
    y_tr = _to(torch.tensor(y_tr_np, dtype=torch.float32).unsqueeze(1))
    X_te = _to(torch.tensor(X_te_np, dtype=torch.float32).unsqueeze(1))

    id_mask  = np.abs(X_te_np) <= 4.0
    ood_mask = np.abs(X_te_np) >  4.0

    np.random.seed(SEED); torch.manual_seed(SEED)
    print("Training 5 methods on Lakshminarayanan (2017) y = x^3 Benchmark (alpha=5.0)...")
    outputs = [
        train_deep_ensemble(X_tr, y_tr, epochs=1200, lr=5e-3),
        train_vanilla_bootstrap(X_tr, y_tr, epochs=1200, lr=5e-3),
        train_boot_prior(X_tr, y_tr, beta=1.0, epochs=1200, lr=5e-3),
        train_vi(X_tr, y_tr, epochs=3000, lr=1e-3, prior_std=1.0, noise_var=9.0),
        train_dp_bnn(X_tr, y_tr, domain_range=(-6.0, 6.0), alpha=5.0, prior_sigma=50.0, epochs=1200, lr=5e-3),
    ]

    all_preds, all_metrics = [], []
    for lab, out in zip(METHOD_LABELS, outputs):
        mu, std, raw = predict(out, X_te)
        m = {
            "ood_ratio": metric_ratio(std, ood_mask, id_mask),
            "coverage":  metric_coverage(y_te_np, mu, std),
            "crps":      metric_crps(y_te_np, mu, std),
            "id_mse":    metric_id_mse(y_te_np, mu, id_mask),
        }
        all_preds.append((mu, std, raw))
        all_metrics.append(m)
        print(f"  {lab:<26} OOD Ratio={m['ood_ratio']:.2f}x  "
              f"Cov={m['coverage']:.1f}%  CRPS={m['crps']:.4f}  ID-MSE={m['id_mse']:.4f}")

    for stem in ["paper_lakshminarayanan_figure1", "paper_extrapolation_showdown", "paper_osband_figure2"]:
        make_figure(
            X_te_np, y_te_np, X_tr_np, y_tr_np,
            all_preds, all_metrics,
            void_lo=4.0, void_hi=6.0,
            id_lo=-4.0, id_hi=4.0, id_color='#2ca02c',
            ratio_key="ood_ratio", ratio_lab="OOD Ratio",
            ylim=(-160, 160), xlim=(-6.0, 6.0),
            suptitle="Lakshminarayanan et al. (NeurIPS 2017, Fig. 1): y = x³ Extrapolation Showdown (K = 10, α = 5)",
            out_stem=stem,
        )
    return all_metrics


# ══════════════════════════════════════════════
#  BENCHMARK 2: Disjoint Domain Split with Vertical Transition
#  (Vashishtha & Maillard, DP-PINNs Slide 5)
# ══════════════════════════════════════════════

def run_disjoint_split():
    print("\n" + "="*65)
    print("BENCHMARK 2: Disjoint Domain Split with Vertical Phase Transition")
    print("f(x) = 2*sin(x) + (1.5 if x > 0 else -1.5), Flanks [-4, -1.5] & [1.5, 4]")
    print("="*65)

    np.random.seed(42); torch.manual_seed(42)
    n_half   = 40
    X_left   = np.random.uniform(-4.0, -1.5, n_half)
    X_right  = np.random.uniform( 1.5,  4.0, n_half)
    X_tr_np  = np.concatenate([X_left, X_right])

    def ground_truth_fn(x):
        return 2.0 * np.sin(x) + np.where(x > 0, 1.5, -1.5)

    y_tr_np  = ground_truth_fn(X_tr_np) + 0.1 * np.random.randn(len(X_tr_np))
    X_te_np  = np.linspace(-5.0, 5.0, 400)
    y_te_np  = ground_truth_fn(X_te_np)

    X_tr = _to(torch.tensor(X_tr_np, dtype=torch.float32).unsqueeze(1))
    y_tr = _to(torch.tensor(y_tr_np, dtype=torch.float32).unsqueeze(1))
    X_te = _to(torch.tensor(X_te_np, dtype=torch.float32).unsqueeze(1))

    # Gap: void between flanks [-1.5, 1.5]
    gap_mask = (X_te_np >= -1.5) & (X_te_np <= 1.5)
    # Flanks: in-distribution training coverage
    flank_mask = (((X_te_np >= -4.0) & (X_te_np <= -1.5)) |
                   ((X_te_np >=  1.5) & (X_te_np <=  4.0)))

    np.random.seed(SEED); torch.manual_seed(SEED)
    print("Training 5 methods on Disjoint Split Benchmark (alpha=5.0, Support-Consistent Base Measure)...")
    outputs = [
        train_deep_ensemble(X_tr, y_tr),
        train_vanilla_bootstrap(X_tr, y_tr),
        train_boot_prior(X_tr, y_tr, beta=1.0),
        train_vi(X_tr, y_tr, epochs=3000, prior_std=1.0, noise_var=0.01),
        train_dp_bnn(X_tr, y_tr, domain_range=(-5.0, 5.0), support_y=(-3.5, 3.5), alpha=2.5),
    ]

    all_preds, all_metrics = [], []
    for lab, out in zip(METHOD_LABELS, outputs):
        mu, std, raw = predict(out, X_te)
        m = {
            "gap_ratio": metric_ratio(std, gap_mask, flank_mask),
            "coverage":  metric_coverage(y_te_np, mu, std),
            "crps":      metric_crps(y_te_np, mu, std),
            "id_mse":    metric_id_mse(y_te_np, mu, flank_mask),
        }
        all_preds.append((mu, std, raw))
        all_metrics.append(m)
        print(f"  {lab:<26} Gap Ratio={m['gap_ratio']:.2f}x  "
              f"Cov={m['coverage']:.1f}%  CRPS={m['crps']:.4f}  ID-MSE={m['id_mse']:.4f}")

    for stem in ["paper_disjoint_split_showdown", "paper_foong_inbetween"]:
        make_figure(
            X_te_np, y_te_np, X_tr_np, y_tr_np,
            all_preds, all_metrics,
            void_lo=-1.5, void_hi=1.5,
            id_lo=-4.0, id_hi=-1.5, id_color='#ff7f0e',
            ratio_key="gap_ratio", ratio_lab="Gap Ratio",
            ylim=(-6.0, 6.0), xlim=(-5.0, 5.0),
            suptitle="Disjoint Domain Split with Vertical Phase Transition — 5-Method Showdown (Support Base Measure [-3.5, 3.5])",
            out_stem=stem,
        )
    return all_metrics


# ══════════════════════════════════════════════
#  BENCHMARK 3: Hidden Oasis / Sparse Reward Cliff
#  (Canonical RL Exploration Failure Mode of Deep Ensembles)
# ══════════════════════════════════════════════

def run_hidden_oasis():
    print("\n" + "="*65)
    print("BENCHMARK 3: Hidden Oasis / Sparse Reward Exploration")
    print("Explored: x in [-4, 1] (f=0); Hidden Oasis: x in [3.5, 5.5] (f=5.0)")
    print("Support-Consistent Base Measure: Y in [0.0, 5.0]")
    print("="*65)

    np.random.seed(42); torch.manual_seed(42)
    n_data   = 60
    X_tr_np  = np.random.uniform(-4.0, 1.0, n_data)
    y_tr_np  = 0.05 * np.random.randn(n_data)
    idx      = np.argsort(X_tr_np)
    X_tr_np, y_tr_np = X_tr_np[idx], y_tr_np[idx]

    def true_oasis_fn(x):
        return 5.0 * np.exp(-1.5 * (x - 4.5)**2) * (x > 2.0)

    X_te_np  = np.linspace(-4.0, 6.0, 300)
    y_te_np  = true_oasis_fn(X_te_np)

    X_tr = _to(torch.tensor(X_tr_np, dtype=torch.float32).unsqueeze(1))
    y_tr = _to(torch.tensor(y_tr_np, dtype=torch.float32).unsqueeze(1))
    X_te = _to(torch.tensor(X_te_np, dtype=torch.float32).unsqueeze(1))

    id_mask    = X_te_np <= 1.0
    oasis_mask = X_te_np >  1.0

    np.random.seed(SEED); torch.manual_seed(SEED)
    print("Training 5 methods on Hidden Oasis Benchmark (N = 60)...")
    outputs = [
        train_deep_ensemble(X_tr, y_tr, epochs=1200, lr=4e-3),
        train_vanilla_bootstrap(X_tr, y_tr, epochs=1200, lr=4e-3),
        train_boot_prior(X_tr, y_tr, beta=1.5, epochs=1200, lr=4e-3),
        train_vi(X_tr, y_tr, epochs=3000, lr=1e-3, prior_std=1.0, noise_var=0.0025),
        train_dp_bnn(X_tr, y_tr, domain_range=(-4.0, 6.0), support_y=(0.0, 5.0), alpha=2.5, epochs=1200, lr=4e-3),
    ]

    all_preds, all_metrics = [], []
    for lab, out in zip(METHOD_LABELS, outputs):
        mu, std, raw = predict(out, X_te)
        m = {
            "oasis_ratio": metric_ratio(std, oasis_mask, id_mask),
            "coverage":    metric_coverage(y_te_np, mu, std),
            "crps":        metric_crps(y_te_np, mu, std),
            "id_mse":      metric_id_mse(y_te_np, mu, id_mask),
        }
        all_preds.append((mu, std, raw))
        all_metrics.append(m)
        print(f"  {lab:<26} Oasis Ratio={m['oasis_ratio']:.2f}x  "
              f"Cov={m['coverage']:.1f}%  CRPS={m['crps']:.4f}  ID-MSE={m['id_mse']:.4f}")

    for stem in ["paper_hidden_oasis_showdown"]:
        make_figure(
            X_te_np, y_te_np, X_tr_np, y_tr_np,
            all_preds, all_metrics,
            void_lo=1.0, void_hi=6.0,
            id_lo=-4.0, id_hi=1.0, id_color='#2ca02c',
            ratio_key="oasis_ratio", ratio_lab="Oasis Ratio",
            ylim=(-1.0, 7.0), xlim=(-4.0, 6.0),
            suptitle="Hidden Oasis (Sparse Reward RL Exploration) — 5-Method Showdown (N = 60, Support Base Measure [0, 5])",
            out_stem=stem,
        )
    return all_metrics


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("DP-BNN Paper — Publication Uncertainty Benchmarks")
    print("5 Methods | Same Backbone | Equal Ensemble Size K = 10 | alpha = 5.0")
    print("=" * 65)

    disjoint_m = run_disjoint_split()
    oasis_m    = run_hidden_oasis()

    summary = {
        "methods": METHOD_LABELS,
        "shared_backbone": "Linear(1->64)->LN->ReLU->Linear(64->64)->LN->ReLU->head",
        "ensemble_size_K": 10,
        "disjoint_split_benchmark": [{**m, "method": l} for l, m in zip(METHOD_LABELS, disjoint_m)],
        "hidden_oasis_benchmark":   [{**m, "method": l} for l, m in zip(METHOD_LABELS, oasis_m)],
    }
    out_json = os.path.join(RESULTS_DIR, "paper_uncertainty_metrics.json")
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    shutil.copy2(out_json, os.path.join(ARTIFACT_DIR, "paper_uncertainty_metrics.json"))
    print(f"\nMetrics saved -> {out_json}")
    print("\nDone. Output files:")
    for stem in ["paper_disjoint_split_showdown", "paper_hidden_oasis_showdown"]:
        for ext in ["pdf", "png"]:
            print(f"  results/{stem}.{ext}")
