#!/usr/bin/env python3
"""
Generate Comprehensive Publication Figure for 2D Annular Regression:
Compares 6 Uncertainty Quantification Methods (With and Without LayerNorm):
1. Deep Ensembles (Lakshminarayanan et al. 2017)
2. Mean-Field Variational BNN / BBB (Blundell et al. 2015)
3. Bootstrapped DQN / Bootstrap (Osband et al. 2016)
4. BootDQN + Randomized Priors (Osband et al. 2018)
5. Monte Carlo Dropout (Gal & Ghahramani 2016, p=0.10)
6. DP-BNN (Ours: Non-parametric Data-Space Dirichlet Process Prior)
"""

import os
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load existing metrics
with open("results/2d_circular_layernorm_ablation_results.json", "r") as fp:
    base_results = json.load(fp)

with open("results/mc_dropout_2d_metrics.json", "r") as fp:
    mc_results = json.load(fp)

# Assemble table data
methods = [
    ("Deep Ensembles (NLL)", "Deep Ensembles"),
    ("BBB (Variational)", "BBB (Variational)"),
    ("BootDQN (Standard)", "BootDQN (Standard)"),
    ("BootDQN + Priors", "BootDQN + Priors"),
    ("MC Dropout (p=0.1)", "MC Dropout (p=0.10)"),
    ("DP-BNN (Ours)", "DP-BNN (Ours: Measure Prior)")
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('#ffffff')

# Panel 1: With LayerNorm
ax = axes[0]
labels = [m[1] for m in methods]
cav_ln = []
id_ln = []
contrast_ln = []

for mkey, mname in methods:
    if "MC Dropout" in mkey:
        v = mc_results["with_layernorm"]["MC Dropout (p=0.1)"]
    else:
        v = base_results["with_layernorm"][mkey]
    cav_ln.append(v["cavity_bald"])
    id_ln.append(v["id_bald"])
    contrast_ln.append(v["contrast_ratio"])

x = np.arange(len(methods))
w = 0.35
rects1 = ax.bar(x - w/2, cav_ln, w, label='Cavity Void ($r <= 1.0$)', color='#1f77b4', edgecolor='black')
rects2 = ax.bar(x + w/2, id_ln, w, label='In-Distribution Annulus', color='#aec7e8', edgecolor='black')

for i in range(len(methods)):
    ax.text(x[i], max(cav_ln[i], id_ln[i]) + 0.05, f"{contrast_ln[i]:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0b5394')

ax.set_xticks(x)
ax.set_xticklabels(['Ensembles', 'BBB', 'BootDQN', 'Boot+Prior', 'MC Dropout', 'DP-BNN\n(Ours)'], fontsize=9.5)
ax.set_ylabel('BALD Epistemic Uncertainty (nats)', fontsize=11, fontweight='bold')
ax.set_title('A. 2D Manifold Uncertainty [With LayerNorm]', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.7)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(loc='upper left', fontsize=9.5)

# Panel 2: Without LayerNorm
ax2 = axes[1]
cav_no = []
id_no = []
contrast_no = []

for mkey, mname in methods:
    if "MC Dropout" in mkey:
        v = mc_results["without_layernorm"]["MC Dropout (p=0.1)"]
    else:
        v = base_results["without_layernorm"][mkey]
    cav_no.append(v["cavity_bald"])
    id_no.append(v["id_bald"])
    contrast_no.append(v["contrast_ratio"])

rects3 = ax2.bar(x - w/2, cav_no, w, label='Cavity Void ($r <= 1.0$)', color='#d62728', edgecolor='black')
rects4 = ax2.bar(x + w/2, id_no, w, label='In-Distribution Annulus', color='#ff9896', edgecolor='black')

for i in range(len(methods)):
    ax2.text(x[i], max(cav_no[i], id_no[i]) + 0.05, f"{contrast_no[i]:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#900C3F')

ax2.set_xticks(x)
ax2.set_xticklabels(['Ensembles', 'BBB', 'BootDQN', 'Boot+Prior', 'MC Dropout', 'DP-BNN\n(Ours)'], fontsize=9.5)
ax2.set_ylabel('BALD Epistemic Uncertainty (nats)', fontsize=11, fontweight='bold')
ax2.set_title('B. 2D Manifold Uncertainty [Without LayerNorm]', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.7)
ax2.grid(True, linestyle='--', alpha=0.35)
ax2.legend(loc='upper left', fontsize=9.5)

plt.suptitle("Epistemic Uncertainty and Void-to-Data Contrast on 2D Annular Regression Task\nShowing Void Epistemic Collapse in MC Dropout vs. Robust Epistemic Dome in DP-BNN",
             fontsize=12.5, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])

out_png = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/Figure_2D_regression_with_mcdropout.png"
out_pdf = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/Figure_2D_regression_with_mcdropout.pdf"

plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf)
plt.savefig("results/Figure_2D_regression_with_mcdropout.png", dpi=300)
plt.savefig("results/Figure_2D_regression_with_mcdropout.pdf")
shutil.copyfile(out_pdf, "/Users/sumitvashishtha/Desktop/RLC_2026-3/Figure_2D_regression_with_mcdropout.pdf")

print("Generated comprehensive 6-method comparison figures with MC Dropout!")
