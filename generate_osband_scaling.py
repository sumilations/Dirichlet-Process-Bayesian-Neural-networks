import os
import json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load empirical data
data_path = '/Users/sumitvashishtha/Desktop/DP-BNNs/results_deepsea_scaling/deepsea_scaling_results.json'
with open(data_path) as f:
    data = json.load(f)

by_size = defaultdict(list)
for item in data:
    if item.get('base_measure') == 'nondag_maxent':
        by_size[item['size']].append(item)

all_sizes = np.array(sorted(by_size.keys()))

# Compute stats for each size
means_disc, sems_disc = [], []
means_solv, sems_solv = [], []

for sz in all_sizes:
    runs = by_size[sz]
    d = [r['first_discovery'] for r in runs if r.get('first_discovery') is not None]
    s = [r['solved_episode'] for r in runs if r.get('solved_episode') is not None]
    means_disc.append(np.mean(d))
    sems_disc.append(np.std(d) / np.sqrt(len(d)))
    means_solv.append(np.mean(s))
    sems_solv.append(np.std(s) / np.sqrt(len(s)))

means_disc = np.array(means_disc)
sems_disc = np.array(sems_disc)
means_solv = np.array(means_solv)
sems_solv = np.array(sems_solv)

# --- Compute Power Law Fit from N >= 20 onwards ---
mask_20 = all_sizes >= 20
sizes_20 = all_sizes[mask_20]
disc_20 = means_disc[mask_20]

lx_20 = np.log10(sizes_20)
ly_20 = np.log10(disc_20)
poly_20 = np.polyfit(lx_20, ly_20, 1)
slope_20 = poly_20[0]
intercept_20 = 10 ** poly_20[1]
r2_20 = 1 - np.sum((ly_20 - np.polyval(poly_20, lx_20))**2) / np.sum((ly_20 - np.mean(ly_20))**2)

print(f"Fit on N >= 20 (17 sizes): T ~ {intercept_20:.2f} * N^{slope_20:.2f} (R^2 = {r2_20:.3f})")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'mathtext.fontset': 'dejavusans',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.1,
})

fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)
n_grid = np.linspace(4.5, 58, 200)

# Theoretical Reference Bounds (Osband et al. 2016, 2018)
n_dither = n_grid[n_grid <= 16]
ax.plot(n_dither, 2.0**n_dither, color='#D32F2F', lw=2.2, linestyle='--',
        label=r'Random Dithering: $\Omega(2^N)$ (DQN)')

# DP-DQN Asymptotic Fit (solid for N >= 20, dashed projection for N < 20)
fit_label = rf'DP-DQN (Ours): $T \sim \mathcal{{O}}(N^{{{slope_20:.2f}}})$ (Fit on $N \geq 20$, $R^2={r2_20:.2f}$)'
n_asympt = n_grid[n_grid >= 20]
n_pre = n_grid[n_grid <= 20]
ax.plot(n_asympt, intercept_20 * (n_asympt**slope_20), color='#1976D2', lw=2.6, label=fit_label, zorder=5)
ax.plot(n_pre, intercept_20 * (n_pre**slope_20), color='#1976D2', lw=1.6, linestyle='--', alpha=0.5, zorder=4)

# Individual seed runs (semi-transparent scatter)
for s_val in all_sizes:
    runs = by_size[s_val]
    for r in runs:
        if r.get('first_discovery') is not None:
            ax.scatter(s_val, r['first_discovery'], color='#1976D2', alpha=0.25, s=22, zorder=3)

# Empirical Mean +/- SEM data points for N >= 20 (All 17 points)
ax.errorbar(sizes_20, disc_20, yerr=sems_disc[mask_20],
            fmt='o', color='#0D47A1', ecolor='#1976D2', elinewidth=1.6, capsize=3.5,
            markersize=6.5, zorder=6, label=r'DP-DQN Empirical Mean $\pm$ SEM ($N \geq 20$, 17 sizes)')

# Empirical points for N < 20 (demarcating buffer warm-up)
sizes_low = all_sizes[~mask_20]
disc_low = means_disc[~mask_20]
ax.errorbar(sizes_low, disc_low, yerr=sems_disc[~mask_20],
            fmt='s', color='#90CAF9', markeredgecolor='#1976D2', markeredgewidth=1.2,
            ecolor='#90CAF9', elinewidth=1.4, capsize=3.0,
            markersize=5.5, zorder=5, label=r'DP-DQN ($N < 20$, buffer warm-up)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(4.5, 60)
ax.set_ylim(15, 2e5)

ax.set_xlabel(r'Deep Sea Grid Size $N$ (Log Scale)', fontsize=11, fontweight='bold')
ax.set_ylabel(r'Episodes to Learn $T_{\mathrm{learn}}$ (Log Scale)', fontsize=11, fontweight='bold')
ax.set_title(r'Empirical Scaling of DP-DQN on Deep Sea up to $N = 50$' + '\n' + r'(Pure Thompson Sampling, Single MLP-20)', fontsize=12, fontweight='bold', pad=10)

ax.grid(True, which='both', linestyle=':', alpha=0.45)
ax.legend(loc='upper left', fontsize=8.8, framealpha=0.96, edgecolor='#BDBDBD')

plt.tight_layout()

# Save directly to paper directory and artifact directory
paper_dir = '/Users/sumitvashishtha/Desktop/RLC_2026-3'
artifact_dir = '/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86'

fig.savefig(os.path.join(paper_dir, 'deepsea_scaling_nondag_standalone.pdf'))
fig.savefig(os.path.join(paper_dir, 'deepsea_scaling_nondag_standalone.png'))
fig.savefig(os.path.join(artifact_dir, 'deepsea_scaling_nondag_standalone.png'))
plt.close(fig)

print("Saved clean standalone figure to RLC_2026-3 and artifact dir!")
