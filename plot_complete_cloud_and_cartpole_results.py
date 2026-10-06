import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#ffffff')

# ----------------------------------------------------
# 1. Scaling Plot: Deep Sea N=10 to N=50
# ----------------------------------------------------
ax = axes[0]
sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]

# Multi-Sample
ms_means = []
ms_stds = []
for n in sizes:
    files = sorted(glob.glob(f'./results_scaling_sampler_comparison/deepsea_N{n}_multi_sample_s*.json'))
    solv = [json.load(open(f))['solved_episode'] for f in files if json.load(open(f)).get('solved_episode') is not None]
    ms_means.append(np.mean(solv))
    ms_stds.append(np.std(solv))

ms_means = np.array(ms_means)
ms_stds = np.array(ms_stds)

# Pure TS (Standard K_prior=50)
pts_sizes = [10, 15, 20, 25, 30, 35, 40, 45]
pts_means = []
pts_stds = []
for n in pts_sizes:
    files = sorted(glob.glob(f'./results_scaling_sampler_comparison/deepsea_N{n}_pure_ts_s*.json'))
    solv = [json.load(open(f))['solved_episode'] for f in files if json.load(open(f)).get('solved_episode') is not None]
    pts_means.append(np.mean(solv))
    pts_stds.append(np.std(solv))

pts_means = np.array(pts_means)
pts_stds = np.array(pts_stds)

# Pure TS (Scaled K_prior = 4N) on N=40
scaled_files = sorted(glob.glob('./results_pure_ts_deepsea/deepsea_N40_pure_ts_scaled_kprior_s*.json'))
scaled_solv = [json.load(open(f))['solved_episode'] for f in scaled_files if json.load(open(f)).get('solved_episode') is not None]
scaled_mean = np.mean(scaled_solv)
scaled_std = np.std(scaled_solv)

ax.errorbar(sizes, ms_means, yerr=ms_stds, fmt='o-', color='#1f77b4', lw=2.2, capsize=4,
            label='Multi-Sample (SampleOnce=False) [Fit: O(N^1.75)]')
ax.errorbar(pts_sizes, pts_means, yerr=pts_stds, fmt='s--', color='#ff7f0e', lw=2.2, capsize=4,
            label='Pure TS (K_prior=50) [Fit: O(N^1.98) ~ O(N²)]')

ax.errorbar([40], [scaled_mean], yerr=[scaled_std], fmt='^', color='#2ca02c', markersize=10, capsize=6,
            label=f'Pure TS (Scaled K_prior=4N): N=40 Solved (Mean: {scaled_mean:.0f} eps)')

# Theoretical Lower Bound curve
n_fine = np.linspace(10, 50, 100)
# Scale O(N^2) to match Pure TS at N=20
c_lower = pts_means[2] / (20.0**2)
ax.plot(n_fine, c_lower * (n_fine**2), ':', color='black', alpha=0.6, lw=1.8, label=r'$\Omega(N^2)$ Lower Bound (Osband et al.)')

ax.set_title('Deep Sea Scaling: Pure TS vs Multi-Sample vs Theoretical Bound', fontsize=11, fontweight='bold')
ax.set_xlabel('Deep Sea Size N (State Space = 2^N)')
ax.set_ylabel('Episodes to Solve (50-MA return >= 0.85)')
ax.set_yscale('log')
ax.grid(True, which='both', alpha=0.25)
ax.legend(loc='upper left', fontsize=8.5)

# ----------------------------------------------------
# 2. Cart-Pole Swing-Up Pure TS Convergence
# ----------------------------------------------------
ax2 = axes[1]
cp_files = sorted(glob.glob('./results_cartpole_pure_ts/*_ckpt.json'))
cp_colors = {'s42': '#1f77b4', 's43': '#ff7f0e', 's44': '#2ca02c', 's45': '#d62728', 's46': '#9467bd'}

def smooth(x, window=10):
    if len(x) < window: return x
    return np.convolve(x, np.ones(window)/window, mode='valid')

for f in cp_files:
    d = json.load(open(f))
    s = f"s{d.get('seed')}"
    col = cp_colors.get(s, '#333333')
    r = np.array(d.get('returns', []))
    solv = d.get('solved_episode')
    disc = d.get('first_discovery')
    lbl = f"Seed {d.get('seed')} (Solved: Ep {solv})" if solv else f"Seed {d.get('seed')} (Disc: Ep {disc})"
    if len(r) > 0:
        ax2.plot(r, color=col, alpha=0.15)
        sm = smooth(r, 10)
        ax2.plot(np.arange(len(r)-len(sm), len(r)), sm, color=col, lw=2.2, label=lbl)

ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.axhline(750, color='forestgreen', linestyle=':', alpha=0.8, lw=1.5, label='Solved Threshold (~750)')
ax2.set_title('Cart-Pole Swing-Up Pure TS (Gaussian State + Optimistic Reward)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Undiscounted Return')
ax2.grid(True, alpha=0.25)
ax2.legend(loc='upper left', fontsize=8.5)

plt.tight_layout()
out_png = '/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/cloud_and_cartpole_complete_results.png'
plt.savefig(out_png, dpi=200)
plt.savefig('cloud_and_cartpole_complete_results.png', dpi=200)
print(f"Saved combined plot to {out_png}")
