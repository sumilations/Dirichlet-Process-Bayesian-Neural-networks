#!/usr/bin/env python3
"""Generate publication-quality figures:
1. DeepSea N=50 Pure TS 10-Seeds Cumulative Regret Curves
2. LayerNorm Ablation: DeepSea N=40 and CartPole Swing-Up (With vs Without LayerNorm)
"""

import os
import glob
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BRAIN_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"

# -------------------------------------------------------------
# FIGURE 1: DeepSea N=50 Pure TS 10-Seeds Regret Curves
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#fbfbfd')

files = sorted(glob.glob('results_deepsea50_pure_ts_10seeds/deepsea_N50_scaled_kprior_s*.json*'))
colors = plt.cm.tab10(np.linspace(0, 1, 10))

solved_info = []

for i, f in enumerate(files):
    d = json.load(open(f))
    seed = d.get('seed')
    disc = d.get('first_discovery')
    solv = d.get('solved_episode')
    rc = d.get('regret_curve', [])
    if not rc:
        continue
    eps = [pt['episode'] for pt in rc]
    regs = [pt['cum_regret'] for pt in rc]
    
    col = colors[i % len(colors)]
    if solv is not None:
        lbl = f"Seed {seed} (SOLVED @ Ep {solv}, Regret {regs[-1]:.0f})"
        ax.plot(eps, regs, lw=2.2, color=col, label=lbl, zorder=5)
        # Mark solve point
        ax.scatter([solv], [regs[-1]], color=col, s=70, edgecolors='black', linewidths=1.2, zorder=6)
        solved_info.append((seed, solv, regs[-1]))
    elif disc is not None:
        lbl = f"Seed {seed} (Discovery @ Ep {disc})"
        ax.plot(eps, regs, lw=1.8, color=col, linestyle='--', label=lbl, zorder=4)
        ax.scatter([disc], [regs[-1]], color=col, s=50, marker='^', zorder=5)
    else:
        lbl = f"Seed {seed} (Exploring)"
        ax.plot(eps, regs, lw=1.2, color='gray', alpha=0.5, linestyle=':', label=lbl, zorder=2)

ax.set_title(r"DeepSea $N=50$: Cumulative Regret under Pure TS ($K_{\mathrm{prior}} = 4N = 200$ atoms)", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Environment Episode", fontsize=11, fontweight='medium')
ax.set_ylabel(r"Cumulative Regret $\sum (V^* - V(\pi_k))$", fontsize=11, fontweight='medium')
ax.grid(True, linestyle='--', alpha=0.35, color='gray')
ax.legend(loc='upper left', fontsize=8.5, framealpha=0.92, facecolor='white', edgecolor='#e0e0e0')
ax.set_xlim(0, 6500)
ax.set_ylim(0, 6500)

# Add annotation box
textstr = (
    r"$\mathbf{Scaling\ Breakthrough:}$" + "\n"
    r"$\bullet\ \mathrm{Seed\ 49:\ Solved\ @\ Ep\ 2,387\ (Regret:\ 2,326)}$" + "\n"
    r"$\bullet\ \mathrm{Seed\ 43:\ Solved\ @\ Ep\ 2,479\ (Regret:\ 2,418)}$" + "\n"
    r"$\bullet\ \mathrm{Seed\ 45:\ Solved\ @\ Ep\ 4,588\ (Regret:\ 4,500)}$" + "\n"
    r"$\bullet\ \mathrm{Seed\ 42:\ Discovery\ @\ Ep\ 5,959}$"
)
props = dict(boxstyle='round,pad=0.6', facecolor='#e8f4fd', alpha=0.9, edgecolor='#7bb5e8')
ax.text(0.55, 0.15, textstr, transform=ax.transAxes, fontsize=8.5, verticalalignment='bottom', bbox=props)

plt.tight_layout()
f1_local = "deepsea50_pure_ts_10seeds_regret.png"
f1_dest = os.path.join(BRAIN_DIR, "deepsea50_pure_ts_10seeds_regret.png")
fig.savefig(f1_local, dpi=300)
fig.savefig(f1_dest, dpi=300)
plt.close(fig)
print(f"Saved {f1_dest}")

# -------------------------------------------------------------
# FIGURE 2: LayerNorm Ablation (DeepSea N=40 & Cart-Pole Swing-Up)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Subplot A: DeepSea N=40 With vs Without LayerNorm
ax = axes[0]
ax.set_facecolor('#fbfbfd')

# Data for DeepSea 40
with_ln_s42 = json.load(open('results_pure_ts_deepsea/deepsea_N40_pure_ts_scaled_kprior_s42.json'))
no_ln_s42 = json.load(open('results_no_ln/deepsea40_no_ln_s42.json'))
no_ln_s43 = json.load(open('results_no_ln/deepsea40_no_ln_s43.json'))

bars = [
    ("With LayerNorm\n(Seed 42)", with_ln_s42["solved_episode"], with_ln_s42["first_discovery"], with_ln_s42["elapsed_seconds"], '#1f77b4'),
    ("Without LayerNorm\n(Seed 42)", no_ln_s42["solved_episode"], no_ln_s42["first_discovery"], no_ln_s42["elapsed_seconds"], '#2ca02c'),
    ("Without LayerNorm\n(Seed 43)", no_ln_s43["solved_episode"], no_ln_s43["first_discovery"], no_ln_s43["elapsed_seconds"], '#2ca02c'),
]

x = np.arange(len(bars))
w = 0.35
solv_vals = [b[1] for b in bars]
disc_vals = [b[2] for b in bars]
times = [b[3] for b in bars]

rects1 = ax.bar(x - w/2, disc_vals, w, label='Discovery Episode', color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.8)
rects2 = ax.bar(x + w/2, solv_vals, w, label='Solved Episode', color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.8)

for i, b in enumerate(bars):
    ax.text(x[i] - w/2, disc_vals[i] + 40, f"Ep {disc_vals[i]}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.text(x[i] + w/2, solv_vals[i] + 40, f"Ep {solv_vals[i]}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.text(x[i], max(solv_vals[i], disc_vals[i]) + 220, f"{times[i]:.0f}s ({times[i]/60:.1f}m)", ha='center', va='bottom', fontsize=8, color='#333333', style='italic')

ax.set_title(r"DeepSea $N=40$: LayerNorm Ablation (Pure TS)", fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in bars], fontsize=9)
ax.set_ylabel("Episode Count", fontsize=10)
ax.set_ylim(0, 2600)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

# Subplot B: Cart-Pole Swing-Up Upright Balancing With vs Without LayerNorm
ax = axes[1]
ax.set_facecolor('#fbfbfd')

# Load CartPole data
cp_with_ln = json.load(open('results_cartpole_pure_ts/cartpole_swingup_dp_dqn_deepsea_construct_gaussian_pure_ts_a15.0_s42.json'))
cp_no_ln_s43 = json.load(open('results_no_ln/cartpole_no_ln_gaussian_pure_ts_a15.0_s43_ckpt.json'))
cp_no_ln_s44 = json.load(open('results_no_ln/cartpole_no_ln_gaussian_pure_ts_a15.0_s44.json'))

def smooth(x, window=25):
    if len(x) < window: return x
    return np.convolve(x, np.ones(window)/window, mode='valid')

u_with = np.array(cp_with_ln.get('upright_steps', []))[:1500]
u_no_43 = np.array(cp_no_ln_s43.get('upright_steps', []))
u_no_44 = np.array(cp_no_ln_s44.get('upright_steps', []))

if len(u_with) > 0:
    sm = smooth(u_with, 25)
    ax.plot(np.arange(len(u_with)-len(sm), len(u_with)), sm, label='With LayerNorm (Seed 42, Peak: 789)', color='#1f77b4', lw=2.0)
if len(u_no_43) > 0:
    sm = smooth(u_no_43, 25)
    ax.plot(np.arange(len(u_no_43)-len(sm), len(u_no_43)), sm, label='Without LayerNorm (Seed 43, Peak: 782)', color='#2ca02c', lw=2.0)
if len(u_no_44) > 0:
    sm = smooth(u_no_44, 25)
    ax.plot(np.arange(len(u_no_44)-len(sm), len(u_no_44)), sm, label='Without LayerNorm (Seed 44, Peak: 787)', color='#ff7f0e', lw=2.0)

ax.set_title(r"Cart-Pole Swing-Up: Upright Steps (25-MA, Pure TS)", fontsize=11, fontweight='bold')
ax.set_xlabel("Episode", fontsize=10)
ax.set_ylabel(r"Upright Steps per Episode (Max: 1000)", fontsize=10)
ax.set_ylim(-20, 700)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

plt.tight_layout()
f2_local = "no_layernorm_deepsea_cartpole_comparison.png"
f2_dest = os.path.join(BRAIN_DIR, "no_layernorm_deepsea_cartpole_comparison.png")
fig.savefig(f2_local, dpi=300)
fig.savefig(f2_dest, dpi=300)
plt.close(fig)
print(f"Saved {f2_dest}")
