import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import shutil

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor('#ffffff')

files = [
    ('Gaussian α=3.0 (B1 - SOLVED)', './results_deepsea_construct_cartpole/cartpole_swingup_dp_dqn_deepsea_construct_gaussian_alpha3_s42_ckpt.json', '#1f77b4'),
    ('Gaussian α=15.0 (B2)', './results_deepsea_construct_cartpole/cartpole_swingup_dp_dqn_deepsea_construct_gaussian_alpha15_s42_ckpt.json', '#ff7f0e'),
    ('Empirical α=3.0 (A1)', './results_deepsea_construct_cartpole/cartpole_swingup_dp_dqn_deepsea_construct_empirical_alpha3_s42_ckpt.json', '#2ca02c'),
    ('Empirical α=15.0 (A2)', './results_deepsea_construct_cartpole/cartpole_swingup_dp_dqn_deepsea_construct_empirical_alpha15_s42_ckpt.json', '#d62728'),
]

def smooth(x, window=10):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='valid')

# 1. Episode Returns
ax = axes[0, 0]
for name, fpath, col in files:
    if os.path.exists(fpath):
        d = json.load(open(fpath))
        r = np.array(d.get('returns', []))
        if len(r) > 0:
            ax.plot(r, color=col, alpha=0.25)
            sm = smooth(r, 10)
            ax.plot(np.arange(len(r)-len(sm), len(r)), sm, color=col, lw=2.2, label=name)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(750, color='forestgreen', linestyle=':', alpha=0.7, label='Solved Threshold (~750)')
ax.set_title('Episode Return: Deep Sea Construct on Cart-Pole (Seed 42)', fontsize=11, fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Undiscounted Return')
ax.grid(True, alpha=0.25)
ax.legend(loc='upper left', fontsize=8.5)

# 2. Upright Steps
ax = axes[0, 1]
for name, fpath, col in files:
    if os.path.exists(fpath):
        d = json.load(open(fpath))
        u = np.array(d.get('upright_steps', []))
        if len(u) > 0:
            ax.plot(u, color=col, lw=2.0, label=f"{name} (Max: {np.max(u)})")
ax.set_title('Upright Balancing Steps per Episode (Max: 1000)', fontsize=11, fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Steps (|cos θ| > 0.95)')
ax.grid(True, alpha=0.25)
ax.legend(loc='upper left', fontsize=8.5)

# 3. Cumulative Regret
ax = axes[1, 0]
for name, fpath, col in files:
    if os.path.exists(fpath):
        d = json.load(open(fpath))
        reg = np.array(d.get('cumulative_regrets', []))
        if len(reg) > 0:
            ax.plot(reg, color=col, lw=2.0, label=name)
ax.set_title('Cumulative Regret vs. Episodes', fontsize=11, fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Cumulative Regret')
ax.grid(True, alpha=0.25)
ax.legend(loc='upper left', fontsize=8.5)

# 4. Summary Bar Chart
ax = axes[1, 1]
names = []
max_ups = []
first_discs = []
cols = []
for name, fpath, col in files:
    if os.path.exists(fpath):
        d = json.load(open(fpath))
        u = np.array(d.get('upright_steps', []))
        names.append(name.replace(' (', '\n(').replace(' - SOLVED', ''))
        max_u = int(np.max(u)) if len(u) > 0 else 0
        max_ups.append(max_u)
        disc = d.get('first_discovery')
        solv = d.get('solved_episode')
        sub = f"Disc: Ep {disc}\nSolved: Ep {solv}" if solv else (f"Disc: Ep {disc}" if disc else "Undiscovered")
        first_discs.append(sub)
        cols.append(col)

bars = ax.bar(names, max_ups, color=cols, width=0.5, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Peak Upright Steps Achieved')
ax.set_title('Exploration Breakthrough: Deep Sea Construct with Gaussian Base Measure', fontsize=11, fontweight='bold')
ax.grid(True, axis='y', alpha=0.25)
for b, val, sub in zip(bars, max_ups, first_discs):
    y = b.get_height()
    ax.text(b.get_x() + b.get_width()/2., y + 15, f"{val} steps\n{sub}",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_ylim(0, 950)

plt.tight_layout()
local_png = 'cartpole_deepsea_construct_breakthrough.png'
dest_png = '/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/cartpole_deepsea_construct_breakthrough.png'
plt.savefig(local_png, dpi=200)
shutil.copyfile(local_png, dest_png)
print("Successfully generated breakthrough plot!")
