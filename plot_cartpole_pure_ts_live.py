import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor('#ffffff')

colors = {
    's42': '#1f77b4',
    's43': '#ff7f0e',
    's44': '#2ca02c',
    's45': '#d62728',
    's46': '#9467bd',
}

files = sorted(glob.glob('./results_cartpole_pure_ts/*_ckpt.json'))

def smooth(x, window=10):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='valid')

# 1. Episode Returns
ax = axes[0, 0]
for fpath in files:
    d = json.load(open(fpath))
    s = f"s{d.get('seed')}"
    col = colors.get(s, '#333333')
    r = np.array(d.get('returns', []))
    solv = d.get('solved_episode')
    disc = d.get('first_discovery')
    lbl = f"Seed {d.get('seed')} (Solved: Ep {solv})" if solv else f"Seed {d.get('seed')} (Disc: Ep {disc})"
    if len(r) > 0:
        ax.plot(r, color=col, alpha=0.15)
        sm = smooth(r, 10)
        ax.plot(np.arange(len(r)-len(sm), len(r)), sm, color=col, lw=2.2, label=lbl)

ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(750, color='forestgreen', linestyle=':', alpha=0.8, lw=1.5, label='Solved Threshold (~750)')
ax.set_title('Episode Return: Cart-Pole Swing-Up Pure TS (Gaussian State + Optimistic Reward)', fontsize=11, fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Undiscounted Return')
ax.grid(True, alpha=0.25)
ax.legend(loc='upper left', fontsize=8.5)

# 2. Upright Steps
ax = axes[0, 1]
for fpath in files:
    d = json.load(open(fpath))
    s = f"s{d.get('seed')}"
    col = colors.get(s, '#333333')
    u = np.array(d.get('upright_steps', []))
    max_u = max(u) if len(u) > 0 else 0
    if len(u) > 0:
        ax.plot(u, color=col, alpha=0.2)
        sm = smooth(u, 10)
        ax.plot(np.arange(len(u)-len(sm), len(u)), sm, color=col, lw=2.0, label=f"Seed {d.get('seed')} (Max: {max_u})")

ax.set_title('Upright Balancing Steps per Episode (Max: 1000)', fontsize=11, fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Steps (|cos θ| > 0.95)')
ax.grid(True, alpha=0.25)
ax.legend(loc='upper left', fontsize=8.5)

# 3. Cumulative Regret
ax = axes[1, 0]
for fpath in files:
    d = json.load(open(fpath))
    s = f"s{d.get('seed')}"
    col = colors.get(s, '#333333')
    reg = np.array(d.get('cumulative_regrets', []))
    if len(reg) > 0:
        ax.plot(reg / 1e3, color=col, lw=2.0, label=f"Seed {d.get('seed')}")

ax.set_title('Cumulative Regret vs. Episodes (Lower = Sublinear Convergence)', fontsize=11, fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Cumulative Regret (x10³)')
ax.grid(True, alpha=0.25)
ax.legend(loc='upper left', fontsize=8.5)

# 4. Summary Bar Chart
ax = axes[1, 1]
names = []
max_ups = []
annotations = []
cols = []

for fpath in files:
    d = json.load(open(fpath))
    s = f"s{d.get('seed')}"
    col = colors.get(s, '#333333')
    u = np.array(d.get('upright_steps', []))
    max_u = int(max(u)) if len(u) > 0 else 0
    disc = d.get('first_discovery')
    solv = d.get('solved_episode')
    sub = f"Disc: Ep {disc}\nSolved: Ep {solv}" if solv else (f"Disc: Ep {disc}" if disc else "Exploring")
    names.append(f"Seed {d.get('seed')}")
    max_ups.append(max_u)
    annotations.append(sub)
    cols.append(col)

bars = ax.bar(names, max_ups, color=cols, width=0.5, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Peak Upright Steps Achieved (out of 1000)')
ax.set_title('Pure TS Exploration Efficiency on Cart-Pole', fontsize=11, fontweight='bold')
ax.grid(True, axis='y', alpha=0.25)
ax.set_ylim(0, 950)

for b, val, sub in zip(bars, max_ups, annotations):
    y = b.get_height()
    ax.text(b.get_x() + b.get_width()/2., y + 15, f"{val} steps\n{sub}",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
out_png = '/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/cartpole_pure_ts_live_progress.png'
plt.savefig(out_png, dpi=200)
plt.savefig('cartpole_pure_ts_live_progress.png', dpi=200)
print(f"Saved plot to {out_png}")
