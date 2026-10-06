import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def smooth(y, box_pts=25):
    box = np.ones(box_pts)/box_pts
    return np.convolve(y, box, mode='same')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. First Discovery Episodes comparison
ax1 = axes[0]
cases = [
    ('Dirichlet α=15 (50x50)', 'results_dirichlet_vs_stick/cartpole_swingup_dp_dqn_dirichlet_s42_ckpt.json', 'forestgreen'),
    ('BootDQN K=20', 'results_boot_dqn_2500ep/cartpole_swingup_boot_dqn_s42_ckpt.json', 'crimson'),
    ('Dirichlet α=3 NoLN (64x64)', 'results_64x64_alpha3/cartpole_swingup_dp_dqn_dir_noln_64_s42_ckpt.json', 'limegreen'),
    ('Dirichlet α=3 LN (64x64)', 'results_64x64_alpha3/cartpole_swingup_dp_dqn_dir_ln_64_s42_ckpt.json', 'darkgreen'),
    ('VM Stick α=3 NoLN (64x64)', 'results_64x64_alpha3/cartpole_swingup_dp_dqn_vm_noln_64_s42_ckpt.json', 'dodgerblue'),
    ('VM Stick α=3 LN (64x64)', 'results_64x64_alpha3/cartpole_swingup_dp_dqn_vm_ln_64_s42_ckpt.json', 'royalblue'),
]

disc_names, disc_eps, bar_colors = [], [], []
for label, path, col in cases:
    if os.path.exists(path):
        d = json.load(open(path))
        ep = d.get('first_discovery', None)
        disc_names.append(label)
        disc_eps.append(ep if ep is not None else 700) # 700 = not discovered yet
        bar_colors.append(col)

bars = ax1.barh(disc_names[::-1], disc_eps[::-1], color=bar_colors[::-1], alpha=0.85, edgecolor='black')
for bar, ep_val in zip(bars, disc_eps[::-1]):
    text = f'Ep {ep_val}' if ep_val < 700 else 'No Discovery Yet (>700)'
    ax1.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2, text, va='center', fontweight='bold', fontsize=9)
ax1.set_xlim(0, 850)
ax1.set_title('First Goal Discovery Episode (Seed 42)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Episode of First Upright Arrival (|θ| < 0.2 rad)')
ax1.grid(True, linestyle='--', alpha=0.5)

# 2. Return Curves for Seed 42 across alpha=15 vs alpha=3 vs BootDQN
ax2 = axes[1]
for label, path, col in cases[:4]:
    if os.path.exists(path):
        d = json.load(open(path))
        rets = d['returns']
        eps = np.arange(1, len(rets) + 1)
        ax2.plot(eps, smooth(rets, box_pts=25), label=f'{label} (ep={len(rets)})', color=col, linewidth=2.0)

ax2.set_title('Episodic Return on Seed 42', fontsize=12, fontweight='bold')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Undiscounted Return')
ax2.set_xlim(0, 1000)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper left', fontsize=8.5)

# 3. Upright Steps on Seed 42
ax3 = axes[2]
for label, path, col in cases[:4]:
    if os.path.exists(path):
        d = json.load(open(path))
        uprs = d['upright_steps']
        eps = np.arange(1, len(uprs) + 1)
        ax3.plot(eps, smooth(uprs, box_pts=25), label=f'{label}', color=col, linewidth=2.0)

ax3.axhline(200, color='gray', linestyle=':', label='Upright Benchmark (200)')
ax3.set_title('Upright Balancing Steps on Seed 42', fontsize=12, fontweight='bold')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Upright Steps / Episode')
ax3.set_xlim(0, 1000)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(loc='upper left', fontsize=8.5)

plt.suptitle('Diagnostic Investigation: Impact of Concentration α (15 vs 3), LayerNorm, and Architecture on Seed 42', fontsize=13, y=1.02)
plt.tight_layout()

out_path = '/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/diagnostic_investigation_alpha_ln.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print('Saved diagnostic comparison to', out_path)
