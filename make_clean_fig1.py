import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

methods = ['Deep Ensembles\n(Gaussian NLL)', 'BBB\n(Variational)', 'BootDQN\n+ Rand Priors', 'MC Dropout\n(p=0.10)', 'DP-BNN\n(Ours)']
cavity = [0.2004, 0.1097, 0.4627, 0.2847, 1.4072]
in_dist = [0.0282, 0.1577, 0.1227, 0.1213, 0.1597]

fig, ax = plt.subplots(figsize=(8.8, 4.0), dpi=300)
fig.patch.set_facecolor('white')

x = np.arange(len(methods))
w = 0.35

r1 = ax.bar(x - w/2, cavity, w, label=r'Cavity Void ($r \leq 1.0$)', color='#1b7837', edgecolor='black', alpha=0.9, zorder=3)
r2 = ax.bar(x + w/2, in_dist, w, label=r'Observed Annulus ($1.3 \leq r \leq 2.45$)', color='#a6dba0', edgecolor='black', alpha=0.9, zorder=3)

# Highlight DP-BNN bars
r1[4].set_color('#b2182b')
r1[4].set_edgecolor('black')
r2[4].set_color('#fddbc7')
r2[4].set_edgecolor('black')

# Annotate exact numbers for BOTH bars
for rect in r1:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                color='#990000' if rect == r1[4] else '#1b7837')

for rect in r2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                color='#b2182b' if rect == r2[4] else '#2b83ba')

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10, fontweight='bold')
ax.set_ylabel('BALD Epistemic Uncertainty (nats)', fontsize=11, fontweight='bold')
ax.set_title('2D Annular Manifold: Epistemic Uncertainty in Unobserved Void vs. Data Support', fontsize=11.5, fontweight='bold', pad=12)
ax.set_ylim(0, 1.70)
ax.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

plt.tight_layout()
for p in ['/Users/sumitvashishtha/Desktop/RLC_2026-3/Figure_1_2D_regression_NoLN.pdf',
          '/Users/sumitvashishtha/Desktop/RLC_2026-3/Figure_1_2D_regression_NoLN.png',
          '/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/Figure_1_2D_regression_NoLN.png',
          '/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/Figure_1_2D_regression_NoLN.pdf']:
    fig.savefig(p, bbox_inches='tight')
print('Successfully generated Figure_1_2D_regression_NoLN with numbers for both void and data!')
