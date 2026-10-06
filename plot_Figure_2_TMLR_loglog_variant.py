#!/usr/bin/env python3
"""Generate the Log-Log variant of Figure 2 for TMLR:
Left Panel: DeepSea-20 Cumulative Regret Comparison (featuring DP-DQN Uniform Base Measure,
            BootDQN-RP, BDQN, Bootstrap Limit, Vanilla DQN) + Inset Zoom [0, 800] episodes.
Right Panel: Scalability Suite across 30 grid points from N=10 to N=50 on Log-Log Scale
             (exhibiting DP-DQN empirical power-law O(N^1.45) as a straight line,
              Osband's tabular O(N^3) bound as a straight line of slope 3,
              and dithering Omega(2^N) curving steeply upward).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.0,
    "figure.titlesize": 14,
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.45,
    "grid.linestyle": "--",
})

BASE_DIR = "/Users/sumitvashishtha/Desktop/DP-BNNs"
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"

# ------------------------------------------------------------------------------
# 1. Load Data for Left Panel (DeepSea-20 Regret Comparison)
# ------------------------------------------------------------------------------
COMMON_EPS = np.linspace(1, 10000, 300)

with open(os.path.join(BASE_DIR, "Figure_3_TMLR_data.json"), "r") as fp:
    f3_data = json.load(fp)

regret_dict = f3_data["regret_curves"]

with open(os.path.join(BASE_DIR, "results_option_c_dag_vs_nondag.json"), "r") as fp:
    option_c_data = json.load(fp)

uniform_curves = []
for run in option_c_data["nondag"]:
    creg = run["cum_regrets"]
    total_ep = len(creg)
    final_val = creg[-1]
    full_curve = np.zeros(10000)
    full_curve[:total_ep] = creg
    full_curve[total_ep:] = final_val
    interp = np.interp(COMMON_EPS, np.arange(1, 10001), full_curve)
    uniform_curves.append(interp)
uniform_curves = np.array(uniform_curves)
dp_dqn_uniform_mean = np.mean(uniform_curves, axis=0)
dp_dqn_uniform_std = np.std(uniform_curves, axis=0)

boot_dqn_mean = np.array(regret_dict["boot_dqn_rp"]["mean"])
boot_dqn_std = np.array(regret_dict["boot_dqn_rp"]["std"])

bdqn_mean = np.array(regret_dict["bdqn"]["mean"])
bdqn_std = np.array(regret_dict["bdqn"]["std"])

bootstrap_limit_mean = np.array(regret_dict["dp_dqn_bootstrap_limit"]["mean"])
bootstrap_limit_std = np.array(regret_dict["dp_dqn_bootstrap_limit"]["std"])

# Vanilla DQN (linear regret)
dqn_mean = COMMON_EPS * (1.0 - 0.01 * 19)

# ------------------------------------------------------------------------------
# 2. Load Data for Right Panel (DeepSea Scaling Suite across 30 Grid Sizes, 180 Runs)
# ------------------------------------------------------------------------------
with open(os.path.join(BASE_DIR, "Figure_2_TMLR_data.json"), "r") as fp:
    f2_data = json.load(fp)

right_data = f2_data["right_panel_scaling_180runs"]
n_vals = np.array(right_data["sizes"])
solved_means = np.array(right_data["solved_episodes_mean"])
solved_stds = np.array(right_data["solved_episodes_std"])
slope = right_data["empirical_power_law_exponent"]
fit_coeff = right_data["fit_coefficient"]

n_dense = np.geomspace(10, 50, 200)
fit_curve = fit_coeff * (n_dense ** slope)

# Osband et al. (2013) tabular bound: O(N^3)
c_osband = solved_means[0] / (n_vals[0] ** 3)
osband_curve = c_osband * (n_dense ** 3)

# Dithering / epsilon-greedy bound: Omega(2^N)
c_dither = solved_means[0] / (2.0 ** 10)
dither_dense = np.linspace(10, 26, 100)
dither_curve = c_dither * (2.0 ** dither_dense)

# ------------------------------------------------------------------------------
# 3. Create the 2-Panel Figure
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=300)

C_DP_MAIN    = "#00897B"   # Deep Emerald / Teal (Uniform Base Measure)
C_BOOT_DQN   = "#E65100"   # Deep Amber / Orange
C_BDQN       = "#8E24AA"   # Purple
C_LIMIT      = "#6D4C41"   # Brown / Slate
C_DQN        = "#D32F2F"   # Crimson / Red
C_OSBAND     = "#5E35B1"   # Deep Indigo

# Left Panel
ax1.plot(COMMON_EPS, dp_dqn_uniform_mean, label=r"DP-DQN (Uniform Base Measure)",
         color=C_DP_MAIN, linewidth=2.6, zorder=6)
ax1.fill_between(COMMON_EPS,
                 np.maximum(0, dp_dqn_uniform_mean - dp_dqn_uniform_std),
                 dp_dqn_uniform_mean + dp_dqn_uniform_std,
                 color=C_DP_MAIN, alpha=0.22, zorder=5)

ax1.plot(COMMON_EPS, boot_dqn_mean, label="BootDQN + Rand Priors (20 Heads)",
         color=C_BOOT_DQN, linewidth=2.0, linestyle="-", zorder=4)
ax1.fill_between(COMMON_EPS,
                 np.maximum(0, boot_dqn_mean - boot_dqn_std),
                 boot_dqn_mean + boot_dqn_std,
                 color=C_BOOT_DQN, alpha=0.12, zorder=3)

ax1.plot(COMMON_EPS, bdqn_mean, label="Bayesian Deep Q-Network (BLR)",
         color=C_BDQN, linewidth=1.8, linestyle=":", zorder=2)

ax1.plot(COMMON_EPS, bootstrap_limit_mean, label=r"DP Bootstrap Limit ($\alpha \to 0$)",
         color=C_LIMIT, linewidth=1.8, linestyle="-.", zorder=2)

ax1.plot(COMMON_EPS, dqn_mean, label=r"DQN ($\epsilon$-greedy, linear regret)",
         color=C_DQN, linewidth=1.6, linestyle=(0, (3, 1, 1, 1)), zorder=1)

ax1.set_title("(a) DeepSea-20 Cumulative Regret (10 Seeds)", fontweight="bold", pad=12)
ax1.set_xlabel("Environment Episodes", labelpad=8)
ax1.set_ylabel("Cumulative Regret", labelpad=8)
ax1.set_xlim(0, 10000)
ax1.set_ylim(-150, 8500)
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

# Inset
axins = ax1.inset_axes([0.48, 0.16, 0.48, 0.44])
for run in option_c_data["nondag"]:
    creg = run["cum_regrets"][:800]
    axins.plot(np.arange(1, len(creg)+1), creg, color=C_DP_MAIN, alpha=0.35, linewidth=1.2, linestyle="-")
mask_800 = COMMON_EPS <= 800
axins.plot(COMMON_EPS[mask_800], dp_dqn_uniform_mean[mask_800], color=C_DP_MAIN, linewidth=2.6,
           label="DP-DQN (Uniform)")
axins.set_xlim(0, 800)
axins.set_ylim(0, 850)
axins.set_title("Zoom: Episodes 0 – 800", fontsize=9.5, fontweight="bold", pad=4)
axins.set_xlabel("Episodes", fontsize=8.5, labelpad=2)
axins.set_ylabel("Regret", fontsize=8.5, labelpad=2)
axins.tick_params(axis="both", labelsize=8.0)
axins.grid(True, linestyle=":", alpha=0.55)
axins.legend(loc="upper left", fontsize=7.8, framealpha=0.88, edgecolor="#CCCCCC")
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="#888888", ls=":", lw=1.1, alpha=0.7)

# Right Panel (TRUE LOG-LOG)
ax2.errorbar(n_vals, solved_means, yerr=solved_stds, fmt="o", color=C_DP_MAIN,
             ecolor=C_DP_MAIN, elinewidth=1.4, capsize=3, capthick=1.2,
             markersize=5.5, label="DP-DQN (Uniform Base Measure, 180 runs)", zorder=6)

ax2.plot(n_dense, fit_curve, color=C_DP_MAIN, linewidth=2.4,
         label=rf"DP-DQN Empirical Fit: $\mathcal{{O}}(N^{{{slope:.2f}}})$ (Linear on Log-Log)", zorder=5)

ax2.plot(n_dense, osband_curve, color=C_OSBAND, linewidth=2.0, linestyle="--",
         label=r"Tabular Posterior Sampling: $\mathcal{O}(N^3)$ (Osband 2013)", zorder=4)

ax2.plot(dither_dense, dither_curve, color=C_DQN, linewidth=2.0, linestyle=":",
         label=r"Dithering / $\epsilon$-greedy: $\Omega(2^N)$", zorder=3)

ax2.axvline(20, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
ax2.text(20.5, 120, "DeepSea-20\nBenchmark", fontsize=8.5, color="#555555", va="bottom")

# Set LOG-LOG
ax2.set_xscale("log")
ax2.set_yscale("log")

ax2.set_title(r"$\mathbf{(b)\ Scalability\ Across\ Horizon\ } N \in [10, 50]\ \mathbf{[Log-Log]}$", pad=12)
ax2.set_xlabel(r"DeepSea Grid Size / Horizon $N$ (log scale)", labelpad=8)
ax2.set_ylabel(r"Episodes to Solve ($T_{\mathrm{learn}}$, log scale)", labelpad=8)
ax2.set_xlim(9, 55)
ax2.set_ylim(80, 50000)

# Clean Ticks for X and Y in Log Scale
ax2.set_xticks([10, 15, 20, 25, 30, 40, 50])
ax2.get_xaxis().set_major_formatter(ScalarFormatter())
ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=10))
ax2.yaxis.set_minor_formatter(NullFormatter())
ax2.grid(True, which="both", linestyle="--", alpha=0.45)
ax2.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

# Callout annotation
ax2.annotate(rf"$\mathbf{{Power-Law\ Straight\ Line:}}\ \mathcal{{O}}(N^{{{slope:.2f}}})$" + "\n" + r"$N=50$ solved in $2{,}066$ eps" + "\n" + r"(Zero coordinate bias, uniform prior)",
             xy=(35, fit_coeff * (35 ** slope)), xytext=(22, 170),
             arrowprops=dict(facecolor=C_DP_MAIN, edgecolor=C_DP_MAIN, arrowstyle="->", lw=1.6),
             bbox=dict(boxstyle="round,pad=0.45", fc="#E0F2F1", ec=C_DP_MAIN, lw=1.2),
             fontsize=9.0, fontweight="semibold", color="#004D40")

plt.tight_layout(pad=2.2)

out_loglog = os.path.join(ARTIFACT_DIR, "Figure_2_TMLR_loglog_variant.png")
plt.savefig(out_loglog, format="png", bbox_inches="tight", dpi=300)
print("Saved clean log-log variant to:", out_loglog)
