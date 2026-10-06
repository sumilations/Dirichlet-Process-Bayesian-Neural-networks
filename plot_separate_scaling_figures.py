"""Generate separate, publication-quality scaling figures for DAG and Non-DAG priors."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
})

# Load benchmark results
json_path = "results_deepsea_scaling/deepsea_scaling_results.json"
with open(json_path, "r") as fp:
    data = json.load(fp)

# Organize by base measure and size
data_by_variant = {
    "dag": {"disc": defaultdict(list), "solv": defaultdict(list)},
    "nondag": {"disc": defaultdict(list), "solv": defaultdict(list)}
}

for item in data:
    bm = item["base_measure"]
    sz = item["size"]
    d = item.get("first_discovery")
    s = item.get("solved_episode")

    var_key = "dag" if "dag" in bm and "nondag" not in bm else "nondag"
    if d is not None:
        data_by_variant[var_key]["disc"][sz].append(d)
    if s is not None:
        data_by_variant[var_key]["solv"][sz].append(s)


def get_stats(data_dict):
    sizes = sorted(data_dict.keys())
    means, stds = [], []
    for sz in sizes:
        vals = data_dict[sz]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(sizes), np.array(means), np.array(stds)


# ==============================================================================
# 1. Figure: DAG Prior Scaling
# ==============================================================================
dag_disc_sz, dag_disc_m, dag_disc_s = get_stats(data_by_variant["dag"]["disc"])
dag_solv_sz, dag_solv_m, dag_solv_s = get_stats(data_by_variant["dag"]["solv"])

p_dag_disc = np.polyfit(np.log(dag_disc_sz), np.log(dag_disc_m), 1)
p_dag_solv = np.polyfit(np.log(dag_solv_sz), np.log(dag_solv_m), 1)

fig_dag, axes_dag = plt.subplots(1, 2, figsize=(14, 5.2))
fig_dag.patch.set_facecolor("#ffffff")

# Panel 1: Log-Log Sample Complexity
ax1 = axes_dag[0]
ax1.errorbar(dag_disc_sz, dag_disc_m, yerr=dag_disc_s, fmt='o-', color='#1f77b4',
             lw=2.2, capsize=4, label=f'Discovery: $T \\sim \\mathcal{{O}}(N^{{{p_dag_disc[0]:.2f}}})$')
ax1.errorbar(dag_solv_sz, dag_solv_m, yerr=dag_solv_s, fmt='s--', color='#2ca02c',
             lw=2.2, capsize=4, label=f'Solved (50-MA): $T \\sim \\mathcal{{O}}(N^{{{p_dag_solv[0]:.2f}}})$')

# Fits and theoretical references
n_fine = np.linspace(5, 50, 100)
fit_disc = np.exp(p_dag_disc[1]) * (n_fine ** p_dag_disc[0])
ax1.plot(n_fine, fit_disc, '-', color='#1f77b4', alpha=0.5, lw=1.8)

# Lower bound: O(N^2)
c_lower = dag_disc_m[4] / (dag_disc_sz[4] ** 2.0)
ax1.plot(n_fine, c_lower * (n_fine ** 2.0), ':', color='black', alpha=0.7, lw=2.0,
         label=r'$\Omega(N^2)$ Information-Theoretic Lower Bound')

# Random walk baseline
n_rw = np.linspace(5, 13, 50)
ax1.plot(n_rw, 2.0 ** (n_rw - 1), 'r:', lw=2.0, label=r'Random Walk: $\Omega(2^N)$')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Deep Sea Size $N$ (State Space $|\mathcal{S}| = 2^N$)')
ax1.set_ylabel('Episodes (Log Scale)')
ax1.set_title('DAG Prior: Log-Log Sample Complexity Scaling', fontweight='bold')
ax1.legend(loc='upper left', frameon=True)
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax1.set_ylim(10, 1e5)

# Panel 2: Linear Scale Discovery and Solved Trajectories
ax2 = axes_dag[1]
ax2.plot(dag_disc_sz, dag_disc_m, 'o-', color='#1f77b4', lw=2.2, label='Discovery Episodes (Mean ± 1 std)')
ax2.fill_between(dag_disc_sz, np.maximum(0, dag_disc_m - dag_disc_s), dag_disc_m + dag_disc_s,
                 color='#1f77b4', alpha=0.2)

ax2.plot(dag_solv_sz, dag_solv_m, 's--', color='#2ca02c', lw=2.2, label='Solved Episodes (Mean ± 1 std)')
ax2.fill_between(dag_solv_sz, np.maximum(0, dag_solv_m - dag_solv_s), dag_solv_m + dag_solv_s,
                 color='#2ca02c', alpha=0.15)

ax2.set_xlabel('Deep Sea Size $N$')
ax2.set_ylabel('Episodes to Discover / Solve')
ax2.set_title('DAG Prior: Linear Scaling vs. Depth $N$', fontweight='bold')
ax2.legend(loc='upper left', frameon=True)
ax2.grid(True, ls="--", alpha=0.4)

plt.tight_layout()
dag_out = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/deepsea_scaling_dag.png"
fig_dag.savefig(dag_out, dpi=300)
fig_dag.savefig("deepsea_scaling_dag.png", dpi=300)
print(f"Saved DAG scaling figure to {dag_out}")


# ==============================================================================
# 2. Figure: Non-DAG Model-Free Prior Scaling
# ==============================================================================
nondag_disc_sz, nondag_disc_m, nondag_disc_s = get_stats(data_by_variant["nondag"]["disc"])
nondag_solv_sz, nondag_solv_m, nondag_solv_s = get_stats(data_by_variant["nondag"]["solv"])

p_nondag_disc = np.polyfit(np.log(nondag_disc_sz), np.log(nondag_disc_m), 1)
p_nondag_solv = np.polyfit(np.log(nondag_solv_sz), np.log(nondag_solv_m), 1)

fig_nondag, axes_nondag = plt.subplots(1, 2, figsize=(14, 5.2))
fig_nondag.patch.set_facecolor("#ffffff")

# Panel 1: Log-Log Sample Complexity
ax1_nd = axes_nondag[0]
ax1_nd.errorbar(nondag_disc_sz, nondag_disc_m, yerr=nondag_disc_s, fmt='o-', color='#d62728',
                lw=2.2, capsize=4, label=f'Discovery: $T \\sim \\mathcal{{O}}(N^{{{p_nondag_disc[0]:.2f}}}) \\approx \\mathcal{{O}}(N^2)$')
ax1_nd.errorbar(nondag_solv_sz, nondag_solv_m, yerr=nondag_solv_s, fmt='s--', color='#9467bd',
                lw=2.2, capsize=4, label=f'Solved (50-MA): $T \\sim \\mathcal{{O}}(N^{{{p_nondag_solv[0]:.2f}}})$')

# Fits and theoretical references
fit_disc_nd = np.exp(p_nondag_disc[1]) * (n_fine ** p_nondag_disc[0])
ax1_nd.plot(n_fine, fit_disc_nd, '-', color='#d62728', alpha=0.5, lw=1.8)

# Lower bound: O(N^2)
c_lower_nd = nondag_disc_m[4] / (nondag_disc_sz[4] ** 2.0)
ax1_nd.plot(n_fine, c_lower_nd * (n_fine ** 2.0), ':', color='black', alpha=0.7, lw=2.0,
            label=r'$\Omega(N^2)$ Information-Theoretic Lower Bound')

# Random walk baseline
ax1_nd.plot(n_rw, 2.0 ** (n_rw - 1), 'r:', lw=2.0, label=r'Random Walk: $\Omega(2^N)$')

ax1_nd.set_xscale('log')
ax1_nd.set_yscale('log')
ax1_nd.set_xlabel('Deep Sea Size $N$ (State Space $|\mathcal{S}| = 2^N$)')
ax1_nd.set_ylabel('Episodes (Log Scale)')
ax1_nd.set_title('Non-DAG Model-Free: Log-Log Sample Complexity Scaling', fontweight='bold')
ax1_nd.legend(loc='upper left', frameon=True)
ax1_nd.grid(True, which="both", ls="--", alpha=0.4)
ax1_nd.set_ylim(10, 1e5)

# Panel 2: Linear Scale Discovery and Solved Trajectories
ax2_nd = axes_nondag[1]
ax2_nd.plot(nondag_disc_sz, nondag_disc_m, 'o-', color='#d62728', lw=2.2, label='Discovery Episodes (Mean ± 1 std)')
ax2_nd.fill_between(nondag_disc_sz, np.maximum(0, nondag_disc_m - nondag_disc_s), nondag_disc_m + nondag_disc_s,
                    color='#d62728', alpha=0.2)

ax2_nd.plot(nondag_solv_sz, nondag_solv_m, 's--', color='#9467bd', lw=2.2, label='Solved Episodes (Mean ± 1 std)')
ax2_nd.fill_between(nondag_solv_sz, np.maximum(0, nondag_solv_m - nondag_solv_s), nondag_solv_m + nondag_solv_s,
                    color='#9467bd', alpha=0.15)

ax2_nd.set_xlabel('Deep Sea Size $N$')
ax2_nd.set_ylabel('Episodes to Discover / Solve')
ax2_nd.set_title('Non-DAG Model-Free: Linear Scaling vs. Depth $N$', fontweight='bold')
ax2_nd.legend(loc='upper left', frameon=True)
ax2_nd.grid(True, ls="--", alpha=0.4)

plt.tight_layout()
nondag_out = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/deepsea_scaling_nondag.png"
fig_nondag.savefig(nondag_out, dpi=300)
fig_nondag.savefig("deepsea_scaling_nondag.png", dpi=300)
print(f"Saved Non-DAG scaling figure to {nondag_out}")


# ==============================================================================
# 3. Standalone Single-Panel Scaling Figures (Compact Column Format)
# ==============================================================================
# DAG Single Panel
fig_dag_s, ax_dag_s = plt.subplots(figsize=(7.5, 5.5))
fig_dag_s.patch.set_facecolor("#ffffff")
ax_dag_s.errorbar(dag_disc_sz, dag_disc_m, yerr=dag_disc_s, fmt='o-', color='#1f77b4',
                  lw=2.4, capsize=4, label=f'Empirical Discovery: $T \\sim \\mathcal{{O}}(N^{{{p_dag_disc[0]:.2f}}})$')
ax_dag_s.plot(n_fine, c_lower * (n_fine ** 2.0), ':', color='black', alpha=0.7, lw=2.0,
              label=r'$\Omega(N^2)$ Lower Bound (Osband et al.)')
ax_dag_s.plot(n_rw, 2.0 ** (n_rw - 1), 'r:', lw=2.0, label=r'Random Walk: $\Omega(2^N)$')
ax_dag_s.set_xscale('log')
ax_dag_s.set_yscale('log')
ax_dag_s.set_xlabel('Deep Sea Size $N$ ($|\mathcal{S}| = 2^N$)')
ax_dag_s.set_ylabel('Episodes to Find Treasure (Log Scale)')
ax_dag_s.set_title('DAG Prior: Empirical Scaling vs. Theoretical Bound', fontweight='bold')
ax_dag_s.legend(loc='upper left', frameon=True)
ax_dag_s.grid(True, which="both", ls="--", alpha=0.4)
ax_dag_s.set_ylim(10, 1e5)
plt.tight_layout()
dag_single_out = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/deepsea_scaling_dag_standalone.png"
fig_dag_s.savefig(dag_single_out, dpi=300)
fig_dag_s.savefig("deepsea_scaling_dag_standalone.png", dpi=300)

# Non-DAG Single Panel
fig_nd_s, ax_nd_s = plt.subplots(figsize=(7.5, 5.5))
fig_nd_s.patch.set_facecolor("#ffffff")
ax_nd_s.errorbar(nondag_disc_sz, nondag_disc_m, yerr=nondag_disc_s, fmt='o-', color='#d62728',
                 lw=2.4, capsize=4, label=f'Empirical Discovery: $T \\sim \\mathcal{{O}}(N^{{{p_nondag_disc[0]:.2f}}}) \\approx \\mathcal{{O}}(N^2)$')
ax_nd_s.plot(n_fine, c_lower_nd * (n_fine ** 2.0), ':', color='black', alpha=0.7, lw=2.0,
             label=r'$\Omega(N^2)$ Lower Bound (Osband et al.)')
ax_nd_s.plot(n_rw, 2.0 ** (n_rw - 1), 'r:', lw=2.0, label=r'Random Walk: $\Omega(2^N)$')
ax_nd_s.set_xscale('log')
ax_nd_s.set_yscale('log')
ax_nd_s.set_xlabel('Deep Sea Size $N$ ($|\mathcal{S}| = 2^N$)')
ax_nd_s.set_ylabel('Episodes to Find Treasure (Log Scale)')
ax_nd_s.set_title('Non-DAG Model-Free: Empirical Scaling vs. Theoretical Bound', fontweight='bold')
ax_nd_s.legend(loc='upper left', frameon=True)
ax_nd_s.grid(True, which="both", ls="--", alpha=0.4)
ax_nd_s.set_ylim(10, 1e5)
plt.tight_layout()
nd_single_out = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/deepsea_scaling_nondag_standalone.png"
fig_nd_s.savefig(nd_single_out, dpi=300)
fig_nd_s.savefig("deepsea_scaling_nondag_standalone.png", dpi=300)

print("Generated all standalone scaling figures successfully!")
