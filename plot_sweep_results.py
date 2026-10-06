import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_sweep():
    with open("results/dp_sweep_results.json", "r") as f:
        data = json.load(f)

    mean_sweep = data["sweep_prior_mean"]
    alpha_sweep = data["sweep_alpha"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Base Measure Prior Mean
    ax1 = axes[0]
    p_means = [x["prior_mean"] for x in mean_sweep]
    regrets_pm = [x["mean_regret"] for x in mean_sweep]
    errs_pm = [x["std_err"] for x in mean_sweep]
    opt_pm = [x["mean_outside_opt"] for x in mean_sweep]

    color = "#1f77b4"
    ax1.errorbar(p_means, regrets_pm, yerr=errs_pm, fmt='-o', color=color, linewidth=2.2, markersize=7, capsize=5, label="Cumulative Regret")
    ax1.set_xlabel(r"Base Measure Optimistic Prior Mean ($\mu_0$)", fontsize=11)
    ax1.set_ylabel("Final Regret (Lower is Better)", fontsize=11, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title(r"Sensitivity to Base Measure Optimism ($\mu_0$)", fontsize=12, fontweight="bold")

    # Secondary axis for discovery rate
    ax1_twin = ax1.twinx()
    color_twin = "#2ca02c"
    ax1_twin.plot(p_means, opt_pm, '--s', color=color_twin, linewidth=2.0, markersize=6, label="Outer Opt Discovery %")
    ax1_twin.set_ylabel("Outer Quadrant Discovery Rate (%)", fontsize=11, color=color_twin)
    ax1_twin.tick_params(axis='y', labelcolor=color_twin)

    # Highlight optimal point
    best_idx = regrets_pm.index(min(regrets_pm))
    ax1.annotate(f"Optimal $\\mu_0 = {p_means[best_idx]}$\n({regrets_pm[best_idx]:.0f} regret)",
                 xy=(p_means[best_idx], regrets_pm[best_idx]),
                 xytext=(-20, 25), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                 fontsize=9, fontweight="bold")

    # 2. DP Concentration Parameter Alpha
    ax2 = axes[1]
    alphas = [x["alpha"] for x in alpha_sweep]
    regrets_a = [x["mean_regret"] for x in alpha_sweep]
    errs_a = [x["std_err"] for x in alpha_sweep]
    opt_a = [x["mean_outside_opt"] for x in alpha_sweep]

    color = "#9467bd"
    ax2.errorbar(alphas, regrets_a, yerr=errs_a, fmt='-o', color=color, linewidth=2.2, markersize=7, capsize=5, label="Cumulative Regret")
    ax2.set_xlabel(r"DP Concentration Parameter ($\alpha$)", fontsize=11)
    ax2.set_ylabel("Final Regret (Lower is Better)", fontsize=11, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title(r"Sensitivity to Prior Mass ($\alpha$)", fontsize=12, fontweight="bold")

    ax2_twin = ax2.twinx()
    color_twin = "#ff7f0e"
    ax2_twin.plot(alphas, opt_a, '--s', color=color_twin, linewidth=2.0, markersize=6, label="Outer Opt Discovery %")
    ax2_twin.set_ylabel("Outer Quadrant Discovery Rate (%)", fontsize=11, color=color_twin)
    ax2_twin.tick_params(axis='y', labelcolor=color_twin)

    best_a_idx = regrets_a.index(min(regrets_a))
    ax2.annotate(f"Optimal $\\alpha = {alphas[best_a_idx]}$\n({regrets_a[best_a_idx]:.0f} regret)",
                 xy=(alphas[best_a_idx], regrets_a[best_a_idx]),
                 xytext=(-20, 25), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                 fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_file = "results/dp_hyperparameter_sweep.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Sweep plot saved to {out_file}")


if __name__ == "__main__":
    plot_sweep()
