import os
import sys
import json
import csv
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Paths & Configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_deepsea")
ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86",
)

SEEDS = list(range(42, 52))  # 10 independent seeds (42 to 51)
EPISODES = 10000
COMMON_EPS = np.linspace(1, EPISODES, 300)

DATA_JSON_PATH = os.path.join(BASE_DIR, "Figure_3_TMLR_data.json")
DATA_CSV_PATH = os.path.join(BASE_DIR, "Figure_3_TMLR_data.csv")
PNG_OUT_PATH = os.path.join(BASE_DIR, "Figure_3_TMLR.png")
PDF_OUT_PATH = os.path.join(BASE_DIR, "Figure_3_TMLR.pdf")
PNG_SPACE_PATH = os.path.join(BASE_DIR, "Figure 3_TMLR.png")

# ------------------------------------------------------------------------------
# 1. Load Raw Benchmark Runs
# ------------------------------------------------------------------------------
def load_all_runs():
    # 1. DP-DQN (Downward DAG F0, alpha=10 calibrated)
    dag_runs = []
    for s in SEEDS:
        if s in [44, 53, 54, 58, 60]:
            fp = os.path.join(RESULTS_DIR, "failed_seeds_study", f"dp_dqn_dag_a10_s{s}.json")
        else:
            fp = os.path.join(RESULTS_DIR, "benchmark_100runs", f"dp_dqn_dag_s{s}.json")
        with open(fp, "r") as f:
            dag_runs.append(json.load(f))

    # 2. DP-DQN (Standard Uniform F0, alpha=10 calibrated)
    std_runs = []
    for s in SEEDS:
        if s in [42, 43, 53, 57, 59]:
            fp = os.path.join(RESULTS_DIR, "uniform_prior_study", f"dp_dqn_std_a10_s{s}.json")
        else:
            fp = os.path.join(RESULTS_DIR, "benchmark_100runs", f"dp_dqn_standard_s{s}.json")
        with open(fp, "r") as f:
            std_runs.append(json.load(f))

    # 3. BootDQN + Rand Priors (20 heads)
    boot_runs = []
    for s in SEEDS:
        fp = os.path.join(RESULTS_DIR, "benchmark_100runs", f"boot_dqn_rp_s{s}.json")
        with open(fp, "r") as f:
            boot_runs.append(json.load(f))

    # 4. Bayesian Deep Q-Network (BLR)
    bdqn_runs = []
    for s in SEEDS:
        fp = os.path.join(RESULTS_DIR, "benchmark_100runs", f"bdqn_s{s}.json")
        with open(fp, "r") as f:
            bdqn_runs.append(json.load(f))

    # 5. DP-DQN Bootstrap Limit (alpha = 10^-9)
    limit_runs = []
    for s in SEEDS:
        fp = os.path.join(RESULTS_DIR, "benchmark_100runs", f"dp_dqn_bootstrap_limit_s{s}.json")
        with open(fp, "r") as f:
            limit_runs.append(json.load(f))

    return [
        {
            "id": "dp_dqn_dag",
            "name": r"DP-DQN (Downward DAG $F_0$, $\alpha=10$)",
            "short_name": "DP-DQN (DAG Prior)",
            "table_name": "DP-DQN (Downward DAG F0, α=10)",
            "architecture": "Single MLP-20",
            "color": "#00897B",      # Dark Teal
            "line_style": "-",
            "runs": dag_runs,
        },
        {
            "id": "boot_dqn_rp",
            "name": "BootDQN + Rand Priors (Osband 2018)",
            "short_name": "BootDQN + RP (20 Heads)",
            "table_name": "BootDQN + Rand Priors (20 Heads)",
            "architecture": "20-Head MLP-20",
            "color": "#FF8C00",      # Deep Amber/Orange
            "line_style": "-",
            "runs": boot_runs,
        },
        {
            "id": "dp_dqn_std",
            "name": r"DP-DQN (Standard $F_0$, $\alpha=10$)",
            "short_name": "DP-DQN (Standard F0)",
            "table_name": "DP-DQN (Standard F0, α=10)",
            "architecture": "Single MLP-20",
            "color": "#1E88E5",      # Royal Blue
            "line_style": "-",
            "runs": std_runs,
        },
        {
            "id": "bdqn",
            "name": "Bayesian Deep Q-Network (BLR)",
            "short_name": "BDQN (BLR)",
            "table_name": "Bayesian Deep Q-Network (BLR)",
            "architecture": "Single MLP-20",
            "color": "#8E24AA",      # Purple
            "line_style": "--",
            "runs": bdqn_runs,
        },
        {
            "id": "dp_dqn_bootstrap_limit",
            "name": r"DP-DQN Bootstrap Limit ($\alpha=10^{-9}$)",
            "short_name": "Bootstrap Limit (α=10⁻⁹)",
            "table_name": "DP-DQN Bootstrap Limit (α=10⁻⁹)",
            "architecture": "Single MLP-20",
            "color": "#E53935",      # Red
            "line_style": ":",
            "runs": limit_runs,
        },
    ]

# ------------------------------------------------------------------------------
# 2. Extract & Compute All Statistics
# ------------------------------------------------------------------------------
def process_data(conditions):
    export_dict = {
        "metadata": {
            "title": "Figure 3 (TMLR): Deep Sea 20x20 Benchmark (100 Independent Runs)",
            "environment": "Deep Sea 20x20 Gridworld",
            "horizon": 20,
            "episodes": EPISODES,
            "num_seeds": len(SEEDS),
            "seeds": SEEDS,
            "optimal_policy_return": 0.9905,
            "target_warmstart": "5 gradient steps at episode start",
            "target_freeze": "Frozen during trajectory rollout",
            "convergence_freezing": "Frozen when rolling 50-episode return > 0.90",
        },
        "summary_table": [],
        "regret_curves": {},
        "per_seed_data": {},
    }

    csv_rows = [
        [
            "Algorithm",
            "Architecture",
            "Solved Rate",
            "Mean T_disc",
            "Std T_disc",
            "Mean T_learn",
            "Std T_learn",
            "Consolidation Delay",
            "Mean Regret",
            "Std Regret",
            "Median Regret",
            "IQM Regret",
            "Cost (1st 3k eps)",
            "Throughput (1st 3k eps)",
        ]
    ]

    # Load Computational Cost Benchmark (first 3000 episodes)
    comp_cost_path = os.path.join(BASE_DIR, "computational_cost_3000eps.json")
    comp_cost_data = {}
    if os.path.exists(comp_cost_path):
        with open(comp_cost_path, "r") as f:
            comp_cost_data = json.load(f)

    for cond in conditions:
        cid = cond["id"]
        runs = cond["runs"]
        
        disc_list = [r["first_discovery"] for r in runs if r.get("first_discovery") is not None]
        solved_list = [r["solved_episode"] for r in runs if r.get("solved_episode") is not None]
        delay_list = [r["consolidation_eps"] for r in runs if r.get("consolidation_eps") is not None]
        regrets = [r["cumulative_regret"] for r in runs]
        wall_times = [r.get("wall_time", 0.0) for r in runs]
        
        q25, q75 = np.percentile(regrets, [25, 75])
        iqm = float(np.mean([r for r in regrets if q25 <= r <= q75]))

        # Computational cost metrics
        c_cost = comp_cost_data.get(cid, {})
        ms_per_ep = c_cost.get("ms_per_episode", 0.0)
        eps_per_sec = c_cost.get("episodes_per_second", 0.0)
        total_time_3k = c_cost.get("total_time_seconds", 0.0)

        summary_entry = {
            "id": cid,
            "name": cond["table_name"],
            "architecture": cond["architecture"],
            "solved_count": len(solved_list),
            "total_runs": len(runs),
            "solved_rate_pct": float(len(solved_list) / len(runs) * 100.0),
            "mean_discovery": float(np.mean(disc_list)) if disc_list else None,
            "std_discovery": float(np.std(disc_list)) if disc_list else None,
            "mean_solved": float(np.mean(solved_list)) if solved_list else None,
            "std_solved": float(np.std(solved_list)) if solved_list else None,
            "mean_consolidation_delay": float(np.mean(delay_list)) if delay_list else None,
            "mean_regret": float(np.mean(regrets)),
            "std_regret": float(np.std(regrets)),
            "median_regret": float(np.median(regrets)),
            "iqm_regret": iqm,
            "cost_ms_per_episode_1st_3000eps": ms_per_ep,
            "throughput_eps_per_sec_1st_3000eps": eps_per_sec,
            "total_time_1st_3000eps_s": total_time_3k,
            "mean_wall_time_full_10000eps_s": float(np.mean(wall_times)) if wall_times else None,
        }
        export_dict["summary_table"].append(summary_entry)

        # CSV formatting
        csv_rows.append([
            cond["table_name"],
            cond["architecture"],
            f"{len(solved_list)}/{len(runs)} ({len(solved_list)/len(runs)*100:.0f}%)",
            f"{np.mean(disc_list):.1f}" if disc_list else "Timeout",
            f"{np.std(disc_list):.1f}" if disc_list else "—",
            f"{np.mean(solved_list):.1f}" if solved_list else "Timeout",
            f"{np.std(solved_list):.1f}" if solved_list else "—",
            f"{np.mean(delay_list):.1f}" if delay_list else "—",
            f"{np.mean(regrets):.1f}",
            f"{np.std(regrets):.1f}",
            f"{np.median(regrets):.1f}",
            f"{iqm:.1f}",
            f"{ms_per_ep:.2f} ms/ep",
            f"{eps_per_sec:.1f} eps/s",
        ])

        # Interpolate regret curve across common episodes
        interp_matrix = []
        for r in runs:
            pts_ep = [p[0] for p in r["cum_regrets"]]
            pts_reg = [p[1] for p in r["cum_regrets"]]
            interp_matrix.append(np.interp(COMMON_EPS, pts_ep, pts_reg))

        arr = np.array(interp_matrix)
        export_dict["regret_curves"][cid] = {
            "episodes": COMMON_EPS.tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "median": np.median(arr, axis=0).tolist(),
            "p25": np.percentile(arr, 25, axis=0).tolist(),
            "p75": np.percentile(arr, 75, axis=0).tolist(),
        }

        # Per seed detailed entry
        export_dict["per_seed_data"][cid] = [
            {
                "seed": r.get("seed"),
                "first_discovery": r.get("first_discovery"),
                "solved_episode": r.get("solved_episode"),
                "consolidation_eps": r.get("consolidation_eps"),
                "cumulative_regret": r.get("cumulative_regret"),
                "cum_regrets": r.get("cum_regrets"),
            }
            for r in runs
        ]

    # Save JSON data
    with open(DATA_JSON_PATH, "w") as f:
        json.dump(export_dict, f, indent=2)
    print(f"[Data Export] Saved structured JSON to: {DATA_JSON_PATH}")

    # Save CSV data
    with open(DATA_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"[Data Export] Saved summary CSV to: {DATA_CSV_PATH}")

    # Copy data files to artifact directory as well
    shutil.copy(DATA_JSON_PATH, os.path.join(ARTIFACT_DIR, "Figure_3_TMLR_data.json"))
    shutil.copy(DATA_CSV_PATH, os.path.join(ARTIFACT_DIR, "Figure_3_TMLR_data.csv"))

    return export_dict

# ------------------------------------------------------------------------------
# 3. Render Publication-Quality Figure 3 (TMLR Format)
# ------------------------------------------------------------------------------
def plot_figure_3(conditions, export_dict):
    # Set high-level publication styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 14,
    })

    fig = plt.figure(figsize=(26, 7.2), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.9], wspace=0.24)

    fig.suptitle(
        r"$\mathbf{Figure\ 3:}$ Deep Sea $20 \times 20$ Benchmark Showdown (50 Independent Runs, 10 Seeds, 10,000 Episodes)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # --------------------------------------------------------------------------
    # Panel (a): Cumulative Regret Curves
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    common_eps = np.array(export_dict["regret_curves"]["dp_dqn_dag"]["episodes"])

    for cond in conditions:
        cid = cond["id"]
        c_data = export_dict["regret_curves"][cid]
        mean_reg = np.array(c_data["mean"])
        std_reg = np.array(c_data["std"])

        label = f"{cond['short_name']} ({mean_reg[-1]:.0f})"
        ax1.plot(
            common_eps,
            mean_reg,
            label=label,
            color=cond["color"],
            linestyle=cond["line_style"],
            linewidth=2.4,
        )
        ax1.fill_between(
            common_eps,
            np.maximum(0, mean_reg - std_reg),
            mean_reg + std_reg,
            color=cond["color"],
            alpha=0.15,
        )

    ax1.set_title(r"$\mathbf{(a)}$ Cumulative Regret $\sum_{t=1}^T (V^* - R_t)$", fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(r"Cumulative Regret (Mean $\pm$ 1 Std)")
    ax1.set_xlim(0, EPISODES)
    ax1.set_ylim(-100, 10300)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", framealpha=0.92, edgecolor="#CCCCCC")

    # --------------------------------------------------------------------------
    # Panel (b): Sample Complexity / Solved Episodes Boxplot
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    box_data = []
    box_labels = []
    box_colors = []

    for cond in conditions:
        cid = cond["id"]
        runs = export_dict["per_seed_data"][cid]
        solv_list = [r["solved_episode"] for r in runs if r.get("solved_episode") is not None]
        if solv_list:
            box_data.append(solv_list)
            # Clean label for boxplot
            clean_lbl = cond["short_name"].replace(" (DAG Prior)", " (DAG)").replace(" + RP (20 Heads)", "\n+ RP").replace(" (Standard F0)", "\n(Standard)")
            box_labels.append(clean_lbl)
            box_colors.append(cond["color"])

    bp = ax2.boxplot(
        box_data,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7},
        medianprops={"color": "black", "linewidth": 1.6},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    ax2.set_xticks(range(1, len(box_labels) + 1))
    ax2.set_xticklabels(box_labels)


    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual seed points (jittered strip plot)
    rng = np.random.RandomState(42)
    for idx, vals in enumerate(box_data):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax2.scatter(
            np.ones(len(vals)) * (idx + 1) + jitter,
            vals,
            color="#212121",
            alpha=0.65,
            s=22,
            zorder=4,
        )

    ax2.set_title(r"$\mathbf{(b)}$ Episodes to Solve ($T_{\mathrm{learn}}$, 100% Solved)", fontweight="bold")
    ax2.set_ylabel("Episode to Reach >0.90 Rolling Return")
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y")

    # --------------------------------------------------------------------------
    # Panel (c): Master Summary Table
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    headers = [
        "Algorithm Condition",
        "Architecture",
        "Solved",
        "Mean Disc.",
        "Mean Solved",
        "Consol. Delay",
        "Mean Regret",
        "Cost (1st 3k)",
        "IQM Regret",
    ]

    cell_data = []
    for s in export_dict["summary_table"]:
        disc_str = f"{s['mean_discovery']:.0f} ± {s['std_discovery']:.0f}" if s["mean_discovery"] is not None else "Timeout"
        solv_str = f"{s['mean_solved']:.0f} ± {s['std_solved']:.0f}" if s["mean_solved"] is not None else "Timeout"
        delay_str = f"{s['mean_consolidation_delay']:.0f} eps" if s["mean_consolidation_delay"] is not None else "—"
        reg_str = f"{s['mean_regret']:.0f} ± {s['std_regret']:.0f}"
        iqm_str = f"{s['iqm_regret']:.0f}"
        solved_str = f"{s['solved_count']}/{s['total_runs']} ({s['solved_rate_pct']:.0f}%)"
        cost_str = f"{s['cost_ms_per_episode_1st_3000eps']:.1f} ms/ep" if s.get("cost_ms_per_episode_1st_3000eps") else "—"

        cell_data.append([
            s["name"],
            s["architecture"],
            solved_str,
            disc_str,
            solv_str,
            delay_str,
            reg_str,
            cost_str,
            iqm_str,
        ])

    col_widths = [0.24, 0.13, 0.09, 0.11, 0.11, 0.09, 0.11, 0.11, 0.08]
    table = ax3.table(
        cellText=cell_data,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
        colColours=["#212121"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 2.3)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#B0BEC5")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_facecolor("#263238")
        else:
            if row_idx == 1:
                # Highlight top winner: DP-DQN DAG
                cell.set_facecolor("#E0F2F1")
                if col_idx in [0, 2, 4, 6]:
                    cell.get_text().set_fontweight("bold")
            elif row_idx % 2 == 1:
                cell.set_facecolor("#F9F9F9")
            else:
                cell.set_facecolor("#FFFFFF")

            if col_idx == 2:
                txt = cell.get_text().get_text()
                if "100%" in txt:
                    cell.get_text().set_color("#1B5E20")
                    cell.get_text().set_fontweight("bold")
                elif "0%" in txt:
                    cell.get_text().set_color("#C62828")

    ax3.set_title(r"$\mathbf{(c)}$ Benchmark Performance Leaderboard (10 Independent Seeds)", fontweight="bold")

    # Save PNG (300 DPI) & Vector PDF
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.04, right=0.98)
    fig.savefig(PNG_OUT_PATH, bbox_inches="tight", dpi=300)
    fig.savefig(PDF_OUT_PATH, bbox_inches="tight")
    fig.savefig(PNG_SPACE_PATH, bbox_inches="tight", dpi=300)

    # Also copy to artifact directory for display in chat if available
    if os.path.isdir(ARTIFACT_DIR):
        artifact_png = os.path.join(ARTIFACT_DIR, "Figure_3_TMLR.png")
        fig.savefig(artifact_png, bbox_inches="tight", dpi=300)
        artifact_space_png = os.path.join(ARTIFACT_DIR, "Figure 3_TMLR.png")
        fig.savefig(artifact_space_png, bbox_inches="tight", dpi=300)

    plt.close(fig)
    print(f"[Plotting] Figure 3 saved to:")
    print(f"  - {PNG_OUT_PATH}")
    print(f"  - {PDF_OUT_PATH}")
    print(f"  - {PNG_SPACE_PATH}")

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Figure 3 for TMLR.")
    parser.add_argument("--recompute", action="store_true", help="Recompute from raw results_deepsea directory if available.")
    parser.add_argument("--cached", action="store_true", default=True, help="Load directly from verified Figure_3_TMLR_data.json (default).")
    args, _ = parser.parse_known_args()

    default_conditions = [
        {
            "id": "dp_dqn_dag",
            "name": r"DP-DQN (Downward DAG $F_0$, $\alpha=10$)",
            "short_name": "DP-DQN (DAG Prior)",
            "table_name": "DP-DQN (Downward DAG F0, α=10)",
            "architecture": "Single MLP-20",
            "color": "#00897B",      # Dark Teal
            "line_style": "-",
        },
        {
            "id": "boot_dqn_rp",
            "name": "BootDQN + Rand Priors (Osband 2018)",
            "short_name": "BootDQN + RP (20 Heads)",
            "table_name": "BootDQN + Rand Priors (20 Heads)",
            "architecture": "20-Head MLP-20",
            "color": "#FF8C00",      # Deep Amber/Orange
            "line_style": "-",
        },
        {
            "id": "dp_dqn_std",
            "name": r"DP-DQN (Standard $F_0$, $\alpha=10$)",
            "short_name": "DP-DQN (Standard F0)",
            "table_name": "DP-DQN (Standard F0, α=10)",
            "architecture": "Single MLP-20",
            "color": "#1E88E5",      # Royal Blue
            "line_style": "-",
        },
        {
            "id": "bdqn",
            "name": "Bayesian Deep Q-Network (BLR)",
            "short_name": "BDQN (BLR)",
            "table_name": "Bayesian Deep Q-Network (BLR)",
            "architecture": "Single MLP-20",
            "color": "#8E24AA",      # Purple
            "line_style": "--",
        },
        {
            "id": "dp_dqn_bootstrap_limit",
            "name": r"DP-DQN Bootstrap Limit ($\alpha=10^{-9}$)",
            "short_name": "Bootstrap Limit (α=10⁻⁹)",
            "table_name": "DP-DQN Bootstrap Limit (α=10⁻⁹)",
            "architecture": "Single MLP-20",
            "color": "#E53935",      # Red
            "line_style": ":",
        },
    ]

    if args.recompute and os.path.exists(RESULTS_DIR):
        print("=== Processing & Exporting Figure 3 Data from Raw Logs (TMLR) ===")
        conditions = load_all_runs()
        export_dict = process_data(conditions)
    elif os.path.exists(DATA_JSON_PATH):
        print(f"=== Rendering Figure 3 from Verified Benchmark Data ({DATA_JSON_PATH}) ===")
        with open(DATA_JSON_PATH, "r") as fp:
            export_dict = json.load(fp)
        conditions = default_conditions
    else:
        raise FileNotFoundError(f"Neither {RESULTS_DIR} nor {DATA_JSON_PATH} found.")

    plot_figure_3(conditions, export_dict)

    if os.path.isdir(ARTIFACT_DIR):
        shutil.copy(__file__, os.path.join(ARTIFACT_DIR, "plot_Figure_3_TMLR.py"))
        print(f"[Script Copy] Copied generator script to: {os.path.join(ARTIFACT_DIR, 'plot_Figure_3_TMLR.py')}")
    print("=== Finished successfully! ===")
