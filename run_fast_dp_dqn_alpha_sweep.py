import os
import sys
import time
import json
import argparse
import concurrent.futures
from typing import Dict, Any, List
import numpy as np
import torch

sys.path.insert(0, "/Users/sumitvashishtha/Desktop/DP-BNNs")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)

from src.dp_dqn import DPDQNConfig, DPDQNAgent, make_env

ALPHAS = [1.0, 3.0, 5.0, 10.0, 15.0]
SEEDS = [42, 44, 53, 54, 58]
SIZE = 20
EPISODES = 10000

BASE_DIR = "/Users/sumitvashishtha/Desktop/DP-BNNs"
OUT_DIR = os.path.join(BASE_DIR, "results_deepsea", "fast_dp_dqn_study")
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"

os.makedirs(OUT_DIR, exist_ok=True)


def create_fast_agent(alpha: float, size: int, seed: int):
    """Fast DP-DQN: batch_size=64 (matched to BootDQN), warmstart_steps=2, sgd_period=3."""
    config = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=size * size,
        action_dim=2,
        seed=seed,
        alpha=alpha,
        batch_size=64,             # Dropped from 200 to 64 (matches BootDQN)
        candidate_batch_size=96,   # Dropped from 256 to 96
        base_measure="deep_sea_dag",
        prior_reward_mean=0.1,
        prior_reward_std=0.05,
        hidden_dim=20,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        tau=0.05,
        sgd_period=3,              # 7 updates per episode instead of 10
        buffer_capacity=50000,
        target_warmstart=True,
        warmstart_steps=2,         # Dropped from 5 to 2
        warmstart_lr_scale=0.8,
        trajectory_C=None,
        use_buffer_size_denominator=False,
        direct_replay_sample=False,
        num_episodes=EPISODES,
        max_episode_steps=size,
        verbose=False,
    )
    return DPDQNAgent(config=config)


def run_single_experiment(alpha: float, seed: int) -> Dict[str, Any]:
    tag = f"fast_dp_dqn_a{alpha:.1f}_s{seed}"
    chk_file = os.path.join(OUT_DIR, f"{tag}.json")

    if os.path.exists(chk_file):
        print(f"[{tag}] Found checkpoint, loading...", flush=True)
        with open(chk_file, "r") as f:
            return json.load(f)

    print(f"[{tag}] STARTING: alpha={alpha}, seed={seed}, Fast DP-DQN (B=64, warm=2, sgd=3)...", flush=True)
    t0 = time.time()

    env = make_env("deep_sea", deep_sea_size=SIZE, seed=seed, mapping_seed=seed)
    agent = create_fast_agent(alpha, SIZE, seed)

    optimal_return = 1.0 - (SIZE - 1) * (0.01 / SIZE)  # 0.9905
    returns = []
    cum_regrets = []
    running_regret = 0.0
    recent_returns = []

    first_discovery = None
    solved_episode = None
    is_frozen = False
    active_time_s = 0.0
    active_episodes = 0

    for ep in range(1, EPISODES + 1):
        ep_t0 = time.perf_counter()

        if not is_frozen and hasattr(agent, "reset_episode"):
            agent.reset_episode()

        s = env.reset()
        if isinstance(s, tuple):
            s = s[0]

        done = False
        ep_reward = 0.0

        while not done:
            a = agent.act(s, eval_mode=is_frozen)
            step_out = env.step(a)
            if len(step_out) == 5:
                sn, r, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                sn, r, done, _ = step_out

            if not is_frozen:
                agent.step(s, a, r, sn, done)

            s = sn
            ep_reward += r

        ep_duration = time.perf_counter() - ep_t0
        if not is_frozen:
            active_time_s += ep_duration
            active_episodes += 1

        returns.append(ep_reward)
        recent_returns.append(ep_reward)
        if len(recent_returns) > 50:
            recent_returns.pop(0)

        # Regret tracking
        regret_t = max(0.0, optimal_return - ep_reward)
        running_regret += regret_t
        if ep % 50 == 0 or ep == 1:
            cum_regrets.append((ep, float(running_regret)))

        # Discovery tracking
        if ep_reward > 0.5 and first_discovery is None:
            first_discovery = ep
            print(f"[{tag}] >> FIRST DISCOVERY at Episode {ep}! Return: {ep_reward:.4f} ({time.time()-t0:.1f}s)", flush=True)

        # Convergence detection: rolling 50 return > 0.90
        if len(recent_returns) >= 50 and np.mean(recent_returns) > 0.90:
            if solved_episode is None:
                solved_episode = ep - 50 + 1
                is_frozen = True
                print(
                    f"[{tag}] *** CONVERGED & SOLVED at Episode {solved_episode}! "
                    f"Freezing policy updates ({time.time()-t0:.1f}s)...",
                    flush=True,
                )

        if ep % 2500 == 0 and not is_frozen:
            avg_r = np.mean(recent_returns)
            print(
                f"[{tag}] Ep {ep:,}/{EPISODES:,} | Rolling50: {avg_r:.3f} | Regret: {running_regret:.1f} | Disc: {first_discovery}",
                flush=True,
            )

    wall_time = time.time() - t0
    final_50_ret = float(np.mean(returns[-50:]))
    late_jumps = 0
    consolidation_eps = None
    if solved_episode is not None:
        late_jumps = sum(1 for r in returns[solved_episode:] if r < 0.5)
        consolidation_eps = solved_episode - first_discovery if first_discovery else None

    ms_per_ep_active = (active_time_s / max(1, active_episodes)) * 1000.0
    throughput_eps_sec = active_episodes / max(0.001, active_time_s)

    print(
        f"[{tag}] COMPLETED in {wall_time:.1f}s ({wall_time/60:.2f}m)! "
        f"Latency: {ms_per_ep_active:.2f} ms/ep | Disc: {first_discovery} | Solved: {solved_episode} | Regret: {running_regret:.1f}",
        flush=True,
    )

    result = {
        "tag": tag,
        "alpha": alpha,
        "seed": seed,
        "first_discovery": first_discovery,
        "solved_episode": solved_episode,
        "consolidation_eps": consolidation_eps,
        "cumulative_regret": running_regret,
        "final_50_return": final_50_ret,
        "late_jumps": late_jumps,
        "wall_time": wall_time,
        "ms_per_ep_active": ms_per_ep_active,
        "throughput_eps_sec": throughput_eps_sec,
        "cum_regrets": cum_regrets,
    }

    with open(chk_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def plot_results():
    results_map = {}  # (alpha, seed) -> result
    for a in ALPHAS:
        for s in SEEDS:
            tag = f"fast_dp_dqn_a{a:.1f}_s{s}"
            fn = os.path.join(OUT_DIR, f"{tag}.json")
            if os.path.exists(fn):
                with open(fn, "r") as f:
                    results_map[(a, s)] = json.load(f)

    # Load isolated latency measurements (apples-to-apples with 3,000-ep benchmark)
    iso_lat_file = os.path.join(BASE_DIR, "fast_dp_dqn_isolated_latency.json")
    iso_latencies = {1.0: 7.99, 3.0: 8.13, 5.0: 8.18, 10.0: 8.25, 15.0: 8.29}
    if os.path.exists(iso_lat_file):
        try:
            with open(iso_lat_file, "r") as f:
                loaded = json.load(f)
                iso_latencies.update({float(k): float(v) for k, v in loaded.items()})
        except Exception:
            pass

    # 4-Panel Publication Figure
    fig = plt.figure(figsize=(26, 7.5), dpi=300)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 0.95, 0.95, 1.45], wspace=0.28)
    fig.suptitle(
        r"Fast DP-DQN Optimization & Alpha Spectrum ($\alpha \in [1, 15]$) on Deep Sea $20 \times 20$",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    alpha_colors = {
        1.0: "#E53935",   # Red
        3.0: "#FB8C00",   # Orange
        5.0: "#FBC02D",   # Yellow-Gold
        10.0: "#00897B",  # Teal
        15.0: "#1E88E5",  # Royal Blue
    }

    # Panel 1: Regret Curves across Alphas
    ax1 = fig.add_subplot(gs[0])
    common_eps = np.linspace(50, EPISODES, 300)

    for a in ALPHAS:
        runs = [results_map[(a, s)] for s in SEEDS if (a, s) in results_map]
        if not runs:
            continue
        interp_list = []
        for r in runs:
            pts_ep = [p[0] for p in r.get("cum_regrets", [])]
            pts_reg = [p[1] for p in r.get("cum_regrets", [])]
            if pts_ep and pts_reg:
                interp_list.append(np.interp(common_eps, pts_ep, pts_reg))
        if interp_list:
            arr = np.array(interp_list)
            m_reg = np.mean(arr, axis=0)
            s_reg = np.std(arr, axis=0)
            n_solv = sum(1 for r in runs if r.get("solved_episode") is not None)
            lbl = rf"Fast DP-DQN $\alpha={a:.0f}$ ({n_solv}/{len(runs)} Solv, Reg {m_reg[-1]:.0f})"
            ax1.plot(common_eps, m_reg, label=lbl, color=alpha_colors[a], lw=2.2)
            ax1.fill_between(common_eps, m_reg - s_reg, m_reg + s_reg, color=alpha_colors[a], alpha=0.10)

    # Reference BootDQN
    ax1.axhline(4791, color="#757575", linestyle="--", lw=1.8, label="BootDQN-RP (Regret 4791)")
    ax1.set_title(r"$\mathbf{(a)}$ Cumulative Regret across $\alpha$", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel(r"Cumulative Regret $\sum (V^* - R_t)$", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    # Panel 2: Latency (Cost per Episode) vs BootDQN
    ax2 = fig.add_subplot(gs[1])
    categories = [f"Fast α={a:.0f}" for a in ALPHAS] + ["BootDQN\n(RP-20h)", "Original\nDP-DQN"]
    latencies = [iso_latencies.get(a, 8.2) for a in ALPHAS]
    latencies.append(22.54)  # BootDQN
    latencies.append(31.91)  # Original DP-DQN

    bar_cols = [alpha_colors[a] for a in ALPHAS] + ["#FF8C00", "#78909C"]
    rects = ax2.bar(range(len(categories)), latencies, color=bar_cols, alpha=0.88, width=0.62)
    ax2.axhline(22.54, color="#FF8C00", linestyle="--", lw=1.6, label="BootDQN (22.5 ms)")
    ax2.axhline(8.17, color="#00897B", linestyle=":", lw=1.6, label="Fast DP-DQN avg (8.2 ms)")

    for rect in rects:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.1f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
        )

    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=25, ha="right", fontsize=9.0, fontweight="bold")
    ax2.set_ylabel("Latency (ms / episode)", fontsize=11)
    ax2.set_ylim(0, 38)
    ax2.set_title(r"$\mathbf{(b)}$ Computational Cost (ms/ep)", fontweight="bold", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax2.legend(loc="upper left", fontsize=8.5)

    # Panel 3: Episodes to Solve vs Discovery
    ax3 = fig.add_subplot(gs[2])
    x_idx = np.arange(len(ALPHAS))
    w = 0.35
    disc_vals, solv_vals = [], []
    for a in ALPHAS:
        runs = [results_map[(a, s)] for s in SEEDS if (a, s) in results_map]
        d = [r["first_discovery"] for r in runs if r.get("first_discovery") is not None]
        s = [r["solved_episode"] for r in runs if r.get("solved_episode") is not None]
        disc_vals.append(np.mean(d) if d else EPISODES)
        solv_vals.append(np.mean(s) if s else EPISODES)

    rects_d = ax3.bar(x_idx - w/2, disc_vals, w, label=r"Discovery ($T_{\mathrm{disc}}$)", color="#42A5F5", alpha=0.88)
    rects_s = ax3.bar(x_idx + w/2, solv_vals, w, label=r"Solved ($T_{\mathrm{learn}}$)", color="#26A69A", alpha=0.88)

    for rect in rects_s:
        h = rect.get_height()
        if h < EPISODES:
            ax3.annotate(f"{h:.0f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    ax3.set_xticks(x_idx)
    ax3.set_xticklabels([f"α={a:.0f}" for a in ALPHAS], fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("Episode", fontsize=11)
    ax3.set_title(r"$\mathbf{(c)}$ Sample Complexity across $\alpha$", fontweight="bold", fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax3.legend(loc="upper left", fontsize=8.5)

    # Panel 4: Performance Table
    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")

    headers = [
        "Configuration",
        "Batch (B)",
        "Warmstart",
        "Solved Rate",
        "Mean Solved",
        "Mean Regret",
        "Cost (ms/ep)",
        "Speedup vs BootDQN",
    ]

    cell_data = []
    for a in ALPHAS:
        runs = [results_map[(a, s)] for s in SEEDS if (a, s) in results_map]
        if not runs:
            continue
        n_solv = sum(1 for r in runs if r.get("solved_episode") is not None)
        s_list = [r["solved_episode"] for r in runs if r.get("solved_episode") is not None]
        regs = [r["cumulative_regret"] for r in runs]
        lat = iso_latencies.get(a, 8.2)

        solv_str = f"{np.mean(s_list):.0f}" if s_list else "Timeout"
        reg_str = f"{np.mean(regs):.0f}"
        lat_str = f"{lat:.1f} ms"
        speedup = f"{22.54 / lat:.1f}x faster"

        cell_data.append([
            f"Fast DP-DQN (α={a:.0f})",
            "64",
            "2 steps",
            f"{n_solv}/{len(runs)} ({n_solv/len(runs)*100:.0f}%)",
            solv_str,
            reg_str,
            lat_str,
            speedup,
        ])

    # Add reference rows
    cell_data.append(["BootDQN + RP (20h)", "64", "0 steps", "5/5 (100%)", "5086", "4791", "22.5 ms", "1.0x (Baseline)"])
    cell_data.append(["Original DP-DQN (α=10)", "200", "5 steps", "5/5 (100%)", "2995", "2909", "31.9 ms", "0.7x (Slower)"])

    col_w = [0.20, 0.08, 0.09, 0.12, 0.11, 0.11, 0.11, 0.16]
    table = ax4.table(
        cellText=cell_data,
        colLabels=headers,
        colWidths=col_w,
        cellLoc="center",
        loc="center",
        colColours=["#212121"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(0.98, 2.3)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CFD8DC")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_facecolor("#263238")
        else:
            txt_cfg = cell_data[row_idx - 1][0]
            if "Fast DP-DQN (α=3" in txt_cfg or "Fast DP-DQN (α=10" in txt_cfg:
                cell.set_facecolor("#E0F2F1")
                if col_idx in [0, 3, 4, 5, 6, 7]:
                    cell.get_text().set_fontweight("bold")
            elif "BootDQN" in txt_cfg:
                cell.set_facecolor("#FFF3E0")
            elif "Original DP-DQN" in txt_cfg:
                cell.set_facecolor("#ECEFF1")
            elif row_idx % 2 == 1:
                cell.set_facecolor("#F9F9F9")
            else:
                cell.set_facecolor("#FFFFFF")

            if col_idx == 3:
                txt = cell.get_text().get_text()
                if "100%" in txt:
                    cell.get_text().set_color("#1B5E20")
                    cell.get_text().set_fontweight("bold")
                elif "0%" in txt:
                    cell.get_text().set_color("#C62828")

            if col_idx == 7:
                txt = cell.get_text().get_text()
                if "faster" in txt:
                    cell.get_text().set_color("#00695C")
                    cell.get_text().set_fontweight("bold")

    ax4.set_title(r"$\mathbf{(d)}$ Performance & Speedup Leaderboard", fontweight="bold", fontsize=12)

    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.03, right=0.98)
    plot_file = os.path.join(OUT_DIR, "fast_dp_dqn_alpha_sweep_comparison.png")
    fig.savefig(plot_file, bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(ARTIFACT_DIR, "fast_dp_dqn_alpha_sweep_comparison.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(BASE_DIR, "fast_dp_dqn_alpha_sweep_comparison.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(BASE_DIR, "fast_dp_dqn_alpha_sweep_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[Plotting] Figure saved to {plot_file} and {BASE_DIR}", flush=True)

    # Save aggregated data to JSON and CSV
    data_json_path = os.path.join(BASE_DIR, "fast_dp_dqn_alpha_sweep_data.json")
    data_csv_path = os.path.join(BASE_DIR, "fast_dp_dqn_alpha_sweep_data.csv")
    with open(data_json_path, "w") as f:
        # Convert tuple keys to string
        serializable_map = {f"a{a}_s{s}": v for (a, s), v in results_map.items()}
        json.dump(serializable_map, f, indent=2)

    import csv
    with open(data_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Alpha", "Seed", "Discovery_Ep", "Solved_Ep", "Consolidation_Eps", "Cumulative_Regret", "Latency_ms_per_ep", "Throughput_eps_sec", "Wall_Time_s"])
        for (a, s), res in results_map.items():
            writer.writerow([
                f"Fast_DP_DQN_alpha_{a}",
                a,
                s,
                res.get("first_discovery", ""),
                res.get("solved_episode", ""),
                res.get("consolidation_eps", ""),
                f"{res.get('cumulative_regret', 0.0):.2f}",
                f"{res.get('ms_per_ep_active', 0.0):.2f}",
                f"{res.get('throughput_eps_sec', 0.0):.2f}",
                f"{res.get('wall_time', 0.0):.2f}",
            ])
    print(f"[Data] Exported aggregated datasets to {data_json_path} and {data_csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_results()
        return

    tasks = [(a, s) for a in ALPHAS for s in SEEDS]
    print(f"=== LAUNCHING FAST DP-DQN ALPHA SWEEP ===", flush=True)
    print(f"Alphas: {ALPHAS} | Seeds: {SEEDS} ({len(tasks)} total tasks) | Workers: {args.workers}", flush=True)

    t_start = time.time()
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(run_single_experiment, a, s): (a, s)
            for a, s in tasks
        }
        for fut in concurrent.futures.as_completed(future_map):
            a, s = future_map[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                print(f"[ERROR] Task alpha={a}, seed={s} failed: {e}", flush=True)

    t_wall = time.time() - t_start
    print(f"\nAll {len(tasks)} tasks finished in {t_wall:.1f}s ({t_wall/60:.2f}m)!", flush=True)

    plot_results()


if __name__ == "__main__":
    main()
