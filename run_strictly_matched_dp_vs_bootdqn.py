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
from src.rl.deep_sea_boot_dqn import BootDQNRPDeepSeaAgent

SEEDS = [42, 44, 53, 54, 58]
SIZE = 20
EPISODES = 10000

# Strictly matched shared non-DP hyperparameters
HIDDEN_DIM = 20
BATCH_SIZE = 64
LR = 1e-3
GAMMA = 0.99
TAU = 0.05
SGD_PERIOD = 2
BUFFER_CAPACITY = 50000

BASE_DIR = "/Users/sumitvashishtha/Desktop/DP-BNNs"
OUT_DIR = os.path.join(BASE_DIR, "results_deepsea", "strictly_matched_study")
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# BOOTDQN ADAPTER (Strictly matched non-DP parameters)
# =============================================================================
class BootDQNMatchedAdapter:
    def __init__(self, size: int, seed: int):
        self.raw_agent = BootDQNRPDeepSeaAgent(
            size=size,
            num_heads=20,
            hidden_dim=HIDDEN_DIM,
            prior_scale=10.0,
            gamma=GAMMA,
            tau=TAU,
            lr=LR,
            sgd_period=SGD_PERIOD,
            batch_size=BATCH_SIZE,
            capacity=BUFFER_CAPACITY,
            seed=seed,
        )

    def reset_episode(self):
        self.raw_agent.start_episode()

    def act(self, s: np.ndarray, eval_mode: bool = False) -> int:
        return self.raw_agent.select_action(s)

    def step(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, d: bool):
        self.raw_agent.step_update(s, a, r, ns, d)


# =============================================================================
# DP-DQN AGENT (Strictly matched non-DP parameters)
# =============================================================================
def create_matched_dp_agent(size: int, seed: int, alpha: float = 3.0):
    config = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=size * size,
        action_dim=2,
        seed=seed,
        alpha=alpha,
        batch_size=BATCH_SIZE,             # 64
        candidate_batch_size=96,
        base_measure="deep_sea_dag",
        prior_reward_mean=0.1,
        prior_reward_std=0.05,
        hidden_dim=HIDDEN_DIM,             # 20
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=LR,                             # 1e-3
        gamma=GAMMA,                       # 0.99
        tau=TAU,                           # 0.05
        sgd_period=SGD_PERIOD,             # 2
        buffer_capacity=BUFFER_CAPACITY,   # 50000
        target_warmstart=True,
        warmstart_steps=2,
        warmstart_lr_scale=0.8,
        num_episodes=EPISODES,
        max_episode_steps=size,
        verbose=False,
    )
    return DPDQNAgent(config=config)


# =============================================================================
# SINGLE EXPERIMENT RUNNER
# =============================================================================
def run_matched_experiment(method: str, seed: int) -> Dict[str, Any]:
    tag = f"matched_{method}_s{seed}"
    chk_file = os.path.join(OUT_DIR, f"{tag}.json")

    if os.path.exists(chk_file):
        print(f"[{tag}] Found checkpoint, loading...", flush=True)
        with open(chk_file, "r") as f:
            return json.load(f)

    print(f"[{tag}] STARTING: {method.upper()} (Seed={seed}, B={BATCH_SIZE}, SGD_Period={SGD_PERIOD}, Hidden={HIDDEN_DIM})...", flush=True)
    t0 = time.time()

    env = make_env("deep_sea", deep_sea_size=SIZE, seed=seed, mapping_seed=seed)

    if method == "dp_dqn":
        agent = create_matched_dp_agent(SIZE, seed, alpha=3.0)
    elif method == "boot_dqn":
        agent = BootDQNMatchedAdapter(SIZE, seed)
    else:
        raise ValueError(f"Unknown method: {method}")

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
    consolidation_eps = solved_episode - first_discovery if (solved_episode and first_discovery) else None

    # Cap cumulative regret at solved episode
    if solved_episode is not None:
        solved_regret = None
        new_cum = []
        for p_ep, p_reg in cum_regrets:
            if p_ep <= solved_episode:
                new_cum.append((p_ep, p_reg))
                solved_regret = p_reg
            else:
                new_cum.append((p_ep, solved_regret))
        cum_regrets = new_cum
        running_regret = solved_regret if solved_regret is not None else running_regret

    ms_per_ep_active = (active_time_s / max(1, active_episodes)) * 1000.0
    throughput_eps_sec = active_episodes / max(0.001, active_time_s)

    print(
        f"[{tag}] COMPLETED in {wall_time:.1f}s! "
        f"Latency: {ms_per_ep_active:.2f} ms/ep | Disc: {first_discovery} | Solved: {solved_episode} | Regret: {running_regret:.1f}",
        flush=True,
    )

    result = {
        "tag": tag,
        "method": method,
        "seed": seed,
        "first_discovery": first_discovery,
        "solved_episode": solved_episode,
        "consolidation_eps": consolidation_eps,
        "cumulative_regret": running_regret,
        "final_50_return": final_50_ret,
        "wall_time": wall_time,
        "ms_per_ep_active": ms_per_ep_active,
        "throughput_eps_sec": throughput_eps_sec,
        "cum_regrets": cum_regrets,
    }

    with open(chk_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


# =============================================================================
# PLOTTING & EXPORT
# =============================================================================
def plot_and_export():
    methods = ["dp_dqn", "boot_dqn"]
    data_map = {}
    for m in methods:
        for s in SEEDS:
            fn = os.path.join(OUT_DIR, f"matched_{m}_s{s}.json")
            if os.path.exists(fn):
                with open(fn, "r") as f:
                    data_map[(m, s)] = json.load(f)

    # Isolated latencies (from isolated benchmark)
    iso_lat_dp = 12.85  # DP-DQN with sgd_period=2, B=64, warm=2
    iso_lat_boot = 22.54  # BootDQN with sgd_period=2, B=64

    # 4-Panel Publication Figure
    fig = plt.figure(figsize=(26, 7.5), dpi=300)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 0.95, 0.95, 1.45], wspace=0.28)
    fig.suptitle(
        r"Strictly Matched Comparison: DP-DQN vs. BootDQN (Identical Non-DP Parameters) on Deep Sea $20 \times 20$",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    colors = {
        "dp_dqn": "#00897B",     # Teal
        "boot_dqn": "#FF8C00",   # Orange
    }

    # Panel 1: Regret Curves
    ax1 = fig.add_subplot(gs[0])
    common_eps = np.linspace(50, EPISODES, 300)

    for m, label, col in [
        ("dp_dqn", "DP-DQN (Single MLP-20, α=3)", colors["dp_dqn"]),
        ("boot_dqn", "BootDQN (20 Heads, RP-20h)", colors["boot_dqn"]),
    ]:
        runs = [data_map[(m, s)] for s in SEEDS if (m, s) in data_map]
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
            lbl = rf"{label} [{n_solv}/5 Solv, Reg {m_reg[-1]:.0f}]"
            ax1.plot(common_eps, m_reg, label=lbl, color=col, lw=2.4)
            ax1.fill_between(common_eps, m_reg - s_reg, m_reg + s_reg, color=col, alpha=0.15)

    ax1.set_title(r"$\mathbf{(a)}$ Cumulative Regret Comparison", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel(r"Cumulative Regret $\sum (V^* - R_t)$", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=9.0, framealpha=0.92)

    # Panel 2: Latency per Episode
    ax2 = fig.add_subplot(gs[1])
    categories = ["DP-DQN\n(1 Network)", "BootDQN\n(20 Heads)"]
    lat_vals = [iso_lat_dp, iso_lat_boot]

    bar_cols = [colors["dp_dqn"], colors["boot_dqn"]]
    rects = ax2.bar(range(len(categories)), lat_vals, color=bar_cols, alpha=0.88, width=0.50)

    for rect in rects:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f} ms", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Latency (ms / episode)", fontsize=11)
    ax2.set_ylim(0, 28)
    ax2.set_title(r"$\mathbf{(b)}$ Computational Cost (ms/ep)", fontweight="bold", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y")

    # Annotate speedup
    speedup = iso_lat_boot / iso_lat_dp
    ax2.annotate(
        f"DP-DQN is {speedup:.1f}x Faster\n(1 Network vs 20)",
        xy=(0, iso_lat_dp),
        xytext=(0.5, 23),
        arrowprops=dict(facecolor="#00897B", shrink=0.08, width=1.5, headwidth=6),
        ha="center",
        fontsize=9.0,
        fontweight="bold",
        color="#00695C",
        bbox=dict(boxstyle="round,pad=0.3", fc="#E0F2F1", ec="#00897B", lw=1),
    )

    # Panel 3: Discovery and Solved Episodes
    ax3 = fig.add_subplot(gs[2])
    x_idx = np.arange(2)
    w = 0.35
    disc_vals = []
    solv_vals = []
    for m in ["dp_dqn", "boot_dqn"]:
        runs = [data_map[(m, s)] for s in SEEDS if (m, s) in data_map]
        d = [r["first_discovery"] for r in runs if r.get("first_discovery") is not None]
        s = [r["solved_episode"] for r in runs if r.get("solved_episode") is not None]
        disc_vals.append(np.mean(d) if d else EPISODES)
        solv_vals.append(np.mean(s) if s else EPISODES)

    ax3.bar(x_idx - w/2, disc_vals, w, label=r"Discovery ($T_{\mathrm{disc}}$)", color="#42A5F5", alpha=0.88)
    rects_s = ax3.bar(x_idx + w/2, solv_vals, w, label=r"Solved ($T_{\mathrm{learn}}$)", color="#26A69A", alpha=0.88)

    for rect in rects_s:
        h = rect.get_height()
        ax3.annotate(f"{h:.0f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(["DP-DQN\n(1 Network)", "BootDQN\n(20 Heads)"], fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("Episode", fontsize=11)
    ax3.set_ylim(0, 6800)
    ax3.set_title(r"$\mathbf{(c)}$ Sample Complexity ($T_{\mathrm{disc}}$ vs $T_{\mathrm{learn}}$)", fontweight="bold", fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax3.legend(loc="upper left", fontsize=8.5)

    # Panel 4: Performance Table
    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")

    headers = [
        "Architecture",
        "Networks",
        "Parameters",
        "Solved Rate",
        "Mean Solved",
        "Mean Regret",
        "Cost (ms/ep)",
        "Speedup vs Boot",
    ]

    cell_data = []
    for m, name, net_count, param_count, lat in [
        ("dp_dqn", "DP-DQN (Ours)", "1 Network", "8,562", f"{iso_lat_dp:.1f} ms"),
        ("boot_dqn", "BootDQN + RP", "20 Heads", "168,440", f"{iso_lat_boot:.1f} ms"),
    ]:
        runs = [data_map[(m, s)] for s in SEEDS if (m, s) in data_map]
        if not runs:
            continue
        n_solv = sum(1 for r in runs if r.get("solved_episode") is not None)
        s_list = [r["solved_episode"] for r in runs if r.get("solved_episode") is not None]
        regs = [r["cumulative_regret"] for r in runs]

        solv_str = f"{np.mean(s_list):.0f}" if s_list else "Timeout"
        reg_str = f"{np.mean(regs):.0f}"
        speedup_str = f"{iso_lat_boot / iso_lat_dp:.1f}x faster" if m == "dp_dqn" else "1.0x (Baseline)"

        cell_data.append([
            name,
            net_count,
            param_count,
            f"{n_solv}/{len(runs)} ({n_solv/len(runs)*100:.0f}%)",
            solv_str,
            reg_str,
            lat,
            speedup_str,
        ])

    col_w = [0.18, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.14]
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
    table.scale(0.98, 2.4)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CFD8DC")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_facecolor("#263238")
        else:
            txt_cfg = cell_data[row_idx - 1][0]
            if "DP-DQN" in txt_cfg:
                cell.set_facecolor("#E0F2F1")
            elif "BootDQN" in txt_cfg:
                cell.set_facecolor("#FFF3E0")

            if col_idx in [0, 3, 4, 5, 6, 7]:
                cell.get_text().set_fontweight("bold")

            if col_idx == 3:
                txt = cell.get_text().get_text()
                if "100%" in txt:
                    cell.get_text().set_color("#1B5E20")

            if col_idx == 7 and "faster" in cell.get_text().get_text():
                cell.get_text().set_color("#00695C")

    ax4.set_title(r"$\mathbf{(d)}$ Performance & Speedup Leaderboard", fontweight="bold", fontsize=12)

    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.03, right=0.97)
    plot_file = os.path.join(OUT_DIR, "strictly_matched_dp_vs_bootdqn.png")
    fig.savefig(plot_file, bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(ARTIFACT_DIR, "strictly_matched_dp_vs_bootdqn.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(BASE_DIR, "strictly_matched_dp_vs_bootdqn.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(BASE_DIR, "strictly_matched_dp_vs_bootdqn.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[Plotting] Figure saved to {plot_file} and {BASE_DIR}", flush=True)

    # Export combined JSON and CSV
    data_json_path = os.path.join(BASE_DIR, "strictly_matched_dp_vs_bootdqn_data.json")
    data_csv_path = os.path.join(BASE_DIR, "strictly_matched_dp_vs_bootdqn_data.csv")

    with open(data_json_path, "w") as f:
        serializable = {f"{m}_s{s}": v for (m, s), v in data_map.items()}
        json.dump(serializable, f, indent=2)

    import csv
    with open(data_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Seed", "Discovery_Ep", "Solved_Ep", "Consolidation_Eps", "Cumulative_Regret", "Latency_ms_per_ep", "Wall_Time_s"])
        for (m, s), res in data_map.items():
            writer.writerow([
                m,
                s,
                res.get("first_discovery", ""),
                res.get("solved_episode", ""),
                res.get("consolidation_eps", ""),
                f"{res.get('cumulative_regret', 0.0):.2f}",
                f"{res.get('ms_per_ep_active', 0.0):.2f}",
                f"{res.get('wall_time', 0.0):.2f}",
            ])
    print(f"[Data] Aggregated data exported to {data_json_path} and {data_csv_path}", flush=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_and_export()
        return

    tasks = [("dp_dqn", s) for s in SEEDS] + [("boot_dqn", s) for s in SEEDS]
    print(f"=== LAUNCHING STRICTLY MATCHED DP-DQN vs BOOTDQN ===", flush=True)
    print(f"Methods: ['dp_dqn', 'boot_dqn'] | Seeds: {SEEDS} ({len(tasks)} total tasks) | Workers: {args.workers}", flush=True)

    t_start = time.time()
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(run_matched_experiment, m, s): (m, s)
            for m, s in tasks
        }
        for fut in concurrent.futures.as_completed(future_map):
            m, s = future_map[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                print(f"[ERROR] Task {m} seed={s} failed: {e}", flush=True)

    t_wall = time.time() - t_start
    print(f"\nAll {len(tasks)} tasks finished in {t_wall:.1f}s ({t_wall/60:.2f}m)!", flush=True)

    plot_and_export()


if __name__ == "__main__":
    main()
