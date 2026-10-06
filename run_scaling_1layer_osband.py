#!/usr/bin/env python3
"""Deep Sea Scaling Benchmark: Exact Single-Layer (1-Layer) Architecture matching Osband et al.

Tests the Unified Modern DP-DQN across scales N in [5, 10, 15, 20, 30, 40, 50]:
- 1 Hidden Layer of 20 units: Linear(N^2, 20) -> LayerNorm -> ReLU -> Linear(20, 2)
- One Living Q-Network (in-place continuous weights)
- Fast Polyak Target Tracking (tau = 0.05)
- Option 3: TD Information Gain Prior Decay (no reward gates)
- 5 Standard Seeds: [42, 44, 53, 54, 58]
- Two Variants:
    1. With DAG (dag_uniform_reward): Causal DAG, optimistic reward, NO chest
    2. Without DAG (deep_sea): Generic uniform grid, NO DAG, NO chest
"""

import os
import sys
import time
import json
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

sys.path.insert(0, "/Users/sumitvashishtha/Desktop/DP-BNNs")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.agent import DPDQNAgent

BASE_DIR = "/Users/sumitvashishtha/Desktop/DP-BNNs"
RESULTS_DIR = os.path.join(BASE_DIR, "results_deepsea")
DATA_DIR = os.path.join(RESULTS_DIR, "scaling_1layer_osband")
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
os.makedirs(DATA_DIR, exist_ok=True)

SCALES = [5, 10, 15, 20, 30, 40, 50]
SEEDS = [42, 44, 53, 54, 58]
VARIANTS = ["dag_uniform_reward", "deep_sea"]

VARIANT_LABELS = {
    "dag_uniform_reward": "With DAG (Uniform Optimistic, 1-Layer)",
    "deep_sea": "Without DAG (Generic Grid, 1-Layer)",
}

VARIANT_COLORS = {
    "dag_uniform_reward": "#2E7D32",  # Dark Green
    "deep_sea": "#1976D2",            # Blue
}

SEED_COLORS = {
    42: "#E91E63",
    44: "#2196F3",
    53: "#4CAF50",
    54: "#FF9800",
    58: "#9C27B0",
}


def get_max_episodes(size: int) -> int:
    if size <= 10:
        return 1000
    elif size <= 15:
        return 1500
    elif size <= 20:
        return 2500
    elif size <= 30:
        return 4000
    elif size <= 40:
        return 6000
    else:
        return 8000


def run_single_experiment(size: int, variant: str, seed: int) -> Dict[str, Any]:
    """Execute a single (N, variant, seed) trial with atomic disk caching."""
    out_file = os.path.join(DATA_DIR, f"run_N{size}_{variant}_s{seed}.json")
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                cached = json.load(f)
            if "total_episodes" in cached or cached.get("solved"):
                return cached
        except Exception:
            pass

    max_episodes = get_max_episodes(size)
    buffer_cap = min(120000, 2500 * size)

    # 1 Hidden Layer of 20 units (exact Osband setup)
    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=size * size,
        action_dim=2,
        seed=seed,
        alpha=3.0,
        batch_size=64,
        candidate_batch_size=96,
        base_measure=variant,
        prior_reward_mean=0.1,
        prior_reward_std=0.05,
        hidden_dim=20,
        num_layers=1,  # 1 HIDDEN LAYER
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.05,
        sgd_period=2,
        buffer_capacity=buffer_cap,
        target_warmstart=True,
        warmstart_steps=2,
        warmstart_lr_scale=0.8,
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        dp_sampled_target=False,
        use_td_info_gain_decay=True,
        td_info_scale=1.0,
        verbose=False,
    )

    env = make_env("deep_sea", size=size, seed=seed, randomize_actions=True)
    agent = DPDQNAgent(cfg)

    optimal_return = 1.0 - 0.01
    recent_rewards: List[float] = []
    first_discovery: Optional[int] = None
    solved_episode: Optional[int] = None
    cum_regret = 0.0
    info_at_discovery = None
    info_at_solve = None

    t0 = time.time()
    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        s = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            a = agent.act(s)
            ns, r, done, _ = env.step(a)
            agent.step(s, a, float(r), ns, done)
            ep_ret += float(r)
            s = ns

        recent_rewards.append(ep_ret)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        regret_step = max(0.0, optimal_return - ep_ret)
        cum_regret += regret_step

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep
            info_at_discovery = float(getattr(agent, "cumulative_info", 0.0))

        if solved_episode is None and len(recent_rewards) == 100 and np.mean(recent_rewards) >= 0.85:
            solved_episode = ep - 100 + 1
            info_at_solve = float(getattr(agent, "cumulative_info", 0.0))

        # Early stopping: 80 episodes of post-solve verification
        if solved_episode is not None and ep >= solved_episode + 80:
            break

    elapsed = time.time() - t0
    is_solved = (solved_episode is not None)
    final_learn_time = solved_episode if is_solved else max_episodes

    result = {
        "size": size,
        "variant": variant,
        "seed": seed,
        "num_layers": 1,
        "hidden_dim": 20,
        "solved": is_solved,
        "first_discovery": first_discovery,
        "solved_episode": final_learn_time,
        "cum_regret": float(cum_regret),
        "info_at_discovery": info_at_discovery,
        "info_at_solve": info_at_solve,
        "final_info": float(getattr(agent, "cumulative_info", 0.0)),
        "elapsed_sec": elapsed,
        "total_episodes": ep,
    }

    tmp_path = out_file + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, out_file)

    v_str = "DAG" if variant == "dag_uniform_reward" else "No-DAG"
    status_str = f"SOLVED @ ep {final_learn_time}" if is_solved else "TIMEOUT"
    print(f"  [{v_str:6s}] N={size:2d} | Seed {seed} | {status_str:15s} | Regret: {cum_regret:7.1f} | {elapsed:5.1f}s", flush=True)

    return result


def gather_and_save_data() -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    results = {var: {s: [] for s in SCALES} for var in VARIANTS}
    for sz in SCALES:
        for var in VARIANTS:
            for s in SEEDS:
                fpath = os.path.join(DATA_DIR, f"run_N{sz}_{var}_s{s}.json")
                if os.path.exists(fpath):
                    try:
                        with open(fpath, "r") as f:
                            d = json.load(f)
                        results[var][sz].append(d)
                    except Exception:
                        pass

    summary_file = os.path.join(RESULTS_DIR, "scaling_1layer_osband_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    return results


def render_scaling_plots(all_results: Dict[str, Dict[int, List[Dict[str, Any]]]]):
    """Render 3-panel publication scaling figure."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.5), dpi=250)

    fits = {}
    for var in VARIANTS:
        sizes = sorted(all_results[var].keys())
        if not sizes:
            continue
        means = []
        errs = []
        for s in sizes:
            runs = all_results[var][s]
            lts = [r["solved_episode"] for r in runs if r.get("solved_episode")]
            if len(lts) >= 1:
                means.append(np.mean(lts))
                errs.append(np.std(lts) / np.sqrt(len(lts)) if len(lts) > 1 else 0.0)
            else:
                means.append(np.nan)
                errs.append(np.nan)

        valid = ~np.isnan(means)
        s_arr = np.array(sizes)[valid]
        m_arr = np.array(means)[valid]
        e_arr = np.array(errs)[valid]

        if len(s_arr) >= 2:
            lx = np.log10(s_arr)
            ly = np.log10(m_arr)
            poly = np.polyfit(lx, ly, 1)
            r2 = 1 - np.sum((ly - np.polyval(poly, lx)) ** 2) / np.sum((ly - np.mean(ly)) ** 2)
            fits[var] = (poly[0], 10 ** poly[1], r2, s_arr, m_arr, e_arr)

    # Panel 1: Log-Log Empirical Scaling
    n_space = np.linspace(4.5, 55, 100)
    ax1.plot(n_space[n_space <= 16], 2.0 ** n_space[n_space <= 16], "--", color="#D32F2F", lw=1.8, label=r"Random Dithering $\Omega(2^N)$ (Exponential)")
    ax1.plot(n_space, 0.75 * (n_space ** 3), ":", color="#9C27B0", lw=2.0, label=r"BootDQN-RP (Osband 2018): $\tilde{\mathcal{O}}(N^3)$")

    for var in VARIANTS:
        if var in fits:
            slope, intercept, r2, s_arr, m_arr, e_arr = fits[var]
            col = VARIANT_COLORS[var]
            lbl = VARIANT_LABELS[var] + f"\n  Fit: $\mathcal{{O}}(N^{{{slope:.2f}}})$ ($R^2={r2:.3f}$)"
            ax1.errorbar(s_arr, m_arr, yerr=e_arr, fmt="o-", color=col, ecolor=col,
                         elinewidth=1.8, capsize=4, ms=7, lw=2.0, label=lbl)
            ax1.plot(n_space, intercept * (n_space ** slope), "--", color=col, lw=1.2, alpha=0.6)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(4.5, 60)
    ax1.set_ylim(20, 5e5)
    ax1.set_xlabel("Deep Sea Grid Size $N$ (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Episodes to Learn $T_{\mathrm{learn}}$ (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Empirical Scaling Law: 1-Layer MLP (Exact Osband Scale)\n" + r"$\log T_{\mathrm{learn}} = d \log N + \log c$", fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.94)

    # Panel 2: Per-Seed Scaling Curves
    for var in VARIANTS:
        col = VARIANT_COLORS[var]
        ls = "-" if var == "dag_uniform_reward" else "--"
        for s in SEEDS:
            pts_x = []
            pts_y = []
            for sz in SCALES:
                runs = [r for r in all_results[var][sz] if r["seed"] == s]
                if runs and runs[0].get("solved_episode"):
                    pts_x.append(sz)
                    pts_y.append(runs[0]["solved_episode"])
            if len(pts_x) >= 2:
                ax2.plot(pts_x, pts_y, ls, color=col, alpha=0.55, lw=1.5, marker="o", ms=4)

    ax2.plot([], [], "-", color=VARIANT_COLORS["dag_uniform_reward"], lw=2, label="With DAG (5 Seeds)")
    ax2.plot([], [], "--", color=VARIANT_COLORS["deep_sea"], lw=2, label="Without DAG (5 Seeds)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(4.5, 60)
    ax2.set_ylim(20, 1.5e4)
    ax2.set_xlabel("Deep Sea Grid Size $N$ (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Episodes to Learn $T_{\mathrm{learn}}$ (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Consistency Across All 5 Seeds (1-Layer)\n(Parallel slopes demonstrate invariant degree $d$)", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", linestyle=":", alpha=0.35)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.94)

    # Panel 3: Linear Scale Sample Complexity
    for var in VARIANTS:
        sizes = sorted(all_results[var].keys())
        means = []
        for s in sizes:
            runs = all_results[var][s]
            lts = [r["solved_episode"] for r in runs if r.get("solved_episode")]
            means.append(np.mean(lts) if lts else np.nan)
        col = VARIANT_COLORS[var]
        lbl = "With DAG (1-Layer)" if var == "dag_uniform_reward" else "Without DAG (1-Layer)"
        valid = ~np.isnan(means)
        ax3.plot(np.array(sizes)[valid], np.array(means)[valid], "o-", color=col, lw=2.5, ms=8, label=lbl)

    ax3.set_xlim(0, 55)
    ax3.set_xlabel("Deep Sea Grid Size $N$ (Linear Scale)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Episodes to Learn ($T_{\mathrm{learn}}$)", fontsize=11, fontweight="bold")
    ax3.set_title("(c) Sample Complexity up to $N = 50$ (1-Layer)\n(Linear Scale: Measuring the Topological Gap)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.35)
    ax3.legend(loc="upper left", fontsize=9, framealpha=0.94)

    plt.tight_layout()
    plot_file = os.path.join(RESULTS_DIR, "scaling_1layer_osband_plot.png")
    fig.savefig(plot_file, bbox_inches="tight")
    artifact_file = os.path.join(ARTIFACT_DIR, "scaling_1layer_osband_plot.png")
    fig.savefig(artifact_file, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] Saved scaling figure to: {plot_file}", flush=True)


def main():
    print("=" * 85)
    print("DEEP SEA SCALING BENCHMARK: EXACT 1-LAYER MLP (OSBAND SCALE)")
    print("Scales: [5, 10, 15, 20, 30, 40, 50] | Seeds: [42, 44, 53, 54, 58] (70 total runs)")
    print("Architecture: Linear(N^2, 20) -> LayerNorm -> ReLU -> Linear(20, 2)")
    print("=" * 85)

    results = gather_and_save_data()

    tasks: List[Tuple[int, str, int]] = []
    cached_count = 0
    for sz in SCALES:
        for var in VARIANTS:
            for s in SEEDS:
                fpath = os.path.join(DATA_DIR, f"run_N{sz}_{var}_s{s}.json")
                if os.path.exists(fpath):
                    try:
                        with open(fpath, "r") as f:
                            d = json.load(f)
                        if "total_episodes" in d or d.get("solved"):
                            cached_count += 1
                            continue
                    except Exception:
                        pass
                tasks.append((sz, var, s))

    print(f"Status: {cached_count} runs already cached. {len(tasks)} runs queued to execute.")

    if not tasks:
        print("\nAll 70 runs are already complete!")
        render_scaling_plots(results)
        return

    # Sort tasks: smaller sizes first
    tasks.sort(key=lambda t: (t[0], 0 if t[1] == "dag_uniform_reward" else 1, t[2]))

    # Use 6 CPU workers (Mac has 10 cores)
    max_workers = min(6, len(tasks))
    print(f"\nLaunching {len(tasks)} tasks across {max_workers} CPU worker processes...")
    t_start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_experiment, sz, var, s): (sz, var, s) for (sz, var, s) in tasks}

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            sz, var, s = futures[future]
            try:
                future.result()
                completed += 1
                results = gather_and_save_data()
                render_scaling_plots(results)
                print(f"  --> Progress: {completed}/{len(tasks)} tasks finished ({completed + cached_count}/70 total) | Elapsed: {time.time() - t_start:.1f}s", flush=True)
            except Exception as e:
                print(f"  [ERROR] Task ({sz}, {var}, {s}) failed: {e}", flush=True)

    print("\n" + "=" * 85)
    print("BENCHMARK EXECUTION COMPLETE! Rendering final publication plots...")
    final_results = gather_and_save_data()
    render_scaling_plots(final_results)
    print("=" * 85)


if __name__ == "__main__":
    main()
