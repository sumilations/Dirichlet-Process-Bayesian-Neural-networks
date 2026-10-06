#!/usr/bin/env python3
"""Base Measure Ablation Benchmark for One Living Network + TD Info Gain Prior Decay.

Compares 3 different base measures under the exact same architecture:
- One Living Q-Network (in-place continuous updates)
- Standard Polyak target tracking (tau = 0.05)
- Option 3: TD Information Gain Prior Decay (no reward gates)
Tested base measures:
1. "deep_sea_dag": Causal DAG with Goal Chest Bonus (Maximum Domain Knowledge)
2. "dag_uniform_reward": Causal DAG with Uniform Reward, No Chest (Physical Transition Structure Only)
3. "deep_sea": Generic Uniform Grid, No DAG Column Physics, No Chest (Coarse Time Advancement Only)
"""

import os
import sys
import time
import json
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

SEEDS = [42, 44, 53, 54, 58]
SIZE = 20
EPISODES = 10000

HIDDEN_DIM = 20
BATCH_SIZE = 64
LR = 1e-3
GAMMA = 0.99
TAU = 0.05
SGD_PERIOD = 2
BUFFER_CAPACITY = 50000
ALPHA = 3.0
WARMSTART_STEPS = 2
WARMSTART_LR_SCALE = 0.8

BASE_DIR = "/Users/sumitvashishtha/Desktop/DP-BNNs"
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"

BASE_MEASURE_VARIANTS = [
    ("bm_dag_with_goal", "deep_sea_dag", "Causal DAG + Goal Chest"),
    ("bm_dag_uniform", "dag_uniform_reward", "Causal DAG (No Chest Bonus)"),
    ("bm_uniform_grid", "deep_sea", "Generic Uniform Grid (No DAG, No Chest)"),
]


def create_agent(variant_key: str, base_measure_name: str, seed: int) -> DPDQNAgent:
    """Factory for agent with specified base measure."""
    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=SIZE,
        state_dim=SIZE * SIZE,
        action_dim=2,
        seed=seed,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        candidate_batch_size=96,
        base_measure=base_measure_name,
        prior_reward_mean=0.1,
        prior_reward_std=0.05,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=LR,
        gamma=GAMMA,
        tau=TAU,  # Standard Polyak tracking
        sgd_period=SGD_PERIOD,
        buffer_capacity=BUFFER_CAPACITY,
        target_warmstart=True,
        warmstart_steps=WARMSTART_STEPS,
        warmstart_lr_scale=WARMSTART_LR_SCALE,
        num_episodes=EPISODES,
        max_episode_steps=SIZE,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        dp_sampled_target=False,  # Use Polyak tracking
        use_td_info_gain_decay=True,  # Option 3 TD Information Gain
        td_info_scale=1.0,
        verbose=False,
    )
    return DPDQNAgent(cfg)


def run_single_experiment(variant_key: str, base_measure_name: str, seed: int) -> Dict[str, Any]:
    """Train single agent on Deep Sea 20, exactly matched."""
    t0 = time.time()
    env = make_env("deep_sea", size=SIZE, seed=seed, randomize_actions=True)
    agent = create_agent(variant_key, base_measure_name, seed)

    rewards = []
    recent_rewards = []
    cum_regrets = []
    first_discovery = None
    solved_ep = None
    late_jumps = 0
    cum_regret = 0.0
    info_at_discovery = None
    info_at_solve = None

    optimal_return = 1.0 - 0.01

    for ep in range(1, EPISODES + 1):
        agent.reset_episode()
        s = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            a = agent.act(s)
            ns, r, done, _ = env.step(a)
            agent.step(s, a, float(r), ns, done)
            ep_reward += float(r)
            s = ns

        rewards.append(ep_reward)
        recent_rewards.append(ep_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        regret_step = max(0.0, optimal_return - ep_reward)
        cum_regret += regret_step

        if ep_reward > 0.5 and first_discovery is None:
            first_discovery = ep
            info_at_discovery = float(getattr(agent, "cumulative_info", 0.0))
            print(f"  [{variant_key}|s{seed}] DISCOVERY at ep {ep}! (info={info_at_discovery:.1f})", flush=True)

        if first_discovery is not None and solved_ep is None and ep > first_discovery:
            if ep_reward < 0.5:
                late_jumps += 1

        if solved_ep is None and len(recent_rewards) == 100 and np.mean(recent_rewards) >= 0.85:
            solved_ep = ep - 100 + 1
            info_at_solve = float(getattr(agent, "cumulative_info", 0.0))
            print(f"  [{variant_key}|s{seed}] SOLVED at ep {solved_ep}! (info={info_at_solve:.1f})", flush=True)

        if ep % 50 == 0 or ep == EPISODES:
            cum_regrets.append((ep, float(cum_regret)))

    wall_time = time.time() - t0
    ms_per_ep = (wall_time / EPISODES) * 1000.0

    res = {
        "variant": variant_key,
        "base_measure": base_measure_name,
        "seed": seed,
        "first_discovery": first_discovery,
        "solved_episode": solved_ep,
        "consolidation_eps": (solved_ep - first_discovery) if (solved_ep and first_discovery) else None,
        "cumulative_regret": float(cum_regret),
        "final_50_return": float(np.mean(rewards[-50:])),
        "late_jumps": late_jumps,
        "wall_time": wall_time,
        "ms_per_ep_active": ms_per_ep,
        "final_cumulative_info": float(getattr(agent, "cumulative_info", 0.0)),
        "info_at_discovery": info_at_discovery,
        "info_at_solve": info_at_solve,
        "cum_regrets": cum_regrets,
    }
    print(f"[{variant_key}|s{seed}] Done: Solved={solved_ep}, Disc={first_discovery}, Regret={cum_regret:.1f}, Info={agent.cumulative_info:.1f}, Time={wall_time:.1f}s", flush=True)
    return res


def main():
    print("=" * 80)
    print("BASE MEASURE ABLATION BENCHMARK (ONE LIVING NET + TD INFO GAIN)")
    print(f"Deep Sea {SIZE}x{SIZE} | Seeds: {SEEDS} | Episodes: {EPISODES}")
    print("=" * 80)

    # Load baseline cached runs (BootDQN, etc.)
    baseline_path = os.path.join(BASE_DIR, "dp_target_and_info_study_data.json")
    cached_data = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            cached_data = json.load(f)

    tasks = []
    for var_key, bm_name, _ in BASE_MEASURE_VARIANTS:
        for s in SEEDS:
            tasks.append((var_key, bm_name, s))

    print(f"Launching {len(tasks)} tasks concurrently across 10 cores...")
    new_results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(run_single_experiment, vk, bm, s): f"{vk}_s{s}" for vk, bm, s in tasks}
        for future in concurrent.futures.as_completed(futures):
            k = futures[future]
            res = future.result()
            new_results[k] = res

    # Combine cached and new results
    combined_data = dict(cached_data)
    combined_data.update(new_results)

    # Save data
    out_json = os.path.join(BASE_DIR, "base_measures_td_info_study_data.json")
    with open(out_json, "w") as f:
        json.dump(combined_data, f, indent=2)

    # Print Comparative Table
    variants_to_compare = [
        ("boot_dqn", "BootDQN (10-head)"),
        ("vm_dp_target_plus_td_info", "VM No-Polyak (DAG + Chest)"),
        ("bm_dag_with_goal", "VM Polyak (DAG + Chest)"),
        ("bm_dag_uniform", "VM Polyak (DAG Uniform, No Chest)"),
        ("bm_uniform_grid", "VM Polyak (Uniform Grid, No DAG)"),
    ]

    print("\n" + "=" * 125)
    print(f"{'Variant':<38} | {'Solved':<8} | {'First Disc':<16} | {'Solved Ep':<16} | {'Consolidation ΔT':<18} | {'Cum Regret':<14}")
    print("=" * 125)

    summary_stats = {}
    for var_key, var_label in variants_to_compare:
        discs = [combined_data[f"{var_key}_s{s}"]["first_discovery"] or EPISODES for s in SEEDS]
        solves = [combined_data[f"{var_key}_s{s}"]["solved_episode"] or EPISODES for s in SEEDS]
        conss = [combined_data[f"{var_key}_s{s}"]["consolidation_eps"] or EPISODES for s in SEEDS]
        regrets = [combined_data[f"{var_key}_s{s}"]["cumulative_regret"] for s in SEEDS]
        num_solved = sum(1 for s in SEEDS if combined_data[f"{var_key}_s{s}"]["solved_episode"] is not None)

        summary_stats[var_key] = {
            "label": var_label,
            "solved_rate": f"{num_solved}/{len(SEEDS)}",
            "disc_mean": np.mean(discs), "disc_std": np.std(discs),
            "solve_mean": np.mean(solves), "solve_std": np.std(solves),
            "cons_mean": np.mean(conss), "cons_std": np.std(conss),
            "regret_mean": np.mean(regrets), "regret_std": np.std(regrets),
        }

        print(
            f"{var_label:<38} | "
            f"{num_solved}/{len(SEEDS):<6} | "
            f"{np.mean(discs):5.0f} ± {np.std(discs):4.0f}     | "
            f"{np.mean(solves):5.0f} ± {np.std(solves):4.0f}     | "
            f"{np.mean(conss):5.0f} ± {np.std(conss):4.0f}       | "
            f"{np.mean(regrets):6.0f} ± {np.std(regrets):4.0f}"
        )
    print("=" * 125)

    # Generate Publication-Quality Figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = {
        "boot_dqn": "#E69F00",
        "vm_dp_target_plus_td_info": "#0072B2",
        "bm_dag_with_goal": "#009E73",
        "bm_dag_uniform": "#56B4E9",
        "bm_uniform_grid": "#CC79A7",
    }

    linestyles = {
        "boot_dqn": "--",
        "vm_dp_target_plus_td_info": "-.",
        "bm_dag_with_goal": "-",
        "bm_dag_uniform": "-",
        "bm_uniform_grid": "-",
    }

    # Panel 1: Cumulative Regret
    ax1 = axes[0, 0]
    for var_key, var_label in variants_to_compare:
        all_curves = []
        for s in SEEDS:
            k = f"{var_key}_s{s}"
            pts = combined_data[k]["cum_regrets"]
            eps_x = [p[0] for p in pts]
            vals = [p[1] for p in pts]
            all_curves.append(vals)
        all_curves = np.array(all_curves)
        mean_reg = np.mean(all_curves, axis=0)
        std_reg = np.std(all_curves, axis=0)
        ax1.plot(eps_x, mean_reg, label=var_label, color=colors[var_key], lw=2.2, linestyle=linestyles[var_key])
        ax1.fill_between(eps_x, np.maximum(0, mean_reg - std_reg), mean_reg + std_reg, color=colors[var_key], alpha=0.12)
    ax1.set_title("A. Cumulative Regret (Mean ± 1 SD, 5 Seeds)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Cumulative Regret", fontsize=11)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.0)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10000)

    # Panel 2: Consolidation Window ΔT
    ax2 = axes[0, 1]
    x_pos = np.arange(len(variants_to_compare))
    bar_width = 0.55
    cons_means = [summary_stats[vk]["cons_mean"] for vk, _ in variants_to_compare]
    cons_stds = [summary_stats[vk]["cons_std"] for vk, _ in variants_to_compare]
    bar_colors = [colors[vk] for vk, _ in variants_to_compare]
    bars = ax2.bar(x_pos, cons_means, bar_width, yerr=cons_stds, color=bar_colors, capsize=5, alpha=0.85, edgecolor="black", lw=1.2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([vl.split("(")[0].strip() for _, vl in variants_to_compare], rotation=25, ha="right", fontsize=8.5)
    ax2.set_ylabel("Episodes to Consolidate", fontsize=11)
    ax2.set_title(r"B. Consolidation Window $\Delta T = T_{\mathrm{learn}} - T_{\mathrm{disc}}$", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, cons_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, f"{m:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 3: Discovery vs. Solved Episodes
    ax3 = axes[1, 0]
    b_w = 0.35
    t_disc_means = [summary_stats[vk]["disc_mean"] for vk, _ in variants_to_compare]
    t_disc_stds = [summary_stats[vk]["disc_std"] for vk, _ in variants_to_compare]
    t_solve_means = [summary_stats[vk]["solve_mean"] for vk, _ in variants_to_compare]
    t_solve_stds = [summary_stats[vk]["solve_std"] for vk, _ in variants_to_compare]

    ax3.bar(x_pos - b_w/2, t_disc_means, b_w, yerr=t_disc_stds, label=r"Discovery $T_{\mathrm{disc}}$", color="#56B4E9", capsize=4, alpha=0.85, edgecolor="black")
    ax3.bar(x_pos + b_w/2, t_solve_means, b_w, yerr=t_solve_stds, label=r"Solved $T_{\mathrm{learn}}$", color="#D55E00", capsize=4, alpha=0.85, edgecolor="black")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([vl.split("(")[0].strip() for _, vl in variants_to_compare], rotation=25, ha="right", fontsize=8.5)
    ax3.set_ylabel("Episode Index", fontsize=11)
    ax3.set_title(r"C. Discovery vs. Convergence Episodes ($T_{\mathrm{disc}}$ vs. $T_{\mathrm{learn}}$)", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True, fontsize=9.0)
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Late Regret Drift (Stability Test: Ep 2,000 to 10,000)
    ax4 = axes[1, 1]
    drift_vals = []
    for var_key, _ in variants_to_compare:
        drifts = []
        for s in SEEDS:
            k = f"{var_key}_s{s}"
            pts = combined_data[k]["cum_regrets"]
            r2k = [p[1] for p in pts if p[0] <= 2000][-1]
            r10k = [p[1] for p in pts if p[0] <= 10000][-1]
            drifts.append(r10k - r2k)
        drift_vals.append((np.mean(drifts), np.std(drifts)))
    
    d_means = [d[0] for d in drift_vals]
    d_stds = [d[1] for d in drift_vals]
    bars_d = ax4.bar(x_pos, d_means, bar_width, yerr=d_stds, color=bar_colors, capsize=5, alpha=0.85, edgecolor="black", lw=1.2)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([vl.split("(")[0].strip() for _, vl in variants_to_compare], rotation=25, ha="right", fontsize=8.5)
    ax4.set_ylabel("Late Regret Drift (Ep 2k to 10k)", fontsize=11)
    ax4.set_title("D. Late-Stage Regret Drift (Ep 2,000 to 10,000)", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    for bar, dm in zip(bars_d, d_means):
        ax4.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 15, f"{dm:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "base_measures_td_info_study.png")
    plot_pdf = os.path.join(BASE_DIR, "base_measures_td_info_study.pdf")
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_pdf)
    plt.close()

    # Copy to artifact directory
    os.system(f"cp '{plot_path}' '{ARTIFACT_DIR}/base_measures_td_info_study.png'")
    os.system(f"cp '{plot_pdf}' '{ARTIFACT_DIR}/base_measures_td_info_study.pdf'")

    print(f"\nPlots saved to {plot_path} and copied to artifacts!")


if __name__ == "__main__":
    main()
