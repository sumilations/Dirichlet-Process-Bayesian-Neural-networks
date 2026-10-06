#!/usr/bin/env python3
"""Replay-Scaled Dirichlet Posterior with Subsampling Study for DP-DQN on Deep Sea 20.

Benchmarks:
1. BootDQN (10-head ensemble)
2. Baseline DP-DQN (Sethuraman Stick-Breaking)
3. Vashishtha & Maillard (2025) Fixed-Budget (N=64, K_prior=30)
4. Vashishtha & Maillard Pure Replay-Scaled N (N=|D| from ep 1, subsample B=64)
5. Vashishtha & Maillard Post-Discovery Replay-Scaled N (N=64 pre-discovery -> N=|D| post-discovery, subsample B=64)
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


def create_agent(variant_name: str, seed: int) -> DPDQNAgent:
    """Factory for agent variants."""
    common_args = dict(
        env_name="deep_sea",
        deep_sea_size=SIZE,
        state_dim=SIZE * SIZE,
        action_dim=2,
        seed=seed,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        candidate_batch_size=96,
        base_measure="deep_sea_dag",
        prior_reward_mean=0.1,
        prior_reward_std=0.05,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=LR,
        gamma=GAMMA,
        tau=TAU,
        sgd_period=SGD_PERIOD,
        buffer_capacity=BUFFER_CAPACITY,
        target_warmstart=True,
        warmstart_steps=WARMSTART_STEPS,
        warmstart_lr_scale=WARMSTART_LR_SCALE,
        num_episodes=EPISODES,
        max_episode_steps=SIZE,
        verbose=False,
    )

    if variant_name == "vm_replay_scaled":
        cfg = DPDQNConfig(
            **common_args,
            sampler_type="vashishtha_maillard",
            vm_prior_multiplier=10.0,
            vm_replay_scaled_n=True,
            vm_scale_post_discovery_only=False,
        )
    elif variant_name == "vm_post_discovery":
        cfg = DPDQNConfig(
            **common_args,
            sampler_type="vashishtha_maillard",
            vm_prior_multiplier=10.0,
            vm_replay_scaled_n=True,
            vm_scale_post_discovery_only=True,
        )
    elif variant_name == "vashishtha_maillard":
        cfg = DPDQNConfig(
            **common_args,
            sampler_type="vashishtha_maillard",
            vm_prior_multiplier=10.0,
            vm_replay_scaled_n=False,
        )
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    return DPDQNAgent(cfg)


def run_single_experiment(variant_name: str, seed: int) -> Dict[str, Any]:
    """Train single agent on Deep Sea 20, exactly matched to run_variance_reduction_study."""
    t0 = time.time()
    env = make_env("deep_sea", size=SIZE, seed=seed, randomize_actions=True)
    agent = create_agent(variant_name, seed)

    rewards = []
    recent_rewards = []
    cum_regrets = []
    w_prior_history = []
    first_discovery = None
    solved_ep = None
    late_jumps = 0
    cum_regret = 0.0

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

        # Track W_prior if available
        if hasattr(agent.sampler, "last_prob_syn"):
            w_prior_history.append(float(agent.sampler.last_prob_syn))

        if ep_reward > 0.5 and first_discovery is None:
            first_discovery = ep
            print(f"  [{variant_name}|s{seed}] DISCOVERY at ep {ep}!", flush=True)

        if first_discovery is not None and solved_ep is None and ep > first_discovery:
            if ep_reward < 0.5:
                late_jumps += 1

        if solved_ep is None and len(recent_rewards) == 100 and np.mean(recent_rewards) >= 0.85:
            solved_ep = ep - 100 + 1
            print(f"  [{variant_name}|s{seed}] SOLVED at ep {solved_ep}!", flush=True)

        if ep % 50 == 0 or ep == EPISODES:
            cum_regrets.append((ep, float(cum_regret)))

    wall_time = time.time() - t0
    ms_per_ep = (wall_time / EPISODES) * 1000.0

    res = {
        "variant": variant_name,
        "seed": seed,
        "first_discovery": first_discovery,
        "solved_episode": solved_ep,
        "consolidation_eps": (solved_ep - first_discovery) if (solved_ep and first_discovery) else None,
        "cumulative_regret": float(cum_regret),
        "final_50_return": float(np.mean(rewards[-50:])),
        "late_jumps": late_jumps,
        "wall_time": wall_time,
        "ms_per_ep_active": ms_per_ep,
        "cum_regrets": cum_regrets,
        "w_prior_sub": [w_prior_history[i] for i in range(0, len(w_prior_history), 50)] if w_prior_history else [],
    }
    print(f"[{variant_name}|s{seed}] Done: Solved={solved_ep}, Disc={first_discovery}, Regret={cum_regret:.1f}, Time={wall_time:.1f}s", flush=True)
    return res


def main():
    print("=" * 80)
    print("REPLAY-SCALED DIRICHLET POSTERIOR WITH SUBSAMPLING BENCHMARK")
    print(f"Deep Sea {SIZE}x{SIZE} | Seeds: {SEEDS} | Episodes: {EPISODES}")
    print("=" * 80)

    # Load existing cached runs
    cached_data = {}
    with open(os.path.join(BASE_DIR, "variance_reduction_study_data.json"), "r") as f:
        cached_data = json.load(f)

    # Tasks to run: pure replay scaled and post-discovery replay scaled
    variants_to_run = ["vm_replay_scaled", "vm_post_discovery"]
    tasks = [(v, s) for v in variants_to_run for s in SEEDS]

    new_results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_single_experiment, v, s): f"{v}_s{s}" for v, s in tasks}
        for future in concurrent.futures.as_completed(futures):
            k = futures[future]
            res = future.result()
            new_results[k] = res

    # Combine all results
    combined_data = dict(cached_data)
    combined_data.update(new_results)

    # Save updated data
    out_json = os.path.join(BASE_DIR, "replay_scaled_study_data.json")
    with open(out_json, "w") as f:
        json.dump(combined_data, f, indent=2)

    # Print Summary Comparison Table
    variants_to_compare = [
        ("boot_dqn", "BootDQN (10-head)"),
        ("baseline_dp_dqn", "Baseline DP-DQN (Old Stick)"),
        ("vashishtha_maillard", "VM Fixed-Budget (N=64)"),
        ("vm_replay_scaled", "VM Pure Replay-Scaled N (|D|)"),
        ("vm_post_discovery", "VM Post-Discovery Scaled N"),
    ]

    print("\n" + "=" * 110)
    print(f"{'Variant':<34} | {'First Disc (T_disc)':<18} | {'Solved (T_learn)':<18} | {'Consolidation ΔT':<18} | {'Cum Regret':<14}")
    print("=" * 110)

    summary_stats = {}
    for var_key, var_label in variants_to_compare:
        discs = [combined_data[f"{var_key}_s{s}"]["first_discovery"] or EPISODES for s in SEEDS]
        solves = [combined_data[f"{var_key}_s{s}"]["solved_episode"] or EPISODES for s in SEEDS]
        conss = [combined_data[f"{var_key}_s{s}"]["consolidation_eps"] or EPISODES for s in SEEDS]
        regrets = [combined_data[f"{var_key}_s{s}"]["cumulative_regret"] for s in SEEDS]

        summary_stats[var_key] = {
            "label": var_label,
            "disc_mean": np.mean(discs), "disc_std": np.std(discs),
            "solve_mean": np.mean(solves), "solve_std": np.std(solves),
            "cons_mean": np.mean(conss), "cons_std": np.std(conss),
            "regret_mean": np.mean(regrets), "regret_std": np.std(regrets),
        }

        print(
            f"{var_label:<34} | "
            f"{np.mean(discs):6.0f} ± {np.std(discs):4.0f}       | "
            f"{np.mean(solves):6.0f} ± {np.std(solves):4.0f}       | "
            f"{np.mean(conss):6.0f} ± {np.std(conss):4.0f}       | "
            f"{np.mean(regrets):6.0f} ± {np.std(regrets):4.0f}"
        )
    print("=" * 110)

    # Generate Publication-Quality Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    colors = {
        "boot_dqn": "#E69F00",
        "baseline_dp_dqn": "#D55E00",
        "vashishtha_maillard": "#0072B2",
        "vm_replay_scaled": "#009E73",
        "vm_post_discovery": "#CC79A7",
    }

    # Panel 1: Cumulative Regret
    ax1 = axes[0, 0]
    for var_key, var_label in variants_to_compare:
        # cum_regrets is list of (ep, val) tuples
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
        ax1.plot(eps_x, mean_reg, label=var_label, color=colors[var_key], lw=2.2)
        ax1.fill_between(eps_x, np.maximum(0, mean_reg - std_reg), mean_reg + std_reg, color=colors[var_key], alpha=0.15)
    ax1.set_title("A. Cumulative Regret (Mean ± 1 SD, 5 Seeds)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Cumulative Regret", fontsize=11)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
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
    ax2.set_xticklabels([vl.split("(")[0].strip() for _, vl in variants_to_compare], rotation=20, ha="right", fontsize=9.5)
    ax2.set_ylabel("Episodes to Consolidate", fontsize=11)
    ax2.set_title(r"B. Consolidation Window $\Delta T = T_{\mathrm{learn}} - T_{\mathrm{disc}}$", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, cons_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, f"{m:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 3: T_disc vs T_learn bar chart
    ax3 = axes[1, 0]
    b_w = 0.35
    t_disc_means = [summary_stats[vk]["disc_mean"] for vk, _ in variants_to_compare]
    t_disc_stds = [summary_stats[vk]["disc_std"] for vk, _ in variants_to_compare]
    t_solve_means = [summary_stats[vk]["solve_mean"] for vk, _ in variants_to_compare]
    t_solve_stds = [summary_stats[vk]["solve_std"] for vk, _ in variants_to_compare]

    ax3.bar(x_pos - b_w/2, t_disc_means, b_w, yerr=t_disc_stds, label=r"Discovery $T_{\mathrm{disc}}$", color="#56B4E9", capsize=4, alpha=0.85, edgecolor="black")
    ax3.bar(x_pos + b_w/2, t_solve_means, b_w, yerr=t_solve_stds, label=r"Solved $T_{\mathrm{learn}}$", color="#D55E00", capsize=4, alpha=0.85, edgecolor="black")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([vl.split("(")[0].strip() for _, vl in variants_to_compare], rotation=20, ha="right", fontsize=9.5)
    ax3.set_ylabel("Episode Index", fontsize=11)
    ax3.set_title(r"C. Discovery vs. Convergence Episodes ($T_{\mathrm{disc}}$ vs. $T_{\mathrm{learn}}$)", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Prior Weight W_prior Trajectory
    ax4 = axes[1, 1]
    eps_sub = np.arange(0, EPISODES, 50)
    for var_key in ["vashishtha_maillard", "vm_replay_scaled", "vm_post_discovery"]:
        w_histories = []
        for s in SEEDS:
            k = f"{var_key}_s{s}"
            if "w_prior_sub" in combined_data[k] and combined_data[k]["w_prior_sub"]:
                w_histories.append(combined_data[k]["w_prior_sub"])
        if w_histories:
            # find min length
            min_len = min(len(w) for w in w_histories)
            w_hist_arr = np.array([w[:min_len] for w in w_histories])
            mean_w = np.mean(w_hist_arr, axis=0)
            ax4.plot(eps_sub[:min_len], mean_w, label=summary_stats[var_key]["label"], color=colors[var_key], lw=2.2)
        elif var_key == "vashishtha_maillard":
            ax4.plot(eps_sub, [4.0/68.0] * len(eps_sub), label="VM Fixed-Budget (N=64, ~5.9%)", color=colors[var_key], lw=2.2, linestyle="--")

    ax4.set_title(r"D. Dynamic Prior Weight Contraction ($W_{\mathrm{prior}}$)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Episode", fontsize=11)
    ax4.set_ylabel(r"Prior Mass Weight $W_{\mathrm{prior}}$", fontsize=11)
    ax4.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 3000)
    ax4.set_yscale("log")

    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "replay_scaled_study.png")
    plot_pdf = os.path.join(BASE_DIR, "replay_scaled_study.pdf")
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_pdf)
    plt.close()

    # Copy to artifact directory
    os.system(f"cp '{plot_path}' '{ARTIFACT_DIR}/replay_scaled_study.png'")
    os.system(f"cp '{plot_pdf}' '{ARTIFACT_DIR}/replay_scaled_study.pdf'")

    print(f"\nPlots saved to {plot_path} and copied to artifacts!")


if __name__ == "__main__":
    main()
