"""Deep Sea 20 Live Benchmark: DP-DQN across 5 Seeds with Real-Time Regret Plots.

Hyperparameters:
- N = 20 (Deep Sea 20x20, 2^-20 exploration hurdle)
- Architecture: 2-layer MLP (hidden_dim=64, LayerNorm, ReLU, lr=1e-3)
- Target tracking: Fast Polyak (tau=0.05) on One Living Network
- Replay Buffer: 100,000 transitions (zero FIFO eviction up to ep 5,000)
- Prior: Uniformly Optimistic DAG Prior (dag_uniform_reward: r ~ N(0.1, 0.05), NO chest bonus)
- Prior Decay: TD Information Gain
- Seeds: [42, 43, 44, 45, 46] (5 seeds in parallel)
- Live Plotting: Real-time plot generation every 25 episodes to deepsea20_live_plot.png
"""

import concurrent.futures
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env

# Deep Sea 20 Configuration
SIZE = 20
EPISODES = 5000
SEEDS = [42, 43, 44, 45, 46]
BATCH_SIZE = 64
LR = 1e-3
HIDDEN_DIM = 64
GAMMA = 0.99
TAU = 0.05
SGD_PERIOD = 2
ALPHA = 3.0
WARMSTART_STEPS = 2
WARMSTART_LR_SCALE = 0.8
BUFFER_CAPACITY = 100000  # 100k transitions = 5,000 full episodes of zero eviction

BASE_DIR = "/Users/sumitvashishtha/Desktop/DP-BNNs"
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
SCRATCH_DIR = os.path.join(ARTIFACT_DIR, "scratch", "deepsea20_live")
os.makedirs(SCRATCH_DIR, exist_ok=True)


def create_agent(seed: int) -> DPDQNAgent:
    """Factory for DP-DQN on Deep Sea 20 with winning paper configuration."""
    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=SIZE,
        state_dim=SIZE * SIZE,
        action_dim=2,
        seed=seed,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        candidate_batch_size=96,
        base_measure="dag_uniform_reward",  # Uniformly optimistic DAG prior, zero chest bonus
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
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        dp_sampled_target=False,
        use_td_info_gain_decay=True,
        td_info_scale=1.0,
        verbose=False,
    )
    return DPDQNAgent(cfg)


def run_worker_seed(seed: int):
    """Run one seed and periodically flush progress to a dedicated JSON file."""
    env = make_env("deep_sea", size=SIZE, seed=seed, randomize_actions=True)
    agent = create_agent(seed)
    out_file = os.path.join(SCRATCH_DIR, f"seed_{seed}_progress.json")

    rewards = []
    recent_rewards = []
    cum_regrets = []
    first_discovery = None
    solved_ep = None
    late_jumps = 0
    cum_regret = 0.0
    info_at_discovery = None
    info_at_solve = None

    optimal_return = 1.0 - 0.01  # 0.99 for Deep Sea

    t0 = time.time()
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
            print(f"\n  >>> [SEED {seed}] *** FIRST DISCOVERY at Episode {ep}! *** (TD Info = {info_at_discovery:.1f})\n", flush=True)

        if first_discovery is not None and ep > first_discovery:
            if ep_reward < 0.5:
                late_jumps += 1

        if solved_ep is None and len(recent_rewards) == 100 and np.mean(recent_rewards) >= 0.85:
            solved_ep = ep - 100 + 1
            info_at_solve = float(getattr(agent, "cumulative_info", 0.0))
            print(f"\n  >>> [SEED {seed}] *** SOLVED at Episode {solved_ep}! *** (TD Info = {info_at_solve:.1f})\n", flush=True)

        if ep % 25 == 0 or ep == EPISODES:
            cum_regrets.append((ep, float(cum_regret)))

            elapsed = time.time() - t0
            progress_data = {
                "seed": seed,
                "current_ep": ep,
                "total_ep": EPISODES,
                "first_discovery": first_discovery,
                "solved_ep": solved_ep,
                "cum_regret": float(cum_regret),
                "late_jumps": late_jumps,
                "info_at_discovery": info_at_discovery,
                "info_at_solve": info_at_solve,
                "current_info": float(getattr(agent, "cumulative_info", 0.0)),
                "recent_mean_return": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
                "cum_regrets": cum_regrets,
                "elapsed_sec": elapsed,
                "ms_per_ep": (elapsed / ep) * 1000.0,
                "done": (ep == EPISODES),
            }
            tmp_path = out_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(progress_data, f)
            os.replace(tmp_path, out_file)

    return progress_data


def render_live_plot(seed_data_map: Dict[int, Dict[str, Any]], output_png: str):
    """Render live 3-panel progress plot focused on cumulative regret."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=200)

    colors = {42: "#E91E63", 43: "#2196F3", 44: "#4CAF50", 45: "#FF9800", 46: "#9C27B0"}

    # Panel 1: Cumulative Regret Curves per Seed + Mean Curve
    all_trajs = []
    max_ep = 50
    min_len = 999999

    for s in sorted(seed_data_map.keys()):
        data = seed_data_map[s]
        if "cum_regrets" in data and data["cum_regrets"]:
            eps = [pt[0] for pt in data["cum_regrets"]]
            regs = [pt[1] for pt in data["cum_regrets"]]
            max_ep = max(max_ep, max(eps))
            min_len = min(min_len, len(regs))
            col = colors.get(s, "#333333")
            lbl = f"Seed {s}"
            if data.get("solved_ep"):
                lbl += f" (Solved @ ep {data['solved_ep']}, Regret: {data['cum_regret']:.0f})"
            elif data.get("first_discovery"):
                lbl += f" (Found @ ep {data['first_discovery']})"
            ax1.plot(eps, regs, color=col, lw=2.0, label=lbl, alpha=0.9)
            
            # Mark discovery
            disc = data.get("first_discovery")
            if disc:
                # Find closest ep
                closest_idx = np.argmin(np.abs(np.array(eps) - disc))
                ax1.scatter([eps[closest_idx]], [regs[closest_idx]], color=col, s=70, marker="o", edgecolors="black", zorder=5)

            # Mark solve
            solv = data.get("solved_ep")
            if solv:
                closest_idx = np.argmin(np.abs(np.array(eps) - solv))
                ax1.scatter([eps[closest_idx]], [regs[closest_idx]], color=col, s=120, marker="*", edgecolors="black", zorder=6)

            all_trajs.append(regs)

    # If we have multiple seeds at the same length, plot mean ± std
    if len(all_trajs) >= 2 and min_len > 0:
        common_eps = [pt[0] for pt in list(seed_data_map.values())[0]["cum_regrets"][:min_len]]
        mat = np.array([traj[:min_len] for traj in all_trajs])
        mean_reg = np.mean(mat, axis=0)
        std_reg = np.std(mat, axis=0)
        ax1.plot(common_eps, mean_reg, color="black", lw=2.8, linestyle="--", label=f"Mean (5 Seeds): {mean_reg[-1]:.0f}")
        ax1.fill_between(common_eps, mean_reg - std_reg, mean_reg + std_reg, color="gray", alpha=0.18)

    ax1.set_title(r"Deep Sea $20 \times 20$: Cumulative Regret vs. Episodes" + f"\n(Horizon: {max_ep}/{EPISODES} Episodes)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Cumulative Regret", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, max(200, max_ep * 1.05))
    ax1.grid(True, alpha=0.25, linestyle=":")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.92)

    # Panel 2: 100-Episode Moving Average Return
    for s in sorted(seed_data_map.keys()):
        data = seed_data_map[s]
        col = colors.get(s, "#333333")
        cur_ep = data.get("current_ep", 0)
        recent_ret = data.get("recent_mean_return", 0.0)
        ax2.scatter([cur_ep], [recent_ret], color=col, s=90, edgecolors="black", zorder=5, label=f"Seed {s}: {recent_ret:+.2f}")

    ax2.axhline(0.99, color="#2E7D32", linestyle="--", lw=1.5, label="Optimal Policy (+0.99)")
    ax2.axhline(0.0, color="#C62828", linestyle=":", lw=1.2, label="Failure (0.0)")
    ax2.set_title("Current 100-Episode Mean Return\n(Target: Optimal Diagonal Path +0.99)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Current Episode", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Return", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, max(200, max_ep * 1.05))
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.25, linestyle=":")
    ax2.legend(loc="lower right", fontsize=8)

    # Panel 3: Live Summary Table / Regret Bar Chart
    labels = [f"Seed {s}" for s in sorted(seed_data_map.keys())]
    regrets = [seed_data_map[s].get("cum_regret", 0.0) for s in sorted(seed_data_map.keys())]
    status_colors = [colors.get(s, "#555") for s in sorted(seed_data_map.keys())]
    
    bars = ax3.bar(labels, regrets, color=status_colors, width=0.55, edgecolor="black", alpha=0.85)
    for b, s in zip(bars, sorted(seed_data_map.keys())):
        d = seed_data_map[s]
        solv = d.get("solved_ep")
        disc = d.get("first_discovery")
        txt = f"{d.get('cum_regret', 0.0):.0f}"
        if solv:
            sub = f"SOLVED (ep {solv})"
        elif disc:
            sub = f"Found (ep {disc})"
        else:
            sub = f"Ep {d.get('current_ep', 0)}"
        ax3.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 15.0, f"{txt}\n{sub}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax3.set_ylabel("Cumulative Regret", fontsize=10, fontweight="bold")
    ax3.set_title("Cumulative Regret per Seed\n(Sub-linear Plateau upon Convergence)", fontsize=11, fontweight="bold")
    if regrets:
        ax3.set_ylim(0, max(max(regrets) * 1.3, 100))
    ax3.grid(True, axis="y", alpha=0.25, linestyle=":")

    plt.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 85)
    print(f"DEEP SEA {SIZE}x{SIZE} LIVE BENCHMARK: DP-DQN (5 SEEDS x {EPISODES:,} EPISODES)")
    print(f"Specs: MLP-64 | Polyak tau={TAU} | Buffer={BUFFER_CAPACITY:,} | TD Info Gain | Uniform DAG Prior")
    print(f"Seeds: {SEEDS}")
    print("=" * 85)

    # Clean old progress files
    for s in SEEDS:
        p = os.path.join(SCRATCH_DIR, f"seed_{s}_progress.json")
        if os.path.exists(p):
            os.remove(p)

    live_png_base = os.path.join(BASE_DIR, "deepsea20_live_plot.png")
    live_png_art = os.path.join(ARTIFACT_DIR, "deepsea20_live_plot.png")
    status_json_path = os.path.join(ARTIFACT_DIR, "deepsea20_live_status.json")

    # Launch 5 workers across 5 CPU cores
    print(f"Launching {len(SEEDS)} parallel workers across 5 CPU cores...")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=len(SEEDS))
    futures = {executor.submit(run_worker_seed, s): s for s in SEEDS}

    t_start = time.time()
    last_plot_time = 0

    while True:
        done_count = sum(1 for f in futures if f.done())

        # Read available progress data
        current_data = {}
        for s in SEEDS:
            fpath = os.path.join(SCRATCH_DIR, f"seed_{s}_progress.json")
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r") as fp:
                        current_data[s] = json.load(fp)
                except Exception:
                    pass

        # Update live plot every 3 seconds
        now = time.time()
        if current_data and (now - last_plot_time >= 3.0):
            try:
                render_live_plot(current_data, live_png_base)
                if os.path.exists(live_png_base):
                    shutil.copyfile(live_png_base, live_png_art)

                status_summary = {
                    "timestamp": now,
                    "elapsed_sec": now - t_start,
                    "completed_seeds": done_count,
                    "total_seeds": len(SEEDS),
                    "seeds": current_data,
                }
                with open(status_json_path, "w") as sf:
                    json.dump(status_summary, sf, indent=2)

                cur_eps = [current_data[s].get("current_ep", 0) for s in current_data]
                mean_ep = np.mean(cur_eps) if cur_eps else 0
                discs = sum(1 for s in current_data if current_data[s].get("first_discovery"))
                solves = sum(1 for s in current_data if current_data[s].get("solved_ep"))
                mean_reg = np.mean([current_data[s].get("cum_regret", 0.0) for s in current_data])
                print(
                    f"  [LIVE UPDATE] Ep ~{mean_ep:5.0f}/{EPISODES} | "
                    f"Discovered: {discs}/{len(SEEDS)} | Solved: {solves}/{len(SEEDS)} | "
                    f"Mean Regret: {mean_reg:6.1f} | Elapsed: {now - t_start:.1f}s | Plot updated.",
                    flush=True,
                )
                last_plot_time = now
            except Exception as e:
                print(f"Plotting error: {e}", flush=True)

        if done_count == len(SEEDS):
            break

        time.sleep(1.0)

    # Final gather and save
    final_results = {}
    for f in futures:
        s = futures[f]
        final_results[f"deepsea20_s{s}"] = f.result()

    final_json = os.path.join(BASE_DIR, "deepsea20_final_data.json")
    with open(final_json, "w") as f:
        json.dump(final_results, f, indent=2)

    render_live_plot(
        {s: final_results[f"deepsea20_s{s}"] for s in SEEDS},
        live_png_base,
    )
    shutil.copyfile(live_png_base, live_png_art)

    print("\n" + "=" * 85)
    print("DEEP SEA 20 BENCHMARK COMPLETE!")
    print(f"Results saved to {final_json}")
    print(f"Final plot saved to {live_png_art}")
    print("=" * 85)


if __name__ == "__main__":
    main()
