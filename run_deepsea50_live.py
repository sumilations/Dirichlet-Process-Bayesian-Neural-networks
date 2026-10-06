"""Deep Sea 50x50 Live Benchmark: DP-DQN with exact MLP-20, Polyak, TD Info Gain, and Uniform DAG Prior.

Exact Specifications:
- Size N = 50 (50x50 Deep Sea, 2^-50 exploration hurdle)
- Architecture: Exact MLP-20 (hidden_dim=20, 2 layers, LayerNorm, ReLU, lr=1e-3)
- Target Tracking: Fast Polyak (tau=0.05) on One Living Network
- Replay Buffer: 120,000 transitions (2.4 GB RAM per worker)
- Prior: Uniformly Optimistic DAG Prior (dag_uniform_reward: r ~ N(0.1, 0.05), NO chest bonus)
- Prior Decay: TD Information Gain (Option 3)
- Horizon: 15,000 episodes
- Seeds: [42, 44, 53, 54] (4 seeds in parallel across 4 CPU cores)
- Live Plotting: Real-time plot generation every 50 episodes to deepsea50_live_plot.png
"""

import concurrent.futures
import json
import os
import sys
import time
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env

# Deep Sea 50 Configuration
SIZE = 50
EPISODES = 10000
SEEDS = [42, 44, 53, 54]
BATCH_SIZE = 64
LR = 1e-3
HIDDEN_DIM = 20  # EXACT MLP-20
GAMMA = 0.99
TAU = 0.05
SGD_PERIOD = 2
ALPHA = 3.0
WARMSTART_STEPS = 2
WARMSTART_LR_SCALE = 0.8
BUFFER_CAPACITY = 120000

BASE_DIR = "/Users/sumitvashishtha/Desktop/DP-BNNs"
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
SCRATCH_DIR = os.path.join(ARTIFACT_DIR, "scratch", "deepsea50_live")
os.makedirs(SCRATCH_DIR, exist_ok=True)


def create_agent(seed: int) -> DPDQNAgent:
    """Factory for DP-DQN on Deep Sea 50 with exact MLP-20 architecture."""
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
        hidden_dim=HIDDEN_DIM,  # Exact MLP-20
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
        dp_sampled_target=False,  # Polyak tracking
        use_td_info_gain_decay=True,  # TD Information Gain
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

    optimal_return = 1.0 - 0.01

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

        if ep % 50 == 0 or ep == EPISODES or (solved_ep is not None and ep >= solved_ep + 300):
            cum_regrets.append((ep, float(cum_regret)))
            
            elapsed = time.time() - t0
            is_done = (ep == EPISODES) or (solved_ep is not None and ep >= solved_ep + 300)
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
                "done": is_done,
            }
            tmp_path = out_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(progress_data, f)
            os.replace(tmp_path, out_file)

            if is_done and solved_ep is not None:
                print(f"\n  >>> [SEED {seed}] Completed with post-solve verification at ep {ep}! Regret = {cum_regret:.1f}\n", flush=True)
                break

    return progress_data


def render_live_plot(seed_data_map: Dict[int, Dict[str, Any]], output_png: str):
    """Render live 3-panel progress plot for N=50."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=200)

    colors = {42: "#E91E63", 44: "#2196F3", 53: "#4CAF50", 54: "#FF9800"}

    # Panel 1: Cumulative Regret Curves per Seed
    max_ep = 100
    for s, data in seed_data_map.items():
        if "cum_regrets" in data and data["cum_regrets"]:
            eps = [pt[0] for pt in data["cum_regrets"]]
            max_ep = max(max_ep, max(eps))

    for s, data in seed_data_map.items():
        if "cum_regrets" in data and data["cum_regrets"]:
            eps = [pt[0] for pt in data["cum_regrets"]]
            regs = [pt[1] for pt in data["cum_regrets"]]
            if data.get("solved_ep") and eps[-1] < max_ep:
                eps = list(eps) + [max_ep]
                regs = list(regs) + [regs[-1]]
            col = colors.get(s, "#333333")
            lbl = f"Seed {s}"
            if data.get("solved_ep"):
                lbl += f" (Solved @ ep {data['solved_ep']})"
            elif data.get("first_discovery"):
                lbl += f" (Found @ ep {data['first_discovery']})"
            ax1.plot(eps, regs, color=col, lw=2.2, label=lbl, alpha=0.9)

    ax1.set_title(r"Deep Sea $50 \times 50$: Cumulative Regret vs. Episodes" + f"\n(Horizon: {max_ep}/{EPISODES} Episodes | Scale: $2^{{-50}}$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Cumulative Regret", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, max(500, max_ep))
    ax1.grid(True, alpha=0.25, linestyle=":")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.92)

    # Panel 2: 100-Episode Moving Average Return
    for s, data in seed_data_map.items():
        col = colors.get(s, "#333333")
        cur_ep = data.get("current_ep", 0)
        recent_ret = data.get("recent_mean_return", 0.0)
        ax2.scatter([cur_ep], [recent_ret], color=col, s=80, edgecolors="black", zorder=5)

    ax2.axhline(0.99, color="#2E7D32", linestyle="--", lw=1.5, label="Optimal Policy (+0.99)")
    ax2.axhline(0.0, color="#C62828", linestyle=":", lw=1.2, label="Failure (0.0)")
    ax2.set_title("Current 100-Episode Mean Return\n(Target: Optimal 50-Step Diagonal)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Current Episode", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Return", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, max(500, max_ep))
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.25, linestyle=":")
    ax2.legend(loc="lower right", fontsize=8)

    # Panel 3: TD Information Gain
    labels = [f"Seed {s}" for s in SEEDS]
    infos = [seed_data_map[s].get("current_info", 0.0) if s in seed_data_map else 0.0 for s in SEEDS]
    bars = ax3.bar(labels, infos, color=[colors.get(s, "#555") for s in SEEDS], width=0.55, edgecolor="black", alpha=0.85)
    for b, inf in zip(bars, infos):
        ax3.text(b.get_x() + b.get_width() / 2.0, inf + 2.0, f"{inf:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax3.set_ylabel("Cumulative TD Information Gain", fontsize=10, fontweight="bold")
    ax3.set_title("TD Information Gain per Seed\n(Prior Decay Metric)", fontsize=11, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.25, linestyle=":")

    plt.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 85)
    print(f"DEEP SEA {SIZE}x{SIZE} LIVE BENCHMARK (4 SEEDS x {EPISODES:,} EPISODES)")
    print(f"Specs: Exact MLP-20 | Polyak tau={TAU} | Buffer={BUFFER_CAPACITY:,} | TD Info Gain | Uniform DAG Prior")
    print("=" * 85)

    # Check for already solved seeds
    active_seeds = []
    for s in SEEDS:
        p = os.path.join(SCRATCH_DIR, f"seed_{s}_progress.json")
        if os.path.exists(p):
            try:
                with open(p, "r") as fp:
                    d = json.load(fp)
                if d.get("solved_ep") is not None:
                    print(f"Seed {s} ALREADY SOLVED at episode {d['solved_ep']} (regret: {d['cum_regret']:.1f}). Reusing cached solution!")
                    continue
            except Exception:
                pass
        if os.path.exists(p):
            os.remove(p)
        active_seeds.append(s)

    live_png_base = os.path.join(BASE_DIR, "deepsea50_live_plot.png")
    live_png_art = os.path.join(ARTIFACT_DIR, "deepsea50_live_plot.png")
    status_json_path = os.path.join(ARTIFACT_DIR, "deepsea50_live_status.json")

    futures = {}
    if active_seeds:
        print(f"Launching {len(active_seeds)} active workers for seeds {active_seeds}...")
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=len(active_seeds))
        futures = {executor.submit(run_worker_seed, s): s for s in active_seeds}
    else:
        print("All seeds are already solved! No active workers needed.")
        executor = None

    t_start = time.time()
    last_plot_time = 0

    while True:
        done_count = sum(1 for f in futures if f.done())
        
        current_data = {}
        for s in SEEDS:
            fpath = os.path.join(SCRATCH_DIR, f"seed_{s}_progress.json")
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r") as fp:
                        current_data[s] = json.load(fp)
                except Exception:
                    pass

        now = time.time()
        if current_data and (now - last_plot_time >= 4.0):
            try:
                render_live_plot(current_data, live_png_base)
                if os.path.exists(live_png_base):
                    import shutil
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
                print(
                    f"  [LIVE N=50] Ep ~{mean_ep:5.0f}/{EPISODES} | "
                    f"Discovered: {discs}/{len(SEEDS)} | Solved: {solves}/{len(SEEDS)} | "
                    f"Elapsed: {now - t_start:.1f}s | Plot updated.",
                    flush=True,
                )
                last_plot_time = now
            except Exception as e:
                print(f"Plotting error: {e}", flush=True)

        if done_count == len(active_seeds):
            break

        time.sleep(1.0)

    # Final gather and save
    final_results = {}
    for s in SEEDS:
        p = os.path.join(SCRATCH_DIR, f"seed_{s}_progress.json")
        if os.path.exists(p):
            with open(p, "r") as fp:
                final_results[f"deepsea50_s{s}"] = json.load(fp)

    final_json = os.path.join(BASE_DIR, "deepsea50_final_data.json")
    with open(final_json, "w") as f:
        json.dump(final_results, f, indent=2)

    # Final high-res plot
    render_live_plot(
        {s: final_results[f"deepsea50_s{s}"] for s in SEEDS if f"deepsea50_s{s}" in final_results},
        os.path.join(BASE_DIR, "deepsea50_live_plot.png"),
    )
    import shutil
    shutil.copyfile(os.path.join(BASE_DIR, "deepsea50_live_plot.png"), live_png_art)

    print("\n" + "=" * 85)
    print("DEEP SEA 50 BENCHMARK COMPLETE!")
    print(f"Results saved to {final_json}")
    print(f"Final plot saved to {live_png_art}")
    print("=" * 85)


if __name__ == "__main__":
    main()
