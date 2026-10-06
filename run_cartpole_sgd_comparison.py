#!/usr/bin/env python3
"""Cart-Pole Swing-Up Benchmark: sgd_period=2 vs sgd_period=4.

Evaluates the Modern Unified DP-DQN on arXiv:1703.07608 Section 7.2.2:
- One Living Q-Network (in-place continuous weights)
- Polyak Target Tracking (tau = 0.02)
- TD Information Gain Prior Decay (Option 3)
- CartPoleResonantBaseMeasure (50% upright anchors, 50% resonant swing dynamics)
- Seed: 42
- Comparison: sgd_period=2 (500 SGD steps/ep) vs sgd_period=4 (250 SGD steps/ep)
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

from src.envs.cartpole_jmlr import CartpoleSwingupJMLR
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.agent import DPDQNAgent

BASE_DIR = "/Users/sumitvashishtha/Desktop/DP-BNNs"
RESULTS_DIR = os.path.join(BASE_DIR, "results_rl")
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Benchmark Configuration
SEED = 42
NUM_EPISODES = 500
STATE_DIM = 6
ACTION_DIM = 3
HIDDEN_DIM = 64
NUM_LAYERS = 2
ACTIVATION = "relu"
USE_LAYER_NORM = True
LR = 1e-3
GAMMA = 0.99
TAU = 0.02
BUFFER_CAPACITY = 100000
BATCH_SIZE = 64
ALPHA = 15.0
BASE_MEASURE = "cartpole"
USE_TD_INFO_GAIN_DECAY = True
TD_INFO_SCALE = 1.0
WARMSTART_STEPS = 2

CONFIGS = [
    {
        "label": "Modern DP-DQN (sgd_period=2)",
        "sgd_period": 2,
        "color": "#1E88E5",  # Vibrant Blue
    },
    {
        "label": "Modern DP-DQN (sgd_period=4)",
        "sgd_period": 4,
        "color": "#E53935",  # Vibrant Red
    },
]


def run_trial(sgd_period: int, seed: int = SEED, num_episodes: int = NUM_EPISODES) -> Dict[str, Any]:
    """Execute single run for specified sgd_period."""
    config = DPDQNConfig(
        env_name="cartpole_swingup",
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        activation=ACTIVATION,
        use_layer_norm=USE_LAYER_NORM,
        one_living_network=True,
        warmstart_steps=WARMSTART_STEPS,
        use_td_info_gain_decay=USE_TD_INFO_GAIN_DECAY,
        td_info_scale=TD_INFO_SCALE,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        lr=LR,
        gamma=GAMMA,
        tau=TAU,
        sgd_period=sgd_period,
        buffer_capacity=BUFFER_CAPACITY,
        base_measure=BASE_MEASURE,
        seed=seed,
    )

    env = CartpoleSwingupJMLR(seed=seed)
    agent = DPDQNAgent(config)

    returns = []
    upright_steps = []
    peak_cos_thetas = []
    prob_syn_history = []
    cumulative_info_history = []
    ep_times = []

    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Launching sgd_period={sgd_period} (Seed={seed}, Episodes={num_episodes})...", flush=True)

    for ep in range(num_episodes):
        ep_start = time.time()
        agent.reset_episode()
        state = env.reset()
        done = False
        ep_reward = 0.0
        upr_count = 0
        peak_cos = -1.0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if info.get("is_upright", False):
                upr_count += 1
            cos_th = float(info.get("cos_theta", state[0]))
            if cos_th > peak_cos:
                peak_cos = cos_th

        ep_duration = time.time() - ep_start
        returns.append(float(ep_reward))
        upright_steps.append(int(upr_count))
        peak_cos_thetas.append(float(peak_cos))
        prob_syn_history.append(float(agent.current_prob_syn))
        cumulative_info_history.append(float(agent.cumulative_info))
        ep_times.append(ep_duration)

        if (ep + 1) % 25 == 0 or ep == 0 or (ep + 1) == num_episodes:
            recent_ret = np.mean(returns[-25:]) if ep >= 24 else np.mean(returns)
            recent_upr = np.mean(upright_steps[-25:]) if ep >= 24 else np.mean(upright_steps)
            elapsed = time.time() - start_time
            eps_per_sec = (ep + 1) / elapsed
            print(
                f"  [sgd_period={sgd_period}] Ep {ep+1:4d}/{num_episodes} | "
                f"25-Ep Return: {recent_ret:6.1f} | Upright Steps: {recent_upr:4.1f}/1000 | "
                f"Peak cos: {peak_cos:5.2f} | P(syn): {agent.current_prob_syn:5.3f} | "
                f"Rate: {eps_per_sec:.2f} ep/s | Elapsed: {elapsed:4.0f}s",
                flush=True,
            )

    total_time = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Completed sgd_period={sgd_period} in {total_time:.1f}s ({total_time/num_episodes:.3f}s/ep)", flush=True)

    return {
        "sgd_period": sgd_period,
        "seed": seed,
        "num_episodes": num_episodes,
        "returns": returns,
        "upright_steps": upright_steps,
        "peak_cos_thetas": peak_cos_thetas,
        "prob_syn_history": prob_syn_history,
        "cumulative_info_history": cumulative_info_history,
        "total_time": total_time,
        "avg_time_per_ep": total_time / num_episodes,
    }


def make_plots(results: Dict[int, Dict[str, Any]], out_path: str):
    """Generate 4-panel publication-grade comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def smooth(vals, w=15):
        if len(vals) < w:
            return vals
        box = np.ones(w) / w
        return np.convolve(vals, box, mode="same")

    colors = {2: "#1E88E5", 4: "#E53935"}
    labels = {2: "Modern DP-DQN (sgd_period=2, 500 updates/ep)", 4: "Modern DP-DQN (sgd_period=4, 250 updates/ep)"}

    # (a) Episodic Return
    ax = axes[0, 0]
    for sgd, res in results.items():
        rets = res["returns"]
        ax.plot(smooth(rets), label=labels[sgd], color=colors[sgd], lw=2.2)
        ax.plot(rets, color=colors[sgd], alpha=0.15, lw=0.7)
    ax.set_title("(a) Episodic Return (Smoothed, w=15)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    # (b) Time Upright per Episode
    ax = axes[0, 1]
    for sgd, res in results.items():
        upr = res["upright_steps"]
        ax.plot(smooth(upr), label=labels[sgd], color=colors[sgd], lw=2.2)
        ax.plot(upr, color=colors[sgd], alpha=0.15, lw=0.7)
    ax.set_title("(b) Upright Equilibrium Steps / Episode (max=1000)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Upright Steps (cos θ > 0.95)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    # (c) Cumulative Upright Steps
    ax = axes[1, 0]
    for sgd, res in results.items():
        cum_upr = np.cumsum(res["upright_steps"])
        ax.plot(cum_upr, label=f"{labels[sgd]} (Final: {cum_upr[-1]:,})", color=colors[sgd], lw=2.5)
    ax.set_title("(c) Cumulative Time Spent Upright (Sample Efficiency)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Steps (cos θ > 0.95)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    # (d) Prior Weight Decay P(syn)
    ax = axes[1, 1]
    for sgd, res in results.items():
        psyn = res["prob_syn_history"]
        ax.plot(psyn, label=labels[sgd], color=colors[sgd], lw=2.2)
    ax.set_title("(d) Dirichlet Prior Weight Decay P(syn) vs. Episodes", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("P(syn) = α / (α + n_effective)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.suptitle(
        "Cart-Pole Swing-Up (arXiv:1703.07608 Sec 7.2.2)\n"
        "Modern DP-DQN Architecture Ablation: sgd_period=2 vs sgd_period=4 (Seed 42)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {out_path}", flush=True)


def main():
    print("=" * 80)
    print("CART-POLE SWING-UP BENCHMARK: SGD PERIOD 2 vs 4 (SEED 42)")
    print("=" * 80)

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_trial, cfg["sgd_period"]): cfg["sgd_period"]
            for cfg in CONFIGS
        }
        for fut in concurrent.futures.as_completed(futures):
            sgd = futures[fut]
            res = fut.result()
            results[sgd] = res

    # Save results JSON
    json_path = os.path.join(RESULTS_DIR, "cartpole_sgd2_vs_sgd4.json")
    with open(json_path, "w") as f:
        def serialize(obj):
            if isinstance(obj, (np.ndarray, list)):
                return [float(x) for x in obj]
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj
        json.dump(results, f, default=serialize, indent=2)
    print(f"Results saved to: {json_path}")

    # Generate plot
    plot_path = os.path.join(RESULTS_DIR, "cartpole_sgd2_vs_sgd4.png")
    make_plots(results, plot_path)

    # Copy to artifact dir
    artifact_plot = os.path.join(ARTIFACT_DIR, "cartpole_sgd2_vs_sgd4.png")
    import shutil
    shutil.copy2(plot_path, artifact_plot)
    print(f"Plot copied to artifact: {artifact_plot}")

    print("=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    for sgd in [2, 4]:
        res = results[sgd]
        cum_upr = int(np.sum(res["upright_steps"]))
        final_ret = float(np.mean(res["returns"][-50:]))
        print(f"  sgd_period={sgd}: Total Time = {res['total_time']:.1f}s | Final 50-Ep Return = {final_ret:.1f} | Cumulative Upright = {cum_upr:,} steps")
    print("=" * 80)


if __name__ == "__main__":
    main()
