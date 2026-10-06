"""RiverSwim Scaling Suite for DP-DQN (Dirichlet Process Deep Q-Networks).

Evaluates DP-DQN on RiverSwim across scaling sizes:
N in {6, 10, 14, 18, 22, 26, 30}
across 5 independent random seeds per size.

Tracks:
1. First Upstream Goal Discovery Episode
2. Episodes to Solve (First reward >= 1.0)
3. Cumulative Return & Final Return
4. Wallclock Computation Scaling
5. Fits Empirical Power Law O(N^gamma)

Generates publication-quality 3-panel vector figures (.pdf and .png).
"""

import os
import sys
import time
import json
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, "/Users/sumitvashishtha/Desktop/DP-BNNs")
from src.bayesian_distributional_rl.environments import RiverSwimEnv
from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from run_riverswim_misspecification_benchmark import RiverSwimBaseMeasure

SEEDS = [42, 101, 2024, 7, 99]
SIZES = [6, 10, 14, 18, 22, 26, 30]
MAX_EPISODES = 100


def run_scaling_benchmark() -> Dict[str, Any]:
    print("=" * 70)
    print("RIVERSWIM SCALING SUITE FOR DP-DQN (ONLY DP-DQN, NO BASELINES)")
    print(f"Sizes N: {SIZES}")
    print(f"Seeds: {SEEDS} ({len(SEEDS)} per size)")
    print(f"Max Episodes per run: {MAX_EPISODES}")
    print("=" * 70)

    summary = {
        "sizes": SIZES,
        "discovery_episodes_mean": [],
        "discovery_episodes_std": [],
        "discovery_episodes_all": {},
        "cumulative_returns_mean": [],
        "cumulative_returns_std": [],
        "learning_curves": {},
        "wallclock_times_mean": [],
    }

    for N in SIZES:
        H = 10 * N
        print(f"\n--- Testing RiverSwim N = {N} (Horizon H = {H}) ---")
        
        discovery_eps = []
        cum_returns = []
        seed_learning_curves = []
        runtimes = []

        for seed in SEEDS:
            t0 = time.time()
            env = RiverSwimEnv(n_states=N, max_steps=H, seed=seed)
            cfg = DPDQNConfig(
                state_dim=N, action_dim=2, hidden_dim=32, num_layers=2, use_layer_norm=True,
                activation="relu", alpha=3.0, base_measure="custom", sampler_type="vashishtha_maillard",
                vm_prior_multiplier=4.0, seed=seed, buffer_capacity=50000, one_living_network=True,
                warmstart_steps=5, episodic_sgd=True, tau=0.05, lr=5e-3, gamma=0.98, batch_size=32,
                candidate_batch_size=64, sgd_period=2,
            )
            bm = RiverSwimBaseMeasure(n_states=N, misspecified=False)
            agent = DPDQNAgent(cfg, base_measure=bm)
            for _ in range(50):
                agent.update()

            solved_at = None
            ep_returns = []
            cum_ret = 0.0

            for ep in range(1, MAX_EPISODES + 1):
                agent.reset_episode()
                s_idx = env.reset()
                s = np.zeros(N, dtype=np.float32)
                s[s_idx] = 1.0
                done = False
                ep_ret = 0.0
                steps = 0

                while not done:
                    a = agent.act(s)
                    sn_idx, r, done, _ = env.step(a)
                    sn = np.zeros(N, dtype=np.float32)
                    sn[sn_idx] = 1.0
                    agent.step(s, a, float(r), sn, done)
                    ep_ret += float(r)
                    s = sn
                    steps += 1
                    if r >= 1.0 and solved_at is None:
                        solved_at = ep

                agent.end_episode(steps)
                ep_returns.append(ep_ret)
                cum_ret += ep_ret

            elapsed = time.time() - t0
            runtimes.append(elapsed)
            # If not solved within MAX_EPISODES, penalize by MAX_EPISODES
            discovery_ep = solved_at if solved_at is not None else MAX_EPISODES
            discovery_eps.append(discovery_ep)
            cum_returns.append(cum_ret)
            seed_learning_curves.append(ep_returns)
            print(f"  [N={N} | Seed {seed:4d}] Solved at Ep {discovery_ep:2d} | Cum Return: {cum_ret:5.1f} | Time: {elapsed:.2f}s")

        summary["discovery_episodes_mean"].append(float(np.mean(discovery_eps)))
        summary["discovery_episodes_std"].append(float(np.std(discovery_eps)))
        summary["discovery_episodes_all"][str(N)] = [int(x) for x in discovery_eps]
        summary["cumulative_returns_mean"].append(float(np.mean(cum_returns)))
        summary["cumulative_returns_std"].append(float(np.std(cum_returns)))
        summary["learning_curves"][str(N)] = np.mean(seed_learning_curves, axis=0).tolist()
        summary["wallclock_times_mean"].append(float(np.mean(runtimes)))

        print(f"  => N={N} Summary: Discovery Ep = {np.mean(discovery_eps):.1f} +/- {np.std(discovery_eps):.1f} | Mean Runtime: {np.mean(runtimes):.2f}s")

    # Save JSON results
    with open("riverswim_scaling_dpdqn.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved statistical summary to riverswim_scaling_dpdqn.json")

    # Fit empirical power law: Discovery = c * N^gamma
    N_arr = np.array(SIZES, dtype=np.float64)
    disc_arr = np.array(summary["discovery_episodes_mean"], dtype=np.float64)

    def power_law(x, c, gamma):
        return c * (x ** gamma)

    try:
        popt, _ = curve_fit(power_law, N_arr, disc_arr, p0=[1.0, 1.0])
        gamma_fit = popt[1]
        c_fit = popt[0]
        print(f"\nEmpirical Power-Law Fit for Discovery: Ep(N) = {c_fit:.3f} * N^{gamma_fit:.2f}")
    except Exception as e:
        gamma_fit = 1.0
        c_fit = 1.0
        print(f"Power law fit failed: {e}")

    # Generate 3-Panel Publication Figure
    plot_results(summary, c_fit, gamma_fit)
    return summary


def plot_results(summary: Dict[str, Any], c_fit: float, gamma_fit: float):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    N_arr = np.array(summary["sizes"])
    disc_mean = np.array(summary["discovery_episodes_mean"])
    disc_std = np.array(summary["discovery_episodes_std"])
    times_mean = np.array(summary["wallclock_times_mean"])

    # Panel 1: Discovery Episodes vs N (Power Law)
    ax1 = axes[0]
    ax1.errorbar(N_arr, disc_mean, yerr=disc_std, fmt="o-", color="#0072B2", linewidth=2.2, capsize=5, capthick=1.5, label="DP-DQN (Ours)")
    
    # Plot power law fit
    N_dense = np.linspace(min(N_arr), max(N_arr), 100)
    fit_curve = c_fit * (N_dense ** gamma_fit)
    ax1.plot(N_dense, fit_curve, "--", color="#D55E00", linewidth=2.0, label=f"Fit: $\\mathcal{{O}}(N^{{{gamma_fit:.2f}}})$")
    
    ax1.set_xlabel("River Length $N$ (States)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Episodes to First Upstream Goal", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Sample Complexity Scaling on RiverSwim", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Learning Curves Across River Sizes
    ax2 = axes[1]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(summary["sizes"]))]

    ep_axis = np.arange(1, MAX_EPISODES + 1)
    for i, N in enumerate(summary["sizes"]):
        curve = np.array(summary["learning_curves"][str(N)])
        ax2.plot(ep_axis, curve, label=f"N={N}", color=colors[i], linewidth=1.8)

    ax2.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Episode Return", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Learning Curves Across River Lengths", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, fontsize=9, ncol=2)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Wallclock Runtime vs N
    ax3 = axes[2]
    ax3.plot(N_arr, times_mean, "s-", color="#009E73", linewidth=2.2, markersize=7, label="Wallclock Time (s)")
    ax3.set_xlabel("River Length $N$ (States)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Time for 100 Episodes (s)", fontsize=11, fontweight="bold")
    ax3.set_title("(c) Wallclock Computation Scaling", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True, fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_path = "Figure_RiverSwim_Scaling.pdf"
    png_path = "Figure_RiverSwim_Scaling.png"
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    print(f"\nSaved figures to {pdf_path} and {png_path}")


if __name__ == "__main__":
    run_scaling_benchmark()
