"""Benchmark: DP-DQN vs Parametric PSRL under Prior Misspecification on RiverSwim.

Canonical Environment: 6-State RiverSwim (Strehl & Littman 2008, Osband et al. 2013).
Compares:
1. DP-DQN (Misspecified Base Measure) - Falsely believes downstream (a=0) is good (+0.5)
2. DP-DQN (Optimistic Prior) - Standard uninformative base measure
3. Parametric PSRL (Misspecified Conjugate Prior) - Dirichlet/Beta prior biased toward downstream
4. Vanilla DQN (eps-greedy annealing)

Generates:
- Statistical summary (mean, std, discovery rates across 5 seeds)
- Publication-grade figures (.png and .pdf)
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/sumitvashishtha/Desktop/DP-BNNs")
from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.base_measures import BaseMeasure
from src.bayesian_distributional_rl.environments import RiverSwimEnv
from unified_dp_dqn.baselines import StandardDQNAgent

ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
SEEDS = [42, 101, 2024, 7, 99]


# ==============================================================================
# 1. Base Measures for RiverSwim
# ==============================================================================
class RiverSwimBaseMeasure(BaseMeasure):
    def __init__(self, n_states: int = 6, misspecified: bool = False):
        super().__init__(state_dim=n_states, action_dim=2)
        self.n_states = n_states
        self.misspecified = misspecified

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.n_states), dtype=np.float32)
        sn = np.zeros((count, self.n_states), dtype=np.float32)
        idx = rng.randint(0, self.n_states, size=count)
        s[np.arange(count), idx] = 1.0
        a = rng.randint(0, 2, size=count)
        
        # Next state transitions
        next_idx = np.where(a == 1, np.minimum(idx + 1, self.n_states - 1), np.maximum(idx - 1, 0))
        sn[np.arange(count), next_idx] = 1.0
        
        if self.misspecified:
            # Misspecified prior: falsely believes downstream (a=0) is lucrative (+0.5) and upstream (a=1) is 0
            r = np.where(a == 0, rng.normal(0.5, 0.1, size=count), rng.normal(0.0, 0.1, size=count)).astype(np.float32)
        else:
            # Standard optimistic prior: upstream has reward potential
            r = np.where((idx == self.n_states - 1) & (a == 1), rng.normal(1.0, 0.2, size=count), rng.normal(0.05, 0.05, size=count)).astype(np.float32)
            
        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done


# ==============================================================================
# 2. Parametric PSRL with Misspecified Conjugate Prior
# ==============================================================================
class MisspecifiedPSRL:
    """Parametric PSRL using Dirichlet transitions and Beta rewards with misspecified pseudo-counts."""
    def __init__(self, n_states: int = 6, misspecified: bool = True, seed: int = 42):
        self.n_states = n_states
        self.rng = np.random.default_rng(seed)
        self.r_counts = np.ones((n_states, 2, 2))
        self.t_counts = np.ones((n_states, 2, n_states)) * 0.1
        if misspecified:
            self.r_counts[:, 0, 0] = 50.0  # Strong prior: a=0 gives reward
            self.r_counts[:, 1, 1] = 50.0  # Strong prior: a=1 fails
            for s in range(n_states):
                self.t_counts[s, 0, max(0, s - 1)] = 50.0
                self.t_counts[s, 1, max(0, s - 1)] = 50.0  # Strong prior: upstream washes you back

    def sample_mdp_and_solve(self, gamma: float = 0.95) -> np.ndarray:
        P = np.zeros((self.n_states, 2, self.n_states))
        R = np.zeros((self.n_states, 2))
        for s in range(self.n_states):
            for a in range(2):
                P[s, a] = self.rng.dirichlet(np.maximum(self.t_counts[s, a], 1e-3))
                R[s, a] = self.rng.beta(max(self.r_counts[s, a, 0], 1e-3), max(self.r_counts[s, a, 1], 1e-3))

        V = np.zeros(self.n_states)
        for _ in range(50):
            Q = R + gamma * np.einsum("san,n->sa", P, V)
            V = np.max(Q, axis=1)
        return np.argmax(Q, axis=1)

    def update(self, s: int, a: int, r: float, sn: int):
        self.t_counts[s, a, sn] += 1.0
        if r > 0.1:
            self.r_counts[s, a, 0] += 1.0
        else:
            self.r_counts[s, a, 1] += 1.0


# ==============================================================================
# 3. Experiment Runner
# ==============================================================================
def run_riverswim_benchmark(seeds: List[int] = SEEDS, max_episodes: int = 100, max_steps: int = 40) -> Dict[str, Any]:
    print(f"\n==============================================================")
    print(f"RUNNING RIVERSWIM PRIOR MISSPECIFICATION BENCHMARK ({len(seeds)} SEEDS)")
    print(f"==============================================================")
    
    results = {
        "DP-DQN (Misspecified Base Measure)": {"returns": [], "solved_ep": [], "time": 0.0},
        "DP-DQN (Optimistic Prior)": {"returns": [], "solved_ep": [], "time": 0.0},
        "Parametric PSRL (Misspecified Prior)": {"returns": [], "solved_ep": [], "time": 0.0},
        "Vanilla DQN (eps-greedy)": {"returns": [], "solved_ep": [], "time": 0.0},
    }

    # 1. DP-DQN (Misspecified)
    t0 = time.time()
    for seed in seeds:
        env = RiverSwimEnv(n_states=6, max_steps=max_steps, seed=seed)
        cfg = DPDQNConfig(
            state_dim=6, action_dim=2, hidden_dim=32, num_layers=2, use_layer_norm=True,
            activation="relu", alpha=3.0, base_measure="custom", sampler_type="vashishtha_maillard",
            vm_prior_multiplier=4.0, seed=seed, buffer_capacity=50000, one_living_network=True,
            warmstart_steps=5, episodic_sgd=True, tau=0.05, lr=5e-3, gamma=0.95, batch_size=32,
            candidate_batch_size=64, sgd_period=2,
        )
        bm = RiverSwimBaseMeasure(n_states=6, misspecified=True)
        agent = DPDQNAgent(cfg, base_measure=bm)
        for _ in range(50):
            agent.update()

        seed_returns = []
        solved_at = None
        for ep in range(1, max_episodes + 1):
            agent.reset_episode()
            s_idx = env.reset()
            s = np.zeros(6, dtype=np.float32)
            s[s_idx] = 1.0
            done = False
            ep_ret = 0.0
            steps = 0
            while not done:
                a = agent.act(s)
                sn_idx, r, done, _ = env.step(a)
                sn = np.zeros(6, dtype=np.float32)
                sn[sn_idx] = 1.0
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r)
                s = sn
                steps += 1
                if r >= 1.0 and solved_at is None:
                    solved_at = ep
            agent.end_episode(steps)
            seed_returns.append(ep_ret)

        results["DP-DQN (Misspecified Base Measure)"]["returns"].append(seed_returns)
        results["DP-DQN (Misspecified Base Measure)"]["solved_ep"].append(solved_at)
        print(f"  [DP-DQN Misspecified] Seed {seed} solved at Episode {solved_at} (Best: {max(seed_returns):.1f})")
    results["DP-DQN (Misspecified Base Measure)"]["time"] = time.time() - t0

    # 2. DP-DQN (Optimistic)
    t0 = time.time()
    for seed in seeds:
        env = RiverSwimEnv(n_states=6, max_steps=max_steps, seed=seed)
        cfg = DPDQNConfig(
            state_dim=6, action_dim=2, hidden_dim=32, num_layers=2, use_layer_norm=True,
            activation="relu", alpha=3.0, base_measure="custom", sampler_type="vashishtha_maillard",
            vm_prior_multiplier=4.0, seed=seed, buffer_capacity=50000, one_living_network=True,
            warmstart_steps=5, episodic_sgd=True, tau=0.05, lr=5e-3, gamma=0.95, batch_size=32,
            candidate_batch_size=64, sgd_period=2,
        )
        bm = RiverSwimBaseMeasure(n_states=6, misspecified=False)
        agent = DPDQNAgent(cfg, base_measure=bm)
        for _ in range(50):
            agent.update()

        seed_returns = []
        solved_at = None
        for ep in range(1, max_episodes + 1):
            agent.reset_episode()
            s_idx = env.reset()
            s = np.zeros(6, dtype=np.float32)
            s[s_idx] = 1.0
            done = False
            ep_ret = 0.0
            steps = 0
            while not done:
                a = agent.act(s)
                sn_idx, r, done, _ = env.step(a)
                sn = np.zeros(6, dtype=np.float32)
                sn[sn_idx] = 1.0
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r)
                s = sn
                steps += 1
                if r >= 1.0 and solved_at is None:
                    solved_at = ep
            agent.end_episode(steps)
            seed_returns.append(ep_ret)

        results["DP-DQN (Optimistic Prior)"]["returns"].append(seed_returns)
        results["DP-DQN (Optimistic Prior)"]["solved_ep"].append(solved_at)
        print(f"  [DP-DQN Optimistic] Seed {seed} solved at Episode {solved_at} (Best: {max(seed_returns):.1f})")
    results["DP-DQN (Optimistic Prior)"]["time"] = time.time() - t0

    # 3. Parametric PSRL (Misspecified)
    t0 = time.time()
    for seed in seeds:
        env = RiverSwimEnv(n_states=6, max_steps=max_steps, seed=seed)
        psrl = MisspecifiedPSRL(n_states=6, misspecified=True, seed=seed)
        seed_returns = []
        solved_at = None
        for ep in range(1, max_episodes + 1):
            pi = psrl.sample_mdp_and_solve(gamma=0.95)
            s = env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                a = pi[s]
                sn, r, done, _ = env.step(a)
                psrl.update(s, a, r, sn)
                ep_ret += r
                s = sn
                if r >= 1.0 and solved_at is None:
                    solved_at = ep
            seed_returns.append(ep_ret)

        results["Parametric PSRL (Misspecified Prior)"]["returns"].append(seed_returns)
        results["Parametric PSRL (Misspecified Prior)"]["solved_ep"].append(solved_at)
        print(f"  [PSRL Misspecified] Seed {seed} solved at Episode {solved_at} (Best: {max(seed_returns):.1f})")
    results["Parametric PSRL (Misspecified Prior)"]["time"] = time.time() - t0

    # 4. Vanilla DQN
    t0 = time.time()
    for seed in seeds:
        env = RiverSwimEnv(n_states=6, max_steps=max_steps, seed=seed)
        agent = StandardDQNAgent(
            state_dim=6, action_dim=2, hidden_dim=32, epsilon_start=1.0, epsilon_end=0.05,
            anneal_episodes=50, gamma=0.95, tau=0.05, lr=5e-3, batch_size=32, capacity=50000, seed=seed
        )
        seed_returns = []
        solved_at = None
        for ep in range(1, max_episodes + 1):
            agent.reset_episode()
            s_idx = env.reset()
            s = np.zeros(6, dtype=np.float32)
            s[s_idx] = 1.0
            done = False
            ep_ret = 0.0
            while not done:
                a = agent.act(s)
                sn_idx, r, done, _ = env.step(a)
                sn = np.zeros(6, dtype=np.float32)
                sn[sn_idx] = 1.0
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r)
                s = sn
                if r >= 1.0 and solved_at is None:
                    solved_at = ep
            seed_returns.append(ep_ret)

        results["Vanilla DQN (eps-greedy)"]["returns"].append(seed_returns)
        results["Vanilla DQN (eps-greedy)"]["solved_ep"].append(solved_at)
        print(f"  [Vanilla DQN] Seed {seed} solved at Episode {solved_at} (Best: {max(seed_returns):.1f})")
    results["Vanilla DQN (eps-greedy)"]["time"] = time.time() - t0

    return results


# ==============================================================================
# 4. Plotter and Exporter
# ==============================================================================
def plot_and_export_riverswim(results: Dict[str, Any]):
    colors = {
        "DP-DQN (Misspecified Base Measure)": "#1b9e77", # Forest Green
        "DP-DQN (Optimistic Prior)": "#377eb8",          # Blue
        "Parametric PSRL (Misspecified Prior)": "#e41a1c", # Red
        "Vanilla DQN (eps-greedy)": "#999999",           # Gray
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # Plot 1: Cumulative Return over Episodes
    ax1 = axes[0]
    for name, data in results.items():
        arr = np.array(data["returns"])
        cum_ret = np.cumsum(arr, axis=1)
        mean_cr = np.mean(cum_ret, axis=0)
        std_cr = np.std(cum_ret, axis=0) / np.sqrt(len(SEEDS))
        col = colors.get(name, "#333333")
        ax1.plot(mean_cr, label=f"{name} (Tot: {mean_cr[-1]:.1f})", color=col, lw=2.2)
        ax1.fill_between(range(len(mean_cr)), mean_cr - std_cr, mean_cr + std_cr, color=col, alpha=0.18)

    ax1.set_title("(a) Cumulative Return on 6-State RiverSwim ($5$ Seeds)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=10)
    ax1.set_ylabel("Cumulative Return", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # Plot 2: Smoothed Episode Return
    ax2 = axes[1]
    w = 5
    for name, data in results.items():
        arr = np.array(data["returns"])
        mean_r = np.mean(arr, axis=0)
        std_r = np.std(arr, axis=0) / np.sqrt(len(SEEDS))
        col = colors.get(name, "#333333")
        
        smooth_mean = np.convolve(mean_r, np.ones(w)/w, mode="valid")
        smooth_std = np.convolve(std_r, np.ones(w)/w, mode="valid")
        x_range = np.arange(w - 1, len(mean_r))
        
        valid_solved = [s for s in data["solved_ep"] if s is not None]
        avg_solve = f"Ep {np.mean(valid_solved):.1f}" if valid_solved else "Never"
        ax2.plot(x_range, smooth_mean, label=f"{name} (Jackpot: {avg_solve})", color=col, lw=2.2)
        ax2.fill_between(x_range, smooth_mean - smooth_std, smooth_mean + smooth_std, color=col, alpha=0.18)

    ax2.axhline(0.05 * 40, color="gray", linestyle=":", alpha=0.6, label="Downstream Trap Return (2.0)")
    ax2.set_title("(b) Per-Episode Return (5-step avg)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=10)
    ax2.set_ylabel("Episode Return", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()

    out_base = os.path.join(ARTIFACT_DIR, "riverswim_misspecification_benchmark")
    plt.savefig(f"{out_base}.png", dpi=300)
    plt.savefig(f"{out_base}.pdf")
    plt.close()

    ws_base = "/Users/sumitvashishtha/Desktop/DP-BNNs/riverswim_misspecification_benchmark"
    fig.savefig(f"{ws_base}.png", dpi=300)
    fig.savefig(f"{ws_base}.pdf")
    print(f"\n--> Successfully saved RiverSwim publication figures to {out_base}.pdf and {ws_base}.pdf")

    # Export JSON
    summary_data = {
        "seeds": SEEDS,
        "benchmark": "RiverSwim-6 Prior Misspecification",
        "results": {
            k: {
                "solved_episodes": v["solved_ep"],
                "total_cumulative_return_mean": float(np.mean([np.sum(r) for r in v["returns"]])),
                "final_10_return_mean": float(np.mean([np.mean(np.array(r)[-10:]) for r in v["returns"]])),
                "runtime_sec": v["time"],
            }
            for k, v in results.items()
        }
    }
    with open(f"{out_base}.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    with open(f"{ws_base}.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"--> Saved numeric summary to {out_base}.json")


if __name__ == "__main__":
    t_start = time.time()
    results = run_riverswim_benchmark()
    plot_and_export_riverswim(results)
    print(f"\nRIVERSWIM BENCHMARK COMPLETED IN {time.time() - t_start:.1f}s TOTAL!")
