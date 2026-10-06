"""Canonical RiverSwim Benchmark: DP-DQN under Standard Normal Base Measure and Prior Misspecification.

Canonical Environment: 6-State RiverSwim (Strehl & Littman 2008, Osband et al. 2013).
Network Architecture: Single hidden layer of 20 units (MLP-20) with LayerNorm.

Evaluates 5 distinct paradigms across 5 random seeds {42, 101, 2024, 7, 99}:
1. DP-DQN (Standard Normal Prior) - Zero jackpot bias, uninformative N(0, 1) reward base measure.
2. DP-DQN (Misspecified Base Measure) - Falsely believes downstream (a=0) is lucrative (+0.5).
3. Parametric PSRL (Misspecified Prior) - Conjugate Dirichlet/Beta prior heavily biased downstream.
4. Parametric PSRL (Standard Prior) - Canonical diffuse uninformative prior.
5. Vanilla DQN (eps-greedy) - Standard DQN with single MLP-20 and epsilon annealing.

Generates:
- Statistical summary (mean, std, discovery rates across 5 seeds).
- Publication-grade figures (.png and .pdf) saved to root and artifact directory.
- JSON summary for paper inclusion.
"""

import os
import sys
import time
import json
from typing import Dict, List, Any
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
N_STATES = 6
HORIZON = 50
MAX_EPISODES = 100


class CanonicalRiverSwimBaseMeasure(BaseMeasure):
    def __init__(self, n_states: int = 6, mode: str = "std_normal"):
        super().__init__(state_dim=n_states, action_dim=2)
        self.n_states = n_states
        self.mode = mode

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.n_states), dtype=np.float32)
        sn = np.zeros((count, self.n_states), dtype=np.float32)
        idx = rng.randint(0, self.n_states, size=count)
        s[np.arange(count), idx] = 1.0
        a = rng.randint(0, 2, size=count)
        
        # Next state transitions: a=1 moves right, a=0 moves left
        next_idx = np.where(a == 1, np.minimum(idx + 1, self.n_states - 1), np.maximum(idx - 1, 0))
        sn[np.arange(count), next_idx] = 1.0
        
        if self.mode == "misspecified":
            # Falsely believes downstream (a=0) is lucrative (+0.5) and upstream (a=1) gives zero
            r = np.where(a == 0, rng.normal(0.5, 0.1, size=count), rng.normal(0.0, 0.1, size=count)).astype(np.float32)
        else:
            # Standard Normal Base Measure: N(0, 1) across all transitions (ZERO jackpot bias)
            r = rng.normal(0.0, 1.0, size=count).astype(np.float32)
            
        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done


class RiverSwimPSRL:
    def __init__(self, n_states: int = 6, misspecified: bool = False, seed: int = 42):
        self.n_states = n_states
        self.rng = np.random.default_rng(seed)
        self.r_counts = np.ones((n_states, 2, 2)) * 0.5
        self.t_counts = np.ones((n_states, 2, n_states)) * (1.0 / n_states)
        if misspecified:
            self.r_counts[:, 0, 0] = 50.0  # Strong prior: downstream gives reward
            self.r_counts[:, 1, 1] = 50.0  # Strong prior: upstream fails
            for s in range(n_states):
                self.t_counts[s, 0, max(0, s - 1)] = 50.0
                self.t_counts[s, 1, max(0, s - 1)] = 50.0

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


def run_benchmark() -> Dict[str, Any]:
    print("=" * 70)
    print(f"CANONICAL RIVERSWIM BENCHMARK (N={N_STATES}, H={HORIZON})")
    print(f"Architecture: Single MLP-20 (hidden_dim=20, num_layers=1, LayerNorm)")
    print(f"Seeds: {SEEDS} ({len(SEEDS)} seeds, {MAX_EPISODES} episodes)")
    print("=" * 70)

    results = {
        "DP-DQN (Standard Normal Prior)": {"returns": [], "solved_ep": [], "time": 0.0},
        "DP-DQN (Misspecified Prior)": {"returns": [], "solved_ep": [], "time": 0.0},
        "Parametric PSRL (Standard Prior)": {"returns": [], "solved_ep": [], "time": 0.0},
        "Parametric PSRL (Misspecified Prior)": {"returns": [], "solved_ep": [], "time": 0.0},
        "Vanilla DQN (eps-greedy)": {"returns": [], "solved_ep": [], "time": 0.0},
    }

    # 1. DP-DQN (Standard Normal Prior)
    print("\n--- Running DP-DQN (Standard Normal Prior, Zero Jackpot Bias) ---")
    t0 = time.time()
    for seed in SEEDS:
        env = RiverSwimEnv(n_states=N_STATES, max_steps=HORIZON, seed=seed)
        cfg = DPDQNConfig(
            state_dim=N_STATES, action_dim=2, hidden_dim=20, num_layers=1, use_layer_norm=True,
            activation="relu", alpha=3.0, base_measure="custom", sampler_type="vashishtha_maillard",
            vm_prior_multiplier=4.0, seed=seed, buffer_capacity=50000, one_living_network=True,
            warmstart_steps=5, episodic_sgd=True, tau=0.05, lr=5e-3, gamma=0.95, batch_size=32,
            candidate_batch_size=64, sgd_period=2, sample_once_per_episode=False,
        )
        agent = DPDQNAgent(cfg, base_measure=CanonicalRiverSwimBaseMeasure(N_STATES, mode="std_normal"))
        for _ in range(50):
            agent.update()

        seed_returns = []
        solved_at = None
        for ep in range(1, MAX_EPISODES + 1):
            agent.reset_episode()
            s_idx = env.reset()
            s = np.zeros(N_STATES, dtype=np.float32); s[s_idx] = 1.0
            done = False; ep_ret = 0.0; steps = 0
            while not done:
                a = agent.act(s)
                sn_idx, r, done, _ = env.step(a)
                sn = np.zeros(N_STATES, dtype=np.float32); sn[sn_idx] = 1.0
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r); s = sn; steps += 1
                if r >= 1.0 and solved_at is None:
                    solved_at = ep
            agent.end_episode(steps)
            seed_returns.append(ep_ret)
        results["DP-DQN (Standard Normal Prior)"]["returns"].append(seed_returns)
        results["DP-DQN (Standard Normal Prior)"]["solved_ep"].append(solved_at)
        print(f"  Seed {seed} solved at Episode {solved_at} (CumRet: {sum(seed_returns):.1f})")
    results["DP-DQN (Standard Normal Prior)"]["time"] = time.time() - t0

    # 2. DP-DQN (Misspecified Prior)
    print("\n--- Running DP-DQN (Misspecified Prior, Falsely Favors Downstream) ---")
    t0 = time.time()
    for seed in SEEDS:
        env = RiverSwimEnv(n_states=N_STATES, max_steps=HORIZON, seed=seed)
        cfg = DPDQNConfig(
            state_dim=N_STATES, action_dim=2, hidden_dim=20, num_layers=1, use_layer_norm=True,
            activation="relu", alpha=3.0, base_measure="custom", sampler_type="vashishtha_maillard",
            vm_prior_multiplier=4.0, seed=seed, buffer_capacity=50000, one_living_network=True,
            warmstart_steps=5, episodic_sgd=True, tau=0.05, lr=5e-3, gamma=0.95, batch_size=32,
            candidate_batch_size=64, sgd_period=2, sample_once_per_episode=False,
        )
        agent = DPDQNAgent(cfg, base_measure=CanonicalRiverSwimBaseMeasure(N_STATES, mode="misspecified"))
        for _ in range(50):
            agent.update()

        seed_returns = []
        solved_at = None
        for ep in range(1, MAX_EPISODES + 1):
            agent.reset_episode()
            s_idx = env.reset()
            s = np.zeros(N_STATES, dtype=np.float32); s[s_idx] = 1.0
            done = False; ep_ret = 0.0; steps = 0
            while not done:
                a = agent.act(s)
                sn_idx, r, done, _ = env.step(a)
                sn = np.zeros(N_STATES, dtype=np.float32); sn[sn_idx] = 1.0
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r); s = sn; steps += 1
                if r >= 1.0 and solved_at is None:
                    solved_at = ep
            agent.end_episode(steps)
            seed_returns.append(ep_ret)
        results["DP-DQN (Misspecified Prior)"]["returns"].append(seed_returns)
        results["DP-DQN (Misspecified Prior)"]["solved_ep"].append(solved_at)
        print(f"  Seed {seed} solved at Episode {solved_at} (CumRet: {sum(seed_returns):.1f})")
    results["DP-DQN (Misspecified Prior)"]["time"] = time.time() - t0

    # 3. Parametric PSRL (Standard Prior)
    print("\n--- Running Parametric PSRL (Standard Diffuse Prior) ---")
    t0 = time.time()
    for seed in SEEDS:
        env = RiverSwimEnv(n_states=N_STATES, max_steps=HORIZON, seed=seed)
        psrl = RiverSwimPSRL(n_states=N_STATES, misspecified=False, seed=seed)
        seed_returns = []
        solved_at = None
        for ep in range(1, MAX_EPISODES + 1):
            pi = psrl.sample_mdp_and_solve(gamma=0.95)
            s = env.reset(); done = False; ep_ret = 0.0
            while not done:
                a = pi[s]
                sn, r, done, _ = env.step(a)
                psrl.update(s, a, r, sn)
                ep_ret += r; s = sn
                if r >= 1.0 and solved_at is None:
                    solved_at = ep
            seed_returns.append(ep_ret)
        results["Parametric PSRL (Standard Prior)"]["returns"].append(seed_returns)
        results["Parametric PSRL (Standard Prior)"]["solved_ep"].append(solved_at)
        print(f"  Seed {seed} solved at Episode {solved_at} (CumRet: {sum(seed_returns):.1f})")
    results["Parametric PSRL (Standard Prior)"]["time"] = time.time() - t0

    # 4. Parametric PSRL (Misspecified Prior)
    print("\n--- Running Parametric PSRL (Misspecified Conjugate Prior) ---")
    t0 = time.time()
    for seed in SEEDS:
        env = RiverSwimEnv(n_states=N_STATES, max_steps=HORIZON, seed=seed)
        psrl = RiverSwimPSRL(n_states=N_STATES, misspecified=True, seed=seed)
        seed_returns = []
        solved_at = None
        for ep in range(1, MAX_EPISODES + 1):
            pi = psrl.sample_mdp_and_solve(gamma=0.95)
            s = env.reset(); done = False; ep_ret = 0.0
            while not done:
                a = pi[s]
                sn, r, done, _ = env.step(a)
                psrl.update(s, a, r, sn)
                ep_ret += r; s = sn
                if r >= 1.0 and solved_at is None:
                    solved_at = ep
            seed_returns.append(ep_ret)
        results["Parametric PSRL (Misspecified Prior)"]["returns"].append(seed_returns)
        results["Parametric PSRL (Misspecified Prior)"]["solved_ep"].append(solved_at)
        print(f"  Seed {seed} solved at Episode {solved_at} (CumRet: {sum(seed_returns):.1f})")
    results["Parametric PSRL (Misspecified Prior)"]["time"] = time.time() - t0

    # 5. Vanilla DQN
    print("\n--- Running Vanilla DQN (eps-greedy, MLP-20) ---")
    t0 = time.time()
    for seed in SEEDS:
        env = RiverSwimEnv(n_states=N_STATES, max_steps=HORIZON, seed=seed)
        agent = StandardDQNAgent(
            state_dim=N_STATES, action_dim=2, hidden_dim=20, epsilon_start=1.0, epsilon_end=0.05,
            anneal_episodes=50, gamma=0.95, tau=0.05, lr=5e-3, batch_size=32, capacity=50000, seed=seed
        )
        seed_returns = []
        solved_at = None
        for ep in range(1, MAX_EPISODES + 1):
            s_idx = env.reset()
            s = np.zeros(N_STATES, dtype=np.float32); s[s_idx] = 1.0
            done = False; ep_ret = 0.0
            while not done:
                a = agent.act(s)
                sn_idx, r, done, _ = env.step(a)
                sn = np.zeros(N_STATES, dtype=np.float32); sn[sn_idx] = 1.0
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r); s = sn
                if r >= 1.0 and solved_at is None:
                    solved_at = ep
            seed_returns.append(ep_ret)
        results["Vanilla DQN (eps-greedy)"]["returns"].append(seed_returns)
        results["Vanilla DQN (eps-greedy)"]["solved_ep"].append(solved_at)
        print(f"  Seed {seed} solved at Episode {solved_at} (CumRet: {sum(seed_returns):.1f})")
    results["Vanilla DQN (eps-greedy)"]["time"] = time.time() - t0

    return results


def plot_and_export(results: Dict[str, Any]):
    colors = {
        "DP-DQN (Standard Normal Prior)": "#1b9e77",        # Emerald green
        "DP-DQN (Misspecified Prior)": "#d95f02",           # Ochre / Orange
        "Parametric PSRL (Standard Prior)": "#7570b3",      # Purple
        "Parametric PSRL (Misspecified Prior)": "#e41a1c",  # Red
        "Vanilla DQN (eps-greedy)": "#7f7f7f",              # Gray
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # Panel 1: Cumulative Return
    ax1 = axes[0]
    for name, data in results.items():
        arr = np.array(data["returns"])
        cum_ret = np.cumsum(arr, axis=1)
        mean_cr = np.mean(cum_ret, axis=0)
        std_cr = np.std(cum_ret, axis=0) / np.sqrt(len(SEEDS))
        col = colors.get(name, "#333333")
        ax1.plot(mean_cr, label=f"{name} ({mean_cr[-1]:.1f})", color=col, lw=2.2)
        ax1.fill_between(range(len(mean_cr)), mean_cr - std_cr, mean_cr + std_cr, color=col, alpha=0.15)

    ax1.set_title("(a) Cumulative Return on 6-State RiverSwim ($5$ Seeds)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=10)
    ax1.set_ylabel("Cumulative Return", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # Panel 2: Smoothed Episode Return
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
        ax2.plot(x_range, smooth_mean, label=f"{name} (Solved: {avg_solve})", color=col, lw=2.2)
        ax2.fill_between(x_range, smooth_mean - smooth_std, smooth_mean + smooth_std, color=col, alpha=0.15)

    ax2.axhline(0.05 * HORIZON, color="gray", linestyle=":", alpha=0.7, label=f"Downstream Trap ({0.05*HORIZON:.1f})")
    ax2.set_title("(b) Per-Episode Return (5-Episode Window)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=10)
    ax2.set_ylabel("Episode Return", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()

    out_base = os.path.join(ARTIFACT_DIR, "Figure_RiverSwim_Canonical")
    ws_base = "/Users/sumitvashishtha/Desktop/DP-BNNs/Figure_RiverSwim_Canonical"
    for b in [out_base, ws_base]:
        plt.savefig(f"{b}.png", dpi=300)
        plt.savefig(f"{b}.pdf")
    plt.close()
    print(f"\n--> Successfully saved figures to {out_base}.pdf and {ws_base}.pdf")

    # JSON export
    summary = {
        "n_states": N_STATES,
        "horizon": HORIZON,
        "network": "MLP-20 (hidden_dim=20, num_layers=1)",
        "seeds": SEEDS,
        "results": {
            k: {
                "solved_episodes": v["solved_ep"],
                "solve_rate": float(sum(1 for s in v["solved_ep"] if s is not None) / len(SEEDS)),
                "mean_solve_episode": float(np.mean([s for s in v["solved_ep"] if s is not None])) if any(s is not None for s in v["solved_ep"]) else None,
                "cumulative_return_mean": float(np.mean([np.sum(r) for r in v["returns"]])),
                "cumulative_return_std": float(np.std([np.sum(r) for r in v["returns"]])),
                "runtime_sec": v["time"],
            }
            for k, v in results.items()
        }
    }
    with open(f"{ws_base}.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{out_base}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"--> Saved numeric summary to {ws_base}.json")


if __name__ == "__main__":
    t_start = time.time()
    results = run_benchmark()
    plot_and_export(results)
    print(f"\nRIVERSWIM BENCHMARK COMPLETED IN {time.time() - t_start:.1f}s TOTAL!")
