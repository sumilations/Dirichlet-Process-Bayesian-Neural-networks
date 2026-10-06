"""Master Benchmark Runner: Deceptive N-Chain and MountainCar-v0 across 5 Seeds.

Compares:
1. DP-DQN (Ours, single network with Dirichlet Process prior)
2. BootDQN-RP (Osband et al. 2018, 10-head ensemble + randomized prior)
3. Vanilla DQN (Mnih et al. 2015, eps-greedy annealing)

Generates:
- Statistical summary tables (mean, std, discovery rates)
- Publication-quality figures (.png and .pdf) for TMLR Camera Ready
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/sumitvashishtha/Desktop/DP-BNNs")
from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.base_measures import BaseMeasure
from unified_dp_dqn.baselines import BootDQNRPAgent, StandardDQNAgent

ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"
SEEDS = [42, 101, 2024, 7, 99]


# ==============================================================================
# 1. DECEPTIVE N-CHAIN BENCHMARK
# ==============================================================================
class DeceptiveChainEnv:
    def __init__(self, chain_length: int = 20, max_steps: int = 35):
        self.chain_length = chain_length
        self.max_steps = max_steps
        self.state_dim = chain_length
        self.action_dim = 2
        self.current_state = 0
        self.current_step = 0

    def reset(self) -> np.ndarray:
        self.current_state = 0
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.state_dim, dtype=np.float32)
        obs[self.current_state] = 1.0
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_step += 1
        reward = 0.0
        done = False

        if self.current_state == 0:
            if action == 0:
                reward = 1.0
                self.current_state = 0
                done = True  # Siren trap: episode ends
            else:
                reward = 0.0
                self.current_state = 1
        elif self.current_state == self.chain_length - 1:
            if action == 1:
                reward = 100.0
                done = True  # Grand Treasure
            else:
                self.current_state = max(0, self.current_state - 1)
        else:
            if action == 1:
                self.current_state = min(self.chain_length - 1, self.current_state + 1)
            else:
                self.current_state = max(0, self.current_state - 1)

        if self.current_step >= self.max_steps:
            done = True

        info = {"reached_treasure": bool(reward >= 100.0), "exploited_lure": bool(reward == 1.0)}
        return self._get_obs(), float(reward), done, info


class OptimisticChainBaseMeasure(BaseMeasure):
    def __init__(self, chain_length: int = 20):
        super().__init__(state_dim=chain_length, action_dim=2)
        self.chain_length = chain_length

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.chain_length), dtype=np.float32)
        sn = np.zeros((count, self.chain_length), dtype=np.float32)
        curr_idx = rng.randint(0, self.chain_length, size=count)
        s[np.arange(count), curr_idx] = 1.0
        a = rng.randint(0, 2, size=count)
        next_idx = np.where(a == 1, np.minimum(curr_idx + 1, self.chain_length - 1), np.maximum(curr_idx - 1, 0))
        sn[np.arange(count), next_idx] = 1.0
        r = np.where(a == 1, rng.normal(1.0, 0.2, size=count), rng.normal(0.0, 0.1, size=count)).astype(np.float32)
        done = (curr_idx == self.chain_length - 1).astype(np.float32)
        return s, a, r, sn, done


def run_chain_experiment(seeds: List[int] = SEEDS, max_episodes: int = 100, N: int = 20) -> Dict[str, Any]:
    print(f"\n==========================================")
    print(f"RUNNING DECEPTIVE N-CHAIN (N={N}) 5-SEED BENCHMARK")
    print(f"==========================================")
    
    results = {
        "DP-DQN (Ours)": {"returns": [], "solved_ep": [], "time": 0.0},
        "BootDQN-RP": {"returns": [], "solved_ep": [], "time": 0.0},
        "Vanilla DQN": {"returns": [], "solved_ep": [], "time": 0.0},
    }

    # 1. DP-DQN
    t0 = time.time()
    for s_idx, seed in enumerate(seeds):
        env = DeceptiveChainEnv(chain_length=N)
        cfg = DPDQNConfig(
            state_dim=N,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
            use_layer_norm=True,
            activation="relu",
            alpha=3.0,
            base_measure="custom",
            prior_reward_mean=1.0,
            prior_reward_std=0.2,
            sampler_type="vashishtha_maillard",
            vm_prior_multiplier=4.0,
            seed=seed,
            buffer_capacity=50000,
            one_living_network=True,
            warmstart_steps=5,
            episodic_sgd=True,
            tau=0.05,
            lr=5e-3,
            gamma=0.99,
            batch_size=32,
            candidate_batch_size=64,
            sgd_period=2,
        )
        base_measure = OptimisticChainBaseMeasure(chain_length=N)
        agent = DPDQNAgent(cfg, base_measure=base_measure)
        for _ in range(50):
            agent.update()

        seed_returns = []
        solved_at = None
        for ep in range(1, max_episodes + 1):
            agent.reset_episode()
            s = env.reset()
            done = False
            ep_ret = 0.0
            steps = 0
            while not done:
                a = agent.act(s)
                sn, r, done, info = env.step(a)
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r)
                s = sn
                steps += 1
            agent.end_episode(steps)
            seed_returns.append(ep_ret)
            if ep_ret >= 100.0 and solved_at is None:
                solved_at = ep

        results["DP-DQN (Ours)"]["returns"].append(seed_returns)
        results["DP-DQN (Ours)"]["solved_ep"].append(solved_at)
        print(f"  [DP-DQN] Seed {seed} solved at Episode {solved_at}")
    results["DP-DQN (Ours)"]["time"] = time.time() - t0

    # 2. BootDQN-RP
    t0 = time.time()
    for s_idx, seed in enumerate(seeds):
        env = DeceptiveChainEnv(chain_length=N)
        agent = BootDQNRPAgent(
            state_dim=N,
            action_dim=2,
            num_models=10,
            hidden_dim=32,
            prior_scale=3.0,
            gamma=0.99,
            tau=0.05,
            lr=1e-3,
            batch_size=32,
            capacity=50000,
            seed=seed,
        )
        seed_returns = []
        solved_at = None
        for ep in range(1, max_episodes + 1):
            agent.reset_episode()
            s = env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                a = agent.act(s)
                sn, r, done, info = env.step(a)
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r)
                s = sn
            seed_returns.append(ep_ret)
            if ep_ret >= 100.0 and solved_at is None:
                solved_at = ep

        results["BootDQN-RP"]["returns"].append(seed_returns)
        results["BootDQN-RP"]["solved_ep"].append(solved_at)
        print(f"  [BootDQN-RP] Seed {seed} solved at Episode {solved_at}")
    results["BootDQN-RP"]["time"] = time.time() - t0

    # 3. Vanilla DQN
    t0 = time.time()
    for s_idx, seed in enumerate(seeds):
        env = DeceptiveChainEnv(chain_length=N)
        agent = StandardDQNAgent(
            state_dim=N,
            action_dim=2,
            hidden_dim=32,
            epsilon_start=1.0,
            epsilon_end=0.05,
            anneal_episodes=50,
            gamma=0.99,
            tau=0.05,
            lr=1e-3,
            batch_size=32,
            capacity=50000,
            seed=seed,
        )
        seed_returns = []
        solved_at = None
        for ep in range(1, max_episodes + 1):
            agent.reset_episode()
            s = env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                a = agent.act(s)
                sn, r, done, info = env.step(a)
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r)
                s = sn
            seed_returns.append(ep_ret)
            if ep_ret >= 100.0 and solved_at is None:
                solved_at = ep

        results["Vanilla DQN"]["returns"].append(seed_returns)
        results["Vanilla DQN"]["solved_ep"].append(solved_at)
        print(f"  [Vanilla DQN] Seed {seed} solved at Episode {solved_at}")
    results["Vanilla DQN"]["time"] = time.time() - t0

    return results


# ==============================================================================
# 2. MOUNTAINCAR-V0 BENCHMARK
# ==============================================================================
class MountainCarBaseMeasure(BaseMeasure):
    def __init__(self):
        super().__init__(state_dim=2, action_dim=3)

    def sample(self, count: int, rng: np.random.RandomState):
        x = rng.uniform(-1.2, 0.6, size=count)
        v = rng.uniform(-0.07, 0.07, size=count)
        s = np.column_stack([x, v]).astype(np.float32)
        a = rng.randint(0, 3, size=count)
        force = (a - 1) * 0.001
        vn = np.clip(v + force - np.cos(3 * x) * 0.0025, -0.07, 0.07)
        xn = np.clip(x + vn, -1.2, 0.6)
        sn = np.column_stack([xn, vn]).astype(np.float32)
        is_goal = (xn >= 0.5)
        r = np.where(is_goal, 0.0, rng.normal(-0.1, 0.1, size=count)).astype(np.float32)
        done = is_goal.astype(np.float32)
        return s, a, r, sn, done


def run_mountaincar_experiment(seeds: List[int] = SEEDS, max_episodes: int = 100) -> Dict[str, Any]:
    print(f"\n==========================================")
    print(f"RUNNING MOUNTAINCAR-V0 5-SEED BENCHMARK")
    print(f"==========================================")

    results = {
        "DP-DQN (Ours)": {"returns": [], "solved_ep": [], "time": 0.0},
        "BootDQN-RP": {"returns": [], "solved_ep": [], "time": 0.0},
        "Vanilla DQN": {"returns": [], "solved_ep": [], "time": 0.0},
    }

    # 1. DP-DQN
    t0 = time.time()
    for s_idx, seed in enumerate(seeds):
        env = gym.make("MountainCar-v0")
        cfg = DPDQNConfig(
            state_dim=2,
            action_dim=3,
            hidden_dim=64,
            num_layers=2,
            use_layer_norm=True,
            activation="relu",
            alpha=3.0,
            base_measure="custom",
            sampler_type="vashishtha_maillard",
            vm_prior_multiplier=4.0,
            seed=seed,
            buffer_capacity=50000,
            one_living_network=True,
            warmstart_steps=5,
            episodic_sgd=False,
            tau=0.01,
            lr=1e-3,
            gamma=0.99,
            batch_size=32,
            candidate_batch_size=64,
            sgd_period=4,
        )
        base_measure = MountainCarBaseMeasure()
        agent = DPDQNAgent(cfg, base_measure=base_measure)
        for _ in range(100):
            agent.update()

        seed_returns = []
        solved_at = None
        for ep in range(1, max_episodes + 1):
            agent.reset_episode()
            s = env.reset()
            if isinstance(s, tuple): s = s[0]
            done = False
            ep_ret = 0.0
            steps = 0
            while not done:
                a = agent.act(s)
                res = env.step(a)
                if len(res) == 5:
                    sn, r, term, trunc, _ = res
                    done = term or trunc
                else:
                    sn, r, done, _ = res
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r)
                s = sn
                steps += 1
            seed_returns.append(ep_ret)
            if ep_ret > -200.0 and solved_at is None:
                solved_at = ep

        results["DP-DQN (Ours)"]["returns"].append(seed_returns)
        results["DP-DQN (Ours)"]["solved_ep"].append(solved_at)
        print(f"  [DP-DQN] Seed {seed} reached goal at Episode {solved_at} (Best: {max(seed_returns)})")
    results["DP-DQN (Ours)"]["time"] = time.time() - t0

    # 2. BootDQN-RP
    t0 = time.time()
    for s_idx, seed in enumerate(seeds):
        env = gym.make("MountainCar-v0")
        agent = BootDQNRPAgent(
            state_dim=2,
            action_dim=3,
            num_models=10,
            hidden_dim=64,
            prior_scale=3.0,
            gamma=0.99,
            lr=1e-3,
            seed=seed,
        )
        seed_returns = []
        solved_at = None
        for ep in range(1, max_episodes + 1):
            agent.reset_episode()
            s = env.reset()
            if isinstance(s, tuple): s = s[0]
            done = False
            ep_ret = 0.0
            while not done:
                a = agent.act(s)
                res = env.step(a)
                sn = res[0]
                r = res[1]
                done = res[2] or (res[3] if len(res) == 5 else False)
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r)
                s = sn
            seed_returns.append(ep_ret)
            if ep_ret > -200.0 and solved_at is None:
                solved_at = ep

        results["BootDQN-RP"]["returns"].append(seed_returns)
        results["BootDQN-RP"]["solved_ep"].append(solved_at)
        print(f"  [BootDQN-RP] Seed {seed} reached goal at Episode {solved_at} (Best: {max(seed_returns)})")
    results["BootDQN-RP"]["time"] = time.time() - t0

    # 3. Vanilla DQN
    t0 = time.time()
    for s_idx, seed in enumerate(seeds):
        env = gym.make("MountainCar-v0")
        agent = StandardDQNAgent(
            state_dim=2,
            action_dim=3,
            hidden_dim=64,
            epsilon_start=1.0,
            epsilon_end=0.05,
            anneal_episodes=50,
            gamma=0.99,
            lr=1e-3,
            seed=seed,
        )
        seed_returns = []
        solved_at = None
        for ep in range(1, max_episodes + 1):
            agent.reset_episode()
            s = env.reset()
            if isinstance(s, tuple): s = s[0]
            done = False
            ep_ret = 0.0
            while not done:
                a = agent.act(s)
                res = env.step(a)
                sn = res[0]
                r = res[1]
                done = res[2] or (res[3] if len(res) == 5 else False)
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r)
                s = sn
            seed_returns.append(ep_ret)
            if ep_ret > -200.0 and solved_at is None:
                solved_at = ep

        results["Vanilla DQN"]["returns"].append(seed_returns)
        results["Vanilla DQN"]["solved_ep"].append(solved_at)
        print(f"  [Vanilla DQN] Seed {seed} reached goal at Episode {solved_at} (Best: {max(seed_returns)})")
    results["Vanilla DQN"]["time"] = time.time() - t0

    return results


# ==============================================================================
# 3. PLOTTING AND EXPORTING
# ==============================================================================
def plot_and_export_both(chain_res: Dict[str, Any], mc_res: Dict[str, Any]):
    colors = {
        "DP-DQN (Ours)": "#1b9e77",        # Teal / Green
        "BootDQN-RP": "#d95f02",           # Orange / Red
        "Vanilla DQN": "#7570b3",          # Purple / Slate
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # --- Plot 1: Deceptive N-Chain Cumulative Regret ---
    ax1 = axes[0]
    optimal_ret = 100.0
    for name, data in chain_res.items():
        arr = np.array(data["returns"])  # shape: (n_seeds, n_eps)
        regrets = np.maximum(0.0, optimal_ret - arr)
        cum_regrets = np.cumsum(regrets, axis=1) / 1e3
        mean_cr = np.mean(cum_regrets, axis=0)
        std_cr = np.std(cum_regrets, axis=0) / np.sqrt(len(SEEDS))
        col = colors.get(name, "#333333")
        ax1.plot(mean_cr, label=f"{name} (Final: {mean_cr[-1]:.1f}k)", color=col, lw=2.2)
        ax1.fill_between(range(len(mean_cr)), mean_cr - std_cr, mean_cr + std_cr, color=col, alpha=0.18)

    ax1.set_title("(a) Deceptive N-Chain ($N=20$): Cumulative Regret", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=10)
    ax1.set_ylabel("Cumulative Regret vs Optimal ($\\times 10^3$)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # --- Plot 2: MountainCar-v0 Returns ---
    ax2 = axes[1]
    for name, data in mc_res.items():
        arr = np.array(data["returns"])  # shape: (n_seeds, n_eps)
        # Apply smoothing window of 5
        mean_ret = np.mean(arr, axis=0)
        std_ret = np.std(arr, axis=0) / np.sqrt(len(SEEDS))
        col = colors.get(name, "#333333")
        
        # Smoothed line
        w = 5
        smooth_mean = np.convolve(mean_ret, np.ones(w)/w, mode="valid")
        smooth_std = np.convolve(std_ret, np.ones(w)/w, mode="valid")
        x_range = np.arange(w - 1, len(mean_ret))
        
        # Solved info
        valid_solved = [s for s in data["solved_ep"] if s is not None]
        avg_solve = f"Ep {np.mean(valid_solved):.1f}" if valid_solved else "Never"
        ax2.plot(x_range, smooth_mean, label=f"{name} (First Goal: {avg_solve})", color=col, lw=2.2)
        ax2.fill_between(x_range, smooth_mean - smooth_std, smooth_mean + smooth_std, color=col, alpha=0.18)

    ax2.axhline(-200.0, color="gray", linestyle=":", alpha=0.6, label="Random Exploration Floor (-200)")
    ax2.set_title("(b) MountainCar-v0: Episode Return (5-step avg)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=10)
    ax2.set_ylabel("Episode Return", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()

    out_base = os.path.join(ARTIFACT_DIR, "chain_and_mountaincar_benchmark")
    plt.savefig(f"{out_base}.png", dpi=300)
    plt.savefig(f"{out_base}.pdf")
    plt.close()

    # Also save to workspace
    ws_base = "/Users/sumitvashishtha/Desktop/DP-BNNs/chain_and_mountaincar_benchmark"
    fig.savefig(f"{ws_base}.png", dpi=300)
    fig.savefig(f"{ws_base}.pdf")
    print(f"\n--> Successfully saved publication figures to {out_base}.pdf and {ws_base}.pdf")

    # Save numeric JSON summary
    summary_data = {
        "seeds": SEEDS,
        "chain_results": {
            k: {
                "solved_episodes": v["solved_ep"],
                "final_regret_mean": float(np.mean([np.sum(np.maximum(0, 100.0 - np.array(r))) for r in v["returns"]])),
                "runtime_sec": v["time"],
            }
            for k, v in chain_res.items()
        },
        "mountaincar_results": {
            k: {
                "solved_episodes": v["solved_ep"],
                "final_10_return_mean": float(np.mean([np.mean(np.array(r)[-10:]) for r in v["returns"]])),
                "runtime_sec": v["time"],
            }
            for k, v in mc_res.items()
        },
    }
    with open(f"{out_base}.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    with open(f"{ws_base}.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"--> Saved numeric summary to {out_base}.json")


if __name__ == "__main__":
    t_start = time.time()
    chain_results = run_chain_experiment()
    mc_results = run_mountaincar_experiment()
    plot_and_export_both(chain_results, mc_results)
    print(f"\nALL BENCHMARKS COMPLETED IN {time.time() - t_start:.1f}s TOTAL!")
