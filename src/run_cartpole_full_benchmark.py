"""Unified Cart-Pole Swing-Up Benchmark Runner across 6 Algorithmic Families.

Algorithms:
1. `dp_dqn_haar`: DP-DQN with Haar Base Measure (alpha=3.0)
2. `dp_dqn_non_haar`: DP-DQN with Non-Haar Box Uniform Base Measure (alpha=3.0)
3. `boot_dqn`: Bootstrapped DQN with Additive Randomized Prior (Osband et al., NeurIPS 2018)
4. `bdqn`: Bayesian Deep Q-Networks with Exact Posterior Linear Regression (Azizzadenesheli et al., 2018)
5. `dp_dqn_alpha_small`: DP-DQN with alpha = 1e-10 (Collapsed Base Measure / Zero Epistemic Variance)
6. `vanilla_dqn`: Standard DQN with Linear Epsilon Annealing (Osband et al., Figure 16)
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch

# Ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.envs.cartpole_jmlr import CartpoleSwingupJMLR
from src.rl.dp_dqn import DPDQNAgent
from src.rl.baselines_rl import BootDQNRPAgent, StandardDQNAgent
from src.dp_dqn.bdqn import BayesianDeepQNetworkAgent

R_STAR = 850.0  # Empirical optimal ceiling for 1000-step swing-up episode


class BDQNAdapter:
    """Adapter to unify BDQN interface with DPDQNAgent and BootDQNRPAgent."""
    def __init__(self, agent):
        self.agent = agent

    def start_episode(self):
        self.agent.reset_episode()

    def select_action(self, state: np.ndarray) -> int:
        return self.agent.act(state, eval_mode=False)

    def step_update(self, state, action, reward, next_state, done):
        self.agent.step(state, action, reward, next_state, done)

    def end_episode(self):
        pass


def make_agent(algo: str, seed: int):
    state_dim = 6
    action_dim = 3

    if algo == "dp_dqn_haar":
        return DPDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=3.0,
            truncation_level=10,
            base_measure_type="uniform",
            haar_angle=True,
            gamma=0.99,
            tau=0.05,
            lr=1e-3,
            sgd_period=4,
            batch_size=128,
            capacity=100000,
            seed=seed
        )
    elif algo == "dp_dqn_non_haar":
        return DPDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=3.0,
            truncation_level=10,
            base_measure_type="uniform",
            haar_angle=False,
            gamma=0.99,
            tau=0.05,
            lr=1e-3,
            sgd_period=4,
            batch_size=128,
            capacity=100000,
            seed=seed
        )
    elif algo == "boot_dqn":
        return BootDQNRPAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_models=20,
            hidden_dim=50,
            prior_scale=1.0,
            gamma=0.99,
            tau=0.05,
            lr=1e-3,
            sgd_period=4,
            batch_size=128,
            capacity=100000,
            seed=seed
        )
    elif algo == "bdqn":
        raw_agent = BayesianDeepQNetworkAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=50,
            prior_variance=1.0,
            noise_variance=0.1,
            gamma=0.99,
            tau=0.05,
            lr=1e-3,
            sgd_period=4,
            batch_size=128,
            capacity=100000,
            seed=seed
        )
        return BDQNAdapter(raw_agent)
    elif algo == "dp_dqn_alpha_small":
        return DPDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=1e-10,
            truncation_level=10,
            base_measure_type="uniform",
            haar_angle=True,
            gamma=0.99,
            tau=0.05,
            lr=1e-3,
            sgd_period=4,
            batch_size=128,
            capacity=100000,
            seed=seed
        )
    elif algo == "vanilla_dqn":
        return StandardDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=50,
            epsilon_start=1.0,
            epsilon_end=0.0,
            anneal_episodes=500,
            gamma=0.99,
            tau=0.05,
            lr=1e-3,
            sgd_period=4,
            batch_size=128,
            capacity=100000,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def run_benchmark(algo: str, seed: int, episodes: int = 2500, out_dir: str = "./results_rl/cartpole_60runs"):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{algo}_seed_{seed}.json")
    ckpt_file = os.path.join(out_dir, f"{algo}_seed_{seed}_ckpt.json")

    env = CartpoleSwingupJMLR(seed=seed)
    agent = make_agent(algo, seed)

    returns = []
    upright_steps = []
    regrets = []
    cumulative_regrets = []
    cum_regret = 0.0

    t0 = time.time()
    for ep in range(1, episodes + 1):
        state = env.reset()
        agent.start_episode()

        ep_ret = 0.0
        ep_upr = 0

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.step_update(state, action, reward, next_state, done)

            ep_ret += reward
            if next_state[0] > 0.95:  # cos(theta) > 0.95
                ep_upr += 1
            state = next_state

        agent.end_episode()
        inst_regret = R_STAR - ep_ret
        cum_regret += inst_regret

        returns.append(float(ep_ret))
        upright_steps.append(int(ep_upr))
        regrets.append(float(inst_regret))
        cumulative_regrets.append(float(cum_regret))

        if ep % 50 == 0 or ep == episodes:
            elapsed = time.time() - t0
            ret_50 = np.mean(returns[-50:])
            upr_50 = np.mean(upright_steps[-50:])
            print(f"[{algo} | Seed {seed}] Ep {ep:4d}/{episodes} | "
                  f"Ret(50): {ret_50:6.2f} | Upr(50): {upr_50:5.1f} | "
                  f"CumRegret: {cum_regret/1e3:6.1f}k | Time: {elapsed:5.1f}s")
            
            # Save checkpoint
            payload = {
                "algo": algo,
                "seed": seed,
                "episodes": ep,
                "returns": returns,
                "upright_steps": upright_steps,
                "regrets": regrets,
                "cumulative_regrets": cumulative_regrets,
                "final_cum_regret": cum_regret,
                "elapsed_sec": elapsed
            }
            with open(ckpt_file, "w") as f:
                json.dump(payload, f)

    # Save final results
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{algo} | Seed {seed}] Completed in {time.time() - t0:.1f}s -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cart-Pole Benchmark Runner")
    parser.add_argument("--algo", type=str, required=True,
                        choices=["dp_dqn_haar", "dp_dqn_non_haar", "boot_dqn", "bdqn", "dp_dqn_alpha_small", "vanilla_dqn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2500)
    parser.add_argument("--out_dir", type=str, default="./results_rl/cartpole_60runs")
    args = parser.parse_args()

    run_benchmark(args.algo, args.seed, args.episodes, args.out_dir)
