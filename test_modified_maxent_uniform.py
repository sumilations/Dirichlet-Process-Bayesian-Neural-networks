import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import BaseMeasure
from test_td_info_deepsea20 import HardSnapshotTDInfoAgent

class SupportRespectingUniformBaseMeasure(BaseMeasure):
    """Modified Maximum-Entropy Uniform Base Measure respecting support: r ~ Uniform[-1, 1].

    - Support: [-1, 1] fully contains {-0.01/N, 0.0, 1.0}. Lemma 1 strictly satisfied!
    - Expectation: E[r] = 0.0 (Pure zero-mean, symmetric, no stochastic optimism).
    - Variance: Var[r] = 1/3 ~ 0.333, std = 1/sqrt(3) ~ 0.577.
    - Causal DAG reachability: strictly monotonic depth advance t -> t+1.
    """
    def __init__(self, size: int):
        self.size = size
        state_dim = size * size
        super().__init__(state_dim=state_dim, action_dim=2, prior_reward_mean=0.0, prior_reward_std=float(1.0 / np.sqrt(3)))

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.state_dim), dtype=np.float32)
        sn = np.zeros((count, self.state_dim), dtype=np.float32)
        a = rng.randint(0, 2, size=count)
        done = np.zeros(count, dtype=np.float32)

        # UNIFORM ON [-1, 1]: ZERO-MEAN SYMMETRIC RESPECTING SUPPORT!
        r = rng.uniform(-1.0, 1.0, size=count).astype(np.float32)

        for i in range(count):
            row = rng.randint(0, self.size)
            col = rng.randint(0, row + 1)
            s[i, row * self.size + col] = 1.0

            if row == self.size - 1:
                done[i] = 1.0
            else:
                next_row = row + 1
                step_dir = 1 if rng.rand() < 0.5 else -1
                next_col = max(0, min(self.size - 1, col + step_dir))
                sn[i, next_row * self.size + next_col] = 1.0

        return s, a, r, sn, done


def run_test_uniform_support(size=20, seed=42, max_episodes=800):
    print("=" * 80)
    print(f"RUNNING MODIFIED SUPPORT-RESPECTING MAX-ENT UNIFORM U[-1, 1] ON DEEPSEA N={size}, Seed={seed}")
    print(f"Properties: Support [-1, 1] strictly contains {-0.01/size, 0, 1}, Mean = 0.0 (Zero-Mean!)")
    print("=" * 80)

    state_dim = size * size
    alpha = 5.0
    vm_mult = float(4.0 * size / alpha)

    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=state_dim,
        action_dim=2,
        seed=seed,
        alpha=alpha,
        batch_size=64,
        candidate_batch_size=128,
        base_measure="deep_sea_dag_maxent",
        prior_reward_mean=0.0,
        prior_reward_std=float(1.0 / np.sqrt(3)),
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.0,
        target_warmstart=False,
        warmstart_steps=size // 2,
        sample_once_per_episode=False,
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=vm_mult,
        one_living_network=True,
        episodic_sgd=False,
        use_td_info_gain_decay=True,
        priority_positive_slot=True,
    )

    env = make_env("deep_sea", seed=seed, size=size)
    agent = HardSnapshotTDInfoAgent(cfg, period_k=5)
    # Replace base measure with Support-Respecting Uniform[-1, 1]
    agent.base_measure = SupportRespectingUniformBaseMeasure(size=size)
    agent.sampler.base_measure = agent.base_measure

    returns = []
    t0 = time.time()
    first_discovery = None
    solved_ep = None

    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        s = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            a = agent.act(s)
            sn, r, done, _ = env.step(a)
            agent.replay.push(s, a, float(r), sn, done)
            s = sn
            ep_ret += float(r)

        returns.append(ep_ret)
        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep
            print(f">>> [DISCOVERY!] N={size} reached goal at Episode {ep}! ({time.time()-t0:.2f}s)", flush=True)

        if ep >= 20:
            recent_ret = np.mean(returns[-20:])
            if recent_ret >= 0.8:
                solved_ep = ep
                print(f"*** [SOLVED!] N={size} SOLVED at Episode {ep}! (Avg return: {recent_ret:.3f}, Time: {time.time()-t0:.2f}s) ***", flush=True)
                break

        if ep % 50 == 0:
            print(f"Episode {ep:3d} | Recent Return: {np.mean(returns[-20:]):.4f} | Time: {time.time()-t0:.1f}s", flush=True)

    if solved_ep is None:
        print(f"--> Result: Did not solve in {max_episodes} eps. First discovery: {first_discovery}")

    return first_discovery, solved_ep, time.time() - t0


if __name__ == "__main__":
    print("\n--- 1. DEEPSEA N=10 (MODIFIED MAX-ENT UNIFORM U[-1, 1]) ---")
    run_test_uniform_support(size=10, seed=42)

    print("\n--- 2. DEEPSEA N=20 (MODIFIED MAX-ENT UNIFORM U[-1, 1]) ---")
    run_test_uniform_support(size=20, seed=42)

    print("\n--- 3. DEEPSEA N=30 (MODIFIED MAX-ENT UNIFORM U[-1, 1]) ---")
    run_test_uniform_support(size=30, seed=42)
