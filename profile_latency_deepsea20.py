"""Micro-benchmark: Empirical per-step latency on DeepSea-20 across algorithms.

Measures:
1. End-to-end latency per environment step (wall-clock time / total steps, including warmstart/training).
2. Pure action inference latency per step (time spent in agent.act()).
"""

import time
import numpy as np
import torch
torch.set_num_threads(1)

from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import get_base_measure

from unified_dp_dqn.baselines.boot_dqn import BootDQNRPAgent
from unified_dp_dqn.baselines.bdqn import BayesianDeepQNetworkAgent
from unified_dp_dqn.baselines.standard_dqn import StandardDQNAgent


def benchmark_dp_dqn(warmstart_steps=10, num_episodes=300):
    size = 20
    state_dim = size * size
    action_dim = 2
    env = make_env("deep_sea", size=size, seed=42)
    bm = get_base_measure("deep_sea_nondag_maxent", state_dim, action_dim, deep_sea_size=size)

    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=42,
        alpha=5.0,
        batch_size=64,
        candidate_batch_size=256,
        base_measure="deep_sea_nondag_maxent",
        prior_reward_mean=float(bm.prior_reward_mean),
        prior_reward_std=float(bm.prior_reward_std),
        hidden_dim=20,
        num_layers=1,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.05,
        sgd_period=2,
        episodic_sgd=True,
        buffer_capacity=1000000,
        target_warmstart=True,
        warmstart_steps=warmstart_steps,
        warmstart_lr_scale=1.0,
        sample_once_per_episode=True,
        num_episodes=num_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=float(4.0 * size / 5.0),
        one_living_network=True,
        dp_sampled_target=False,
        use_td_info_gain_decay=True,
        td_info_scale=1.0,
        verbose=False,
    )
    agent = DPDQNAgent(cfg, base_measure=bm)

    # Warmup
    for ep in range(30):
        agent.reset_episode()
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s)
            sn, r, done, _ = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn

    # Timed evaluation
    total_steps = 0
    act_times = []
    t_start = time.perf_counter()

    for ep in range(num_episodes):
        agent.reset_episode()
        s = env.reset()
        done = False
        while not done:
            t_act0 = time.perf_counter()
            a = agent.act(s)
            act_times.append(time.perf_counter() - t_act0)

            sn, r, done, _ = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn
            total_steps += 1

    total_time = time.perf_counter() - t_start
    e2e_ms = (total_time / total_steps) * 1000.0
    act_ms = np.mean(act_times) * 1000.0
    return e2e_ms, act_ms


def benchmark_boot_dqn(num_episodes=300):
    size = 20
    state_dim = size * size
    action_dim = 2
    env = make_env("deep_sea", size=size, seed=42)

    agent = BootDQNRPAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_models=20,
        hidden_dim=20,
        prior_scale=1.0,
        seed=42,
        capacity=1000000
    )

    # Warmup
    for ep in range(30):
        agent.reset_episode()
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s)
            sn, r, done, _ = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn

    # Timed evaluation
    total_steps = 0
    act_times = []
    t_start = time.perf_counter()

    for ep in range(num_episodes):
        agent.reset_episode()
        s = env.reset()
        done = False
        while not done:
            t_act0 = time.perf_counter()
            a = agent.act(s)
            act_times.append(time.perf_counter() - t_act0)

            sn, r, done, _ = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn
            total_steps += 1

    total_time = time.perf_counter() - t_start
    e2e_ms = (total_time / total_steps) * 1000.0
    act_ms = np.mean(act_times) * 1000.0
    return e2e_ms, act_ms


def benchmark_bdqn(num_episodes=300):
    size = 20
    state_dim = size * size
    action_dim = 2
    env = make_env("deep_sea", size=size, seed=42)

    agent = BayesianDeepQNetworkAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=20,
        prior_variance=1.0,
        noise_variance=0.1,
        seed=42,
        capacity=1000000
    )

    # Warmup
    for ep in range(30):
        agent.reset_episode()
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s)
            sn, r, done, _ = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn

    # Timed evaluation
    total_steps = 0
    act_times = []
    t_start = time.perf_counter()

    for ep in range(num_episodes):
        agent.reset_episode()
        s = env.reset()
        done = False
        while not done:
            t_act0 = time.perf_counter()
            a = agent.act(s)
            act_times.append(time.perf_counter() - t_act0)

            sn, r, done, _ = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn
            total_steps += 1

    total_time = time.perf_counter() - t_start
    e2e_ms = (total_time / total_steps) * 1000.0
    act_ms = np.mean(act_times) * 1000.0
    return e2e_ms, act_ms


def benchmark_vanilla_dqn(num_episodes=300):
    size = 20
    state_dim = size * size
    action_dim = 2
    env = make_env("deep_sea", size=size, seed=42)

    agent = StandardDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=20,
        seed=42,
        capacity=1000000
    )

    # Warmup
    for ep in range(30):
        agent.reset_episode()
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s)
            sn, r, done, _ = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn

    # Timed evaluation
    total_steps = 0
    act_times = []
    t_start = time.perf_counter()

    for ep in range(num_episodes):
        agent.reset_episode()
        s = env.reset()
        done = False
        while not done:
            t_act0 = time.perf_counter()
            a = agent.act(s)
            act_times.append(time.perf_counter() - t_act0)

            sn, r, done, _ = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn
            total_steps += 1

    total_time = time.perf_counter() - t_start
    e2e_ms = (total_time / total_steps) * 1000.0
    act_ms = np.mean(act_times) * 1000.0
    return e2e_ms, act_ms


if __name__ == "__main__":
    print("=" * 65)
    print("  EMPIRICAL PER-STEP LATENCY BENCHMARK ON DEEPSEA-20 (MLP-20)")
    print("=" * 65)

    print("\nRunning DP-DQN (Single MLP-20, W=10, Pure TS)...")
    dp_10_e2e, dp_10_act = benchmark_dp_dqn(warmstart_steps=10)
    print(f"  -> End-to-End: {dp_10_e2e:.3f} ms/step | Action Inference: {dp_10_act*1000:.1f} us/step")

    print("\nRunning DP-DQN (Single MLP-20, W=4, Pure TS)...")
    dp_4_e2e, dp_4_act = benchmark_dp_dqn(warmstart_steps=4)
    print(f"  -> End-to-End: {dp_4_e2e:.3f} ms/step | Action Inference: {dp_4_act*1000:.1f} us/step")

    print("\nRunning Vanilla DQN (Single MLP-20)...")
    dqn_e2e, dqn_act = benchmark_vanilla_dqn()
    print(f"  -> End-to-End: {dqn_e2e:.3f} ms/step | Action Inference: {dqn_act*1000:.1f} us/step")

    print("\nRunning BDQN (Azizzadenesheli et al. 2018)...")
    bdqn_e2e, bdqn_act = benchmark_bdqn()
    print(f"  -> End-to-End: {bdqn_e2e:.3f} ms/step | Action Inference: {bdqn_act*1000:.1f} us/step")

    print("\nRunning BootDQN-RP (20 Heads + 20 Prior Nets = 40 Nets)...")
    boot_e2e, boot_act = benchmark_boot_dqn()
    print(f"  -> End-to-End: {boot_e2e:.3f} ms/step | Action Inference: {boot_act*1000:.1f} us/step")

    print("\n" + "=" * 65)
    print("  FINAL LATENCY COMPARISON TABLE (DEEPSEA-20)")
    print("=" * 65)
    print(f"{'Algorithm':<36} | {'Total E2E (ms/step)':<20} | {'Act Inference (us)':<18}")
    print("-" * 78)
    print(f"{'DP-DQN (Single MLP-20, W=4, Pure TS)':<36} | {dp_4_e2e:<20.3f} | {dp_4_act*1000:<18.1f}")
    print(f"{'DP-DQN (Single MLP-20, W=10, Pure TS)':<36} | {dp_10_e2e:<20.3f} | {dp_10_act*1000:<18.1f}")
    print(f"{'Vanilla DQN (Single MLP-20)':<36} | {dqn_e2e:<20.3f} | {dqn_act*1000:<18.1f}")
    print(f"{'BDQN (Azizzadenesheli et al. 2018)':<36} | {bdqn_e2e:<20.3f} | {bdqn_act*1000:<18.1f}")
    print(f"{'BootDQN-RP (20 heads, 40 nets)':<36} | {boot_e2e:<20.3f} | {boot_act*1000:<18.1f}")
    print("=" * 65)
