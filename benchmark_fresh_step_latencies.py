import time
import numpy as np
import torch
import sys

from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from unified_dp_dqn.baselines.standard_dqn import StandardDQNAgent
from unified_dp_dqn.baselines.boot_dqn import BootDQNRPAgent

size = 20
num_episodes = 250
seeds = [42, 43, 44]

latencies = {"DQN": [], "BootDQN": [], "DP-DQN": []}

print(f"Starting Fresh Latency Benchmark on DeepSea-20 ({num_episodes} episodes x {len(seeds)} seeds)...", flush=True)

for s_idx in seeds:
    print(f"\n--- Running Seed {s_idx} ---", flush=True)

    # 1. Standard DQN
    env = make_env("deep_sea", size=size, seed=s_idx)
    dqn = StandardDQNAgent(state_dim=size*size, action_dim=2, hidden_dim=64, seed=s_idx)
    s = env.reset()
    for _ in range(128):
        a = dqn.act(s)
        sn, r, done, _ = env.step(a)
        dqn.step(s, a, r, sn, done)
        s = env.reset() if done else sn

    t0 = time.perf_counter()
    steps = 0
    for ep in range(num_episodes):
        dqn.reset_episode()
        s = env.reset()
        done = False
        while not done:
            a = dqn.act(s)
            sn, r, done, _ = env.step(a)
            dqn.step(s, a, r, sn, done)
            s = sn
            steps += 1
    t_dqn = time.perf_counter() - t0
    ms_dqn = (t_dqn / steps) * 1000.0
    latencies["DQN"].append(ms_dqn)
    print(f"  DQN (Seed {s_idx}): {ms_dqn:.3f} ms/step ({t_dqn:.2f}s for {steps} steps)", flush=True)

    # 2. BootDQN + Rand Priors (20 heads)
    env = make_env("deep_sea", size=size, seed=s_idx)
    boot = BootDQNRPAgent(state_dim=size*size, action_dim=2, num_models=20, hidden_dim=64, seed=s_idx)
    s = env.reset()
    for _ in range(128):
        a = boot.act(s)
        sn, r, done, _ = env.step(a)
        boot.step(s, a, r, sn, done)
        s = env.reset() if done else sn

    t0 = time.perf_counter()
    steps = 0
    for ep in range(num_episodes):
        boot.reset_episode()
        s = env.reset()
        done = False
        while not done:
            a = boot.act(s)
            sn, r, done, _ = env.step(a)
            boot.step(s, a, r, sn, done)
            s = sn
            steps += 1
    t_boot = time.perf_counter() - t0
    ms_boot = (t_boot / steps) * 1000.0
    latencies["BootDQN"].append(ms_boot)
    print(f"  BootDQN-RP (Seed {s_idx}): {ms_boot:.3f} ms/step ({t_boot:.2f}s for {steps} steps)", flush=True)

    # 3. DP-DQN (Pure TS Warmstart, W=4, 1 living net)
    env = make_env("deep_sea", size=size, seed=s_idx)
    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=size*size,
        action_dim=2,
        seed=s_idx,
        alpha=3.0,
        batch_size=64,
        candidate_batch_size=96,
        base_measure="deep_sea_dag_maxent",
        prior_reward_mean=0.5,
        prior_reward_std=0.288,
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        lr=1e-3,
        tau=0.05,
        target_warmstart=True,
        warmstart_steps=4,
        warmstart_lr_scale=1.0,
        one_living_network=True,
        episodic_sgd=False,
    )
    dp_agent = DPDQNAgent(cfg)
    s = env.reset()
    for _ in range(128):
        a = dp_agent.act(s)
        sn, r, done, _ = env.step(a)
        dp_agent.replay.push(s, a, float(r), sn, done)
        s = env.reset() if done else sn

    t0 = time.perf_counter()
    steps = 0
    for ep in range(num_episodes):
        dp_agent.reset_episode()
        s = env.reset()
        done = False
        while not done:
            a = dp_agent.act(s)
            sn, r, done, _ = env.step(a)
            dp_agent.replay.push(s, a, float(r), sn, done)
            s = sn
            steps += 1
    t_dp = time.perf_counter() - t0
    ms_dp = (t_dp / steps) * 1000.0
    latencies["DP-DQN"].append(ms_dp)
    print(f"  DP-DQN (Seed {s_idx}): {ms_dp:.3f} ms/step ({t_dp:.2f}s for {steps} steps)", flush=True)

print("\n" + "=" * 60, flush=True)
print("=== FRESH LATENCY RESULTS (DeepSea-20, Latest Pure TS) ===", flush=True)
print("=" * 60, flush=True)
for k, v in latencies.items():
    print(f"  {k:12s}: {np.mean(v):.3f} +/- {np.std(v):.3f} ms/step (per-seed: {[round(x, 3) for x in v]})", flush=True)
print("=" * 60, flush=True)
