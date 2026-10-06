import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from test_td_info_deepsea20 import HardSnapshotTDInfoAgent

def run_deepsea50_long(seed=42, warmstart_steps=12, prior_std=1.0, max_episodes=8000):
    size = 50
    print("=" * 80)
    print(f"RUNNING LONG DEEPSEA N={size} (ZERO-MEAN REWARDS, NO STOCHASTIC OPTIMISM)")
    print(f"Settings: Seed={seed}, Hard Snapshot (K=5), TD Info Gain Decay, W={warmstart_steps}")
    print(f"Reward Prior: mean=0.0, std={prior_std}, goal_bonus=False, Max Episodes={max_episodes}")
    print("=" * 80)

    state_dim = size * size
    alpha = 5.0
    vm_mult = float(4.0 * size / alpha)  # 40.0

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
        prior_reward_mean=0.0,          # STRICT ZERO MEAN!
        prior_reward_std=prior_std,     # Unit normal variance (std=1.0)
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.0,
        target_warmstart=False,
        warmstart_steps=warmstart_steps, # W = 12 (fast & effective)
        sample_once_per_episode=False,  # Fresh per step!
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
    agent.base_measure.goal_bonus = False

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
            print(f"\n>>> [DISCOVERY!] N={size} reached goal at Episode {ep}! ({time.time()-t0:.1f}s) <<<", flush=True)

        if ep >= 30:
            recent_ret = np.mean(returns[-30:])
            if recent_ret >= 0.85:
                solved_ep = ep
                print(f"\n*** [SOLVED!] N={size} SOLVED at Episode {ep}! (Avg return: {recent_ret:.3f}, Time: {time.time()-t0:.1f}s) ***", flush=True)
                break

        if ep % 100 == 0:
            print(f"Episode {ep:4d} | Recent Return: {np.mean(returns[-20:]):.4f} | Info: {agent.cumulative_info:.1f} | Time: {time.time()-t0:.1f}s", flush=True)

    if solved_ep is None:
        print(f"\n--> Result: Did not solve in {max_episodes} eps. First discovery: {first_discovery}")

    return first_discovery, solved_ep, time.time() - t0

if __name__ == "__main__":
    run_deepsea50_long(seed=42, warmstart_steps=12, prior_std=1.0, max_episodes=8000)
