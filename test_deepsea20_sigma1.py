import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from test_deepsea20_target_variants_fresh import HardSnapshotTargetAgent, DPWarmstartTargetAgent, run_variant

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING TARGET VARIANTS WITH STANDARD NORMAL REWARD PRIOR (SIGMA = 1.0, MEAN = 0.0)")
    print("=" * 80)

    # Modify the config inside run_variant via a wrapper or direct call
    def run_sigma1(agent_cls, name, size=20, seed=42, max_episodes=800, **kwargs):
        state_dim = size * size
        cfg = DPDQNConfig(
            env_name="deep_sea",
            deep_sea_size=size,
            state_dim=state_dim,
            action_dim=2,
            seed=seed,
            alpha=5.0,
            batch_size=64,
            candidate_batch_size=128,
            base_measure="deep_sea_dag_maxent",
            prior_reward_mean=0.0,       # PURE ZERO MEAN
            prior_reward_std=1.0,        # STANDARD UNIT VARIANCE SIGMA = 1.0!
            hidden_dim=64,
            num_layers=2,
            use_layer_norm=True,
            activation="relu",
            lr=1e-3,
            gamma=0.99,
            tau=0.0,
            target_warmstart=False,
            warmstart_steps=size // 2,   # W = 10 steps
            sample_once_per_episode=False, # FRESH PER STEP
            num_episodes=max_episodes,
            max_episode_steps=size,
            sampler_type="vashishtha_maillard",
            vm_prior_multiplier=10.0,
            one_living_network=True,
            episodic_sgd=False,
        )

        env = make_env("deep_sea", seed=seed, size=size)
        agent = agent_cls(cfg, **kwargs)
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
                print(f">>> [DISCOVERY!] {name} reached goal at Episode {ep}! ({time.time()-t0:.2f}s)", flush=True)

            if ep >= 20:
                recent_ret = np.mean(returns[-20:])
                if recent_ret >= 0.8:
                    solved_ep = ep
                    print(f"*** [SOLVED!] {name} SOLVED at Episode {ep}! (Avg return: {recent_ret:.3f}, Time: {time.time()-t0:.2f}s) ***", flush=True)
                    break

            if ep % 50 == 0:
                print(f"Episode {ep:3d} | Recent Return: {np.mean(returns[-20:]):.4f} | Time: {time.time()-t0:.1f}s", flush=True)

        if solved_ep is None:
            print(f"--> Result: Did not solve in {max_episodes} eps. First discovery: {first_discovery}", flush=True)

        return first_discovery, solved_ep, time.time() - t0

    print("\n--- 1. VARIANT D: Hard Snapshot (Period K=5, Sigma=1.0) ---")
    run_sigma1(HardSnapshotTargetAgent, "Hard Snapshot (K=5, sigma=1.0)", size=20, seed=42, period_k=5)

    print("\n--- 2. VARIANT B: DP-Warmstarted Target (2 steps, Sigma=1.0) ---")
    run_sigma1(DPWarmstartTargetAgent, "DP-Warmstarted Target (2 steps, sigma=1.0)", size=20, seed=42, target_steps=2)
