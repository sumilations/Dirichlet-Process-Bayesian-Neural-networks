import os
import sys
import time
import numpy as np
import torch

from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env

def run_deepsea50_stochastic(size=50, seed=43, warmstart_steps=25, max_episodes=3500):
    print("=" * 70)
    print(f"RUNNING DEEPSEA N={size} STOCHASTIC MINI-BATCH DP WARMSTART")
    print(f"Settings: Seed={seed}, Warmstart={warmstart_steps} steps (fresh draw/step), 0 consolidation")
    print("=" * 70)

    state_dim = size * size
    action_dim = 2
    alpha = 5.0
    vm_mult = float(4.0 * size / alpha)  # 40.0 -> K_prior = 200 synthetic atoms

    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=seed,
        alpha=alpha,
        batch_size=128,
        candidate_batch_size=256,
        base_measure="deep_sea_dag_maxent",
        prior_reward_mean=0.5,
        prior_reward_std=0.288,
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.05,
        target_warmstart=True,
        warmstart_steps=warmstart_steps,
        warmstart_lr_scale=1.0,
        sample_once_per_episode=True,
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=vm_mult,
        one_living_network=True,
        episodic_sgd=False,  # NO online SGD during rollout
    )

    env = make_env("deep_sea", seed=seed, size=size)
    agent = DPDQNAgent(cfg)

    t0 = time.time()
    returns = []
    cum_regret = 0.0
    optimal_return = 0.99
    first_discovery = None
    solved_episode = None

    for ep in range(1, max_episodes + 1):
        # 1. Episode boundary: Polyak target update + W fresh stochastic mini-batch DP draws
        agent.reset_episode()

        # 2. Episode rollout: Policy is 100% frozen
        s = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            a = agent.act(s)
            sn, r, done, _ = env.step(a)
            agent.replay.push(s, a, float(r), sn, done)
            s = sn
            ep_ret += float(r)

        returns.append(ep_ret)
        regret = max(0.0, optimal_return - ep_ret)
        cum_regret += regret

        if ep_ret > 0.5 and first_discovery is None:
            first_discovery = ep
            print(f"\n>>> [DISCOVERY!] N={size} Seed={seed} reached goal at Episode {ep}! ({time.time()-t0:.1f}s) <<<", flush=True)

        if ep >= 30:
            recent_ret = np.mean(returns[-30:])
            if recent_ret >= 0.85:
                if solved_episode is None:
                    solved_episode = ep
                    print(f"\n*** [SOLVED!] N={size} Seed={seed} SOLVED at Episode {ep}! (Moving avg return: {recent_ret:.3f}, Time: {time.time()-t0:.1f}s) ***", flush=True)
                elif ep >= solved_episode + 100:
                    print(f"\n--> [CONVERGED] Stabilized post-solve at Episode {ep}. Final cumulative regret: {cum_regret:.1f}.", flush=True)
                    break

        if ep % 100 == 0:
            avg_ret = np.mean(returns[-50:]) if ep >= 50 else np.mean(returns)
            elapsed = time.time() - t0
            print(f"Ep {ep:4d}/{max_episodes} | Regret: {cum_regret:7.1f} | Recent Ret: {avg_ret:+.3f} | Elapsed: {elapsed:5.1f}s", flush=True)

    elapsed_total = time.time() - t0
    print("\n" + "=" * 70)
    print(f"SUMMARY FOR DEEPSEA N={size} (Stochastic Mini-Batch Warmstart):")
    print(f"First Discovery: {first_discovery}")
    print(f"Solved Episode:  {solved_episode}")
    print(f"Total Regret:    {cum_regret:.2f}")
    print(f"Total Time:      {elapsed_total:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    run_deepsea50_stochastic(size=50, seed=43, warmstart_steps=25, max_episodes=3500)
