"""Train Live DP-DQN on DeepSea-50 and Record Exact Real Neural Network Q-Values.

Exact paper configuration:
- Size: 50 (2,500 states, horizon H = 50)
- Architecture: 2-layer MLP-64 + LayerNorm
- Base measure: Non-DAG MaxEnt uniform (zero goal bias)
- Warmstart steps: W = 25
- Sampler: Vashishtha & Maillard fixed budget (vm_prior_multiplier = 40.0)
- Sample once per episode: True (Pure Thompson Sampling)
- One living network: True
- Episodic SGD: True (updates performed at end_episode)
- Seed: 42 (known discovery at Ep 1240, solved at Ep 1517)

Saves real snapshot tensors to real_deepsea50_q_log.npz.
"""

import os
import sys
import time
import numpy as np
import torch

from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import get_base_measure

def run():
    print("=" * 70)
    print(" Starting Exact Real-Data Training: DP-DQN on DeepSea-50 (MLP-64)")
    print(" Using Exact Paper Hyperparameters (Seed 42)")
    print(" Recording True Neural Network Q-Values & Trajectories")
    print("=" * 70)
    
    torch.set_num_threads(4)
    size = 50
    state_dim = size * size
    action_dim = 2
    ws = 25
    seed = 42
    max_episodes = 1550 # Solves at 1517
    
    # Pre-allocate identity matrix for instant forward passes on all 2,500 states
    eye = torch.eye(state_dim, dtype=torch.float32)
    
    env = make_env("deep_sea", size=size, seed=seed)
    base_measure = get_base_measure("deep_sea_nondag_maxent", state_dim, action_dim, deep_sea_size=size)
    
    cfg = DPDQNConfig(
        env_name="deep_sea",
        deep_sea_size=size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=seed,
        alpha=5.0,
        batch_size=64,
        candidate_batch_size=256,
        base_measure="deep_sea_nondag_maxent",
        prior_reward_mean=float(base_measure.prior_reward_mean),
        prior_reward_std=float(base_measure.prior_reward_std),
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.05,
        sgd_period=2,
        episodic_sgd=True,
        buffer_capacity=1000000,
        target_warmstart=True,
        warmstart_steps=ws,
        warmstart_lr_scale=1.0,
        sample_once_per_episode=True,
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=float(4.0 * size / 5.0),
        one_living_network=True,
        dp_sampled_target=False,
        use_td_info_gain_decay=True,
        td_info_scale=1.0,
        verbose=False,
    )
    agent = DPDQNAgent(cfg, base_measure=base_measure)
    
    snapshots = []
    cum_regret = 0.0
    first_discovery_ep = None
    solved_ep = None
    returns_history = []
    
    optimal_return = 1.0 - 0.01 * (size - 1)
    t0 = time.time()
    
    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        s = env.reset()
        done = False
        ep_ret = 0.0
        trajectory = []
        actions = []
        steps = 0
        
        while not done:
            idx = int(np.argmax(s))
            r = idx // size
            c = idx % size
            trajectory.append((r, c))
            
            a = agent.act(s)
            actions.append(a)
            sn, r_rew, done, _ = env.step(a)
            agent.step(s, a, float(r_rew), sn, done)
            ep_ret += float(r_rew)
            s = sn
            steps += 1
            
        agent.end_episode(steps)
        
        # Final terminal state
        idx = int(np.argmax(s))
        trajectory.append((idx // size, idx % size))
        
        regret = optimal_return - ep_ret
        cum_regret += regret
        returns_history.append(ep_ret)
        
        if ep_ret > 0.5 and first_discovery_ep is None:
            first_discovery_ep = ep
            print(f"\n★ [DISCOVERY!] Real Treasure Reached at Episode {ep}! Return: {ep_ret:.3f} ★\n", flush=True)
            
        if solved_ep is None and len(returns_history) >= 20 and np.mean(returns_history[-20:]) > 0.8:
            solved_ep = ep
            print(f"\n✔ [CONVERGED!] Optimal Policy Consolidated at Episode {ep}! ✔\n", flush=True)
            
        # Snapshot selection:
        # Every 25 episodes during struggle, every episode during discovery window, every 10 post-solve
        record = False
        if ep == 1 or ep % 25 == 0:
            record = True
        elif 1230 <= ep <= 1255:
            record = True
        elif ep > 1255 and ep % 10 == 0:
            record = True
        elif ep == first_discovery_ep or ep == solved_ep:
            record = True
            
        if record:
            with torch.no_grad():
                q_online = agent.q_net(eye).cpu().numpy().reshape(size, size, 2)
            delta_q = (q_online[:, :, 1] - q_online[:, :, 0]).astype(np.float32)
            v_val = np.maximum(q_online[:, :, 0], q_online[:, :, 1]).astype(np.float32)
            
            snapshots.append({
                "episode": ep,
                "trajectory": np.array(trajectory, dtype=np.int16),
                "actions": np.array(actions, dtype=np.int8),
                "delta_q": delta_q,
                "v_val": v_val,
                "ep_return": float(ep_ret),
                "cum_regret": float(cum_regret),
                "is_solved": bool(ep_ret > 0.5)
            })
            
        if ep % 50 == 0:
            elapsed = time.time() - t0
            eps_per_sec = ep / elapsed
            eta_sec = (max_episodes - ep) / eps_per_sec
            recent_mean = np.mean(returns_history[-20:]) if returns_history else 0.0
            print(f"Ep {ep:4d} / {max_episodes} | Speed: {eps_per_sec:.2f} ep/s | ETA: {eta_sec/60:.1f} min | Return: {ep_ret:.3f} (Mean: {recent_mean:.2f}) | Regret: {cum_regret:.1f}", flush=True)
            
        if solved_ep and ep >= solved_ep + 25:
            print(f"--> Reached termination criterion at Episode {ep}!", flush=True)
            break
            
    t1 = time.time()
    total_time = t1 - t0
    print(f"\n========================================================")
    print(f" Real Training Complete in {total_time/60:.2f} minutes ({total_time:.1f}s)!")
    print(f" First Discovery: Episode {first_discovery_ep}")
    print(f" Solved Episode:  Episode {solved_ep}")
    print(f" Snapshots Logged: {len(snapshots)}")
    print(f"========================================================")
    
    out_file = "real_deepsea50_q_log.npz"
    np.savez_compressed(
        out_file,
        size=size,
        first_discovery_ep=first_discovery_ep or -1,
        solved_ep=solved_ep or -1,
        total_time=total_time,
        snapshots=snapshots
    )
    print(f"--> Saved exact real Q-value logs to {out_file} ({os.path.getsize(out_file)/1024/1024:.2f} MB)")
    return out_file

if __name__ == "__main__":
    run()
