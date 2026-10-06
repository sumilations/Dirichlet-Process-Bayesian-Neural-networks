"""Train Live DP-DQN Agent on DeepSea and Record Exact Neural Network Q-Values.

Logs:
- Real trajectory (r_t, c_t) and action a_t for each episode
- Full 2D Q-network output Q_theta(r, c, a) across all states
- Empirical advantage Delta Q(r, c) = Q(r, c, Right) - Q(r, c, Left)
- Target network values and episodic regret
- Output saved to real_deepsea_q_log_N{size}.npz
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.base_measures import get_base_measure

def train_and_record(size: int = 20, seed: int = 42, max_episodes: int = 500, log_every: int = 1):
    print("=" * 65)
    print(f" Training Live DP-DQN on DeepSea-N={size} (Seed={seed})")
    print(f" Recording EXACT Neural Network Q-Values at Each Episode")
    print("=" * 65)
    
    state_dim = size * size
    action_dim = 2
    warmstart_steps = max(10, size // 2)
    
    # Vectorized state representations for instant forward passes
    eye = torch.eye(state_dim, dtype=torch.float32)
    all_states = eye.clone() # (size*size, state_dim)
    
    env = make_env("deep_sea", size=size, seed=seed)
    cfg = DPDQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        sample_once_per_episode=True,
        warmstart_steps=warmstart_steps,
        base_measure="deep_sea_nondag_maxent",
        deep_sea_size=size,
        seed=seed
    )
    agent = DPDQNAgent(cfg, seed=seed)
    
    records = []
    cum_regret = 0.0
    first_discovery_ep = None
    solved_ep = None
    returns_history = []
    
    t0 = time.time()
    
    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        s = env.reset()
        done = False
        ep_ret = 0.0
        trajectory = []
        actions = []
        
        while not done:
            # Map one-hot state back to (row, col)
            idx = int(np.argmax(s))
            r = idx // size
            c = idx % size
            trajectory.append((r, c))
            
            a = agent.act(s)
            actions.append(a)
            s_next, r_rew, done, _ = env.step(a)
            agent.step(s, a, r_rew, s_next, done)
            ep_ret += r_rew
            s = s_next
            
        # Add final terminal state
        idx = int(np.argmax(s))
        trajectory.append((idx // size, idx % size))
        
        # Optimal return for DeepSea-N is 1.0 - (N * 0.01 / N) = 0.99
        regret = 0.99 - ep_ret
        cum_regret += regret
        returns_history.append(ep_ret)
        
        if ep_ret > 0.5 and first_discovery_ep is None:
            first_discovery_ep = ep
            print(f"--> [BREAKTHROUGH!] First Discovery at Episode {ep} (Return = {ep_ret:.3f})")
            
        if solved_ep is None and len(returns_history) >= 20 and np.mean(returns_history[-20:]) > 0.8:
            solved_ep = ep
            print(f"--> [SOLVED!] Policy Consistently Converged at Episode {ep}!")
            
        # Record exact neural network Q-values every log_every episodes
        # or on key milestone episodes
        record_this = (ep % log_every == 0) or (ep == first_discovery_ep) or (solved_ep and ep >= solved_ep - 5)
        
        if record_this:
            with torch.no_grad():
                q_online = agent.q_net(all_states).cpu().numpy().reshape(size, size, 2)
                q_target = agent.target_net(all_states).cpu().numpy().reshape(size, size, 2)
                
            delta_q = q_online[:, :, 1] - q_online[:, :, 0] # Q(s, Right) - Q(s, Left)
            v_val = np.maximum(q_online[:, :, 0], q_online[:, :, 1])
            
            records.append({
                "episode": ep,
                "trajectory": np.array(trajectory, dtype=np.int16),
                "actions": np.array(actions, dtype=np.int8),
                "delta_q": delta_q.astype(np.float32),
                "v_val": v_val.astype(np.float32),
                "q_online": q_online.astype(np.float32),
                "ep_return": float(ep_ret),
                "cum_regret": float(cum_regret),
                "is_solved": bool(ep_ret > 0.5)
            })
            
        if ep % 50 == 0 or ep == 1:
            mean_ret = np.mean(returns_history[-20:]) if returns_history else 0.0
            print(f"Episode {ep:4d} / {max_episodes} | Return: {ep_ret:.3f} | Recent Mean: {mean_ret:.3f} | Regret: {cum_regret:.1f}")
            
        # Exit early if solved and run for 30 post-solve episodes to demonstrate convergence
        if solved_ep and ep >= solved_ep + 25:
            print(f"--> Training completed successfully at Episode {ep}!")
            break
            
    t1 = time.time()
    print(f"\n--> Training finished in {t1 - t0:.2f} seconds!")
    print(f"--> First Discovery: Episode {first_discovery_ep}")
    print(f"--> Solved Episode:  Episode {solved_ep}")
    print(f"--> Recorded {len(records)} detailed snapshot frames with exact Q-values.")
    
    out_file = f"real_deepsea_q_log_N{size}.npz"
    np.savez_compressed(
        out_file,
        size=size,
        first_discovery_ep=first_discovery_ep or -1,
        solved_ep=solved_ep or -1,
        total_time=t1 - t0,
        records=records
    )
    print(f"--> Saved exact real Q-value logs to {out_file} ({os.path.getsize(out_file) / 1024:.1f} KB)")
    return out_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episodes", type=int, default=450)
    args = parser.parse_args()
    train_and_record(size=args.size, seed=args.seed, max_episodes=args.max_episodes)
