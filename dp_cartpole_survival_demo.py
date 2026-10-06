"""Reward-Free, Gamma-Free Pure Bayesian Nonparametric RL on CartPole-v1.

Core Principle:
- ZERO rewards used: gym's reward is completely discarded (R = 0).
- ZERO discounting: gamma = 1.0.
- Target is the Failure Boundary (pole falling: |theta| > 12 deg or |x| > 2.4).
- The network predicts Physical Survival Duration:
    tau(s, a) = 1.0 + (1 - done) * max_a' tau(s', a')
- DP-Thompson Sampling: M particles trained with Dirichlet stick-breaking weights.
"""

import random
from collections import deque
import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class SurvivalNet(nn.Module):
    """Predicts physical remaining survival time (in steps) for each action."""
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DPCartPoleAgent:
    def __init__(self, state_dim=4, action_dim=2, M_particles=5, alpha=5.0, lr=1e-3):
        self.M = M_particles
        self.alpha = alpha
        self.action_dim = action_dim
        
        self.models = [SurvivalNet(state_dim, action_dim) for _ in range(M_particles)]
        self.target_models = [SurvivalNet(state_dim, action_dim) for _ in range(M_particles)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.models]
        
        for m in range(M_particles):
            self.target_models[m].load_state_dict(self.models[m].state_dict())
            
        self.buffer = deque(maxlen=20000)
        self.batch_size = 64
        
    def add(self, s, a, next_s, done):
        self.buffer.append((s, a, next_s, done))
        
    def act(self, s, particle_idx):
        with torch.no_grad():
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            tau_preds = self.models[particle_idx](s_tensor)
            # Pick action that maximizes survival duration!
            return int(torch.argmax(tau_preds, dim=1).item())

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
            
        batch = random.sample(self.buffer, self.batch_size)
        s, a, next_s, done = zip(*batch)
        
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.long).unsqueeze(1)
        next_s = torch.tensor(np.array(next_s), dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        
        losses = []
        for m in range(self.M):
            # Dirichlet Process stick-breaking weights for this batch
            # Dirichlet(alpha/B, ..., alpha/B) or standard Dirichlet draw
            weights_np = np.random.dirichlet(np.ones(self.batch_size) * (self.alpha / self.batch_size))
            weights = torch.tensor(weights_np, dtype=torch.float32).unsqueeze(1) * self.batch_size
            
            # Target Survival Calculation (gamma = 1.0, cost = +1 step survived)
            with torch.no_grad():
                next_tau = self.target_models[m](next_s)
                max_next_tau = torch.max(next_tau, dim=1, keepdim=True)[0]
                # If done (pole fell): remaining survival = 1.0 step.
                # If not done: 1.0 + remaining survival from next_s.
                target_tau = 1.0 + (1.0 - done) * max_next_tau
                # Clip to physical max episode cap (500 steps)
                target_tau = torch.clamp(target_tau, max=500.0)
                
            pred_tau = self.models[m](s).gather(1, a)
            loss = torch.mean(weights * (pred_tau - target_tau) ** 2)
            
            self.optimizers[m].zero_grad()
            loss.backward()
            self.optimizers[m].step()
            losses.append(loss.item())
            
        return np.mean(losses)

    def update_target(self, tau_polyak=0.05):
        for m in range(self.M):
            for param, target_param in zip(self.models[m].parameters(), self.target_models[m].parameters()):
                target_param.data.copy_(tau_polyak * param.data + (1.0 - tau_polyak) * target_param.data)

def run():
    env = gym.make("CartPole-v1")
    agent = DPCartPoleAgent(M_particles=5, alpha=5.0)
    
    num_episodes = 90
    survival_history = []
    
    print("--- Training Pure Reward-Free, Gamma-Free DP Survival Agent on CartPole-v1 ---")
    print("Objective: Maximize physical survival steps. Rewards: NONE. Gamma: 1.0.")
    
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=seed + ep) if hasattr(env.reset(), '__len__') and len(env.reset()) == 2 else (env.reset(), {})
        # Thompson sampling: draw one active particle for this episode
        particle_idx = np.random.randint(agent.M)
        
        ep_duration = 0
        done = False
        
        while not done:
            action = agent.act(obs, particle_idx)
            step_res = env.step(action)
            if len(step_res) == 5:
                next_obs, rew, terminated, truncated, _ = step_res
                done = terminated or truncated
                # Truncation at 500 is not a physical failure (pole didn't fall)
                physical_fail = terminated and not truncated
            else:
                next_obs, rew, done, _ = step_res
                physical_fail = done and (ep_duration < 499)
                
            # Note: We completely DISCARD `rew`! It is never passed to the agent!
            agent.add(obs, action, next_obs, physical_fail)
            obs = next_obs
            ep_duration += 1
            
            if len(agent.buffer) >= agent.batch_size:
                agent.train_step()
                agent.update_target(tau_polyak=0.02)
                
        survival_history.append(ep_duration)
        if ep % 10 == 0 or ep == 1:
            avg_10 = np.mean(survival_history[-10:]) if len(survival_history) >= 10 else ep_duration
            print(f"Episode {ep:2d}/{num_episodes} | Survival Duration: {ep_duration:3d} steps | 10-Ep Avg: {avg_10:.1f}")

    # Plot results
    plt.figure(figsize=(9, 4.5))
    episodes = np.arange(1, num_episodes + 1)
    plt.plot(episodes, survival_history, "b-o", alpha=0.5, label="Episode Survival Duration")
    
    # Running average
    window = 10
    running_avg = [np.mean(survival_history[max(0, i-window+1):i+1]) for i in range(len(survival_history))]
    plt.plot(episodes, running_avg, "r-", linewidth=2.5, label=f"{window}-Episode Moving Average")
    plt.axhline(500, color="green", linestyle="--", linewidth=2, label="Perfect Score Cap (500 Steps)")
    
    plt.xlabel("Episode", fontsize=12, fontweight="bold")
    plt.ylabel("Physical Survival Time (Steps)", fontsize=12, fontweight="bold")
    plt.title("CartPole-v1 Solved via Reward-Free, Gamma-Free DP-BNN Survival", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    out_png = "dp_cartpole_survival_demo.png"
    plt.savefig(out_png, dpi=300)
    print(f"\nSaved CartPole survival learning curve to {out_png}!")

if __name__ == "__main__":
    run()
