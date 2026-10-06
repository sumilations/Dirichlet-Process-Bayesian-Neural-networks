"""Reward-Free, Gamma-Free Pure Bayesian Nonparametric RL via DP Hitting-Time.

Environment: Challenging 10-State RiverSwim / Deep Chain
Target Set: G = {L} (State 9)
Rewards: ZERO rewards anywhere! R = 0.
Discounting: ZERO discounting! gamma = 1.0.

Core Principle:
Hitting time: tau(s, a) = 1 + sum_{s' not in G} P(s' | s, a) min_a' tau(s', a')
DP Posterior: P^(m)(. | s, a) ~ Dir(alpha * F_0 + N(s, a, .))
Decision Rule: a_t = argmin_a tau^(m)(s_t, a)  (DP-Thompson Sampling on Time)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

class DeepChainEnv:
    """A 10-state directional chain. Swimming right is against a current."""
    def __init__(self, length=10):
        self.L = length
        self.target = length - 1
        self.state = 0
        
    def reset(self):
        self.state = 0
        return self.state
        
    def step(self, action):
        s = self.state
        if s == self.target:
            return s, True
            
        # Action 0: Left (downstream, effortless)
        # Action 1: Right (upstream, against strong current)
        if action == 0:
            next_s = max(0, s - 1)
        else:
            # 40% advance, 55% stay, 5% swept back
            r = np.random.rand()
            if r < 0.40:
                next_s = min(self.L - 1, s + 1)
            elif r < 0.95:
                next_s = s
            else:
                next_s = max(0, s - 1)
                
        self.state = next_s
        done = (next_s == self.target)
        return next_s, done

class DPHittingTimeAgent:
    """Pure Bayesian Nonparametric Agent with NO rewards and NO gamma."""
    def __init__(self, num_states, num_actions, target_state, alpha=2.0, M_particles=10):
        self.S = num_states
        self.A = num_actions
        self.target = target_state
        self.alpha = alpha
        self.M = M_particles
        
        # Transition counts: N(s, a, s')
        self.counts = np.zeros((self.S, self.A, self.S), dtype=np.float32)
        
        # Base measure F_0: uniform optimistic exploration prior over states
        self.F_0 = np.ones((self.S, self.A, self.S), dtype=np.float32) / self.S
        
    def update(self, s, a, next_s):
        self.counts[s, a, next_s] += 1.0

    def sample_hitting_times(self):
        """Draw M posterior transition measures and solve the hitting time Bellman equation."""
        m_particles = np.zeros((self.M, self.S, self.A))
        
        for m in range(self.M):
            # 1. Sample P^(m)(. | s, a) from Dirichlet posterior
            P_sample = np.zeros((self.S, self.A, self.S))
            for s in range(self.S):
                for a in range(self.A):
                    dirichlet_params = self.alpha * self.F_0[s, a] + self.counts[s, a]
                    P_sample[s, a] = np.random.dirichlet(dirichlet_params)
            
            # 2. Exact Value Iteration for Hitting Time (gamma = 1.0, cost = +1)
            # tau(target, .) = 0
            # tau(s, a) = 1 + sum_{s' != target} P(s' | s, a) min_a' tau(s', a')
            tau = np.zeros((self.S, self.A))
            for _ in range(100):
                v = np.min(tau, axis=1)
                v[self.target] = 0.0  # Boundary condition at target
                
                new_tau = np.zeros_like(tau)
                for s in range(self.S):
                    if s == self.target:
                        new_tau[s] = 0.0
                    else:
                        for a in range(self.A):
                            # Transition into non-target states costs expected remaining time
                            new_tau[s, a] = 1.0 + np.dot(P_sample[s, a], v)
                
                if np.max(np.abs(new_tau - tau)) < 1e-4:
                    break
                tau = new_tau
                
            m_particles[m] = tau
            
        return m_particles

def run_simulation():
    L = 10
    num_episodes = 40
    max_steps_per_episode = 1500
    
    env = DeepChainEnv(length=L)
    agent = DPHittingTimeAgent(num_states=L, num_actions=2, target_state=L-1, alpha=1.5, M_particles=5)
    
    steps_history = []
    
    print(f"--- Starting Reward-Free, Gamma-Free DP-RL on {L}-State Chain ---")
    print(f"Goal: Reach State {L-1} from State 0. Rewards: NONE. Gamma: 1.0 (Undiscounted).")
    
    for ep in range(1, num_episodes + 1):
        s = env.reset()
        
        # Thompson sampling: draw one posterior hitting-time realization for this episode
        hitting_time_particles = agent.sample_hitting_times()
        # Randomly choose particle m for posterior sampling
        m_active = hitting_time_particles[np.random.randint(agent.M)]
        
        ep_steps = 0
        done = False
        
        while not done and ep_steps < max_steps_per_episode:
            # Greedy action w.r.t sampled hitting time: pick action with smallest expected arrival time!
            a = int(np.argmin(m_active[s]))
            
            next_s, done = env.step(a)
            agent.update(s, a, next_s)
            
            s = next_s
            ep_steps += 1
            
        steps_history.append(ep_steps)
        if ep % 5 == 0 or ep == 1:
            print(f"Episode {ep:2d}/{num_episodes} | Steps to reach goal: {ep_steps:4d}")

    # Final evaluation: compute mean and uncertainty of hitting times
    final_particles = agent.sample_hitting_times()  # [M, S, A]
    best_tau_per_particle = np.min(final_particles, axis=2)  # [M, S]
    mean_tau = np.mean(best_tau_per_particle, axis=0)
    std_tau = np.std(best_tau_per_particle, axis=0)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Plot 1: Learning Curve (Steps to Arrival)
    axes[0].plot(range(1, num_episodes + 1), steps_history, "o-", color="#1f77b4", linewidth=2.0, markersize=5)
    axes[0].set_xlabel("Episode", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Steps to Reach Target (Physical Time)", fontsize=12, fontweight="bold")
    axes[0].set_title("Reward-Free DP Hitting-Time Learning Curve", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Plot 2: Epistemic Uncertainty on Time-to-Target
    states = np.arange(L)
    axes[1].plot(states, mean_tau, "r-s", linewidth=2.2, label="Mean Expected Hitting Time $\\mathbb{E}[\\tau(s)]$")
    axes[1].fill_between(states, mean_tau - 1.96 * std_tau, mean_tau + 1.96 * std_tau, color="red", alpha=0.2, label="95% Credible Interval (Epistemic Dome)")
    axes[1].axvline(L - 1, color="green", linestyle="--", linewidth=2, label="Target State $\\mathcal{G}$ ($\\tau = 0$)")
    axes[1].set_xlabel("Chain State $s$", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Expected Steps to Goal $\\tau(s)$", fontsize=12, fontweight="bold")
    axes[1].set_title("Learned Physical Hitting Time Profile", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    output_png = "dp_hitting_time_rl_demo.png"
    plt.savefig(output_png, dpi=300)
    print(f"\nSaved reward-free DP-RL demo figure to {output_png}!")

if __name__ == "__main__":
    run_simulation()
