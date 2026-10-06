import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.agent import DPDQNAgent

# ==============================================================================
# 1. VARIANT A: Polyak Target (Baseline, tau=0.05)
# ==============================================================================
class PolyakTargetAgent(DPDQNAgent):
    """Standard Polyak Target Network with fresh sampling at every warmstart step."""
    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        # Polyak update
        with torch.no_grad():
            tau = 0.05
            for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

        # Fresh prior + empirical sample at EACH warmstart step
        for _ in range(self.config.warmstart_steps):
            s, a, r, sn, done, q_weights = self.sampler.sample(self.replay, device=self.device)
            with torch.no_grad():
                q_next = self.target_net(sn).max(dim=1)[0]
                target_y = r + self.config.gamma * q_next * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * s.shape[0]).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()


# ==============================================================================
# 2. VARIANT B: DP-Warmstarted Target Network
# ==============================================================================
class DPWarmstartTargetAgent(DPDQNAgent):
    """Target net is warmstarted on fresh DP samples before updating living net."""
    def __init__(self, config, target_steps=2, **kwargs):
        super().__init__(config, **kwargs)
        self.target_steps = target_steps
        for p in self.target_net.parameters():
            p.requires_grad = True
        self.target_optimizer = optim.Adam(self.target_net.parameters(), lr=self.config.lr)

    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        # Sync target to online
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Warmstart target net on fresh DP samples
        for _ in range(self.target_steps):
            s_t, a_t, r_t, sn_t, done_t, w_t = self.sampler.sample(self.replay, device=self.device)
            with torch.no_grad():
                q_next_t = self.target_net(sn_t).max(dim=1)[0]
                target_y_t = r_t + self.config.gamma * q_next_t * (1.0 - done_t)
            pred_t = self.target_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
            loss_t = (self.loss_fn(pred_t, target_y_t) * w_t * s_t.shape[0]).mean()
            self.target_optimizer.zero_grad()
            loss_t.backward()
            nn.utils.clip_grad_norm_(self.target_net.parameters(), 1.0)
            self.target_optimizer.step()

        # Warmstart online net on fresh DP samples
        for _ in range(self.config.warmstart_steps):
            s, a, r, sn, done, q_weights = self.sampler.sample(self.replay, device=self.device)
            with torch.no_grad():
                q_next = self.target_net(sn).max(dim=1)[0]
                target_y = r + self.config.gamma * q_next * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * s.shape[0]).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()


# ==============================================================================
# 3. VARIANT C: Living Online Q-Net as Target (No Separate Target Net)
# ==============================================================================
class LivingOnlineTargetAgent(DPDQNAgent):
    """Evaluates target directly on the living online net with fresh sampling at each step."""
    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        for _ in range(self.config.warmstart_steps):
            s, a, r, sn, done, q_weights = self.sampler.sample(self.replay, device=self.device)
            with torch.no_grad():
                # Evaluated on living online network directly (detached)
                q_next = self.q_net(sn).max(dim=1)[0]
                target_y = r + self.config.gamma * q_next * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * s.shape[0]).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()


# ==============================================================================
# 4. VARIANT D: Hard Periodic Snapshot Target (Copy every K episodes)
# ==============================================================================
class HardSnapshotTargetAgent(DPDQNAgent):
    """Copies target_net <- q_net every K episodes (no Polyak decay)."""
    def __init__(self, config, period_k=5, **kwargs):
        super().__init__(config, **kwargs)
        self.period_k = period_k

    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        # Hard snapshot every K episodes
        if self.episodes_completed % self.period_k == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        for _ in range(self.config.warmstart_steps):
            s, a, r, sn, done, q_weights = self.sampler.sample(self.replay, device=self.device)
            with torch.no_grad():
                q_next = self.target_net(sn).max(dim=1)[0]
                target_y = r + self.config.gamma * q_next * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * s.shape[0]).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()


# ==============================================================================
# Runner Function
# ==============================================================================
def run_variant(agent_cls, name, size=20, seed=42, max_episodes=800, **kwargs):
    print("=" * 75)
    print(f"RUNNING: {name} on DeepSea N={size}, Seed={seed}")
    print(f"Settings: ZERO-MEAN REWARDS (mean=0.0, std=0.2), Fresh Sampling (W={size//2})")
    print("=" * 75)

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
        prior_reward_mean=0.0,       # PURE ZERO-MEAN (NO STOCHASTIC OPTIMISM)
        prior_reward_std=0.2,        # Bounded symmetric variance
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.0,
        target_warmstart=False,
        warmstart_steps=size // 2,   # W = 10 steps
        sample_once_per_episode=False, # FRESH SAMPLES AT EACH WARMSTART STEP
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        episodic_sgd=False,
    )

    env = make_env("deep_sea", seed=seed, size=size)
    agent = agent_cls(cfg, **kwargs)
    agent.base_measure.goal_bonus = False  # NO GOAL BONUS

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


if __name__ == "__main__":
    variants = [
        (LivingOnlineTargetAgent, "Variant C: Living Online Q-Net (No Target Net)", {}),
        (HardSnapshotTargetAgent, "Variant D: Hard Snapshot (Period K=5)", {"period_k": 5}),
        (DPWarmstartTargetAgent, "Variant B: DP-Warmstarted Target (2 steps)", {"target_steps": 2}),
        (PolyakTargetAgent, "Variant A: Polyak Target (Baseline, tau=0.05)", {}),
    ]

    results = {}
    for cls, name, kwargs in variants:
        disc, sol, duration = run_variant(cls, name, size=20, seed=42, max_episodes=800, **kwargs)
        results[name] = {"discovery": disc, "solved": sol, "time": duration}

    print("\n" + "=" * 80)
    print("FINAL SUMMARY: TARGET VARIANTS ON DEEPSEA N=20 (ZERO-MEAN REWARDS)")
    print("=" * 80)
    for name, res in results.items():
        print(f"{name:45s} | Discovery: {str(res['discovery']):5s} | Solved: {str(res['solved']):5s} | Time: {res['time']:.1f}s")
