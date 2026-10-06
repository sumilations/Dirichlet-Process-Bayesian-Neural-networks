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

class SameOnlineTargetDPDQNAgent(DPDQNAgent):
    """Single Online Network as Target (Target = Online Q)."""
    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        for _ in range(self.config.warmstart_steps):
            s, a, r, sn, done, q_weights = self.sampler.sample(self.replay, device=self.device)
            effective_batch = s.shape[0]

            with torch.no_grad():
                q_next = self.q_net(sn).max(dim=1)[0]
                target_y = r + self.config.gamma * q_next * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * effective_batch).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()


class TargetWarmstartDPDQNAgent(DPDQNAgent):
    """Target Network Warmstarted on DP posterior."""
    def __init__(self, config, target_warmstart_steps=2, **kwargs):
        super().__init__(config, **kwargs)
        self.target_warmstart_steps = target_warmstart_steps
        for p in self.target_net.parameters():
            p.requires_grad = True
        self.target_optimizer = optim.Adam(self.target_net.parameters(), lr=self.config.lr)

    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        self.target_net.load_state_dict(self.q_net.state_dict())
        for _ in range(self.target_warmstart_steps):
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

        for _ in range(self.config.warmstart_steps):
            s, a, r, sn, done, q_weights = self.sampler.sample(self.replay, device=self.device)
            effective_batch = s.shape[0]
            with torch.no_grad():
                q_next = self.target_net(sn).max(dim=1)[0]
                target_y = r + self.config.gamma * q_next * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * effective_batch).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()


def evaluate_agent(agent_cls, size=20, seed=42, name="Agent", max_episodes=1000):
    print("=" * 70)
    print(f"EVALUATING {name} ON DEEPSEA N={size}, Seed={seed}")
    print(f"Base Reward Prior Mean: 0.0 (Zero-Mean, NO Stochastic Optimism)")
    print("=" * 70)

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
        prior_reward_mean=0.0,       # PURE ZERO MEAN!
        prior_reward_std=0.2,        # Standard variance
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.0,
        target_warmstart=False,
        warmstart_steps=size // 2,
        sample_once_per_episode=False,
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        episodic_sgd=False,
    )

    env = make_env("deep_sea", seed=seed, size=size)
    agent = agent_cls(cfg)
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
            print(f">>> [DISCOVERY!] Episode {ep} reached goal! ({time.time()-t0:.2f}s)", flush=True)

        if ep >= 20:
            recent_ret = np.mean(returns[-20:])
            if recent_ret >= 0.8:
                solved_ep = ep
                print(f"*** [SOLVED!] Episode {ep} SOLVED! (Avg return: {recent_ret:.3f}, Time: {time.time()-t0:.2f}s) ***", flush=True)
                break

        if ep % 50 == 0:
            print(f"Episode {ep} | Recent return: {np.mean(returns[-20:]):.4f}", flush=True)

    if solved_ep is None:
        print(f"Did not solve within {max_episodes} episodes. First discovery: {first_discovery}")
    return first_discovery, solved_ep


if __name__ == "__main__":
    print("\n--- DEEPSEA N=20: TARGET WARMSTART ---")
    evaluate_agent(TargetWarmstartDPDQNAgent, size=20, seed=42, name="Target Warmstart (2 steps)")

    print("\n--- DEEPSEA N=20: SAME ONLINE Q-NET AS TARGET ---")
    evaluate_agent(SameOnlineTargetDPDQNAgent, size=20, seed=42, name="Same Online Q-Net")
