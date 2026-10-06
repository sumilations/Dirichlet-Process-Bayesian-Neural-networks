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

class HardSnapshotTDInfoAgent(DPDQNAgent):
    """Hard Snapshot Target Network with TD Information Gain Decay of W_prior."""
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

        # TD info gain count override
        n_stat = self.cumulative_info if getattr(self.config, "use_td_info_gain_decay", False) else None

        for _ in range(self.config.warmstart_steps):
            # Pass n_stat_override so W_prior decays dynamically with TD information gain!
            s, a, r, sn, done, q_weights = self.sampler.sample(
                self.replay, device=self.device, n_stat_override=n_stat
            )
            with torch.no_grad():
                q_next = self.target_net(sn).max(dim=1)[0]
                target_y = r + self.config.gamma * q_next * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * s.shape[0]).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate TD Information Gain on empirical data
            with torch.no_grad():
                n_emp = getattr(self.sampler, "last_n_emp", s.shape[0])
                if n_emp > 0:
                    emp_delta = (target_y[:n_emp] - pred_q[:n_emp]).detach()
                    surprise = torch.clamp(emp_delta.abs() / 1.0, max=1.0).sum().item()
                    self.cumulative_info += float(surprise)


def test_td_info_deepsea20(size=20, seed=42, max_episodes=800):
    print("=" * 80)
    print(f"TESTING HARD SNAPSHOT (K=5) + TD INFO GAIN DECAY ON DEEPSEA N={size}, Seed={seed}")
    print(f"Settings: ZERO-MEAN REWARDS (mean=0.0, std=0.2), Fresh Sampling (W={size//2})")
    print("=" * 80)

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
        prior_reward_mean=0.0,          # PURE ZERO MEAN
        prior_reward_std=0.2,           # Standard variance
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.0,
        target_warmstart=False,
        warmstart_steps=size // 2,      # W = 10
        sample_once_per_episode=False,  # Fresh per step
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        episodic_sgd=False,
        use_td_info_gain_decay=True,    # ENABLE TD INFO GAIN!
        priority_positive_slot=True,    # Preserve discovery transition in empirical batch
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
            print(f">>> [DISCOVERY!] Goal reached at Episode {ep}! Cumulative Info: {agent.cumulative_info:.1f} ({time.time()-t0:.2f}s)", flush=True)

        if ep >= 20:
            recent_ret = np.mean(returns[-20:])
            if recent_ret >= 0.8:
                solved_ep = ep
                print(f"*** [SOLVED!] SOLVED at Episode {ep}! (Avg return: {recent_ret:.3f}, Info: {agent.cumulative_info:.1f}, Time: {time.time()-t0:.2f}s) ***", flush=True)
                break

        if ep % 50 == 0:
            print(f"Episode {ep:3d} | Recent Return: {np.mean(returns[-20:]):.4f} | Info: {agent.cumulative_info:.1f} | Time: {time.time()-t0:.1f}s", flush=True)

    if solved_ep is None:
        print(f"--> Result: Did not solve in {max_episodes} eps. First discovery: {first_discovery}")

    return first_discovery, solved_ep, time.time() - t0

if __name__ == "__main__":
    test_td_info_deepsea20(size=20, seed=42)
