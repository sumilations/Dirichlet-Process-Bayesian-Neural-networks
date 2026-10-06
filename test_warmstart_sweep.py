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

        if self.episodes_completed % self.period_k == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        n_stat = self.cumulative_info if getattr(self.config, "use_td_info_gain_decay", False) else None

        for _ in range(self.config.warmstart_steps):
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

            with torch.no_grad():
                n_emp = getattr(self.sampler, "last_n_emp", s.shape[0])
                if n_emp > 0:
                    emp_delta = (target_y[:n_emp] - pred_q[:n_emp]).detach()
                    surprise = torch.clamp(emp_delta.abs() / 1.0, max=1.0).sum().item()
                    self.cumulative_info += float(surprise)


def run_warmstart_experiment(size=20, warmstart_steps=10, seed=42, max_episodes=600):
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
        prior_reward_std=0.2,
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.0,
        target_warmstart=False,
        warmstart_steps=warmstart_steps,  # Swept hyperparameter W!
        sample_once_per_episode=False,    # Fresh per step
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
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

        if ep >= 20:
            recent_ret = np.mean(returns[-20:])
            if recent_ret >= 0.8:
                solved_ep = ep
                break

    duration = time.time() - t0
    return first_discovery, solved_ep, duration


if __name__ == "__main__":
    print("=" * 80)
    print("WARMSTART STEP SWEEP (W) ON DEEPSEA N=20 (ZERO-MEAN REWARDS, SEED 42)")
    print("=" * 80)
    w_list = [2, 4, 6, 8, 10, 14, 18]
    results = []
    for w in w_list:
        disc, sol, dur = run_warmstart_experiment(size=20, warmstart_steps=w, seed=42)
        results.append((w, disc, sol, dur))
        print(f"W={w:2d} steps | Discovery: Ep {str(disc):5s} | Solved: Ep {str(sol):5s} | Time: {dur:.2f}s", flush=True)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY TABLE: WARMSTART SWEEP ON DEEPSEA N=20")
    print("=" * 80)
    for w, disc, sol, dur in results:
        print(f"W = {w:2d} warmstart steps | First Discovery: Ep {str(disc):5s} | Solved: Ep {str(sol):5s} | Wallclock: {dur:.2f}s")
