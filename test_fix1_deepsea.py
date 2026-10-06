import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env
from src.dp_dqn.agent import DPDQNAgent, ReplayBuffer

class EpistemicTargetDPDQNAgent(DPDQNAgent):
    """DP-DQN Agent implementing Fix 1: Epistemic Posterior Bellman Target."""

    def __init__(self, config, epistemic_sigma=0.5, **kwargs):
        super().__init__(config, **kwargs)
        self.epistemic_sigma = epistemic_sigma
        # Track state visitation counts
        self.visited_states = set()

    def reset_episode(self):
        self.episodes_completed += 1
        if len(self.replay) < self.config.batch_size or self.config.warmstart_steps <= 0:
            return

        # Polyak update target network
        if self.config.tau > 0.0:
            with torch.no_grad():
                tau = self.config.tau
                for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

        # Warmstart loop with Fix 1: Epistemic Bellman Target
        for _ in range(self.config.warmstart_steps):
            s, a, r, sn, done, q_weights = self.sampler.sample(self.replay, device=self.device)
            effective_batch = s.shape[0]

            with torch.no_grad():
                # 1. Evaluate target network on sn: shape [batch, action_dim]
                q_next_base = self.target_net(sn)  # [B, A]

                # 2. Compute epistemic mask for sn:
                # In DeepSea, state is one-hot. The index of the active state is argmax(sn, dim=1).
                state_indices = sn.argmax(dim=1).cpu().numpy()
                is_visited = np.array([idx in self.visited_states for idx in state_indices], dtype=np.float32)
                # Unvisited states have sigma = epistemic_sigma, visited have sigma = 0
                sigma_epistemic = (1.0 - torch.from_numpy(is_visited).to(self.device).unsqueeze(1)) * self.epistemic_sigma

                # 3. Sample standard Gaussian epistemic perturbations xi ~ N(0, 1)
                xi = torch.randn_like(q_next_base)

                # 4. Epistemic Q-next: Convexity of max induces emergent optimism only on unvisited states!
                q_next_epistemic = q_next_base + sigma_epistemic * xi
                q_next_val = q_next_epistemic.max(dim=1)[0]

                target_y = r + self.config.gamma * q_next_val * (1.0 - done)

            pred_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = (self.loss_fn(pred_q, target_y) * q_weights * effective_batch).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

    def record_visit(self, state: np.ndarray):
        idx = int(np.argmax(state))
        self.visited_states.add(idx)


def test_fix1_deepsea(size=10, seed=42, epistemic_sigma=0.5, max_episodes=500):
    print("=" * 70)
    print(f"TESTING FIX 1: EPISTEMIC TARGET BELLMAN ON DEEPSEA N={size}, Seed={seed}")
    print(f"Base Reward Prior Mean: 0.0 (Zero-Mean, NO Stochastic Optimism)")
    print(f"Epistemic Sigma: {epistemic_sigma}")
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
        prior_reward_std=0.1,        # Small variance
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=True,
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.05,
        target_warmstart=True,
        warmstart_steps=size // 2,
        sample_once_per_episode=False, # Fresh sampling per step
        num_episodes=max_episodes,
        max_episode_steps=size,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        episodic_sgd=False,
    )

    env = make_env("deep_sea", seed=seed, size=size)
    agent = EpistemicTargetDPDQNAgent(cfg, epistemic_sigma=epistemic_sigma)
    # Ensure base measure has NO goal bonus
    agent.base_measure.goal_bonus = False

    returns = []
    t0 = time.time()
    first_discovery = None
    solved_ep = None

    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        s = env.reset()
        agent.record_visit(s)
        done = False
        ep_ret = 0.0

        while not done:
            a = agent.act(s)
            sn, r, done, _ = env.step(a)
            agent.record_visit(sn)
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
    print("\n--- MULTI-SEED EVALUATION: DEEPSEA N=10 (ZERO MEAN REWARDS) ---")
    seeds_10 = [42, 43, 44, 45, 46]
    solved_10 = []
    disc_10 = []
    for s in seeds_10:
        d, sol = test_fix1_deepsea(size=10, seed=s, epistemic_sigma=0.5, max_episodes=400)
        if sol is not None:
            solved_10.append(sol)
            disc_10.append(d)
    print(f"\nDEEPSEA-10 SUMMARY: Solved {len(solved_10)}/{len(seeds_10)} | Avg Solve: {np.mean(solved_10):.1f} | Avg Disc: {np.mean(disc_10):.1f}")

    print("\n--- TESTING DEEPSEA N=20 (ZERO MEAN REWARDS) ---")
    test_fix1_deepsea(size=20, seed=42, epistemic_sigma=0.5, max_episodes=800)
