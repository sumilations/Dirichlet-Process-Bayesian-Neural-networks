"""Quick test: RiverSwim with single MLP-20 and standard normal base measure (zero jackpot bias).

Tests:
1. RiverSwim-6 and RiverSwim-10 across 5 seeds {42, 101, 2024, 7, 99}.
2. Network: Single MLP-20 (hidden_dim=20, num_layers=1, LayerNorm).
3. Base Measure: Standard Normal Base Measure N(0, 1) for rewards everywhere:
   - ZERO bias encouraging right as jackpot.
   - ZERO location information.
   - r ~ N(0, 1) across all transitions.
4. Algorithm: Multi-Sample DP-DQN (episodic_sgd=True, W=5, lr=5e-3).
"""

import time
import numpy as np
import torch
torch.set_num_threads(1)

from src.bayesian_distributional_rl.environments import RiverSwimEnv
from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.base_measures import BaseMeasure


class StandardNormalRiverSwimBaseMeasure(BaseMeasure):
    """Standard Normal Base Measure for RiverSwim: zero jackpot bias, r ~ N(0, 1) everywhere."""
    def __init__(self, n_states: int):
        super().__init__(state_dim=n_states, action_dim=2, prior_reward_mean=0.0, prior_reward_std=1.0)
        self.n_states = n_states

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.zeros((count, self.n_states), dtype=np.float32)
        sn = np.zeros((count, self.n_states), dtype=np.float32)
        idx = rng.randint(0, self.n_states, size=count)
        s[np.arange(count), idx] = 1.0
        a = rng.randint(0, 2, size=count)
        
        # Simple physical 1D walk transitions: a=1 moves right, a=0 moves left
        next_idx = np.where(a == 1, np.minimum(idx + 1, self.n_states - 1), np.maximum(idx - 1, 0))
        sn[np.arange(count), next_idx] = 1.0
        
        # Standard Normal Rewards N(0, 1): ZERO jackpot bias, zero expectation
        r = rng.normal(0.0, 1.0, size=count).astype(np.float32)
        done = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, done


def run_riverswim_test(n_states=6, seeds=[42, 101, 2024, 7, 99], max_episodes=150):
    H = 10 * n_states
    print("=" * 65)
    print(f"RIVERSWIM N={n_states} (Horizon H={H}) with Single MLP-20 & N(0, 1) Prior")
    print(f"Network: 1 hidden layer of 20 units | Prior: Standard Normal N(0, 1) (ZERO Jackpot Bias)")
    print("=" * 65)

    discovery_eps = []
    solved_eps = []
    cum_returns = []
    runtimes = []

    for seed in seeds:
        t0 = time.time()
        env = RiverSwimEnv(n_states=n_states, max_steps=H, seed=seed)
        cfg = DPDQNConfig(
            state_dim=n_states,
            action_dim=2,
            hidden_dim=20,          # Exactly 20 units!
            num_layers=1,           # Exactly 1 hidden layer!
            use_layer_norm=True,
            activation="relu",
            alpha=3.0,
            base_measure="custom",
            prior_reward_mean=0.0,
            prior_reward_std=1.0,
            sampler_type="vashishtha_maillard",
            vm_prior_multiplier=4.0,
            seed=seed,
            buffer_capacity=50000,
            one_living_network=True,
            warmstart_steps=5,
            episodic_sgd=True,
            tau=0.05,
            lr=5e-3,
            gamma=0.98,
            batch_size=32,
            candidate_batch_size=64,
            sgd_period=2,
            sample_once_per_episode=False, # Multi-Sample
        )
        bm = StandardNormalRiverSwimBaseMeasure(n_states=n_states)
        agent = DPDQNAgent(cfg, base_measure=bm)

        disc_at = None
        solved_at = None
        total_ret = 0.0

        for ep in range(1, max_episodes + 1):
            agent.reset_episode()
            s_idx = env.reset()
            s = np.zeros(n_states, dtype=np.float32)
            s[s_idx] = 1.0
            done = False
            ep_ret = 0.0
            steps = 0

            while not done:
                a = agent.act(s)
                sn_idx, r, done, _ = env.step(a)
                sn = np.zeros(n_states, dtype=np.float32)
                sn[sn_idx] = 1.0
                agent.step(s, a, float(r), sn, done)
                ep_ret += float(r)
                s = sn
                steps += 1
                
                # Check if reached the far right goal
                if sn_idx == n_states - 1 and disc_at is None:
                    disc_at = ep
                if r >= 1.0 and solved_at is None:
                    solved_at = ep

            total_ret += ep_ret

        elapsed = time.time() - t0
        discovery_eps.append(disc_at if disc_at is not None else -1)
        solved_eps.append(solved_at if solved_at is not None else -1)
        cum_returns.append(total_ret)
        runtimes.append(elapsed)

        disc_str = str(disc_at) if disc_at is not None else "NEVER"
        solv_str = str(solved_at) if solved_at is not None else "NEVER"
        print(f"  Seed {seed:<5} -> First Reached Goal: Ep {disc_str:<5} | Got Reward: Ep {solv_str:<5} | Total Return: {total_ret:<6.1f} | Time: {elapsed:.1f}s")

    solved_count = sum(1 for s in solved_eps if s > 0)
    valid_solved = [s for s in solved_eps if s > 0]
    mean_solved = np.mean(valid_solved) if valid_solved else float('nan')
    print("-" * 65)
    print(f"Summary N={n_states}: Solved {solved_count}/{len(seeds)} ({solved_count/len(seeds)*100:.0f}%) | Mean Solved Ep: {mean_solved:.1f} | Mean Return: {np.mean(cum_returns):.1f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_riverswim_test(n_states=6)
    run_riverswim_test(n_states=10)
