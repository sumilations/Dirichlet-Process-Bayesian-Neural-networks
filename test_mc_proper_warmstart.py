import gym
import numpy as np
import torch
import sys
import time

sys.path.insert(0, '.')
from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.base_measures import BaseMeasure

class DiffusiveContinuousBaseMeasure(BaseMeasure):
    """Zero-knowledge local diffusive base measure:
    - NO equation of motion.
    - NO goal bias (zero knowledge of flag x >= 0.5).
    - Pure local continuity: s' = s + delta_s (eliminates teleportation).
    - Uniformly optimistic rewards r ~ U(0, 1).
    - d = 0 everywhere.
    """
    def __init__(self, step_sigma=(0.04, 0.007)):
        super().__init__(state_dim=2, action_dim=3)
        self.step_sigma = np.array(step_sigma, dtype=np.float32)

    def sample(self, count: int, rng: np.random.RandomState):
        x = rng.uniform(-1.2, 0.6, size=count)
        v = rng.uniform(-0.07, 0.07, size=count)
        s = np.column_stack([x, v]).astype(np.float32)
        a = rng.randint(0, 3, size=count)
        
        # Local continuous perturbation without knowing any ODEs
        delta = rng.normal(0.0, self.step_sigma, size=(count, 2)).astype(np.float32)
        delta[:, 1] += (a - 1) * 0.001
        
        sn = np.empty_like(s)
        sn[:, 0] = np.clip(s[:, 0] + delta[:, 0], -1.2, 0.6)
        sn[:, 1] = np.clip(s[:, 1] + delta[:, 1], -0.07, 0.07)
        
        # Uniformly optimistic reward everywhere (no goal bias)
        r = rng.uniform(0.0, 1.0, size=count).astype(np.float32)
        d = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, d

SEEDS = [42, 101, 7, 2024, 99]
solved_eps = []
returns_all = []
t0 = time.time()

print(f"=== Running 5-Seed Benchmark: Diffusive Base Measure (Zero ODE / Zero Goal Knowledge) ===")
env = gym.make("MountainCar-v0")
for seed in SEEDS:
    cfg = DPDQNConfig(
        state_dim=2, action_dim=3, hidden_dim=64, num_layers=2, use_layer_norm=True,
        alpha=3.0, base_measure="custom", sampler_type="vashishtha_maillard",
        vm_prior_multiplier=4.0, seed=seed, buffer_capacity=50000,
        one_living_network=True,
        warmstart_steps=100,  # H / 2 = 100 steps
        episodic_sgd=False, tau=0.01, lr=1e-3, gamma=0.99, batch_size=32, sgd_period=4
    )
    agent = DPDQNAgent(cfg, base_measure=DiffusiveContinuousBaseMeasure())
    for _ in range(100): agent.update()
    
    solved = None
    seed_rets = []
    for ep in range(1, 101):
        agent.reset_episode()
        s = env.reset()
        if isinstance(s, tuple): s = s[0]
        done = False
        steps = 0
        ep_ret = 0.0
        while not done and steps < 200:
            a = agent.act(s)
            res = env.step(a)
            if len(res) == 5:
                sn, r, term, trunc, _ = res
                done = bool(term or trunc)
            else:
                sn, r, done, _ = res
            agent.step(s, a, float(r), sn, done)
            ep_ret += float(r)
            s = sn
            steps += 1
            
        seed_rets.append(ep_ret)
        if ep_ret > -200.0 and solved is None:
            solved = ep
            print(f"  [SOLVED] Seed {seed} solved at Episode {ep}! (Steps: {steps}, Return: {ep_ret:.1f})", flush=True)
            
    solved_eps.append(solved)
    returns_all.append(np.mean(seed_rets[-10:]))
    if solved is None:
        print(f"  [FAILED] Seed {seed} did not solve within 100 eps.", flush=True)

elapsed = time.time() - t0
print(f"\n==========================================")
print(f"FINAL RESULTS across {len(SEEDS)} seeds ({elapsed:.1f}s):")
print(f"Solved Episodes: {solved_eps}")
valid = [s for s in solved_eps if s is not None]
print(f"Success Rate: {len(valid)}/{len(SEEDS)}")
print(f"Mean Solved Episode: {np.mean(valid):.1f} +/- {np.std(valid):.1f}")
print(f"Final 10-Ep Return Mean: {np.mean(returns_all):.1f}")
print(f"==========================================")
