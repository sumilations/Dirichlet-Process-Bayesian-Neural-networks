import gym
import numpy as np
import torch
import sys
import time
sys.path.insert(0, '.')
from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.base_measures import BaseMeasure

class KinematicsUnbiasedBaseMeasure(BaseMeasure):
    """Kinematics-consistent base measure with ZERO goal bias:
    Respects physical 1-step equations of motion, but reward is uniformly stochastically optimistic
    everywhere and done=0 everywhere. Zero knowledge of flag x >= 0.5."""
    def __init__(self):
        super().__init__(state_dim=2, action_dim=3)

    def sample(self, count: int, rng: np.random.RandomState):
        x = rng.uniform(-1.2, 0.6, size=count)
        v = rng.uniform(-0.07, 0.07, size=count)
        s = np.column_stack([x, v]).astype(np.float32)
        a = rng.randint(0, 3, size=count)
        force = (a - 1) * 0.001
        vn = np.clip(v + force - np.cos(3 * x) * 0.0025, -0.07, 0.07)
        xn = np.clip(x + vn, -1.2, 0.6)
        sn = np.column_stack([xn, vn]).astype(np.float32)
        r = rng.uniform(0.0, 1.0, size=count).astype(np.float32)
        d = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, d

class ModelFreeUniformBaseMeasure(BaseMeasure):
    """Pure model-free uniform base measure: completely independent uniform s, a, sn.
    Zero kinematics, zero goal knowledge."""
    def __init__(self):
        super().__init__(state_dim=2, action_dim=3)

    def sample(self, count: int, rng: np.random.RandomState):
        s = np.column_stack([rng.uniform(-1.2, 0.6, size=count), rng.uniform(-0.07, 0.07, size=count)]).astype(np.float32)
        sn = np.column_stack([rng.uniform(-1.2, 0.6, size=count), rng.uniform(-0.07, 0.07, size=count)]).astype(np.float32)
        a = rng.randint(0, 3, size=count)
        r = rng.uniform(0.0, 1.0, size=count).astype(np.float32)
        d = np.zeros(count, dtype=np.float32)
        return s, a, r, sn, d

SEEDS = [42, 101, 7, 2024, 99]
MAX_EPISODES = 100

print(f"=== Running MountainCar Unbiased Base Measures ({MAX_EPISODES} Episodes, {len(SEEDS)} Seeds) ===", flush=True)

for name, bm_cls in [("Kinematics (Zero Goal Bias)", KinematicsUnbiasedBaseMeasure), ("Model-Free Uniform", ModelFreeUniformBaseMeasure)]:
    print(f"\n==================================================", flush=True)
    print(f"EVALUATING: {name}", flush=True)
    print(f"==================================================", flush=True)
    solved_eps = []
    final_returns = []
    t0 = time.time()
    for seed in SEEDS:
        env = gym.make("MountainCar-v0")
        cfg = DPDQNConfig(
            state_dim=2, action_dim=3, hidden_dim=64, num_layers=2, use_layer_norm=True,
            alpha=3.0, base_measure="custom", sampler_type="vashishtha_maillard",
            vm_prior_multiplier=4.0, seed=seed, buffer_capacity=50000,
            warmstart_steps=5, episodic_sgd=False, tau=0.01, lr=1e-3, gamma=0.99, batch_size=32, sgd_period=4
        )
        agent = DPDQNAgent(cfg, base_measure=bm_cls())
        for _ in range(50): agent.update()
        
        solved = None
        seed_rets = []
        for ep in range(1, MAX_EPISODES + 1):
            agent.reset_episode()
            s = env.reset()
            if isinstance(s, tuple): s = s[0]
            done = False
            ep_ret = 0.0
            steps = 0
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
                print(f"  [SOLVED] Seed {seed} solved at Episode {ep} (Steps: {steps}, Return: {ep_ret:.1f})!", flush=True)
            if ep in [10, 25, 50, 75, 100]:
                recent_avg = np.mean(seed_rets[-10:])
                print(f"    Seed {seed} Ep {ep}: Recent 10-Ep Avg Return = {recent_avg:.1f} (Solved: {solved})", flush=True)
        solved_eps.append(solved)
        final_returns.append(np.mean(seed_rets[-10:]))
    elapsed = time.time() - t0
    print(f"\n--> {name} Results across {len(SEEDS)} seeds ({elapsed:.1f}s):", flush=True)
    print(f"    Solved Episodes: {solved_eps}", flush=True)
    print(f"    Success Rate: {sum(1 for s in solved_eps if s is not None)}/{len(SEEDS)}", flush=True)
    valid_solves = [s for s in solved_eps if s is not None]
    if valid_solves:
        print(f"    Mean Solved Episode: {np.mean(valid_solves):.1f} +/- {np.std(valid_solves):.1f}", flush=True)
    print(f"    Final 10-Ep Avg Return: {np.mean(final_returns):.1f}", flush=True)
