import gym
import numpy as np
import torch
import sys
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

print("=== Running MountainCar with Zero Goal-Bias Base Measures ===", flush=True)

for name, bm_cls in [("Kinematics (Zero Goal Bias)", KinematicsUnbiasedBaseMeasure), ("Model-Free Uniform", ModelFreeUniformBaseMeasure)]:
    print(f"\n--- Testing {name} ---", flush=True)
    solved_eps = []
    for seed in [42, 101, 7]:
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
        for ep in range(1, 31):
            agent.reset_episode()
            s = env.reset()
            if isinstance(s, tuple): s = s[0]
            done = False
            ep_ret = 0.0
            max_x = -1.2
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
                max_x = max(max_x, s[0])
            
            if ep_ret > -200.0 and solved is None:
                solved = ep
                print(f"  Seed {seed}: SOLVED at Episode {ep} in {steps} steps (Return: {ep_ret:.1f}, Max X: {max_x:.3f})!", flush=True)
                break
            else:
                if ep % 5 == 0 or ep == 1:
                    print(f"  Seed {seed} Ep {ep}: Ret {ep_ret:.1f}, Max X: {max_x:.3f} (steps: {steps})", flush=True)
        if solved is None:
            print(f"  Seed {seed}: Not solved in 30 eps (Last Max X: {max_x:.3f})", flush=True)
        solved_eps.append(solved)
    print(f"--> {name} Final: Solved at {solved_eps}", flush=True)
