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

env = gym.make("MountainCar-v0")
for seed in [42, 101]:
    cfg = DPDQNConfig(
        state_dim=2, action_dim=3, hidden_dim=64, num_layers=2, use_layer_norm=True,
        alpha=3.0, base_measure="custom", sampler_type="vashishtha_maillard",
        vm_prior_multiplier=4.0, seed=seed, buffer_capacity=50000,
        warmstart_steps=5, episodic_sgd=False, tau=0.01, lr=1e-3, gamma=0.99, batch_size=32, sgd_period=4
    )
    agent = DPDQNAgent(cfg, base_measure=KinematicsUnbiasedBaseMeasure())
    for _ in range(50): agent.update()
    
    solved = None
    max_x_reached = -1.2
    for ep in range(1, 301):
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
            if s[0] > max_x_reached:
                max_x_reached = s[0]
        if ep_ret > -200.0:
            print(f"Seed {seed} SOLVED at Episode {ep} (Steps: {steps}, Return: {ep_ret:.1f}, Max X: {max_x_reached:.2f})!")
            break
        if ep % 25 == 0:
            print(f"Seed {seed} Ep {ep}: Max X reached so far = {max_x_reached:.3f}")
