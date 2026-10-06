import numpy as np
import torch
from src.envs.wheel_bandit import WheelBandit
from src.models.dp_bnn import DPBNNAgent
from src.models.baselines import EpsilonGreedyAgent, DeepEnsembleAgent, NeuralLinearAgent


def test_wheel_bandit():
    env = WheelBandit(delta=0.5, seed=42)

    # 1. Inside delta: radius < 0.5
    inside_ctx = np.array([0.2, 0.2], dtype=np.float32)
    means = env.get_mean_rewards(inside_ctx)
    assert np.isclose(means[0], 1.2)
    assert np.all(np.isclose(means[1:], 1.0))

    # 2. Outside delta in Quadrant 1: x1 > 0, x2 > 0
    q1_ctx = np.array([0.6, 0.6], dtype=np.float32)
    means_q1 = env.get_mean_rewards(q1_ctx)
    assert np.isclose(means_q1[0], 1.2)
    assert np.isclose(means_q1[1], 50.0)
    assert np.all(np.isclose(means_q1[2:], 1.0))

    # Step test
    rew, exp_rew, opt_exp_rew, reg = env.step(q1_ctx, 1)
    assert exp_rew == 50.0
    assert opt_exp_rew == 50.0
    assert reg == 0.0
    print("WheelBandit test passed!")


def test_agents():
    env = WheelBandit(delta=0.5, seed=42)
    agents = [
        ("DP-BNN", DPBNNAgent(seed=42, truncation_K=20, steps_per_decision=2)),
        ("Eps-Greedy", EpsilonGreedyAgent(seed=42, steps_per_decision=2)),
        ("Deep-Ensemble", DeepEnsembleAgent(seed=42, num_models=3, steps_per_decision=2)),
        ("Neural-Linear", NeuralLinearAgent(seed=42, steps_per_decision=2))
    ]

    for name, agent in agents:
        print(f"Testing {name}...")
        for t in range(15):
            ctx = env.sample_context()
            act = agent.select_action(ctx)
            rew, _, _, _ = env.step(ctx, act)
            agent.update(ctx, act, rew)
        print(f"{name} passed 15 steps!")


if __name__ == "__main__":
    test_wheel_bandit()
    test_agents()
    print("All component tests succeeded!")
