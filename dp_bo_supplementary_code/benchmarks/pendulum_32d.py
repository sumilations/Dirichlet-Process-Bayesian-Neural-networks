"""
Continuous Control Policy Optimization Benchmark (Pendulum-v1, d=32).

As described in Appendix E of the paper:
Direct policy search over 32 neural network controller parameters:
pi(s) = 2.0 * tanh(W2 @ ReLU(W1 @ s)), with W1 in R^{3x8} and W2 in R^{8x1}.
Objective: cumulative regret (lower is better; ~150 is near-optimal, ~1600 is failure).
"""

import numpy as np

try:
    import gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


def evaluate_pendulum_policy(w_np: np.ndarray, seeds=(101, 102)) -> float:
    """
    Evaluates 32D policy weights in OpenAI Gym Pendulum-v1.
    """
    if not GYM_AVAILABLE:
        raise ImportError("gym package is required to run live Pendulum evaluations. Please install via `pip install gym`.")

    W1 = w_np[:24].reshape(3, 8)
    W2 = w_np[24:].reshape(8, 1)

    env = gym.make("Pendulum-v1")
    total_ret = 0.0

    for seed in seeds:
        s = env.reset(seed=seed)
        if isinstance(s, tuple):
            s = s[0]
        ret = 0.0
        done = False
        while not done:
            h = np.maximum(0.0, s @ W1)  # ReLU
            a = 2.0 * np.tanh(h @ W2)
            step_out = env.step(a)
            if len(step_out) == 5:
                s, r, done, trunc, _ = step_out
                done = done or trunc
            else:
                s, r, done, _ = step_out
            ret += r
            if done:
                break
        total_ret += ret

    avg_ret = total_ret / len(seeds)
    return -avg_ret  # Regret: lower is better
