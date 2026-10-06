import bsuite
import numpy as np


class BSuiteCartpoleWrapper:
    """Wraps bsuite cartpole_swingup into a standard Gym-compatible interface."""

    def __init__(self, bsuite_id="cartpole_swingup/0"):
        self.raw_env = bsuite.load_from_id(bsuite_id)
        self.action_dim = self.raw_env.action_spec().num_values
        # Observation is shape (1, 8) => flattened to 8
        self.state_dim = int(np.prod(self.raw_env.observation_spec().shape))

    def reset(self):
        timestep = self.raw_env.reset()
        obs = np.array(timestep.observation, dtype=np.float32).flatten()
        return obs

    def step(self, action):
        timestep = self.raw_env.step(int(action))
        obs = np.array(timestep.observation, dtype=np.float32).flatten()
        reward = float(timestep.reward or 0.0)
        done = bool(timestep.last())
        info = {
            "discount": float(timestep.discount if timestep.discount is not None else 1.0),
            "step_type": timestep.step_type,
            "is_upright": bool(obs[3] > 0.0)  # obs[3] is cos(theta) > 0 (upper hemisphere)
        }
        return obs, reward, done, info
