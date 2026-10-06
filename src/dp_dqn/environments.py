"""Environment factory and wrappers for Unified DP-DQN."""

from typing import Any, Optional, Tuple
import numpy as np


class EnvWrapper:
    """Standardized environment interface for DP-DQN."""

    def __init__(self, raw_env, state_dim: int, action_dim: int):
        self.raw_env = raw_env
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError


class DeepSeaWrapper(EnvWrapper):
    """Wrapper for bsuite DeepSea environment."""

    def __init__(self, size: int = 10, seed: int = 42, deterministic: bool = True, randomize_actions: bool = True, mapping_seed: Optional[int] = None):
        try:
            from bsuite.environments.deep_sea import DeepSea
            m_seed = mapping_seed if mapping_seed is not None else seed
            env = DeepSea(size=size, deterministic=deterministic, randomize_actions=randomize_actions, seed=seed, mapping_seed=m_seed)
        except ImportError:
            raise ImportError("bsuite is required for DeepSea. Install with `pip install bsuite`.")

        self.size = size
        state_dim = size * size
        action_dim = 2
        super().__init__(env, state_dim, action_dim)

    def reset(self) -> np.ndarray:
        timestep = self.raw_env.reset()
        obs = np.array(timestep.observation, dtype=np.float32).flatten()
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        timestep = self.raw_env.step(int(action))
        obs = np.array(timestep.observation, dtype=np.float32).flatten()
        reward = float(timestep.reward or 0.0)
        done = bool(timestep.last())
        info = {
            "discount": float(timestep.discount if timestep.discount is not None else 1.0),
            "step_type": timestep.step_type,
        }
        return obs, reward, done, info





class GymEnvWrapper(EnvWrapper):
    """Wrapper for standard OpenAI Gym / Farama Gymnasium environments."""

    def __init__(self, env_id: str):
        try:
            import gymnasium as gym
        except ImportError:
            try:
                import gym
            except ImportError:
                raise ImportError("gym or gymnasium is required. Install with `pip install gymnasium`.")

        env = gym.make(env_id)
        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(env.action_space.n)
        super().__init__(env, state_dim, action_dim)

    def reset(self) -> np.ndarray:
        res = self.raw_env.reset()
        obs = res[0] if isinstance(res, tuple) else res
        return np.array(obs, dtype=np.float32).flatten()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        res = self.raw_env.step(int(action))
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = res
        return np.array(obs, dtype=np.float32).flatten(), float(reward), done, info


class RiverSwimEnv:
    """Canonical RiverSwim Environment (Strehl & Littman, 2008; Osband et al., 2013)."""

    def __init__(self, n_states: int = 6, max_steps: int = 40, seed: Optional[int] = None):
        self.n_states = n_states
        self.num_states = n_states
        self.num_actions = 2
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.current_state = 0
        self.current_step = 0

    def reset(self) -> int:
        self.current_state = 0
        self.current_step = 0
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        self.current_step += 1
        s = self.current_state
        r = 0.0

        if action == 0:
            if s == 0:
                s_next = 0
                r = 0.05
            else:
                s_next = s - 1
                r = 0.0
        else:
            if s == 0:
                s_next = 1 if self.rng.random() < 0.6 else 0
                r = 0.0
            elif s == self.n_states - 1:
                if self.rng.random() < 0.6:
                    s_next = self.n_states - 1
                    r = 1.0
                else:
                    s_next = self.n_states - 2
                    r = 0.0
            else:
                p = self.rng.random()
                if p < 0.1:
                    s_next = s - 1
                elif p < 0.7:
                    s_next = s
                else:
                    s_next = s + 1
                r = 0.0

        self.current_state = s_next
        done = (self.current_step >= self.max_steps)
        return s_next, r, done, {}


def make_env(env_name: str, **kwargs) -> EnvWrapper:
    """Factory creating standardized environment wrapper.

    Supported:
    - 'deep_sea' or 'deepsea': kwargs can include size=10, seed=42
    - 'gym:<env_id>': e.g. 'gym:MountainCar-v0', 'gym:Acrobot-v1'
    """
    name = str(env_name).lower().strip()

    if name in ("deep_sea", "deepsea"):
        size = kwargs.get("deep_sea_size", kwargs.get("size", 10))
        seed = kwargs.get("seed", 42)
        mapping_seed = kwargs.get("mapping_seed", seed)
        return DeepSeaWrapper(size=size, seed=seed, mapping_seed=mapping_seed)
    elif name.startswith("gym:"):
        gym_id = env_name.split("gym:", 1)[1]
        return GymEnvWrapper(gym_id)
    else:
        # Try Gym directly
        return GymEnvWrapper(env_name)
