"""Environment factory for unified DP-DQN."""

from typing import Any
from .cartpole_swingup import CartpoleSwingupEnv
from .deep_sea import DeepSeaEnv


def make_env(env_name: str, **kwargs) -> Any:
    """Factory to create standardized environments."""
    name_clean = env_name.lower().strip()
    if "cartpole" in name_clean:
        seed = kwargs.get("seed", None)
        return CartpoleSwingupEnv(seed=seed)
    elif "deep_sea" in name_clean:
        size = kwargs.get("size", kwargs.get("deep_sea_size", 10))
        seed = kwargs.get("seed", None)
        randomize_actions = kwargs.get("randomize_actions", True)
        return DeepSeaEnv(size=size, seed=seed, randomize_actions=randomize_actions)
    else:
        # Fallback to gymnasium if installed
        try:
            import gymnasium as gym
            return gym.make(env_name, **kwargs)
        except Exception:
            try:
                import gym
                return gym.make(env_name, **kwargs)
            except Exception as e:
                raise ValueError(f"Unknown or unsupported environment: {env_name}. Error: {e}")
