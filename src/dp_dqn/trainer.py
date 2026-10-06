"""Unified training and evaluation loop for DP-DQN."""

import time
from typing import Callable, Dict, List, Optional
import numpy as np

from .config import DPDQNConfig
from .agent import DPDQNAgent
from .environments import make_env, EnvWrapper
from .base_measures import BaseMeasure, get_base_measure


def train(
    config: Optional[DPDQNConfig] = None,
    env: Optional[EnvWrapper] = None,
    agent: Optional[DPDQNAgent] = None,
    base_measure: Optional[BaseMeasure] = None,
    callbacks: Optional[List[Callable[[int, Dict], None]]] = None,
    **kwargs,
) -> Dict:
    """Train a DP-DQN Agent on the specified environment.

    Args:
        config: DPDQNConfig instance (or created from kwargs).
        env: Pre-instantiated environment (or automatically created via make_env).
        agent: Pre-instantiated agent (or automatically created).
        base_measure: Custom base measure instance (optional).
        callbacks: Optional list of callback functions called after each episode: fn(episode, stats).

    Returns:
        Dictionary containing training history and metrics.
    """
    config = config or DPDQNConfig(**kwargs)
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    # 1. Instantiate Environment if not provided
    if env is None:
        env = make_env(
            config.env_name,
            deep_sea_size=config.deep_sea_size,
            seed=config.seed or 42,
        )

    # Auto-infer dimensions
    config.state_dim = env.state_dim
    config.action_dim = env.action_dim

    # Resolve default base measure for the environment
    if base_measure is None:
        bm_name = config.base_measure
        if bm_name == "default":
            if "deep_sea" in config.env_name.lower():
                bm_name = "deep_sea"
            else:
                bm_name = "gaussian"

        base_measure = get_base_measure(
            bm_name,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            prior_reward_mean=config.prior_reward_mean,
            prior_reward_std=config.prior_reward_std,
            deep_sea_size=config.deep_sea_size,
        )

    # 2. Instantiate Agent if not provided
    if agent is None:
        agent = DPDQNAgent(config=config, base_measure=base_measure)

    # 3. Training Loop
    episode_returns = []
    episode_lengths = []
    eval_returns = []
    eval_episodes = []
    cumulative_regret = 0.0
    history_regret = []
    t_learn: Optional[int] = None

    t_start = time.time()
    if config.verbose:
        print(f"=== Starting DP-DQN Training ===")
        print(f"Environment: {config.env_name} | State Dim: {config.state_dim} | Action Dim: {config.action_dim}")
        print(f"Alpha: {config.alpha} | Hidden: {config.hidden_dim} | LayerNorm: {config.use_layer_norm}")
        print(f"Target WarmStart: {config.target_warmstart} | SGD Period: {config.sgd_period}")
        print(f"Base Measure: {base_measure.__class__.__name__}")
        print(f"Episodes: {config.num_episodes} | Device: {agent.device}")
        print("-" * 50)

    for ep in range(1, config.num_episodes + 1):
        agent.reset_episode()
        obs = env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done and step_count < config.max_episode_steps:
            action = agent.act(obs, eval_mode=False)
            next_obs, reward, done, info = env.step(action)
            agent.step(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward
            step_count += 1

        episode_returns.append(ep_reward)
        episode_lengths.append(step_count)

        # Regret tracking (especially for Deep Sea)
        ep_regret = 1.0 - ep_reward
        cumulative_regret += ep_regret
        avg_regret = cumulative_regret / ep
        history_regret.append(avg_regret)

        if t_learn is None and avg_regret < 0.9 and ep >= 10:
            t_learn = ep

        # Periodic Evaluation
        if ep % config.eval_frequency == 0 or ep == config.num_episodes:
            eval_ep_returns = []
            for _ in range(config.eval_episodes):
                e_obs = env.reset()
                e_done = False
                e_rew = 0.0
                e_steps = 0
                while not e_done and e_steps < config.max_episode_steps:
                    e_act = agent.act(e_obs, eval_mode=True)
                    e_next, r, e_done, _ = env.step(e_act)
                    e_rew += r
                    e_obs = e_next
                    e_steps += 1
                eval_ep_returns.append(e_rew)

            mean_eval = float(np.mean(eval_ep_returns))
            eval_returns.append(mean_eval)
            eval_episodes.append(ep)

            if config.verbose:
                avg_train = np.mean(episode_returns[-config.eval_frequency:])
                elapsed = time.time() - t_start
                print(
                    f"Ep {ep:4d}/{config.num_episodes} | Train Return: {avg_train:7.2f} | "
                    f"Eval Return: {mean_eval:7.2f} | Avg Regret: {avg_regret:5.3f} | Time: {elapsed:5.1f}s"
                )

        if callbacks:
            stats = {
                "episode": ep,
                "return": ep_reward,
                "length": step_count,
                "avg_regret": avg_regret,
            }
            for cb in callbacks:
                cb(ep, stats)

    total_time = time.time() - t_start
    if config.verbose:
        print("-" * 50)
        print(f"Training completed in {total_time:.2f}s.")
        if t_learn is not None:
            print(f"T_learn (episodes to average regret < 0.9): {t_learn}")
        print("=" * 50)

    return {
        "config": config.__dict__,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "eval_returns": eval_returns,
        "eval_episodes": eval_episodes,
        "avg_regret_history": history_regret,
        "cumulative_regret": cumulative_regret,
        "t_learn": t_learn,
        "total_training_time_s": total_time,
        "final_mean_return": float(np.mean(episode_returns[-min(100, len(episode_returns)):])),
    }
