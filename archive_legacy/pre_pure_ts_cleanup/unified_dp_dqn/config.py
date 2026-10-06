"""Configuration dataclass for Unified DP-DQN."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DPDQNConfig:
    """Hyperparameter configuration for Unified Environment-Agnostic DP-DQN."""

    # Environment dimensions (auto-inferred or specified)
    state_dim: int = 6
    action_dim: int = 3

    # Neural Network Architecture
    hidden_dim: int = 50
    num_layers: int = 2
    use_layer_norm: bool = True
    activation: str = "relu"

    # Dirichlet Process Parameters
    alpha: float = 3.0                  # DP concentration parameter
    batch_size: int = 64                # Empirical batch size per training step
    candidate_batch_size: int = 256     # Candidate pool size for sampling
    base_measure_type: str = "uniform"  # "uniform", "haar", "gaussian", "zero"
    haar_angle: bool = False            # Enforce S^1 manifold Haar measure on angle states

    # Sampler: Vashishtha & Maillard (2025) recursive stick-breaking vs Unified Dirichlet
    sampler_type: str = "vashishtha_maillard"  # "vashishtha_maillard" or "dirichlet"
    w_min: float = 0.0                  # Minimum prior mass floor (e.g. 0.05 to prevent prior extinction)
    vm_prior_multiplier: float = 10.0   # K_prior = max(8, int(vm_prior_multiplier * alpha))
    use_td_info_gain_decay: bool = True # Enable cumulative TD surprise decay
    td_info_scale: float = 1.0          # Scaling factor sigma_TD for TD surprise

    # Optimization
    lr: float = 1e-3                    # Learning rate for Adam
    gamma: float = 0.99                 # Discount factor
    tau: float = 0.05                   # Continuous Polyak target tracking rate
    sgd_period: int = 2                 # Train online step every sgd_period environment steps
    buffer_capacity: int = 3000000       # Replay buffer capacity (3M steps prevents memory eviction)
    grad_clip: float = 1.0              # Maximum gradient norm

    # Episodic Thompson Sampling Warm-Start
    warmstart_steps: int = 2            # Number of fast gradient steps on q_net at episode reset

    # Stability & Exploration Enhancements
    use_decoupled_actor: bool = False   # Clone acting_net at reset so master q_net is not corrupted by prior noise
    use_double_dqn: bool = False        # Use Double DQN targets (decouple action selection from evaluation)
    use_stratified_replay: bool = False # Sample 50% swing-up and 50% balance data to prevent catastrophic forgetting
    w_min_end: Optional[float] = None   # Optional annealing target for w_min over training
    dp_online_sgd: bool = False         # Use DP Dirichlet sampler for online SGD updates during episodes

    # Reproducibility
    sample_once_per_episode: bool = False  # Sample DP posterior once at reset and hold hypothesis frozen (Pure TS)
    seed: Optional[int] = None
