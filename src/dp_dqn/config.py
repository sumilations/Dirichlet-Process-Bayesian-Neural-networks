"""Configuration dataclass for DP-DQN."""

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class DPDQNConfig:
    """Hyperparameter and environment configuration for Unified DP-DQN."""

    # Environment
    env_name: str = "deep_sea"          # "deep_sea", "riverswim", "gym:<env_id>"
    state_dim: int = 100                # Observation dimension (auto-inferred if env supports it)
    action_dim: int = 2                 # Discrete action count (auto-inferred if env supports it)
    deep_sea_size: int = 10             # N for Deep Sea grid (state_dim = N*N, action_dim = 2)

    # Neural Network Architecture
    hidden_dim: int = 64
    num_layers: int = 2
    use_layer_norm: bool = True
    activation: str = "relu"

    # Dirichlet Process Parameters
    alpha: float = 15.0                 # Concentration parameter: F ~ DP(alpha, F_0)
    batch_size: int = 64                # Minibatch size M for stick-breaking
    candidate_batch_size: int = 256     # Maximum empirical replay transitions to pool
    base_measure: str = "deep_sea"      # "deep_sea", "gaussian", "zero", or custom instance
    prior_reward_mean: float = 0.1      # Optimistic target reward scale in F_0 (default: 0.1)
    prior_reward_std: float = 0.05      # Optimistic target reward std in F_0 (default: 0.05)
    contraction_C: Optional[float] = None  # Scale factor C for posterior contraction (Method 1). If None, no contraction.
    use_buffer_size_denominator: bool = False  # If True, synthetic probability uses len(replay) as denominator: alpha / (alpha + |D|)
    trajectory_C: Optional[float] = None  # If set, prob_syn = alpha / (alpha + (|D| / trajectory_C))
    direct_replay_sample: bool = False  # If True, empirical transitions are sampled directly from full replay buffer (no candidate pool)

    # Hierarchical Bayesian Prior on DP Concentration Parameter alpha:
    # theta = P(syn) = alpha / (alpha + B) ~ Beta(a_0, b_0) => alpha ~ BetaPrime(a_0, b_0) * B
    use_bayesian_alpha: bool = False
    alpha_prior_a: float = 1.0           # Prior shape parameter a_0 (synthetic pseudo-count)
    alpha_prior_b: Optional[float] = None # Prior shape parameter b_0 (if None, set to (B / alpha_0) * a_0)
    alpha_evidence_scale: float = 1.0    # C_scale: scale factor for empirical batch evidence accumulation

    # Optimization
    lr: float = 1e-3                    # Learning rate for Adam
    gamma: float = 0.99                 # Discount factor
    tau: float = 0.01                   # Polyak target tracking rate
    sgd_period: int = 2                 # Perform online TD update every sgd_period environment steps
    episodic_sgd: bool = False          # Option B: Run all SGD updates in a single consolidated burst at episode end
    buffer_capacity: int = 100000       # Maximum transitions in replay buffer

    # Episodic Thompson Sampling (Target Warm-Start)
    target_warmstart: bool = True       # Warm-start target network at episode boundaries
    warmstart_steps: int = 10           # Number of fast gradient steps to perturb Q_target
    warmstart_lr_scale: float = 0.5     # LR multiplier for target network warm-start
    sample_once_per_episode: bool = True  # Canonical Pure TS: 1 DP random measure sampled per episode and fit for all warmstart steps

    # Variance Reduction & Fixed-Budget Sampling Options
    sampler_type: str = "vashishtha_maillard"  # "vashishtha_maillard" (default, Eq. 288) or "sethuraman"
    vm_prior_multiplier: float = 10.0          # For VM sampler: K_prior = max(8, int(vm_prior_multiplier * alpha))
    vm_replay_scaled_n: bool = False     # Scale statistical sample size N with replay buffer size
    vm_scale_post_discovery_only: bool = False # Scale N with replay only after positive reward discovered
    reward_gated_contraction: bool = False # Freeze warm-start perturbation once r > 0 exists in replay
    priority_positive_slot: bool = False   # Guarantee 1 slot in empirical batch for r > 0 transitions
    multi_candidate_k: int = 1           # Number of candidate warm-start perturbations evaluated at s_0 (M=1: standard)
    one_living_network: bool = False     # Use single in-place continuous Q-network (no throwaway clones)
    dp_sampled_target: bool = False      # Sample target network from independent DP draw at episode start (no Polyak)
    use_td_info_gain_decay: bool = False # Option 3: Decay prior using cumulative TD surprise (information gain)
    td_info_scale: float = 1.0           # Scaling factor for TD surprise information gain

    # Training & Evaluation
    num_episodes: int = 1000            # Total training episodes
    max_episode_steps: int = 500        # Maximum steps per episode
    seed: Optional[int] = 42            # Random seed
    eval_frequency: int = 50            # Evaluate agent every eval_frequency episodes
    eval_episodes: int = 5              # Number of evaluation episodes (greedy policy)
    verbose: bool = True                # Print progress to stdout
