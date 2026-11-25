# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom RSL-RL configuration classes for navigation tasks.

These config classes replace the standard Isaac Lab RL configs to support
custom network architectures used in navigation tasks with depth camera inputs.
"""

from dataclasses import MISSING
from typing import Literal, Optional

from isaaclab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks with navigation extensions."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    rnn_type: str = "lstm"
    """The type of RNN to use."""

    rnn_hidden_size: int = 256
    """The hidden size of the RNN."""

    rnn_num_layers: int = 1
    """The number of layers in the RNN."""

    dropout: float = 0.0
    """The dropout rate for the first layer of the actor and critic networks."""

    # Visual inputs
    num_cameras: int = 1
    """Number of depth cameras encoded into the observation (1 or 2)."""
    
    image_input_dims: tuple[int, int, int] = (64, 5, 8)
    """Encoded depth feature shape as (C, H, W)."""
    
    height_input_dims: tuple[int, int, int] = (64, 7, 7)
    """Encoded height scan feature shape as (C, H, W). Default is (64, 7, 7) for 64*7*7=3136 features."""

    path_length: int = 60
    """Flattened length of the path observation the network splits off, in floats.

    Must match `num_waypoints * 4` for the path terms actually configured (global path 15 * 4 = 60;
    adding the local path term makes it 30 waypoints = 120). The network assumes the path block sits
    immediately before the image block - adding path terms without bumping this silently moves them
    into the proprioceptive block instead.
    """

    # Action noise bounds, applied by ActorCriticSRU.update_distribution.
    # Left unbounded, log_std is pushed up by the entropy bonus with no restoring force: the B2W
    # run of 2026-11-23 reached std = 8.49 by iteration 3000, saturating tanh and making
    # action_rate_l1 the largest single cost. The defaults below apply to every task that does not
    # override them and keep the sampled std in [0.082, 1.22].
    log_std_min: float = -2.5
    """Lower bound on the policy's log standard deviation."""

    log_std_max: float = 0.2
    """Upper bound on the policy's log standard deviation."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = MISSING
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    min_clip_param: float = 0.0
    """Target of the `clip_schedule` anneal. Give it a positive value when annealing."""

    clip_schedule: str = "fixed"
    """Anneal `clip_param` from its initial value down to `min_clip_param` ("fixed" | "cosine" |
    "linear" | "exponential"). "fixed" keeps the constant value, which is what every config that
    does not ask otherwise gets. `value_clip_param` follows `clip_param` while annealing."""

    value_clip_param: float = 0.2
    """The value clipping parameter. Default is 0.2."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    min_entropy_coef: float = 0.0
    """Target of the `entropy_schedule` anneal."""

    entropy_hold_iters: int = 0
    """Iterations to hold `entropy_coef` before the anneal starts."""

    entropy_decay_span: int = 0
    """Iterations the anneal takes once the hold is over (0 = the rest of the run)."""

    entropy_schedule: str = "fixed"
    """Anneal `entropy_coef` from its initial value down to `min_entropy_coef` ("fixed" | "cosine"
    | "linear" | "exponential"). "fixed" keeps the constant value."""

    log_std_schedule: str = "fixed"
    """Move the *upper* clamp on the policy's `log_std` over training ("fixed" | "cosine").

    A one-off reduction does not work: `torch.clamp` passes zero gradient for out-of-bounds values, so
    a `log_std` sitting on the wall stays pinned there and never comes back down. Instead the wall
    itself moves: `log_std_max_initial` at step 0, `log_std_max_mid` at `log_std_anneal_mid`, and
    `log_std_max_final` at `log_std_anneal_end`, cosine-interpolated in between. "fixed" keeps the
    static `RslRlPpoActorCriticCfg.log_std_max`, which is what every other task gets."""

    log_std_anneal_start: int = 1000
    """Iteration at which the `log_std_max` schedule starts (held at the initial value before)."""

    log_std_anneal_mid: int = 5000
    """Iteration at which the wall reaches `log_std_max_mid`."""

    log_std_anneal_end: int = 8000
    """Iteration at which the wall reaches `log_std_max_final` and stays there."""

    log_std_max_initial: float = 0.1
    """`log_std_max` used at the start of the schedule."""

    log_std_max_mid: float = -0.6
    """`log_std_max` at `log_std_anneal_mid`."""

    log_std_max_final: float = -1.0
    """`log_std_max` from `log_std_anneal_end` on."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    min_learning_rate: float = 1.0e-7
    """Floor of the `schedule` anneal. `cosine` decays to this value instead of to zero."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: Optional[int] = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    reward_shifting_value: float = 0.0
    """The value to shift the reward by. Default is 0.0."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string."""

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all)."""

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is "model_.*.pt" (all)."""
