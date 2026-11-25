# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""RSL-RL agent configurations for B2W navigation tasks."""

from isaaclab.utils import configclass

from isaaclab_nav_task.navigation.config.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class B2WNavMDPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """MDPO runner configuration for B2W navigation."""

    num_steps_per_env = 16
    max_iterations = 15000
    save_interval = 500
    logger = "wandb"
    seed = 60
    wandb_project = "isaaclab_nav_b2w"
    experiment_name = "b2w_navigation_mdpo"
    empirical_normalization = False
    reward_shifting_value = 0.05
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSRU",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_hidden_size=512,
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        dropout=0.2,
        num_cameras=1,
        image_input_dims=(64, 5, 8),  # depth image: 64 channels * 5 * 8 = 2560
        height_input_dims=(64, 7, 7),  # encoded height_scan_critic: 64*7*7 = 3136
        # 30 waypoints * 4 floats: 15 global path points + 15 local path points (6 m lookahead).
        path_length=120,
        # Bound the exploration noise (std in [0.082, 1.22]). `log_std_max` is only the *initial*
        # ceiling here: the MDPO log_std schedule below moves it down over training.
        log_std_min=-2.5,
        log_std_max=0.1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="MDPO",
        value_loss_coef=0.02,
        use_clipped_value_loss=True,
        clip_param=0.2,
        value_clip_param=0.2,
        # Entropy bonus annealed to zero: it must not keep pushing log_std up for the whole run.
        entropy_coef=0.001,
        entropy_schedule="cosine",
        min_entropy_coef=0.0,
        entropy_hold_iters=500,
        entropy_decay_span=3500,
        num_learning_epochs=5,
        num_mini_batches=4,
        # Cosine decay to a 1e-4 floor. NOT "adaptive": MDPO does not implement it and the LR would
        # silently stay frozen at its initial value for the whole run.
        learning_rate=1.0e-3,
        min_learning_rate=1.0e-4,
        schedule="cosine",
        gamma=0.999,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Moving ceiling on the action noise: 0.1 -> -0.6 (iter 1000-5000) -> -1.0 (iter 5000-8000).
        # A one-off clamp cannot work - log_std parks on the wall with zero gradient.
        log_std_schedule="cosine",
        log_std_anneal_start=1000,
        log_std_anneal_mid=5000,
        log_std_anneal_end=8000,
        log_std_max_initial=0.1,
        log_std_max_mid=-0.6,
        log_std_max_final=-1.0,
    )


@configclass
class B2WNavMDPORunnerDevCfg(B2WNavMDPORunnerCfg):
    """Development configuration for MDPO with reduced iterations."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "b2w_navigation_mdpo_dev"
        self.logger = "tensorboard"


@configclass
class B2WNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for B2W navigation."""

    num_steps_per_env = 16
    max_iterations = 15000
    save_interval = 500
    logger = "wandb"
    seed = 60
    wandb_project = "isaaclab_nav_b2w"
    experiment_name = "b2w_navigation_ppo"
    empirical_normalization = False
    reward_shifting_value = 0.05
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSRU",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_hidden_size=512,
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        dropout=0.2,
        num_cameras=1,
        image_input_dims=(64, 5, 8),  # depth image: 64 channels * 5 * 8 = 2560
        height_input_dims=(64, 7, 7),  # encoded height_scan_critic: 64*7*7 = 3136
        path_length=120,  # 15 global + 15 local path waypoints * 4 floats
        # Same noise bounds as the MDPO task, minus the moving-wall schedule (PPO has no hook).
        log_std_min=-2.5,
        log_std_max=0.1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.02,
        use_clipped_value_loss=True,
        clip_param=0.2,
        value_clip_param=0.2,
        entropy_coef=0.00375,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class B2WNavPPORunnerDevCfg(B2WNavPPORunnerCfg):
    """Development configuration for PPO with reduced iterations."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "b2w_navigation_ppo_dev"
        self.logger = "tensorboard"
