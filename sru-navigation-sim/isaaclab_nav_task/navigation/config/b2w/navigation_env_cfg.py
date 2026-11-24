# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""B2W specific configuration for navigation environment."""

import os

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg

from isaaclab_nav_task.navigation.navigation_env_cfg import NavigationEnvCfg, RewardsCfg
import isaaclab_nav_task.navigation.mdp as mdp

from isaaclab_nav_task.navigation.assets import B2W_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR  # isort: skip


LEG_JOINT_NAMES = [".*hip_joint", ".*thigh_joint", ".*calf_joint"]
WHEEL_JOINT_NAMES = [".*foot_joint"]


@configclass
class B2WRewardsCfg(RewardsCfg):
    """B2W rewards: official reward set, recalibrated so that moving always beats standing.

    Why: the official set pays -5.8 (`action_rate_l1`) -0.5 (`lateral_movement`) per episode for
    moving, while every one of those penalties is exactly 0 when the robot stands still, and the
    goal terms are gated to the last 4 s + 1% of steps - so "stand still for 60 s" was the return-
    optimal policy (measured: standing ~-5.6 vs a moving policy ~-13). The terms below follow the
    reward set that is already validated in-repo on Go2W (SR 0.08 -> 0.98 from scratch):
    dense single-point goal progress, 5x smaller smoothness penalties, and a timeout penalty that
    dominates the cost of moving.

    RewardManager multiplies every term by dt (0.2 s), so per episode:
      progress       : +1.5 per metre advanced towards the goal
      reach_goal     : +2.0 per step inside the 0.5 m goal radius (early-terminates after 4 s)
      action_rate_l1 : ~-1.2 (was ~-5.8)
      lateral        : ~-0.1
      time_out       : -20.0 on a true 60 s horizon timeout
    """

    # Dense, ungated progress towards the goal. `progress_to_goal` returns (d_prev - d_now) / step_dt,
    # so weight * step_dt = 1.5 return per metre. Shares `env._progress_prev_dist` with
    # `gated_progress_to_goal` - only one of the two may be active.
    progress = RewTerm(
        func=mdp.progress_to_goal,
        weight=1.5,
        params={"command_name": "robot_goal"},
    )
    # Sparse terminal bonus for actually entering the goal radius.
    reach_goal = RewTerm(
        func=mdp.reach_goal_direct,
        weight=10.0,
        params={"command_name": "robot_goal", "distance_threshold": 0.5},
    )

    # Lower the cost of moving below the reward for moving (official Go2W values).
    action_rate_l1 = RewTerm(func=mdp.action_rate_l1, weight=-0.02)
    lateral_movement = RewTerm(func=mdp.lateral_movement, weight=-0.02)

    # 60 s horizon timeout penalty. Must exceed the ~6.0 return a standing robot saves in
    # smoothness penalties, with margin for the noisier actions seen early in learning.
    time_out_penalty = RewTerm(func=mdp.episode_time_out, weight=-100.0)


@configclass
class B2WNavigationEnvCfg(NavigationEnvCfg):
    # Go2W's deployment config (`Go2WRewardsCfg`) inherits the base `RewardsCfg` and is untouched.
    # `Go2WNavigationEnvCfg_Paper` deliberately inherits this class, so it gets this reward set too.
    rewards: B2WRewardsCfg = B2WRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator
        from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config

        initialize_depth_noise_generator(robot_name="b2w", use_jit_precompiled=False)

        camera_config = get_camera_config("b2w")
        CAMERA_RESOLUTION = camera_config.resolution

        self.scene.robot = B2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.raycast_camera.offset.pos = (0.387, 0.0, 0.28)
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        self.terminations.base_contact.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*hip", ".*thigh"]), "threshold": 1.0}

        self.actions.velocity_command.low_level_position_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"], scale=0.5, use_default_offset=True)
        self.actions.velocity_command.low_level_velocity_action = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*foot_joint"], scale=5.0, use_default_offset=True)
        self.actions.velocity_command.low_level_policy_file = os.path.join(ISAACLAB_NAV_TASKS_ASSETS_DIR, "Policies", "locomotion", "b2w", "policy_b2w_new_2.pt")

        self.rewards.joint_acc_l2_joint.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES+WHEEL_JOINT_NAMES)}

        self.terminations.base_contact.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*hip", ".*thigh"]), "threshold": 1.0}

        self.events.randomize_low_pass_filter_alpha.params = {
            "alpha_range": (0.1, 0.6),
            "action_term": "velocity_command",
            "per_dimension": True,
            "alpha_range_vx": (0.1, 0.6),
            "alpha_range_vy": (0.1, 0.6),
            "alpha_range_omega": (0.1, 0.6),
        }

        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False

@configclass
class B2WNavigationEnvCfg_DEV(B2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False

@configclass
class B2WNavigationEnvCfg_PLAY(B2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
