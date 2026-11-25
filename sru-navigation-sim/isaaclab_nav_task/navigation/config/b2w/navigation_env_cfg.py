# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""B2W specific configuration for navigation environment."""

import copy
import os

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, ObservationTermCfg as ObsTerm, SceneEntityCfg

from isaaclab_nav_task.navigation.navigation_env_cfg import NavigationEnvCfg, RewardsCfg
import isaaclab_nav_task.navigation.mdp as mdp
from isaaclab_nav_task.navigation.mdp.custom_noise import DeltaTransformPathNoiseCfg

from isaaclab_nav_task.navigation.assets import B2W_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR  # isort: skip


LEG_JOINT_NAMES = [".*hip_joint", ".*thigh_joint", ".*calf_joint"]
WHEEL_JOINT_NAMES = [".*foot_joint"]


def insert_obs_terms(group, before: dict | None = None, after: dict | None = None) -> None:
    """Insert observation terms into an `ObsGroup` config, keeping `target_path` as the anchor.

    `ObservationManager` enumerates the group with `cfg.__dict__.items()`, so insertion order *is*
    concatenation order - and `ActorCriticSRU` slices the observation positionally as
    `[proprio, path, image]`. A term appended the normal way would land after `depth_image` and
    silently corrupt that split (and the critic's `[proprio, path, time, height, image]` layout).

    Terms in `before` go immediately before `target_path` (they belong to the proprioceptive block),
    terms in `after` immediately after it (they extend the path block).
    """
    before = before or {}
    after = after or {}
    existing = set(group.__dict__)
    before = {k: v for k, v in before.items() if k not in existing}
    after = {k: v for k, v in after.items() if k not in existing}
    rebuilt = []
    for name, value in list(group.__dict__.items()):
        if name == "target_path":
            rebuilt.extend(before.items())
        rebuilt.append((name, value))
        if name == "target_path":
            rebuilt.extend(after.items())
    group.__dict__.clear()
    group.__dict__.update(rebuilt)



@configclass
class B2WRewardsCfg(RewardsCfg):
    """B2W rewards: follow the planned path, with bounded autonomy to go around what is not on it.

    Design spec: `docs/agent_doing.md` section 2.5. Every term is multiplied by `dt` (0.2 s) by the
    reward manager and every `Episode_Reward/*` log entry is divided by `max_episode_length_s` (90 s).

    Per-episode budget (an ideal 25 m path is ~125 steps):

      progress_along_path : 1.2 per metre paid by the gated path tracker  ~+30
      progress (direct)   : 0.15 per metre of straight-line gain          ~+2.5
      goal_reached_bonus  : +25 one-shot on completing the 1.5 s dwell     +25
      reach_goal          : +2.0 per step inside 0.5 m (during the dwell)  ~+3
      path_deviation      : -0.15 per metre beyond a 0.5 m corridor        ~0 when on path
      time_out            : -50 on a true 90 s horizon timeout

    The path term now *dominates* the direct one (roughly 10:1 in favour of the path, instead of the
    previous 22:1 against it), which is what makes the path a binding constraint rather than a hint.
    """

    # --- path following ---
    # `per_metre=True` divides by dt so the weight is literally "return per metre of paid progress"
    # (the reward manager would otherwise multiply it back down by dt = 0.2).
    progress_along_path = RewTerm(
        func=mdp.progress_along_path,
        weight=1.2,
        params={"command_name": "robot_goal", "probability": 1.0, "per_metre": True},
    )
    path_deviation_penalty = RewTerm(
        func=mdp.path_deviation_penalty,
        weight=-0.15,
        params={"command_name": "robot_goal", "corridor_radius": 0.5, "cap": 1.5},
    )

    # --- goal ---
    # Kept small on purpose: this is the term that let the robot ignore the path entirely when it
    # outweighed the path reward 22:1 per episode.
    progress = RewTerm(func=mdp.progress_to_goal, weight=0.15, params={"command_name": "robot_goal"})
    reach_goal = RewTerm(
        func=mdp.reach_goal_direct,
        weight=2.0,
        params={"command_name": "robot_goal", "distance_threshold": 0.5},
    )
    goal_reached_bonus = RewTerm(
        func=mdp.goal_reached_bonus, weight=25.0, params={"command_name": "robot_goal"}
    )
    # Terminal shaping around the goal radius. `instant_arrive` drops the legacy "ever entered the
    # radius" latch; T_r 0.1 -> 0.2 halves the per-step peak. The soft term keeps its inherited
    # weight and shaping, but must also gate on the *current* distance: with the dwell rule the robot
    # can leave the goal and keep wandering, and a latched soft term would pay it 0.05/step for the
    # rest of a 90 s episode (up to +17.5 for one touch).
    reach_goal_xy_soft = RewTerm(
        func=mdp.reach_goal_xyz,
        weight=0.25,
        params={
            "command_name": "robot_goal",
            "sigmoid": 2.5,
            "T_r": 1.0,
            "probability": 0.01,
            "flat": False,
            "ratio": False,
            "instant_arrive": True,
        },
    )
    reach_goal_xy_tight = RewTerm(
        func=mdp.reach_goal_xyz,
        weight=0.75,
        params={
            "command_name": "robot_goal",
            "sigmoid": 0.25,
            "T_r": 0.2,
            "probability": 0.01,
            "flat": True,
            "ratio": False,
            "instant_arrive": True,
        },
    )

    # --- motion costs (unchanged: already below the reward for moving) ---
    action_rate_l1 = RewTerm(func=mdp.action_rate_l1, weight=-0.02)
    lateral_movement = RewTerm(func=mdp.lateral_movement, weight=-0.02)

    # 90 s horizon timeout. Still dominant over any single episode's motion penalties, but no longer
    # several times the whole path reward - it must not make a legitimate 40 m detour unaffordable.
    time_out_penalty = RewTerm(func=mdp.episode_time_out, weight=-50.0)


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

        # --- path pipeline v2 (docs/agent_doing.md stages B/C) ---------------------------------
        # Category mix: optimal paths are the majority now, so "follow the path" is learnable; the
        # previous [0, 0, 0.5, 0.5] never produced a single clean path.
        # `deepcopy`: `GlobalPathConfig` is a plain class, so the configclass machinery keeps the
        # *same instance* as the class default. Mutating it in place would leak v2 into any other
        # task config built later in the same process (e.g. a base Go2W one).
        path_cfg = copy.deepcopy(self.commands.robot_goal.path_cfg)
        self.commands.robot_goal.path_cfg = path_cfg
        path_cfg.path_pipeline = "v2"
        path_cfg.category_weights = [0.45, 0.25, 0.20, 0.10]
        # Measured B2W half-width (0.26 m, widest horizontal extent incl. wheels) + 0.05 m margin.
        # Deliberately below the PRM's own 4-cell (0.4 m) dilation, which misjudges real 0.8 m doors.
        path_cfg.seg_padding_m = 0.31

        # --- termination semantics ------------------------------------------------------------
        # Success requires holding the goal radius for 1.5 s (8 steps) with the counter reset on exit,
        # and a timeout only counts as success if the robot is inside the radius *on that step*.
        self.terminations.time_out.params = {"distance_threshold": 0.5, "strict_success": True}
        self.terminations.early_termination.params = {"distance_threshold": 0.5, "reset_on_exit": True}

        # --- horizon ---------------------------------------------------------------------------
        # 60 s at ~0.6 m/s covers ~35 m, i.e. barely more than the longest planned paths: every
        # legitimate detour was a race against the clock. 90 s gives ~55 m of budget.
        # NOTE: `Episode_Reward/*` is divided by `max_episode_length_s`, so per-episode reward
        # decompositions must now be multiplied by 90, not 60.
        self.episode_length_s = 90.0

        # --- observations: local path + distance-to-path ---------------------------------------
        # Both terms are *inserted* (not appended) to keep the network's positional layout
        # [proprio, path, image]. See `insert_obs_terms`.
        def local_path_term():
            return ObsTerm(
                func=mdp.generated_path_local_reshaped,
                params={"command_name": "robot_goal", "flatten": True, "horizon_m": 6.0},
                noise=DeltaTransformPathNoiseCfg(rotation=0.1, translation=0.5, noise_prob=0.1, remove_dist=False),
            )

        def distance_term():
            return ObsTerm(func=mdp.distance_to_path, params={"command_name": "robot_goal"})

        insert_obs_terms(
            self.observations.policy,
            before={"distance_to_path": distance_term()},
            after={"target_path_local": local_path_term()},
        )
        insert_obs_terms(
            self.observations.critic,
            before={"distance_to_path": distance_term()},
            after={"target_path_local": local_path_term()},
        )

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

        # Playback keeps the 60 s protocol so results stay comparable with earlier runs; training
        # uses the 90 s horizon (see `B2WNavigationEnvCfg`).
        self.episode_length_s = 60.0
