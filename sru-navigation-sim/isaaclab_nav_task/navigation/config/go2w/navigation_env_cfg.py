# Copyright (c) 2022-2026, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Go2W specific configuration for navigation environment (SRU-Odin 5-Phase Curriculum)."""

import os

from isaaclab.utils import configclass
from isaaclab.sensors import patterns
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm

from isaaclab_nav_task.navigation.navigation_env_cfg import NavigationEnvCfg, RewardsCfg
from isaaclab_nav_task.navigation.config.path_aware import apply_path_aware_v2
from isaaclab_nav_task.navigation.config.b2w.navigation_env_cfg import (  # isort: skip
    B2WNavigationEnvCfg,
    B2WRewardsCfg,
)
import isaaclab_nav_task.navigation.mdp as mdp

from isaaclab_nav_task.navigation.assets import GO2W_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR  # isort: skip
from isaaclab_nav_task.terrains import (
    GO2W_PHASE1_TERRAIN_CFG,
    GO2W_PHASE2_TERRAIN_CFG,
    GO2W_PHASE3_TERRAIN_CFG,
    GO2W_PHASE4_TERRAIN_CFG,
    GO2W_PHASE5_TERRAIN_CFG,
    apply_go2w_terrain_env_overrides,
)


LEG_JOINT_NAMES = [".*hip_joint", ".*thigh_joint", ".*calf_joint"]
WHEEL_JOINT_NAMES = [".*foot_joint"]


def _get_velocity_scale(default_vx: float = 1.5) -> list[float]:
    """Retrieve velocity scaling with environment variable override support (capped at 1.5 m/s for DreamWaQ)."""
    env_vx = os.environ.get("GO2W_MAX_SPEED", "").strip()
    vx = min(float(env_vx), 1.5) if env_vx else min(default_vx, 1.5)
    vy = float(os.environ.get("GO2W_MAX_LATERAL_SPEED", "0.3").strip())
    wz = float(os.environ.get("GO2W_MAX_YAW_RATE", "1.0").strip())
    return [vx, vy, wz]


def _apply_paper_depth_camera(env_cfg) -> None:
    """Apply the paper's ZED-X depth camera model to a Go2W environment config.

    The camera pose stays at the real Go2W mount; only the intrinsics and range are
    changed to match the paper: 105° H x 78° V over 64x40 and a 10 m maximum range.
    """
    env_cfg.scene.raycast_camera.pattern_cfg = patterns.PinholeCameraPatternCfg.from_ros_camera_info(
        fx=72.7025,
        fy=72.7025,
        cx=94.4457,
        cy=62.5424,
        width=192,
        height=120,
        downsample_factor=3,  # 192x120 -> 64x40
    )
    env_cfg.scene.raycast_camera.max_distance = 10.0


@configclass
class Go2WRewardsCfg(RewardsCfg):
    """Reward terms for Go2W navigation (Refactored: Gated potential + PBRS turning + L2 lateral suppression)."""

    # 1. 门控势能引导：机头对准目标才给推进奖励，严防螃蟹步刷分 (破冰与消灭 Timeout 核心)
    progress = RewTerm(
        func=mdp.gated_progress_to_goal,
        weight=1.5,
        params={"command_name": "robot_goal", "power": 2.0, "distance_threshold": 0.5},
    )
    # 2. 调头主动旋转势能激励：奖励转向对齐目标，消除掉头真空期劣势 (PBRS 保守场无刷分漏洞)
    heading_progress = RewTerm(
        func=mdp.heading_progress_to_goal,
        weight=0.3,
        params={"command_name": "robot_goal", "distance_threshold": 0.5},
    )
    # 3. 静态航向对齐奖励：朝向目标方向提供余弦激励（在到点 0.5m 圈内平滑衰减至 0，消除原地打转）
    heading = RewTerm(
        func=mdp.heading_alignment,
        weight=0.5,
        params={"command_name": "robot_goal", "distance_threshold": 0.5},
    )
    # 4. 目标触达直接奖励 (0.5m 内触发)
    reach_goal = RewTerm(
        func=mdp.reach_goal_direct,
        weight=10.0,
        params={"command_name": "robot_goal", "distance_threshold": 0.5},
    )
    # 5. 每步时间微小成本，鼓励高效移动
    time_cost = RewTerm(func=mdp.time_step_cost, weight=-0.01)

    # 6. 关节加速度惩罚涵盖腿部与轮毂关节
    joint_acc_l2_joint = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES)},
    )
    # 7. 动作平滑与侧移带死区二次方重罚 (允许微小踏步，严惩高速螃蟹步与侧滑)
    action_rate_l1 = RewTerm(func=mdp.action_rate_l1, weight=-0.02)
    lateral_movement = RewTerm(
        func=mdp.lateral_movement_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "deadzone": 0.05},
    )
    rot_movement = RewTerm(func=mdp.rot_movement, weight=-1e-5)
    episode_termination = RewTerm(func=mdp.is_terminated, weight=-50.0)
    # 8. 单向前视深度相机倒车惩罚 (与 SRU-Odin Go2 严格对齐)
    backward_movement_penalty = RewTerm(func=mdp.backward_movement_penalty, weight=-1.0)
    # 9. 到点精准制动惩罚 (在 0.5m 终点圆圈内惩罚线速度与旋转速度，强制机器狗完全停稳)
    velocity_at_goal = RewTerm(
        func=mdp.velocity_at_goal_penalty,
        weight=-1.0,
        params={"command_name": "robot_goal", "distance_threshold": 0.5},
    )

    # 10. 显式禁用原版 99% 时间掩码为 0 的稀疏项，消除干扰与死锁
    reach_goal_xy_soft = None
    reach_goal_xy_tight = None


@configclass
class Go2WNavigationEnvCfg(NavigationEnvCfg):
    rewards: Go2WRewardsCfg = Go2WRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator
        from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config

        initialize_depth_noise_generator(robot_name="go2w", use_jit_precompiled=False)

        camera_config = get_camera_config("go2w")
        CAMERA_RESOLUTION = camera_config.resolution

        self.scene.robot = GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Go2W root link is "base"
        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # RealSense optical center: +1 cm forward, +4 cm above the built-in Go2 camera,
        # centered laterally (assumes built-in camera at (0.275, 0.0, 0.05) in the base frame).
        self.scene.raycast_camera.offset.pos = (0.285, 0.0, 0.09)
        # 20 deg downward pitch, matching the current real-world mount.
        # Domain randomization in EventsCfg still perturbs this nominal pose by +/-5 deg.
        self.scene.raycast_camera.offset.rot = (0.9848078, 0.0, 0.1736482, 0.0)
        # Replace inherited ZED ray-cast pattern with RealSense D435 intrinsics.
        # Values are the 64x40 direct-resize equivalent of 87 deg H x 58 deg V at 640x480,
        # as documented in REALSENSE_DEPLOYMENT_EVAL_AND_PARAM_TUNING_PLAN.md.
        self.scene.raycast_camera.pattern_cfg = patterns.PinholeCameraPatternCfg.from_ros_camera_info(
            fx=33.720964,
            fy=36.080955,
            cx=32.0,
            cy=20.0,
            width=64,
            height=40,
            downsample_factor=1,
        )
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Apply dedicated Go2W rewards (strictly aligned with SRU-Odin)
        self.rewards = Go2WRewardsCfg()

        # 碰撞检测：涵盖 base 与所有 hip/thigh，阈值恢复为 SRU-Odin 原版的 1.0N
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*hip", ".*thigh"]),
            "threshold": 1.0,
        }

        # 跌落检测：紧缩至 -0.35m，准确捕获坑洞跌落
        self.terminations.terrain_fall.params = {"fall_height_threshold": -0.35}

        # 目标点高度范围对齐 Go2W 站高 (0.1, 0.4)m
        self.commands.robot_goal.goal_height_offset_range = (0.1, 0.4)
        self.commands.robot_goal.debug_vis = False

        # 对齐 SRU-Odin 原版每回合超时时间为 60 秒
        self.episode_length_s = 60.0

        # 单前视深度相机：永久锁定倒车惩罚（对齐 SRU-Odin Go2 单相机标准，禁止 1000 步后解除）
        self.curriculum.disable_backward_penalty = None

        # 动作空间与执行器配置（支持环境速度动态缩放）
        self.actions.velocity_command.policy_scaling = _get_velocity_scale(1.5)
        self.actions.velocity_command.low_level_position_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"], scale=1.0, use_default_offset=True
        )
        self.actions.velocity_command.low_level_velocity_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=[".*foot_joint"], scale=1.0, use_default_offset=True
        )
        # 底层 Locomotion 策略：使用 DreamWaQ Go2W 适配模型 (policy_go2w_dreamwaq_adapted.pt)
        self.actions.velocity_command.low_level_policy_file = os.path.join(
            ISAACLAB_NAV_TASKS_ASSETS_DIR, "Policies", "locomotion", "go2w", "policy_go2w_dreamwaq_adapted.pt"
        )

        # 匹配 DreamWaQ 执行器刚度与阻尼 (Kp=50.0, Kd=1.0)
        if hasattr(self.scene.robot, "actuators"):
            if "legs" in self.scene.robot.actuators:
                self.scene.robot.actuators["legs"].stiffness = 50.0
                self.scene.robot.actuators["legs"].damping = 1.0
            if "legs_calf" in self.scene.robot.actuators:
                self.scene.robot.actuators["legs_calf"].stiffness = 50.0
                self.scene.robot.actuators["legs_calf"].damping = 1.0
            if "wheels" in self.scene.robot.actuators:
                self.scene.robot.actuators["wheels"].stiffness = 0.0
                self.scene.robot.actuators["wheels"].damping = 0.5

        # 匹配 DreamWaQ 标称对称站姿与站高 (Z=0.40m, thigh=0.67, calf=-1.30)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.40)
        self.scene.robot.init_state.joint_pos = {
            ".*hip_joint": 0.0,
            ".*thigh_joint": 0.67,
            ".*calf_joint": -1.30,
            ".*foot_joint": 0.0,
        }

        # DreamWaQ 观测配置：输入 57 维 (无 base_lin_vel)，关闭高频关节角度噪声以防静止颤动
        self.observations.low_level_policy.base_lin_vel = None
        self.observations.low_level_policy.enable_corruption = False

        # 禁用大范围关节缩放随机化，防止 DreamWaQ 启动时趴地/反折关节
        self.events.reset_robot_joints = None

        # 动作滤波范围对齐 SRU-Odin 原版 (0.4, 0.9)
        self.events.randomize_low_pass_filter_alpha.params = {
            "alpha_range": (0.4, 0.9),
            "action_term": "velocity_command",
            "per_dimension": True,
            "alpha_range_vx": (0.4, 0.9),
            "alpha_range_vy": (0.4, 0.9),
            "alpha_range_omega": (0.4, 0.9),
        }

        # 课程机制：自适应晋升，默认初始行设为 0
        init_level = int(os.environ.get("GO2W_INIT_LEVEL", "0").strip())
        self.scene.terrain.max_init_terrain_level = init_level
        self.scene.terrain.terrain_generator = apply_go2w_terrain_env_overrides(
            GO2W_PHASE4_TERRAIN_CFG, default_diff=(0.3, 0.8)
        )
        self.scene.terrain.terrain_generator.curriculum = True


@configclass
class Go2WNavigationEnvCfg_Paper(B2WNavigationEnvCfg):
    """Paper-aligned Go2W task: only the robot and its low-level controller differ from B2W.

    Rewards, terrain, curriculum, observation corruption, action filtering and episode settings
    are inherited unchanged from :class:`B2WNavigationEnvCfg`.
    """

    def __post_init__(self):
        super().__post_init__()

        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator

        initialize_depth_noise_generator(robot_name="go2w", use_jit_precompiled=False)

        self.scene.robot = GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.raycast_camera.offset.pos = (0.285, 0.0, 0.09)
        self.scene.raycast_camera.offset.rot = (0.9848078, 0.0, 0.1736482, 0.0)
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base"

        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*hip", ".*thigh"]),
            "threshold": 1.0,
        }

        self.actions.velocity_command.low_level_position_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"], scale=1.0,
            use_default_offset=True
        )
        self.actions.velocity_command.low_level_velocity_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=[".*foot_joint"], scale=1.0, use_default_offset=True
        )
        self.actions.velocity_command.low_level_policy_file = os.path.join(
            ISAACLAB_NAV_TASKS_ASSETS_DIR, "Policies", "locomotion", "go2w", "policy_go2w_dreamwaq_adapted.pt"
        )

        if "legs" in self.scene.robot.actuators:
            self.scene.robot.actuators["legs"].stiffness = 50.0
            self.scene.robot.actuators["legs"].damping = 1.0
        if "legs_calf" in self.scene.robot.actuators:
            self.scene.robot.actuators["legs_calf"].stiffness = 50.0
            self.scene.robot.actuators["legs_calf"].damping = 1.0
        if "wheels" in self.scene.robot.actuators:
            self.scene.robot.actuators["wheels"].stiffness = 0.0
            self.scene.robot.actuators["wheels"].damping = 0.5

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.40)
        self.scene.robot.init_state.joint_pos = {
            ".*hip_joint": 0.0,
            ".*thigh_joint": 0.67,
            ".*calf_joint": -1.30,
            ".*foot_joint": 0.0,
        }

        self.observations.low_level_policy.base_lin_vel = None
        self.observations.low_level_policy.enable_corruption = False
        self.events.reset_robot_joints = None


@configclass
class Go2WNavigationEnvCfg_Paper_DEV(Go2WNavigationEnvCfg_Paper):
    """Smoke-test variant with a reduced terrain grid."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 30


# =============================================================================
# Phase 1: Cold Start (Flat + Sparse Obstacles, [0.0, 0.4], vx=1.5 m/s)
# =============================================================================

@configclass
class Go2WNavigationEnvCfg_Phase1(Go2WNavigationEnvCfg):
    """Phase 1: Cold-start exploration on flat ground + sparse obstacles."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.velocity_command.policy_scaling = _get_velocity_scale(1.5)
        self.scene.terrain.terrain_generator = apply_go2w_terrain_env_overrides(
            GO2W_PHASE1_TERRAIN_CFG, default_diff=(0.0, 0.4)
        )
        init_level = int(os.environ.get("GO2W_INIT_LEVEL", "0").strip())
        self.scene.terrain.max_init_terrain_level = init_level


# =============================================================================
# Phase 2: Stabilization (Same layout, curriculum promotion, vx=1.5 m/s)
# =============================================================================

@configclass
class Go2WNavigationEnvCfg_Phase2(Go2WNavigationEnvCfg):
    """Phase 2: Policy stabilization and convergence on easy terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.velocity_command.policy_scaling = _get_velocity_scale(1.5)
        self.scene.terrain.terrain_generator = apply_go2w_terrain_env_overrides(
            GO2W_PHASE2_TERRAIN_CFG, default_diff=(0.0, 0.4)
        )
        init_level = int(os.environ.get("GO2W_INIT_LEVEL", "0").strip())
        self.scene.terrain.max_init_terrain_level = init_level


# =============================================================================
# Phase 3: Intermediate Mazes + 10-15cm Stairs ([0.2, 0.6], vx=1.5 m/s)
# =============================================================================

@configclass
class Go2WNavigationEnvCfg_Phase3(Go2WNavigationEnvCfg):
    """Phase 3: Stairs climbing (10-15cm) and moderate mazes."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.velocity_command.policy_scaling = _get_velocity_scale(1.5)
        self.scene.terrain.terrain_generator = apply_go2w_terrain_env_overrides(
            GO2W_PHASE3_TERRAIN_CFG, default_diff=(0.2, 0.6)
        )
        init_level = int(os.environ.get("GO2W_INIT_LEVEL", "0").strip())
        self.scene.terrain.max_init_terrain_level = init_level


# =============================================================================
# Phase 4: High Difficulty 20cm Stairs + Dense Mazes + Pits ([0.3, 0.8], vx=1.5 m/s)
# =============================================================================

@configclass
class Go2WNavigationEnvCfg_Phase4(Go2WNavigationEnvCfg):
    """Phase 4: 20cm stairs, dense corridors, and pit edge navigation."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.velocity_command.policy_scaling = _get_velocity_scale(1.5)
        self.scene.terrain.terrain_generator = apply_go2w_terrain_env_overrides(
            GO2W_PHASE4_TERRAIN_CFG, default_diff=(0.3, 0.8)
        )
        init_level = int(os.environ.get("GO2W_INIT_LEVEL", "0").strip())
        self.scene.terrain.max_init_terrain_level = init_level


@configclass
class Go2WNavigationEnvCfg_PathAwareNoCurriculum(Go2WNavigationEnvCfg_Phase4):
    """Paper-aligned Go2W task: full 180-tile grid, no curriculum, ZED-X camera.

    Kept separate from Phase4 so the existing D435 / real-deployment task is unchanged.
    """

    def __post_init__(self):
        super().__post_init__()
        # Paper: all 180 procedurally generated terrains are available from the start;
        # no terrain curriculum and no reward-stage schedule.
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.difficulty_range = (0.5, 1.0)
        self.scene.terrain.max_init_terrain_level = 10  # clamped to all 6 difficulty rows
        _apply_paper_depth_camera(self)


# =============================================================================
# Path-aware v2 (docs/agent_doing.md) - the same layer the B2W task trains with
# =============================================================================


@configclass
class Go2WPathAwareRewardsCfg(B2WRewardsCfg):
    """The B2W path-aware reward set applied to Go2W unchanged.

    Every term is robot-agnostic (metres, seconds, action deltas), and keeping a single definition
    means a future rebalance cannot land on one robot and silently miss the other. Robot-specific
    costs (joint limits, low-level controller scaling) stay in the Go2W base config.
    """


@configclass
class Go2WNavigationEnvCfg_PathAwareV2(Go2WNavigationEnvCfg_Phase4):
    """Go2W with the path-aware v2 pipeline, rewards, termination, 90 s horizon and observations.

    Built on Phase 4 so the terrain (20 cm stairs, dense corridors, pits; difficulty 0.3-0.8) and
    the D435 camera are exactly what the existing path-aware task used - only the path-following
    layer changes.
    """

    rewards: Go2WPathAwareRewardsCfg = Go2WPathAwareRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # `Go2WNavigationEnvCfg.__post_init__` assigns `self.rewards = Go2WRewardsCfg()` unconditionally,
        # so the field default above is overwritten and the path-aware set has to be re-applied here.
        self.rewards = Go2WPathAwareRewardsCfg()

        # B2W parity on terrain coverage. Phase 4 keeps `curriculum=True` with
        # `max_init_terrain_level=GO2W_INIT_LEVEL` (default 0), and the curriculum manager has no
        # `update_terrain_levels` term wired - so every environment would stay on difficulty row 0
        # for the whole run. Turning the curriculum off spreads the environments over all six rows
        # and samples difficulty uniformly inside (0.3, 0.8), which is what the B2W task does.
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = 10  # clamped to the available rows

        apply_path_aware_v2(self)


@configclass
class Go2WNavigationEnvCfg_PathAwareV2_PLAY(Go2WNavigationEnvCfg_PathAwareV2):
    """Playback counterpart: same scene, 60 s horizon, no corruption/pushes, markers on."""

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
        # Playback is for inspection: show the goal/spawn markers and the planned path.
        self.commands.robot_goal.debug_vis = True

        # Playback keeps the 60 s protocol so results stay comparable with earlier runs.
        self.episode_length_s = 60.0


# =============================================================================
# Phase 5: Fine-Grained Refinement on Full Complexity (vx=1.5 m/s)
# =============================================================================
@configclass
class Go2WNavigationEnvCfg_Phase5(Go2WNavigationEnvCfg):
    """Phase 5: Trajectory smoothing and fine refinement for real-world deployment."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.velocity_command.policy_scaling = _get_velocity_scale(1.5)
        self.scene.terrain.terrain_generator = apply_go2w_terrain_env_overrides(
            GO2W_PHASE5_TERRAIN_CFG, default_diff=(0.3, 0.8)
        )
        init_level = int(os.environ.get("GO2W_INIT_LEVEL", "0").strip())
        self.scene.terrain.max_init_terrain_level = init_level


# =============================================================================
# PureMaze & Development & Play
# =============================================================================

@configclass
class Go2WNavigationEnvCfg_PureMaze(Go2WNavigationEnvCfg):
    """100% paper-style pure maze corridors (SRU-Odin special refinement task)."""

    def __post_init__(self):
        super().__post_init__()
        sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
        for name in list(sub_terrains.keys()):
            sub_terrains[name].proportion = 1.0 if name == "maze" else 0.0
        if "maze" in sub_terrains:
            sub_terrains["maze"].randomize_wall = False
            sub_terrains["maze"].random_wall_ratio = 0.0
            sub_terrains["maze"].add_stairs_to_maze = True

        self.actions.velocity_command.policy_scaling = _get_velocity_scale(1.5)
        init_level = int(os.environ.get("GO2W_INIT_LEVEL", "0").strip())
        self.scene.terrain.max_init_terrain_level = init_level


@configclass
class Go2WNavigationEnvCfg_DEV(Go2WNavigationEnvCfg_Phase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class Go2WNavigationEnvCfg_PLAY(Go2WNavigationEnvCfg):
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

        # Playback is for inspection: show goal/spawn markers and the planned A* path. Training
        # keeps the base config's `False` to avoid the per-step marker update cost.
        self.commands.robot_goal.debug_vis = True


@configclass
class Go2WNavigationEnvCfg_PathAwareNoCurriculum_PLAY(Go2WNavigationEnvCfg_PLAY):
    """Play counterpart of the paper-aligned Go2W task (same camera model)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.curriculum = False
        _apply_paper_depth_camera(self)


# Backward Compatibility Aliases
Go2WNavigationEnvCfg_Stage1 = Go2WNavigationEnvCfg_Phase1
Go2WNavigationEnvCfg_Stage2 = Go2WNavigationEnvCfg_Phase3
Go2WNavigationEnvCfg_Stage3 = Go2WNavigationEnvCfg_Phase4


@configclass
class Go2WLowObsNavigationEnvCfg(Go2WNavigationEnvCfg):
    """Go2W navigation env with real-aligned camera pose (body center +10cm, straight ahead) and 40-80cm obstacles."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab_nav_task.terrains import LOW_OBS_MAZE_TERRAIN_CFG

        # 1. Mount 40-80cm low obstacle terrain
        self.scene.terrain.terrain_generator = LOW_OBS_MAZE_TERRAIN_CFG

        # 2. Camera pose: body center +10cm (Z=+0.10m), straight ahead (Pitch=0 deg)
        self.scene.raycast_camera.offset.pos = (0.0, 0.0, 0.10)
        self.scene.raycast_camera.offset.rot = (1.0, 0.0, 0.0, 0.0)


@configclass
class Go2WLowObsNavigationEnvCfg_PLAY(Go2WNavigationEnvCfg_PLAY):
    """Play env for Go2W with 40-80cm obstacles and real-aligned camera pose."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab_nav_task.terrains import LOW_OBS_MAZE_TERRAIN_CFG

        self.scene.terrain.terrain_generator = LOW_OBS_MAZE_TERRAIN_CFG
        self.scene.raycast_camera.offset.pos = (0.0, 0.0, 0.10)
        self.scene.raycast_camera.offset.rot = (1.0, 0.0, 0.0, 0.0)


# =============================================================================
# Arrival Heading Control Fine-Tuning Environment Configs
# =============================================================================

@configclass
class Go2WHeadingRewardsCfg(Go2WRewardsCfg):
    """Reward terms for Go2W arrival heading control."""

    # 1. 到达目标点机身朝向对齐门控奖励 (d <= 1.0m 时平滑开启，d->0 时权重为 1.0)
    target_heading = RewTerm(
        func=mdp.target_heading_alignment,
        weight=1.0,
        params={"command_name": "robot_goal", "distance_threshold": 1.0},
    )

    # 2. 到达目标点制动解耦惩罚 (解耦线速度与角速度，允许未对准朝向时原地自转，对齐后施加角速度阻尼)
    velocity_at_goal = RewTerm(
        func=mdp.velocity_at_goal_heading_penalty,
        weight=-1.0,
        params={"command_name": "robot_goal", "distance_threshold": 0.5, "heading_tolerance": 0.1745},
    )

    # 3. 适度放宽侧移约束 (保留合理横移与原地踏步自由度，死区放宽至 0.10m/s，系数由 -0.5 降至 -0.1)
    lateral_movement = RewTerm(
        func=mdp.lateral_movement_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "deadzone": 0.10},
    )


from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_nav_task.navigation.mdp.custom_noise import DeltaTransformationNoiseCfg
from isaaclab_nav_task.navigation.navigation_env_cfg import ObservationsCfg


@configclass
class PolicyHeadingCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel_delayed, noise=Unoise(n_min=-0.2, n_max=0.2))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel_delayed, noise=Unoise(n_min=-0.1, n_max=0.1))
    projected_gravity = ObsTerm(func=mdp.projected_gravity_delayed, noise=Unoise(n_min=-0.1, n_max=0.1))
    last_action = ObsTerm(func=mdp.last_action)
    target_position = ObsTerm(
        func=mdp.generated_commands_reshaped_delayed,
        params={"command_name": "robot_goal", "flatten": True},
        noise=DeltaTransformationNoiseCfg(rotation=0.1, translation=0.5, noise_prob=0.1, remove_dist=False),
    )
    target_heading = ObsTerm(func=mdp.target_heading_delayed, params={"command_name": "robot_goal"})
    depth_image = ObsTerm(func=mdp.depth_image_noisy_delayed, params={"sensor_cfg": SceneEntityCfg("raycast_camera")})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CriticHeadingCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    last_action = ObsTerm(func=mdp.last_action)
    target_position = ObsTerm(
        func=mdp.generated_commands_reshaped, params={"command_name": "robot_goal", "flatten": True}
    )
    target_heading = ObsTerm(func=mdp.target_heading, params={"command_name": "robot_goal"})
    time_normalized = ObsTerm(func=mdp.time_normalized, params={"command_name": "robot_goal"})
    height_scan_critic = ObsTerm(
        func=mdp.height_scan_feat, params={"sensor_cfg": SceneEntityCfg("height_scanner_critic")}
    )
    depth_image = ObsTerm(func=mdp.depth_image_prefect, params={"sensor_cfg": SceneEntityCfg("raycast_camera")})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class MetricsHeadingCfg(ObsGroup):
    """Observations for heading-aware metrics tracking."""

    in_goal = ObsTerm(
        func=mdp.in_goal_with_heading,
        params={"distance_threshold": 0.4, "heading_threshold": 0.1745, "goal_cmd_name": "robot_goal"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ObservationsHeadingCfg(ObservationsCfg):
    policy: PolicyHeadingCfg = PolicyHeadingCfg()
    critic: CriticHeadingCfg = CriticHeadingCfg()
    metrics: MetricsHeadingCfg = MetricsHeadingCfg()


@configclass
class Go2WNavigationEnvCfg_Heading(Go2WNavigationEnvCfg):
    """Go2W navigation environment for arrival heading control fine-tuning."""

    rewards: Go2WHeadingRewardsCfg = Go2WHeadingRewardsCfg()
    observations: ObservationsHeadingCfg = ObservationsHeadingCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards = Go2WHeadingRewardsCfg()
        self.observations = ObservationsHeadingCfg()
        # CRITICAL: DreamWaQ requires 57-dim low-level observations (no base_lin_vel)
        self.observations.low_level_policy.base_lin_vel = None
        self.observations.low_level_policy.enable_corruption = False
        self.events.reset_robot_joints = None
        # Use heading-aware arrival termination (distance < 0.4m, heading error < 10 deg, settled)
        self.terminations.early_termination.func = mdp.at_goal_with_heading_navigation
        self.terminations.early_termination.params = {
            "distance_threshold": 0.4,
            "heading_threshold": 0.1745,  # ~10 degrees
            "goal_cmd_name": "robot_goal",
            "velocity_threshold": 0.50,
            "fixed_dwell_time": 5.0,
            "required_align_steps": 2,
        }
        # Use heading-aware timeout termination (record success only if heading aligned)
        self.terminations.time_out.func = mdp.time_out_with_heading_navigation
        self.terminations.time_out.params = {
            "distance_threshold": 0.4,
            "heading_threshold": 0.1745,  # ~10 degrees
            "goal_cmd_name": "robot_goal",
        }
        # Flat exploration terrain with sparse obstacles for smooth fine-tuning
        self.actions.velocity_command.policy_scaling = _get_velocity_scale(1.5)
        self.scene.terrain.terrain_generator = apply_go2w_terrain_env_overrides(
            GO2W_PHASE1_TERRAIN_CFG, default_diff=(0.0, 0.4)
        )
        init_level = int(os.environ.get("GO2W_INIT_LEVEL", "0").strip())
        self.scene.terrain.max_init_terrain_level = init_level

