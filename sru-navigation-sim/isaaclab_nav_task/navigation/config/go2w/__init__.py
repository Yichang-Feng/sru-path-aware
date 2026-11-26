# Copyright (c) 2022-2026, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Gym environment registration for Go2W navigation tasks (SRU-Odin 5-Phase Curriculum)."""

import gymnasium as gym

from . import agents, navigation_env_cfg

##
# Register Gym environments.
##

# Paper-controlled comparison: identical to the B2W paper task except for the robot.
gym.register(
    id="Isaac-Nav-MDPO-Go2W-Paper-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Paper,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPaperCfg,
    },
)


gym.register(
    id="Isaac-Nav-MDPO-Go2W-Paper-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Paper_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPaperDevCfg,
    },
)


# =============================================================================
# MDPO 5-Phase Environments (Primary Algorithm)
# =============================================================================

# Phase 1: Cold Start (0 -> 2,000 iter, flat ground + sparse obstacles)
gym.register(
    id="Isaac-Nav-MDPO-Go2W-Phase1-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase1,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPhase1Cfg,
    },
)

# Phase 2: Stabilization & precision stopping (2,000 -> 5,000 iter)
gym.register(
    id="Isaac-Nav-MDPO-Go2W-Phase2-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase2,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPhase2Cfg,
    },
)

# Phase 3: Stairs (10-15cm) & moderate mazes (5,000 -> 14,000 iter)
gym.register(
    id="Isaac-Nav-MDPO-Go2W-Phase3-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase3,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPhase3Cfg,
    },
)

# Phase 4: Full 20cm stairs, pits, and dense mazes (14,000 -> 24,000 iter)
gym.register(
    id="Isaac-Nav-MDPO-Go2W-Phase4-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase4,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPhase4Cfg,
    },
)

# Phase 5: Trajectory smoothing & fine refinement (24,000+ iter)
gym.register(
    id="Isaac-Nav-MDPO-Go2W-Phase5-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase5,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPhase5Cfg,
    },
)

# Default Standard Full-Complexity Task (Phase 4)
gym.register(
    id="Isaac-Nav-MDPO-Go2W-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase4,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPhase4Cfg,
    },
)

# PureMaze Refinement Task
gym.register(
    id="Isaac-Nav-MDPO-Go2W-PureMaze-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PureMaze,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPureMazeCfg,
    },
)

# Play/Evaluation Task
gym.register(
    id="Isaac-Nav-MDPO-Go2W-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPhase4Cfg,
    },
)

# Development/Debug Task
gym.register(
    id="Isaac-Nav-MDPO-Go2W-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerDevCfg,
    },
)

# Backward Compatibility Stage Aliases
gym.register(
    id="Isaac-Nav-MDPO-Go2W-Stage1-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase1,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPhase1Cfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-Go2W-Stage2-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase3,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPhase3Cfg,
    },
)


# =============================================================================
# PPO 5-Phase Environments (Baseline Algorithm)
# =============================================================================

# Phase 1: Cold Start (0 -> 2,000 iter)
gym.register(
    id="Isaac-Nav-PPO-Go2W-Phase1-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase1,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPhase1Cfg,
    },
)

# Phase 2: Stabilization & precision stopping (2,000 -> 5,000 iter)
gym.register(
    id="Isaac-Nav-PPO-Go2W-Phase2-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase2,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPhase2Cfg,
    },
)

# Phase 3: Stairs (10-15cm) & moderate mazes (5,000 -> 14,000 iter)
gym.register(
    id="Isaac-Nav-PPO-Go2W-Phase3-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase3,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPhase3Cfg,
    },
)

# Phase 4: Full 20cm stairs, pits, and dense mazes (14,000 -> 24,000 iter)
gym.register(
    id="Isaac-Nav-PPO-Go2W-Phase4-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase4,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPhase4Cfg,
    },
)

# Phase 5: Trajectory smoothing & fine refinement (24,000+ iter)
gym.register(
    id="Isaac-Nav-PPO-Go2W-Phase5-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase5,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPhase5Cfg,
    },
)

# Default Standard Full-Complexity Task (Phase 4)
gym.register(
    id="Isaac-Nav-PPO-Go2W-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase4,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPhase4Cfg,
    },
)

# PureMaze Refinement Task
gym.register(
    id="Isaac-Nav-PPO-Go2W-PureMaze-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PureMaze,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPureMazeCfg,
    },
)

# Play/Evaluation Task
gym.register(
    id="Isaac-Nav-PPO-Go2W-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPhase4Cfg,
    },
)

# Development/Debug Task
gym.register(
    id="Isaac-Nav-PPO-Go2W-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerDevCfg,
    },
)

# Backward Compatibility Stage Aliases
gym.register(
    id="Isaac-Nav-PPO-Go2W-Stage1-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase1,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPhase1Cfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Go2W-Stage2-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase3,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPhase3Cfg,
    },
)

# Warm-Start Fine-Tuning Task (Porting from B2W, strictly aligned with SRU-Odin)
gym.register(
    id="Isaac-Nav-PPO-Go2W-FineTune-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Phase1,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPOFineTuneRunnerCfg,
    },
)

# 40-80cm Low Obstacle Fine-Tuning Task (Real-aligned camera: body center +10cm, pitch=0)
gym.register(
    id="Isaac-Nav-PPO-Go2W-LowObs-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WLowObsNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPOLowObsRunnerCfg,
    },
)

# 40-80cm Low Obstacle Evaluation/Play Task
gym.register(
    id="Isaac-Nav-PPO-Go2W-LowObs-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WLowObsNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPOLowObsRunnerCfg,
    },
)

# Arrival Heading Control Fine-Tuning Tasks (MDPO & PPO)
gym.register(
    id="Isaac-Nav-MDPO-Go2W-Heading-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Heading,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPOHeadingRunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Go2W-Heading-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_Heading,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPOHeadingRunnerCfg,
    },
)

# =============================================================================
# Path-aware v2 aliases used by this isolated workspace.
#
# These ids now point at the v2 environment (`Go2WNavigationEnvCfg_PathAwareV2`): clearance-adaptive
# path noise with locked endpoints, structured blocking events, per-segment feasibility masks, the
# metre-based gated progress reward, the 1.5 s goal dwell, a 90 s horizon and the 2697/5834-dim
# observations - i.e. exactly what the B2W task trains with (see docs/specs/agent_doing.md). The previous
# 60-dim task is still reachable through the Phase/NoCurriculum ids, which are unchanged.
# =============================================================================
gym.register(
    id="Isaac-Nav-PPO-Go2W-PathAware-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PathAwareV2,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPathAwareV2Cfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Go2W-PathAware-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PathAwareV2_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPathAwareV2Cfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Go2W-PathAware-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PathAwareV2,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPathAwareV2DevCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-Go2W-PathAware-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PathAwareV2,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPathAwareV2Cfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-Go2W-PathAware-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PathAwareV2_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPathAwareV2Cfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-Go2W-PathAware-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PathAwareV2,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavMDPORunnerPathAwareV2DevCfg,
    },
)


# =============================================================================
# Paper-aligned Go2W path-aware aliases (single stage, no curriculum, ZED-X camera)
# =============================================================================
gym.register(
    id="Isaac-Nav-PPO-Go2W-PathAware-NoCurriculum-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PathAwareNoCurriculum,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPathAwareNoCurriculumCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Go2W-PathAware-NoCurriculum-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2WNavigationEnvCfg_PathAwareNoCurriculum_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2WNavPPORunnerPathAwareNoCurriculumCfg,
    },
)


