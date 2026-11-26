# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""The path-aware v2 environment layer, shared by the B2W and Go2W navigation tasks.

Why this module exists
----------------------
The redesign in `docs/agent_doing.md` changes six things at once (path pipeline, progress
accounting, reward balance, goal dwell, horizon, observations). Applying them by hand to each
task is exactly how the two robots drifted apart before: the R2B2W/Go2W configs are separate
files, and a change made in one silently missing from the other is invisible until a run is
finished.

`apply_path_aware_v2()` therefore holds the whole layer, and both tasks call it. Anything that is
genuinely robot-specific (rewards that name a sensor, terrain, action scaling, camera) stays in
each task's own config and is applied *after* this call.

What it does
------------
* path pipeline -> "v2": clearance-adaptive noise on sparse waypoints (endpoints locked), corner
  repair, structured blocking events, and per-segment feasibility/pinch masks
* path categories -> `[0.45, 0.25, 0.20, 0.10]` (optimal paths are the majority again)
* `seg_passable` padding -> 0.31 m (measured body half-width + 0.05, not the PRM's 0.4 m)
* termination -> 1.5 s consecutive goal dwell (counter reset on exit), timeout success judged on
  the timeout step itself
* horizon -> 90 s (the reward-log normaliser changes with it: `Episode_Reward/* x 90`)
* observations -> `distance_to_path` (1) and a 6 m local path (60) join the global path (60), for
  both the policy and the critic group
"""

from __future__ import annotations

import copy

import isaaclab_nav_task.navigation.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab_nav_task.navigation.mdp.custom_noise import DeltaTransformPathNoiseCfg

DEFAULT_PATH_CATEGORY_WEIGHTS = [0.45, 0.25, 0.20, 0.10]
DEFAULT_SEG_PADDING_M = 0.31
DEFAULT_EPISODE_LENGTH_S = 90.0
LOCAL_PATH_HORIZON_M = 6.0
# Lookahead used by both `heading_progress_to_path` (reward) and `path_heading_error` (observation):
# ~1.8 s of travel at 0.8 m/s, so the heading term leads the corner instead of chasing it.
HEADING_LOOKAHEAD_M = 1.5


def insert_obs_terms(group, before: dict | None = None, after: dict | None = None) -> None:
    """Insert observation terms into an `ObsGroup` config, keeping `target_path` as the anchor.

    `ObservationManager` enumerates the group with `cfg.__dict__.items()`, so insertion order *is*
    concatenation order - and `ActorCriticSRU` slices the observation positionally as
    `[proprio, path, image]`. A term appended the normal way would land after `depth_image` and
    silently corrupt that split (and the critic's `[proprio, path, time, height, image]` layout).

    Terms in `before` go immediately before `target_path` (they belong to the proprioceptive block),
    terms in `after` immediately after it (they extend the path block). Idempotent.
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


def apply_path_aware_v2(
    env_cfg,
    episode_length_s: float = DEFAULT_EPISODE_LENGTH_S,
    category_weights: list | None = None,
    seg_padding_m: float = DEFAULT_SEG_PADDING_M,
) -> None:
    """Apply the path-aware v2 layer to a navigation environment config, in place."""
    # --- path pipeline ---------------------------------------------------------------
    # `deepcopy`: `GlobalPathConfig` is a plain class, so the configclass machinery keeps the *same
    # instance* as the class default. Mutating it in place would leak v2 into any other task config
    # built later in the same process.
    path_cfg = copy.deepcopy(env_cfg.commands.robot_goal.path_cfg)
    env_cfg.commands.robot_goal.path_cfg = path_cfg
    path_cfg.path_pipeline = "v2"
    path_cfg.category_weights = list(category_weights or DEFAULT_PATH_CATEGORY_WEIGHTS)
    path_cfg.seg_padding_m = float(seg_padding_m)

    # --- termination ----------------------------------------------------------------
    # Success requires holding the goal radius for 1.5 s (8 steps) with the counter reset on exit,
    # and a timeout only counts as success if the robot is inside the radius on that step.
    if getattr(env_cfg.terminations, "time_out", None) is not None:
        env_cfg.terminations.time_out.params = {"distance_threshold": 0.5, "strict_success": True}
    if getattr(env_cfg.terminations, "early_termination", None) is not None:
        env_cfg.terminations.early_termination.params = {"distance_threshold": 0.5, "reset_on_exit": True}

    # --- horizon --------------------------------------------------------------------
    # 60 s at ~0.6 m/s covers ~35 m, i.e. barely more than the longest planned paths: every
    # legitimate detour was a race against the clock.
    # NOTE: `Episode_Reward/*` is divided by `max_episode_length_s`, so per-episode reward
    # decompositions must now be multiplied by 90, not 60.
    env_cfg.episode_length_s = float(episode_length_s)

    # --- observations ---------------------------------------------------------------
    def local_path_term():
        return ObsTerm(
            func=mdp.generated_path_local_reshaped,
            params={"command_name": "robot_goal", "flatten": True, "horizon_m": LOCAL_PATH_HORIZON_M},
            noise=DeltaTransformPathNoiseCfg(rotation=0.1, translation=0.5, noise_prob=0.1, remove_dist=False),
        )

    def distance_term():
        return ObsTerm(func=mdp.distance_to_path, params={"command_name": "robot_goal"})

    def heading_error_term():
        return ObsTerm(
            func=mdp.path_heading_error,
            params={"command_name": "robot_goal", "lookahead_m": HEADING_LOOKAHEAD_M},
        )

    for group in (env_cfg.observations.policy, env_cfg.observations.critic):
        insert_obs_terms(
            group,
            before={"path_heading_error": heading_error_term(), "distance_to_path": distance_term()},
            after={"target_path_local": local_path_term()},
        )
