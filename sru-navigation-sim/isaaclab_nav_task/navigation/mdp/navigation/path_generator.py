# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Global path generation: A* search over a :class:`Prm` roadmap, line-of-sight smoothing, and
path-category sampling (optimal / mildly non-optimal / highly non-optimal / infeasible).

:class:`MultiPath` owns one :class:`GlobalPath` per environment and is rebuilt whenever an
environment's goal/spawn is resampled - see
:class:`isaaclab_nav_task.navigation.mdp.navigation.goal_commands.RobotNavigationGoalCommand`.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np
import torch

from isaaclab.utils.math import transform_points

from .prm import MultiPRM, Prm


# Training iteration used to ramp the blocking-event probability (v2 pipeline). The goal command
# updates this once per step; a module global keeps `GlobalPath` free of an env reference (it is
# constructed from a `MultiPRM` and a `GlobalPathConfig` only, and is also used offline).
_PATH_CURRICULUM_ITERATION: int = 0


def set_path_curriculum_iteration(iteration: int) -> None:
    """Set the (approximate) training iteration the path pipeline ramps its curricula on."""
    global _PATH_CURRICULUM_ITERATION
    _PATH_CURRICULUM_ITERATION = int(iteration)


# --- v2 progress accounting (metres) ---
SYNC_RATE: float = 1.0
"""Max distance (m) the path tracker may follow the robot's projection in a single step."""

CONT_TOL: float = 0.5
"""Progress is only paid while the tracker is within this distance (m) of the projection."""

PAY_CAP: float = 1.5
"""Max progress (m) paid out on a single step, however far the tracker jumped."""

EPS: float = 0.04
"""Progress below this (m/step) is noise and is not paid."""

LOCAL_HORIZON_M: float = 6.0
"""Lookahead (m) covered by the local path observation."""


class GlobalPathConfig:
    """Configuration for global path generation."""

    def __init__(
        self,
        n_waypoints: int = 1,
        num_smooth_points: int = 15,
        add_noise: float = 10,  # noise level in height-map units, applied to infeasible paths
        z_offset: float = 0.5,  # z offset added to all path points (in world units)
        path_categories: list = ["optimal", "mildly_non_optimal", "highly_non_optimal", "infeasible"],
        category_weights: list = [0, 0, 0.5, 0.5],
        visualise: bool = False,
        # --- path pipeline v2 (see `path_pipeline`) ---
        path_pipeline: str = "legacy",
        seg_padding_m: float = 0.31,
        noise_max_m: float = 0.35,
        noise_clearance_frac: float = 0.5,
        noise_reject_tries: int = 5,
        pinch_clearance_m: float = 0.45,
        gate_normal_m: float = 0.8,
        gate_relaxed_m: float = 1.2,
        gate_invalid_min_m: float = 1.6,
        gate_invalid_push_margin: float = 0.6,
        block_prob_start: float = 0.15,
        block_prob_end: float = 0.30,
        block_prob_ramp_iters: int = 2000,
        num_dense_points: int = 64,
    ):
        self.n_waypoints = n_waypoints
        self.num_smooth_points = num_smooth_points
        self.add_noise = add_noise
        self.z_offset = z_offset
        self.path_categories = path_categories
        if not np.isclose(sum(category_weights), 1.0):
            raise ValueError(f"category_weights must sum to 1.0, got {sum(category_weights)}")
        self.category_weights = category_weights
        self.visualise = visualise

        # "legacy" keeps the original pipeline byte-for-byte (Go2W depends on it). "v2" is the
        # B2W path-aware redesign: clearance-adaptive noise on sparse points, locked endpoints,
        # structured local blocking events, per-segment feasibility masks and a metric progress
        # gate. See docs/agent_doing.md.
        self.path_pipeline = path_pipeline
        self.seg_padding_m = seg_padding_m
        self.noise_max_m = noise_max_m
        self.noise_clearance_frac = noise_clearance_frac
        self.noise_reject_tries = noise_reject_tries
        self.pinch_clearance_m = pinch_clearance_m
        self.gate_normal_m = gate_normal_m
        self.gate_relaxed_m = gate_relaxed_m
        self.gate_invalid_min_m = gate_invalid_min_m
        self.gate_invalid_push_margin = gate_invalid_push_margin
        self.block_prob_start = block_prob_start
        self.block_prob_end = block_prob_end
        self.block_prob_ramp_iters = block_prob_ramp_iters
        self.num_dense_points = num_dense_points

    def copy(self) -> "GlobalPathConfig":
        return GlobalPathConfig(
            n_waypoints=self.n_waypoints,
            num_smooth_points=self.num_smooth_points,
            add_noise=self.add_noise,
            z_offset=self.z_offset,
            path_categories=self.path_categories,
            category_weights=self.category_weights,
            visualise=self.visualise,
            path_pipeline=self.path_pipeline,
            seg_padding_m=self.seg_padding_m,
            noise_max_m=self.noise_max_m,
            noise_clearance_frac=self.noise_clearance_frac,
            noise_reject_tries=self.noise_reject_tries,
            pinch_clearance_m=self.pinch_clearance_m,
            gate_normal_m=self.gate_normal_m,
            gate_relaxed_m=self.gate_relaxed_m,
            gate_invalid_min_m=self.gate_invalid_min_m,
            gate_invalid_push_margin=self.gate_invalid_push_margin,
            block_prob_start=self.block_prob_start,
            block_prob_end=self.block_prob_end,
            block_prob_ramp_iters=self.block_prob_ramp_iters,
            num_dense_points=self.num_dense_points,
        )


class MultiPath:
    """Manager for one :class:`GlobalPath` per environment."""

    def __init__(self, num_envs: int, cfg_path: GlobalPathConfig, prm_manager: MultiPRM, device: torch.device):
        self.path_class = GlobalPath
        self.prm_manager = prm_manager
        self.num_envs = num_envs
        self.device = device
        self.size_path = cfg_path.num_smooth_points
        self.num_seg = max(1, self.size_path - 1)
        self.num_dense = cfg_path.num_dense_points
        self.metrics_v2 = cfg_path.path_pipeline == "v2"

        self.paths: list[GlobalPath] = []
        self.paths_tensor: torch.Tensor = torch.zeros((num_envs, self.size_path, 4), dtype=torch.float32, device=self.device)
        self.paths_w: torch.Tensor = torch.zeros((num_envs, self.size_path, 3), dtype=torch.float32, device=self.device)
        self.paths_unscaled: torch.Tensor = torch.zeros((num_envs, self.size_path, 3), dtype=torch.float32, device=self.device)
        self.last_best_progress: torch.Tensor = torch.zeros((num_envs,), dtype=torch.float32, device=self.device)
        self.closest_point: torch.Tensor = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self.final_lookahead: torch.Tensor = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self.final_lookahead_normalised: torch.Tensor = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)

        # --- v2 per-segment state (see `_store_path`) ---
        self.seg_passable: torch.Tensor = torch.ones((num_envs, self.num_seg), dtype=torch.bool, device=self.device)
        self.pinch_mask: torch.Tensor = torch.zeros((num_envs, self.num_seg), dtype=torch.bool, device=self.device)
        self.push_per_seg: torch.Tensor = torch.zeros((num_envs, self.num_seg), dtype=torch.float32, device=self.device)
        self.seg_gate: torch.Tensor = torch.full(
            (num_envs, self.num_seg), float(cfg_path.gate_normal_m), dtype=torch.float32, device=self.device
        )
        self.relax_mask: torch.Tensor = torch.zeros((num_envs, self.num_seg), dtype=torch.bool, device=self.device)
        self.paths_dense_w: torch.Tensor = torch.zeros(
            (num_envs, self.num_dense, 3), dtype=torch.float32, device=self.device
        )
        self.cum_dense: torch.Tensor = torch.zeros((num_envs, self.num_dense), dtype=torch.float32, device=self.device)
        # Arc-length bookkeeping for the metric progress gate (v2): the tracked position and the
        # position already paid for, both in metres along the planned path.
        self.s_track: torch.Tensor = torch.zeros((num_envs,), dtype=torch.float32, device=self.device)
        self.s_reward: torch.Tensor = torch.zeros((num_envs,), dtype=torch.float32, device=self.device)
        self._last_inv_pos: Optional[torch.Tensor] = None
        self._last_inv_rot: Optional[torch.Tensor] = None
        self._last_metrics: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._last_idx_glob: Optional[torch.Tensor] = None
        self._last_advance: Optional[torch.Tensor] = None

        self.build_all(cfg_path.copy())

    def build_all(self, cfg_path: GlobalPathConfig, tile_ids: Optional[torch.Tensor] = None, starts: Optional[torch.Tensor] = None, goals: Optional[torch.Tensor] = None):
        """Build (or rebuild) paths for all environments.

        Args:
            cfg_path: path configuration.
            tile_ids: terrain-tile index per env (shape `(num_envs,)`), or `None` to build without a PRM yet.
            starts: start position per env, `(num_envs, 2)` in height-map coordinates.
            goals: goal position per env, `(num_envs, 2)` in height-map coordinates.
        """
        if cfg_path.num_smooth_points != self.size_path:
            raise RuntimeError(
                f"cfg_path.num_smooth_points ({cfg_path.num_smooth_points}) must match "
                f"MultiPath.size_path ({self.size_path})."
            )
        for env_id in range(self.num_envs):
            prm = self.prm_manager.get(int(tile_ids[env_id])) if tile_ids is not None else None
            start = tuple(starts[env_id].tolist()) if starts is not None else None
            goal = tuple(goals[env_id].tolist()) if goals is not None else None
            self.paths.append(self.path_class(cfg_path.copy(), env_id, prm, start, goal))
            self._store_path(env_id, cfg_path)
        print(f"[MultiPath] Built paths for {self.num_envs} environments.")

    def rebuild(self, cfg_path: GlobalPathConfig, starts: torch.Tensor, goals: torch.Tensor, tile_ids: torch.Tensor, env_ids: Sequence[int]):
        """Rebuild paths for the given environments.

        Args:
            cfg_path: path configuration.
            starts: start position per selected env, `(k, 2)`, world-frame (x, y) - rescaled to
                height-map coordinates internally via `Prm.rescale_points_to_heightmap`.
            goals: goal position per selected env, `(k, 2)`, world-frame (x, y).
            tile_ids: terrain-tile index per selected env, `(k,)`.
            env_ids: environment indices to rebuild paths for.
        """
        if starts.shape[1] != 2 or goals.shape[1] != 2:
            raise RuntimeError("starts and goals must be of shape (k, 2) representing world-frame (x, y).")

        if cfg_path.num_smooth_points != self.size_path:
            raise RuntimeError(
                f"cfg_path.num_smooth_points ({cfg_path.num_smooth_points}) must match "
                f"MultiPath.size_path ({self.size_path})."
            )

        for i, env_id in enumerate(env_ids):
            prm = self.prm_manager.get(int(tile_ids[i]))
            self.paths[env_id].prm = prm
            self.paths[env_id].modify_cfg(**cfg_path.__dict__)

            start_goal = torch.stack((starts[i], goals[i]), dim=0)
            start_goal = prm.rescale_points_to_heightmap(start_goal).detach().cpu().numpy()
            prm.reset_start_goal(start_goal[0], start_goal[1])

            self.paths[env_id].find_path()
            self._store_path(env_id, cfg_path)

    def _store_path(self, env_id: int, cfg_path: GlobalPathConfig) -> None:
        """Copy one env's freshly generated path (and its v2 per-segment state) into the batched tensors."""
        path = self.paths[env_id]
        self.paths_tensor[env_id, : cfg_path.num_smooth_points] = path.path_b
        self.paths_w[env_id, : cfg_path.num_smooth_points] = path.path_w
        self.paths_unscaled[env_id, : cfg_path.num_smooth_points] = path.path_b_unscaled
        self.last_best_progress[env_id] = 0.0
        self.s_track[env_id] = 0.0
        self.s_reward[env_id] = 0.0

        self.seg_passable[env_id] = torch.as_tensor(path.seg_passable, device=self.device, dtype=torch.bool)
        self.pinch_mask[env_id] = torch.as_tensor(path.pinch_mask, device=self.device, dtype=torch.bool)
        self.push_per_seg[env_id] = torch.as_tensor(path.push_per_seg, device=self.device, dtype=torch.float32)
        # Recompute the derived masks over the whole batch: they are (num_envs x 14) tensor ops, far
        # cheaper than maintaining a second per-env code path with different tensor ranks.
        self.relax_mask = self._compute_relax(self.seg_passable, self.pinch_mask)
        self.seg_gate = self._compute_gates(self.seg_passable, self.relax_mask, self.push_per_seg, cfg_path)

        # Dense path in world coordinates, plus its cumulative arc length (metres).
        if path.prm is not None and np.any(path.path_dense):
            dense_w = path.prm.rescale_points_to_world(
                torch.as_tensor(path.path_dense, dtype=torch.float32, device=self.device)
            )
            self.paths_dense_w[env_id] = dense_w
            seg = torch.norm(dense_w[1:] - dense_w[:-1], dim=-1)
            self.cum_dense[env_id, 0] = 0.0
            self.cum_dense[env_id, 1:] = torch.cumsum(seg, dim=0)

    def _compute_relax(self, seg_passable: torch.Tensor, pinch_mask: torch.Tensor) -> torch.Tensor:
        """Two-sided dilation of the invalid segments, excluding pinch (narrow doorway) neighbourhoods.

        Only expanding downstream would punish a robot that starts swerving 1-2 m early (its projection
        is still on the upstream, valid segment), which trains late "brake at the wall" behaviour.
        `torch.roll` is deliberately avoided: it would wrap segment 0 into the last segment's state.
        """
        inv = ~seg_passable
        pad = torch.zeros((inv.shape[0], 1), dtype=torch.bool, device=inv.device)
        left = torch.cat([pad, inv[:, :-1]], dim=1)
        right = torch.cat([inv[:, 1:], pad], dim=1)
        return (inv | left | right) & (~pinch_mask)

    def _compute_gates(
        self,
        seg_passable: torch.Tensor,
        relax_mask: torch.Tensor,
        push_per_seg: torch.Tensor,
        cfg_path: GlobalPathConfig,
    ) -> torch.Tensor:
        """Three-tier payment gate per segment (normal / relaxed / genuinely-infeasible)."""
        inv = ~seg_passable
        gate = torch.full_like(push_per_seg, float(cfg_path.gate_normal_m))
        gate = torch.where(relax_mask & ~inv, torch.full_like(gate, float(cfg_path.gate_relaxed_m)), gate)
        invalid_gate = torch.maximum(
            torch.full_like(gate, float(cfg_path.gate_invalid_min_m)),
            push_per_seg + float(cfg_path.gate_invalid_push_margin),
        )
        gate = torch.where(inv, invalid_gate, gate)
        return gate

    def get_relax_at_nearest(self) -> torch.Tensor:
        """Whether the nearest segment of each env is in the "relaxed" (post-obstacle) class."""
        if self._last_idx_glob is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.relax_mask.gather(1, self._last_idx_glob.unsqueeze(1)).squeeze(1)

    def get(self, env_id: int) -> "GlobalPath":
        """Access a specific `GlobalPath`."""
        return self.paths[env_id]

    def get_all_tensors(self) -> torch.Tensor:
        """All paths in body frame, `(num_envs, num_smooth_points, 4)` = `(direction_xyz, log_distance)`."""
        return self.paths_tensor

    def update(self, inv_pos: torch.Tensor, inv_rot: torch.Tensor, robot_pos_w: torch.Tensor, robot_quat_w: torch.Tensor) -> torch.Tensor:
        """Vectorized per-step update: transform all paths into the current body frame.

        Args:
            inv_pos: inverse robot position, `(num_envs, 3)`.
            inv_rot: inverse robot rotation, `(num_envs, 4)` quaternion.
            robot_pos_w: robot position in world frame, `(num_envs, 3)` (for marker visualization only).
            robot_quat_w: robot orientation in world frame, `(num_envs, 4)` (for marker visualization only).

        Returns:
            Mean distance from the robot to its path, across all environments (scalar, for logging).
        """
        path_b_unscaled = transform_points(self.paths_w, inv_pos, inv_rot)  # (B, N, 3)
        self.paths_unscaled.copy_(path_b_unscaled)
        self._last_inv_pos = inv_pos
        self._last_inv_rot = inv_rot

        distances = torch.norm(path_b_unscaled, dim=-1, keepdim=True) + 1e-9
        path_dirs = path_b_unscaled / distances
        path_logdist = torch.log(distances + 1.0)
        max_dist = torch.clamp(path_logdist.max(dim=1, keepdim=True)[0], min=1e-9)
        norm_dist = path_logdist / max_dist

        self.paths_tensor[:, :, :3].copy_(path_dirs)
        self.paths_tensor[:, :, 3:].copy_(norm_dist)

        self.compute_lookahead_and_closest()
        distance_look_ahead = torch.norm(self.final_lookahead[:, :3], dim=-1, keepdim=True) + 1e-6
        self.final_lookahead_normalised[:, :3] = self.final_lookahead[:, :3] / distance_look_ahead
        self.final_lookahead_normalised[:, 3:] = torch.log(distance_look_ahead + 1.0)

        distance_to_path = torch.norm(self.closest_point, dim=-1, keepdim=True)
        mean_distance_to_path = torch.mean(distance_to_path)
        return mean_distance_to_path

    def compute_lookahead_and_closest(self, lookahead_dist: float = 3.0):
        """Compute, per env, the closest point on the path and a fixed-distance lookahead point ahead of it.

        Populates `self.closest_point` and `self.final_lookahead`, both `(num_envs, 3)` in body frame.
        """
        B, N, D = self.paths_unscaled.shape
        if D == 2:
            raise RuntimeError("2D paths not supported yet.")

        last_point = self.paths_unscaled[:, -1, :]

        a = self.paths_unscaled[:, :-1, :]
        b = self.paths_unscaled[:, 1:, :]
        seg_vecs = b - a
        seg_lens = torch.norm(seg_vecs, dim=-1) + 1e-9

        t = (-torch.sum(a * seg_vecs, dim=-1)) / (seg_lens**2)
        t_clamped = torch.clamp(t, 0.0, 1.0)

        proj = a + t_clamped.unsqueeze(-1) * seg_vecs
        dist = torch.norm(proj, dim=-1)

        best_dist, best_idx = torch.min(dist, dim=1)
        closest_point = proj[torch.arange(B), best_idx]

        curr_seg = best_idx.clone()
        curr_t = t_clamped.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        remaining = torch.full_like(curr_t, lookahead_dist)
        active = torch.ones_like(curr_t, dtype=torch.bool)
        last_seg_index = seg_vecs.shape[1] - 1

        while active.any():
            seg_len_curr = seg_lens.gather(1, curr_seg.unsqueeze(1)).squeeze(1)
            remain_t = 1.0 - curr_t
            remain_len = remain_t * seg_len_curr

            at_last_seg = curr_seg == last_seg_index
            must_stop = at_last_seg & (remaining > remain_len)
            if must_stop.any():
                active[must_stop] = False
                remaining[must_stop] = -1e9
                curr_t[must_stop] = 1.0

            reached = (remaining <= remain_len) & active
            curr_t[reached] = curr_t[reached] + remaining[reached] / seg_len_curr[reached]
            active[reached] = False

            unfinished = active & (~reached) & (~must_stop)
            remaining = remaining - remain_len
            curr_seg = curr_seg + unfinished * 1
            curr_t[unfinished] = 0.0
            curr_seg = torch.clamp(curr_seg, max=last_seg_index)

        index = curr_seg.unsqueeze(1).unsqueeze(1).expand(-1, -1, D)
        seg_vec_final = seg_vecs.gather(1, index).squeeze(1)
        a_final = a.gather(1, index).squeeze(1)
        final_lookahead = a_final + curr_t.unsqueeze(-1) * seg_vec_final

        beyond_end = curr_t == 1.0
        final_lookahead[beyond_end] = last_point[beyond_end]

        self.closest_point = closest_point
        self.final_lookahead = final_lookahead

    def get_all_path_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Distance-to-path and progress-along-path, across all environments.

        **Not idempotent - call at most once per sim step.** This isn't a pure query: it mutates
        `s_track`/`s_reward` (v2) or `last_best_progress` (legacy) as it computes progress, so the
        returned `progress` is only meaningful relative to whenever this was last called. Calling it
        twice in the same step (e.g. once from a reward term and again from an observation/logging
        term) would silently zero out `progress` for the second caller. `RobotNavigationGoalCommand`
        therefore calls it exactly once per step and caches the result for every consumer.

        Returns:
            dists: distance from the robot to the closest point on its path, `(num_envs,)`.
            progress: v2 - metres advanced along the path and actually paid for on this step (0 if
                not payable, or on regression). Legacy - percent of the total path length.
        """
        if self.metrics_v2:
            return self._v2_path_metrics()
        return self._legacy_path_metrics()

    def _v2_path_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Metre-based progress with a rate-limited tracker and a three-tier payment gate.

        The tracker (`s_track`) follows the robot's projection onto the path at no more than
        `SYNC_RATE` m/step, so a projection that *jumps* across a U-shaped detour cannot bank metres it
        never walked; the only exception is a jump whose interior contains no passable segment (the
        robot genuinely went around a blocked one). Credit is paid at most `PAY_CAP` m/step and only
        while the robot is within `CONT_TOL` m of the tracker and inside the segment's gate.
        `s_reward` is ratcheted up even when nothing is paid - it never owes, and never pays back.
        """
        sync_rate, cont_tol, pay_cap, eps = SYNC_RATE, CONT_TOL, PAY_CAP, EPS

        a = self.paths_unscaled[:, :-1, :]
        b = self.paths_unscaled[:, 1:, :]
        seg_vecs = b - a
        seg_lens = torch.norm(seg_vecs, dim=-1) + 1e-9

        t = (-torch.sum(a * seg_vecs, dim=-1)) / (seg_lens**2)
        t_clamped = torch.clamp(t, 0.0, 1.0)
        proj = a + t_clamped.unsqueeze(-1) * seg_vecs
        dist = torch.norm(proj, dim=-1)
        d_glob, idx_glob = torch.min(dist, dim=1)

        cum_end = torch.cumsum(seg_lens, dim=1)
        cum_start = cum_end - seg_lens
        s_glob = (cum_start + t_clamped * seg_lens).gather(1, idx_glob.unsqueeze(1)).squeeze(1)

        num_seg = seg_lens.shape[1]
        idx_track = (cum_end < self.s_track.unsqueeze(1)).sum(dim=1).clamp(max=num_seg - 1)
        vc = torch.cat(
            [torch.zeros((seg_lens.shape[0], 1), device=seg_lens.device), torch.cumsum(self.seg_passable.float(), dim=1)],
            dim=1,
        )
        lo = torch.minimum(idx_track, idx_glob)
        hi = torch.maximum(idx_track, idx_glob)
        n_valid_interior = (
            vc.gather(1, hi.unsqueeze(1)) - vc.gather(1, (lo + 1).clamp(max=num_seg).unsqueeze(1))
        ).squeeze(1).clamp(min=0)
        jump_over_invalid = (n_valid_interior <= 0) & ((s_glob - self.s_track).abs() > sync_rate)

        rate = torch.where(jump_over_invalid, torch.full_like(s_glob, 1.0e9), torch.full_like(s_glob, sync_rate))
        s_track_new = self.s_track + torch.clamp(s_glob - self.s_track, min=-rate, max=rate)

        gate = self.seg_gate.gather(1, idx_glob.unsqueeze(1)).squeeze(1)
        payable = ((s_glob - s_track_new).abs() <= cont_tol) & (d_glob <= gate)

        advance = torch.clamp(s_track_new - self.s_reward, min=0.0, max=pay_cap)
        progress = torch.where(payable & (advance > eps), advance, torch.zeros_like(advance))

        self.s_reward = torch.maximum(self.s_reward, s_track_new)
        self.s_track = s_track_new
        self._last_idx_glob = idx_glob
        self._last_advance = advance
        return d_glob, progress

    def get_path_local(self, horizon_m: float = LOCAL_HORIZON_M, num_points: Optional[int] = None) -> torch.Tensor:
        """`(num_envs, num_points, 4)` body-frame path from the tracker forward `horizon_m` metres.

        Unlike the global 15-point path (which spans the whole route and therefore thins out to 2-3 m
        spacing on a 40 m path), this always covers the next few metres, so the policy sees the actual
        geometry it is about to drive. Encoded exactly like the global path: unit direction (xyz) plus
        log distance normalised by the largest distance in the row.
        """
        num_points = num_points or self.size_path
        if self._last_inv_pos is None:
            return torch.zeros((self.num_envs, num_points, 4), dtype=torch.float32, device=self.device)

        dense_b = transform_points(self.paths_dense_w, self._last_inv_pos, self._last_inv_rot)  # (B, M, 3)
        targets = self.s_track.unsqueeze(1) + torch.linspace(
            0.0, horizon_m, num_points, device=self.device
        ).unsqueeze(0)
        idx = torch.searchsorted(self.cum_dense.contiguous(), targets.contiguous()).clamp(1, self.num_dense - 1)
        i1 = idx
        i0 = idx - 1
        c0 = self.cum_dense.gather(1, i0)
        c1 = self.cum_dense.gather(1, i1)
        w = ((targets - c0) / (c1 - c0 + 1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
        p0 = dense_b.gather(1, i0.unsqueeze(-1).expand(-1, -1, 3))
        p1 = dense_b.gather(1, i1.unsqueeze(-1).expand(-1, -1, 3))
        local_b = p0 + w * (p1 - p0)

        distance = torch.norm(local_b, dim=-1, keepdim=True) + 1e-9
        log_distance = torch.log(distance + 1.0)
        max_log = torch.clamp(log_distance.max(dim=1, keepdim=True)[0], min=1e-9)
        return torch.cat([local_b / distance, log_distance / max_log], dim=-1)

    def _legacy_path_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Original percent-of-path progress (kept verbatim for the Go2W task)."""
        a = self.paths_unscaled[:, :-1, :]
        b = self.paths_unscaled[:, 1:, :]
        seg_vecs = b - a
        seg_lens = torch.norm(seg_vecs, dim=-1) + 1e-9

        t = (-torch.sum(a * seg_vecs, dim=-1)) / (seg_lens**2)
        t_clamped = torch.clamp(t, 0.0, 1.0)

        proj = a + t_clamped.unsqueeze(-1) * seg_vecs
        dist = torch.norm(proj, dim=-1)
        best_dist, best_idx = torch.min(dist, dim=1)

        cum_lens = torch.cumsum(seg_lens, dim=1)
        cum_lens_prev = torch.cat([torch.zeros((seg_lens.shape[0], 1), device=seg_lens.device), cum_lens[:, :-1]], dim=1)
        total_len = torch.sum(seg_lens, dim=1, keepdim=True)

        # Paper (Haro et al. 2026, eq. 12) defines progress in percent of the total path length.
        # Keeping it as a [0, 1] fraction made the 0.15 shortcut threshold mean 15% per step, which
        # never fires (a 0.2 s step at 1.5 m/s advances <3% of a 10-40 m path), so the term was inert.
        progress_along_path = 100.0 * (cum_lens_prev + t_clamped * seg_lens) / total_len
        best_progress = progress_along_path.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        progress_diff = torch.where(
            best_progress > self.last_best_progress,
            best_progress - self.last_best_progress,
            torch.zeros_like(best_progress),
        )
        best_progress = torch.where(best_progress < self.last_best_progress, torch.zeros_like(best_progress), best_progress)
        self.last_best_progress = torch.max(self.last_best_progress, best_progress)

        # epsilon = 0.15% of the path per step (paper eq. 12), filtering sensor/geometry noise
        progress = torch.where(progress_diff > 0.15, progress_diff, torch.zeros_like(progress_diff))

        return best_dist, progress


class GlobalPath:
    """A single environment's global path: A* over the PRM roadmap + line-of-sight smoothing."""

    prm: Optional[Prm]
    cfg: GlobalPathConfig
    waypoints: list[int]  # indices into prm.nodes
    path: list[tuple]  # path nodes in height-map coordinates
    path_w: torch.Tensor  # path nodes in world coordinates
    path_b: torch.Tensor  # path nodes in body frame, with log-distance as 4th element
    path_b_unscaled: torch.Tensor  # path nodes in body frame, without direction/distance scaling

    def __init__(
        self,
        cfg: GlobalPathConfig,
        env_id: int,
        prm: Optional[Prm] = None,
        start: Optional[tuple[int, int]] = None,
        goal: Optional[tuple[int, int]] = None,
    ) -> None:
        self.cfg = cfg
        self.env_id = env_id
        self.path_w = torch.zeros((cfg.num_smooth_points, 3), dtype=torch.float32)
        self.path_b = torch.zeros((cfg.num_smooth_points, 4), dtype=torch.float32)
        self.path_b_unscaled = torch.zeros((cfg.num_smooth_points, 3), dtype=torch.float32)
        self.path = []
        self.waypoints = []
        self.prm = None

        # v2 pipeline per-segment state. Legacy paths keep the permissive defaults below, which makes
        # them behave exactly as before (no relaxed gate, no corridor penalty, full payment).
        num_seg = cfg.num_smooth_points - 1
        self.seg_passable = np.ones(num_seg, dtype=bool)
        self.pinch_mask = np.zeros(num_seg, dtype=bool)
        self.push_per_seg = np.zeros(num_seg, dtype=np.float32)
        self.path_dense = np.zeros((cfg.num_dense_points, 3), dtype=np.float64)
        self.block_event: Optional[tuple[int, float]] = None
        self.category: Optional[str] = None
        self.a_star_failed: bool = False

        if prm is not None:
            self.prm = prm
            self.sample_waypoints()
        if start is not None and goal is not None:
            start_3d = (start[0], start[1], int(prm.height_map[start[0], start[1]]))
            goal_3d = (goal[0], goal[1], int(prm.height_map[goal[0], goal[1]]))
            self.prm.reset_start_goal(np.array(start_3d), np.array(goal_3d))
            self.find_path()

    def modify_cfg(self, **kwargs):
        """Modify the `GlobalPathConfig`."""
        for key, value in kwargs.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
            else:
                raise AttributeError(f"GlobalPathConfig has no attribute '{key}'")

    def sample_waypoints(self):
        """Sample `cfg.n_waypoints` random intermediate waypoints from the PRM's free nodes."""
        if self.cfg.n_waypoints == 0:
            return
        self.waypoints = random.sample(self.prm.free_nodes, self.cfg.n_waypoints)

    def find_path(self):
        """Find a new path from the PRM's current start/goal and store it in `self.path`/`self.path_w`.

        Samples a path category (optimal / mildly non-optimal / highly non-optimal / infeasible) per
        `cfg.path_categories`/`cfg.category_weights`, runs A* accordingly, smooths with line-of-sight,
        and (for infeasible paths) perturbs the result with noise.

        `cfg.path_pipeline == "v2"` routes to :meth:`_find_path_v2` instead, which keeps the same
        category semantics but replaces the noise/injection/feasibility stages (see that method).
        """
        if self.cfg.path_pipeline == "v2":
            return self._find_path_v2()

        if self.prm.cfg.start_idx is None or self.prm.cfg.goal_idx is None:
            raise RuntimeError("Start and goal indices must be set in the PRM before finding a path.")

        path = np.array([self.prm.nodes[self.prm.cfg.start_idx]])
        choice = random.choices(self.cfg.path_categories, weights=self.cfg.category_weights, k=1)[0]

        if choice == "optimal":
            optimal, los = True, True
        elif choice == "mildly_non_optimal":
            optimal, los = False, True
            if self.cfg.n_waypoints > 0:
                self.sample_waypoints()
        elif choice == "highly_non_optimal":
            optimal, los = False, False
            if self.cfg.n_waypoints > 0:
                self.sample_waypoints()
        elif choice == "infeasible":
            optimal, los = False, False
            if self.cfg.n_waypoints > 0:
                self.sample_waypoints()
        else:
            raise RuntimeError(f"Invalid path category choice: {choice}")

        path_segment = self.astar(self.prm.cfg.start_idx, self.prm.cfg.goal_idx, optimal=optimal)
        if path_segment.size != 0:
            if los:
                path = np.vstack((path, self.smooth_path_los(path_segment)[1:]))
            else:
                path = np.vstack((path, self.smooth_path_los_for_non_optimal(path_segment)[1:]))

        # A* failing to connect start->goal should not happen except in pathological cases; fall back
        # to a direct start->goal segment so the rest of the pipeline always has a valid path.
        if not np.array_equal(np.array(path[-1]), self.prm.nodes[self.prm.cfg.goal_idx]):
            print(f"[GlobalPath] WARNING: Path does not reach goal! Path length: {len(path)}")
            if len(path) == 1:
                path = np.vstack((path, self.prm.nodes[self.prm.cfg.goal_idx]))

        for i in range(len(path) - 1):
            if not self.prm.is_traversable(path[i], path[i + 1], check_padding=False):
                print(f"[GlobalPath] WARNING: Path segment from {i} to {i + 1} is not traversable! (path length: {len(path)})")

        path = self.resample_path(path, num_points=self.cfg.num_smooth_points)

        if not np.array_equal(path[-1], self.prm.nodes[self.prm.cfg.goal_idx]):
            raise RuntimeError(f"[GlobalPath] Path does not reach goal after resampling! Path length: {len(path)}")

        if choice == "infeasible":
            path = self.add_noise_to_path(path, noise_level=self.cfg.add_noise)

        path = [(x, y, z + self.cfg.z_offset / self.prm.v_scale) for (x, y, z) in path]
        self.path = path
        self.scale_path_to_world()
        if self.cfg.visualise:
            self.visualise(color=(1, 1, 1, 1))

    # =========================================================================
    # Path pipeline v2
    # =========================================================================

    def _find_path_v2(self) -> None:
        """v2 path pipeline: A* -> LOS smoothing -> corner-chord repair -> shaped noise -> blocking event.

        Differences from the legacy `find_path` (which stays untouched so Go2W is unaffected):

        * noise is applied to the **sparse** smoothed waypoints (before resampling), its amplitude is
          `min(noise_max_m, noise_clearance_frac * clearance)`, candidates are rejection-sampled, and
          the first/last waypoints are locked to the spawn/goal - so the path can no longer disagree
          with the goal, and noise never squeezes a waypoint through a wall;
        * the `infeasible` category no longer means "1 m of noise everywhere"; it means "GBFS path with
          no LOS smoothing plus one guaranteed structured blocking event" (see `inject_blocking_event`);
        * every one of the `num_smooth_points - 1` segments gets a feasibility / pinch mask and a
          payment gate, consumed by `MultiPath.get_all_path_metrics` and the reward terms.
        """
        if self.prm.cfg.start_idx is None or self.prm.cfg.goal_idx is None:
            raise RuntimeError("Start and goal indices must be set in the PRM before finding a path.")

        choice = random.choices(self.cfg.path_categories, weights=self.cfg.category_weights, k=1)[0]
        self.category = choice
        if choice == "optimal":
            optimal, los = True, True
        elif choice == "mildly_non_optimal":
            optimal, los = False, True
        elif choice in ("highly_non_optimal", "infeasible"):
            optimal, los = False, False
        else:
            raise RuntimeError(f"Invalid path category choice: {choice}")
        if not optimal and self.cfg.n_waypoints > 0:
            self.sample_waypoints()

        start_node = self.prm.nodes[self.prm.cfg.start_idx].astype(np.float64)
        goal_node = self.prm.nodes[self.prm.cfg.goal_idx].astype(np.float64)
        path_segment = self.astar(self.prm.cfg.start_idx, self.prm.cfg.goal_idx, optimal=optimal)

        self.a_star_failed = path_segment.size == 0
        if self.a_star_failed:
            print("[GlobalPath] WARNING: A* failed to find a path; falling back to a direct segment.")
            path = np.vstack([start_node, goal_node])
        elif los:
            smoothed, idx = self.smooth_path_los_with_indices(path_segment)
            path = self.repair_corner_chords(smoothed, path_segment, idx)
        else:
            path = self.smooth_path_los_for_non_optimal(path_segment)

        path = self._lock_endpoints(path)

        # 1) clearance-adaptive noise on the sparse waypoints (endpoints stay locked)
        path = self.add_shaped_noise_to_path(path)

        # 2) resample to the fixed waypoint count, endpoints still locked
        path = self.resample_path(path, num_points=self.cfg.num_smooth_points)
        path = self._lock_endpoints(path)

        # 3) at most one structured local blocking event per episode
        self.block_event = None
        if choice == "infeasible" or random.random() < self._block_probability():
            self.block_event = self.inject_blocking_event(path)

        # 4) per-segment feasibility / pinch masks (after noise + injection, before z_offset)
        self.compute_segment_masks(path)

        # 5) dense equal-arc-length path for progress tracking and the local observation
        self.build_dense_path(path)

        path = [(x, y, z + self.cfg.z_offset / self.prm.v_scale) for (x, y, z) in path]
        self.path = path
        self.scale_path_to_world()
        if self.cfg.visualise:
            self.visualise(color=(1, 1, 1, 1))

    def _lock_endpoints(self, path: np.ndarray) -> np.ndarray:
        """Force the first/last waypoints onto the spawn/goal PRM nodes."""
        path = np.array(path, dtype=np.float64, copy=True)
        path[0] = self.prm.nodes[self.prm.cfg.start_idx]
        path[-1] = self.prm.nodes[self.prm.cfg.goal_idx]
        return path

    def _block_probability(self) -> float:
        """Ramped probability of injecting a blocking event (0.15 -> 0.30 over `block_prob_ramp_iters`)."""
        ramp = max(1, self.cfg.block_prob_ramp_iters)
        progress = min(max(_PATH_CURRICULUM_ITERATION / ramp, 0.0), 1.0)
        return self.cfg.block_prob_start + (self.cfg.block_prob_end - self.cfg.block_prob_start) * progress

    def smooth_path_los_with_indices(self, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`smooth_path_los`, also returning which nodes of `path` each kept point came from.

        Unlike the legacy smoother, the line-of-sight test is the *body-width* `segment_passable`, not
        the PRM's 0.4 m cell dilation. A chord that clips the corner of a 0.8 m doorway can satisfy the
        cell test while being too narrow for the robot; rejecting it here keeps the doorway crossing
        straight, which is the one place the path must stay strict.
        """
        if path.size == 0:
            return path, np.zeros(0, dtype=np.int64)
        smoothed = [path[0]]
        idx = [0]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.prm.segment_passable(path[i], path[j], self.cfg.seg_padding_m):
                    break
                j -= 1
            smoothed.append(path[j])
            idx.append(j)
            i = j
        return np.array(smoothed), np.array(idx, dtype=np.int64)

    def repair_corner_chords(self, path: np.ndarray, raw: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Give an impassable smoothed chord a 50/50 chance of being repaired back into the corner.

        `smooth_path_los` cuts corners; a chord that is not actually passable for the body is exactly
        the "two endpoints free, everything in between blocked" signature a real obstacle produces.
        Half the time we restore the original A* corner nodes (a repaired, passable detour); the other
        half we leave the chord alone and let `compute_segment_masks` mark it invalid.
        """
        if len(path) < 2:
            return path
        out = [path[0]]
        for k in range(len(path) - 1):
            a, b = int(idx[k]), int(idx[k + 1])
            if b <= a + 1:
                out.append(path[k + 1])
                continue
            p1, p2 = path[k], path[k + 1]
            impassable = not self.prm.segment_passable(p1, p2, self.cfg.seg_padding_m)
            is_pinch = self.prm.segment_min_clearance(p1, p2) < self.cfg.pinch_clearance_m
            if impassable and not is_pinch and random.random() < 0.5:
                out.extend(raw[a + 1 : b + 1])
            else:
                out.append(path[k + 1])
        # drop consecutive duplicates (restored corners can coincide with the chord endpoints)
        deduped = [out[0]]
        for p in out[1:]:
            if np.linalg.norm(p[:2] - deduped[-1][:2]) > 1e-6:
                deduped.append(p)
        if len(deduped) < 2:
            deduped = [path[0], path[-1]]
        return np.array(deduped, dtype=np.float64)

    def add_shaped_noise_to_path(self, path: np.ndarray) -> np.ndarray:
        """Clearance-adaptive, rejection-sampled 2D noise on the interior waypoints.

        Amplitude is `min(noise_max_m, noise_clearance_frac * clearance(p_i))` (metres, converted to
        height-map cells), so a waypoint in a 0.8 m doorway gets (almost) no noise while an open-floor
        waypoint gets up to `noise_max_m`. Endpoints are never perturbed: the path's tail must agree
        with the goal the robot is actually rewarded for reaching.
        """
        h_scale = self.prm.h_scale
        noisy = np.array(path, dtype=np.float64, copy=True)
        for i in range(1, len(path) - 1):
            # Never perturb a waypoint that belongs to a pinch (narrow doorway / corridor) segment.
            # Rejection sampling alone is not enough: it validates the *sparse* segments, but after
            # resampling a nudged endpoint can still clip the corner of a 0.8 m doorway.
            if self.prm.segment_min_clearance(path[i - 1], path[i]) < self.cfg.pinch_clearance_m:
                continue
            if self.prm.segment_min_clearance(path[i], path[i + 1]) < self.cfg.pinch_clearance_m:
                continue
            clearance_m = self.prm.clearance_at(path[i])
            amp_m = min(self.cfg.noise_max_m, self.cfg.noise_clearance_frac * clearance_m)
            amp_cells = amp_m / h_scale
            if amp_cells <= 1e-3:
                continue
            for _ in range(max(1, self.cfg.noise_reject_tries)):
                r = amp_cells * np.sqrt(np.random.uniform(0.0, 1.0))
                theta = np.random.uniform(0.0, 2.0 * np.pi)
                candidate = path[i].copy()
                candidate[0] += r * np.cos(theta)
                candidate[1] += r * np.sin(theta)
                if self.prm.segment_passable(path[i - 1], candidate, self.cfg.seg_padding_m) and self.prm.segment_passable(
                    candidate, path[i + 1], self.cfg.seg_padding_m
                ):
                    noisy[i] = candidate
                    break
        return noisy

    def inject_blocking_event(self, path: np.ndarray) -> Optional[tuple[int, float]]:
        """Push one interior waypoint into the nearest wall, faking a newly appeared obstacle.

        This is the *only* way the pipeline produces a genuinely infeasible segment: the path is
        otherwise geometrically valid. Constraints from the design (all of them protect something
        specific):

        * clearance in [0.30, 1.00] m - close enough that pushing lands inside the obstacle, and far
          enough that there is room on the other side to detour;
        * at least 1.5 m of free space on the opposite side - a detour must exist;
        * no |dz| > 0.15 m over the three neighbouring waypoints - never inject on stairs or pits;
        * at least two segments away from any pinch segment - narrow doorways stay strictly valid.

        Returns `(waypoint_index, push_distance_m)` or `None`.
        """
        n = len(path)
        h_scale = self.prm.h_scale
        candidates = list(range(2, min(n - 2, 12) + 1))
        random.shuffle(candidates)
        for i in candidates:
            p = path[i]
            clearance_m = self.prm.clearance_at(p)
            if not (0.30 <= clearance_m <= 1.00):
                continue

            # never inject on stairs / pit edges
            dz = max(abs(path[j][2] - p[2]) for j in (i - 1, i, i + 1))
            if dz > 0.15 / self.prm.v_scale:
                continue

            # stay at least two segments away from any pinch segment
            near_pinch = False
            for j in range(max(0, i - 2), min(n - 1, i + 2)):
                if self.prm.segment_min_clearance(path[j], path[j + 1]) < self.cfg.pinch_clearance_m:
                    near_pinch = True
                    break
            if near_pinch:
                continue

            # perpendicular to the local path direction, pointing at the tighter side
            d = path[i + 1][:2] - path[i - 1][:2]
            norm = np.linalg.norm(d)
            if norm < 1e-6:
                continue
            perp = np.array([-d[1], d[0]]) / norm
            probe = np.array([perp[0], perp[1], 0.0]) * (0.2 / h_scale)
            if self.prm.clearance_at(p + probe) > self.prm.clearance_at(p - probe):
                perp = -perp

            # the opposite side must stay free for at least 1.5 m
            free_behind = True
            for s in np.linspace(0.2, 1.5, 7):
                behind = p + np.array([-perp[0], -perp[1], 0.0]) * (s / h_scale)
                if self.prm.clearance_at(behind) <= 0.0:
                    free_behind = False
                    break
            if not free_behind:
                continue

            push_m = clearance_m + 0.30
            new_xy = p[:2] + perp * (push_m / h_scale)
            if self.prm.clearance_at([new_xy[0], new_xy[1], p[2]]) != 0.0:
                continue

            # z is deliberately left untouched: re-reading the height map here would teleport the
            # waypoint onto the wall top or into a pit floor.
            path[i][0], path[i][1] = new_xy
            return (i, push_m)
        return None

    def compute_segment_masks(self, path: np.ndarray) -> None:
        """Per-segment feasibility, pinch and push masks over the final waypoint set."""
        n = len(path)
        num_seg = n - 1
        seg_passable = np.zeros(num_seg, dtype=bool)
        pinch_mask = np.zeros(num_seg, dtype=bool)
        push_per_seg = np.zeros(num_seg, dtype=np.float32)
        for i in range(num_seg):
            seg_passable[i] = self.prm.segment_passable(path[i], path[i + 1], self.cfg.seg_padding_m)
            pinch_mask[i] = self.prm.segment_min_clearance(path[i], path[i + 1]) < self.cfg.pinch_clearance_m

        if self.block_event is not None:
            i, push_m = self.block_event
            for seg in (i - 1, i):
                if 0 <= seg < num_seg:
                    seg_passable[seg] = False
                    push_per_seg[seg] = push_m

        self.seg_passable = seg_passable
        self.pinch_mask = pinch_mask
        self.push_per_seg = push_per_seg

    def build_dense_path(self, path: np.ndarray) -> None:
        """Equal-arc-length dense path in height-map coordinates, with `z_offset` already applied."""
        dense = self.resample_path(np.array(path, dtype=np.float64), self.cfg.num_dense_points)
        dense[:, 2] += self.cfg.z_offset / self.prm.v_scale
        self.path_dense = dense

    def euclidean(self, p1: np.ndarray, p2: np.ndarray, optimal: bool = True) -> float:
        """Distance heuristic between two points: true Euclidean distance if `optimal`, otherwise a
        blend biased towards the first sampled waypoint (to encourage non-optimal routes)."""
        assert len(p1) == len(p2) and len(p1) in (2, 3), "Points must both be 2D or 3D."
        true_distance = float(np.linalg.norm(p1 - p2))
        if optimal:
            return true_distance

        wp = self.prm.nodes[self.waypoints[0]]
        distance_waypoint = float(np.linalg.norm(p1 - wp))
        beta = 0.1
        return beta * true_distance + (1 - beta) * distance_waypoint

    def astar(self, start: int, goal: int, optimal: bool = True) -> np.ndarray:
        """A* search over the PRM roadmap, returning the path as an array of height-map coordinates.

        When `optimal=False`, this isn't really A* despite the `g_score`/`f_score` naming: edge costs
        are never accumulated into `tentative_g` (see below), so the search degenerates to greedy
        best-first search (GBFS), biased towards `self.waypoints[0]` via `euclidean`'s non-optimal
        heuristic. That's deliberate - it's what produces the intentionally suboptimal/meandering
        paths for the non-"optimal" path categories - but worth knowing before assuming `g_score`
        here means real path cost.
        """
        num_nodes = self.prm.nodes.shape[0]

        if not hasattr(self, "_came_from") or self._came_from.shape[0] != num_nodes:
            self._came_from = np.empty(num_nodes, dtype=np.int32)
        came_from = self._came_from
        came_from[:] = -1  # -1 means "no parent"

        if not hasattr(self, "_open_mask") or self._open_mask.shape[0] != num_nodes:
            self._open_mask = np.zeros(num_nodes, dtype=bool)
        open_mask = self._open_mask
        open_mask[:] = False
        open_mask[start] = True

        if not hasattr(self, "_g_score"):
            self._g_score = np.full(num_nodes, np.inf, dtype=np.float32)
        g_score = self._g_score
        g_score[:] = np.inf
        g_score[start] = 0.0

        if not hasattr(self, "_f_score"):
            self._f_score = np.full(num_nodes, np.inf, dtype=np.float32)
        f_score = self._f_score
        f_score[:] = np.inf
        f_score[start] = self.euclidean(self.prm.nodes[start], self.prm.nodes[goal], optimal=optimal)

        while open_mask.any():
            candidates = np.where(open_mask, f_score, np.inf)
            current = np.argmin(candidates)

            if current == goal:
                path_nodes = [current]
                while came_from[current] != -1:
                    current = came_from[current]
                    path_nodes.append(current)
                path_nodes.reverse()
                return self.prm.nodes[path_nodes]

            open_mask[current] = False
            for neighbor in self.prm.roadmap[current]:
                tentative_g = g_score[current]
                if optimal:
                    tentative_g += self.euclidean(self.prm.nodes[current], self.prm.nodes[neighbor])
                # else: no edge cost added - g_score stays 0 along the frontier, so ordering is driven
                # purely by f_score's (biased) heuristic below. This is what makes optimal=False GBFS,
                # not A* - see the docstring.

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.euclidean(self.prm.nodes[neighbor], self.prm.nodes[goal], optimal=optimal)
                    open_mask[neighbor] = True

        print("[GlobalPath] WARNING: A* failed to find a path.")
        return np.array([])

    def smooth_path_los(self, path: np.ndarray) -> np.ndarray:
        """Simplify a path by connecting waypoints directly whenever there's line-of-sight.

        From each waypoint, tries to jump as far ahead as possible (all the way to the end first,
        backing off until a traversable straight line is found).
        """
        if path.size == 0:
            return path
        assert path.ndim == 2 and path.shape[1] == 3, f"Expected path shape (N, 3), got {path.shape}"

        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1  # try to jump straight to the end
            while j > i + 1:
                if self.prm.is_traversable(path[i], path[j], check_padding=True):
                    break
                j -= 1
            smoothed_path.append(path[j])
            i = j
        return np.array(smoothed_path)

    def smooth_path_los_for_non_optimal(self, path: np.ndarray) -> np.ndarray:
        """Line-of-sight smoothing variant that keeps the path non-optimal.

        Unlike `smooth_path_los` (which jumps as far as possible), this walks forward one waypoint
        at a time and stops advancing as soon as line-of-sight to the next point would break,
        preserving more of the original (longer, less direct) route.
        """
        if path.size == 0:
            return path
        assert path.ndim == 2 and path.shape[1] == 3, f"Expected path shape (N, 3), got {path.shape}"

        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 2:
            j = i
            traversable = True
            while j < len(path) - 1:
                j += 1
                traversable = self.prm.is_traversable(path[i], path[j], check_padding=True)
                if not traversable:
                    break
            smoothed_path.append(path[j - 1])
            if (not traversable) and j - 1 == i:
                raise RuntimeError(f"Cannot progress from point {i} to {i + 1} in path smoothing!")
            i = j - 1
        smoothed_path.append(path[-1])
        return np.array(smoothed_path)

    def resample_path(self, path: np.ndarray, num_points: int) -> np.ndarray:
        """Resample a path to a fixed number of points, evenly spaced by arc length (straight-line interpolation)."""
        distances = np.cumsum(np.array([0] + [np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))]))
        total_dist = distances[-1]
        if total_dist == 0:
            first_pt = path[0]
            return np.repeat(first_pt[np.newaxis, :], num_points, axis=0)

        sample_distances = np.linspace(0, total_dist, num_points)
        resampled_path = []
        for d in sample_distances:
            idx = np.searchsorted(distances, d)
            if idx == 0:
                resampled_path.append(path[0])
            elif idx >= len(path):
                resampled_path.append(path[-1])
            else:
                t = (d - distances[idx - 1]) / (distances[idx] - distances[idx - 1])
                resampled_path.append((1 - t) * path[idx - 1] + t * path[idx])
        return np.array(resampled_path)

    def add_noise_to_path(self, path: np.ndarray, noise_level: float = 1) -> np.ndarray:
        """Add random 2D (x, y) noise to every waypoint, sampled uniformly within `noise_level` radius."""
        assert path.ndim == 2 and path.shape[1] == 3, f"Expected path shape (N, 3), got {path.shape}"

        n_points = path.shape[0]
        r = noise_level * np.sqrt(np.random.uniform(0, 1, size=n_points))
        theta = np.random.uniform(0, 2 * np.pi, size=n_points)

        noisy_path = path.copy()
        noisy_path[:, 0] += r * np.cos(theta)
        noisy_path[:, 1] += r * np.sin(theta)
        return noisy_path

    def scale_path_to_world(self) -> None:
        """Rescale `self.path` (height-map coordinates) to world coordinates, stored in `self.path_w`."""
        self.path_w = self.prm.rescale_points_to_world(torch.tensor(self.path, dtype=torch.float32))

    def visualise(self, color: tuple = (1, 1, 1, 1), print_wp: bool = False) -> None:
        from omni.isaac.debug_draw import _debug_draw

        rescaled_path = self.prm.rescale_points_to_world(torch.tensor(self.path, dtype=torch.float32)).tolist()
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.draw_lines(rescaled_path[:-1], rescaled_path[1:], [color] * len(rescaled_path[1:]), [2] * len(rescaled_path[1:]))

        if self.cfg.n_waypoints > 0 and print_wp:
            waypoint_coords = [self.prm.nodes[idx] for idx in self.waypoints]
            print("Waypoints (bin coords):", waypoint_coords)
