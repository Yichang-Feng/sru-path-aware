"""Unit tests for the v2 path pipeline, the metric progress accounting and the goal dwell.

These are the acceptance tests from `docs/agent_doing.md` stages A4, B9 and C1. They run against a
synthetic height field and a hand-built roadmap, so they exercise the real `Prm` / `GlobalPath` /
`MultiPath` code without needing a full Isaac Lab environment.

Run with:
    cd IsaacLab && ./isaaclab.sh -p -m pytest ../sru-navigation-sim/tests -q
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_nav_task.navigation.mdp.navigation.path_generator import GlobalPath, GlobalPathConfig, MultiPath
from isaaclab_nav_task.navigation.mdp.navigation.prm import Prm, PrmConfig

H_SCALE = 0.1
V_SCALE = 0.005
FREE = 0
WALL = 300


# ---------------------------------------------------------------------------
# Synthetic terrain + roadmap helpers
# ---------------------------------------------------------------------------


def make_height_map(size_cells: int = 200) -> np.ndarray:
    """A flat free 20x20 m field with a vertical wall at x=100 that has a 0.8 m doorway.

    The doorway is 8 cells wide (0.8 m), i.e. exactly the "tight but passable for a 0.52 m body"
    case the whole design is built around.
    """
    h = np.zeros((size_cells, size_cells), dtype=np.int16)
    wall_col = size_cells // 2
    h[:, wall_col] = WALL
    door_y = size_cells // 2
    h[door_y - 4 : door_y + 4, wall_col] = FREE
    return h


def make_prm(height_map: np.ndarray) -> Prm:
    """Build a `Prm` without running the (slow) sampling-based constructor."""
    prm = Prm.__new__(Prm)
    prm.cfg = PrmConfig(fpr=0.0, fnr=0.0, padding=4, max_height_diff=50.0, N=10)
    prm._height_np = height_map
    prm.h_scale = H_SCALE
    prm.v_scale = V_SCALE
    prm.size = (height_map.shape[0] * H_SCALE, height_map.shape[1] * H_SCALE)
    prm.terrain_origin = torch.zeros(3)
    free = (height_map >= 0) & (height_map <= 10)
    prm.free_mask = torch.from_numpy(free)
    prm._free_np = free
    prm._padding_np = np.zeros_like(free)
    prm.compute_clearance_map()
    return prm


def build_grid_roadmap(prm: Prm, spacing: int = 10) -> dict:
    """A regular 1 m grid of free nodes, 4-connected where the segment is traversable."""
    rows, cols = prm._height_np.shape
    index: dict[tuple[int, int], int] = {}
    nodes = []
    for x in range(0, rows, spacing):
        for y in range(0, cols, spacing):
            if prm._free_np[x, y] and prm._clearance_np[x, y] >= 0.3:
                index[(x, y)] = len(nodes)
                nodes.append((float(x), float(y), 0.0))
    prm.nodes = np.array(nodes, dtype=np.float64)
    prm.free_nodes = list(range(len(nodes)))
    prm.roadmap = {i: [] for i in range(len(nodes))}
    for (x, y), i in index.items():
        for dx, dy in ((spacing, 0), (-spacing, 0), (0, spacing), (0, -spacing)):
            j = index.get((x + dx, y + dy))
            if j is None:
                continue
            if prm.is_traversable(prm.nodes[i], prm.nodes[j], check_padding=False):
                prm.roadmap[i].append(j)
                prm.roadmap[j].append(i)
    return index


def v2_cfg(**overrides) -> GlobalPathConfig:
    # n_waypoints=0: the synthetic roads have no sampled free-node list to bias GBFS towards, and
    # every test here uses the optimal (A* + LOS) category.
    cfg = GlobalPathConfig(n_waypoints=0, category_weights=[1.0, 0.0, 0.0, 0.0], path_pipeline="v2")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_optimal_path(**cfg_overrides):
    """An optimal (A* + LOS) path across the doorway on the synthetic terrain."""
    prm = make_prm(make_height_map())
    index = build_grid_roadmap(prm)
    prm.cfg.start_idx = index[(10, 10)]
    prm.cfg.goal_idx = index[(190, 190)]
    path = GlobalPath(v2_cfg(**cfg_overrides), env_id=0, prm=prm)
    path.find_path()
    return prm, path


# ---------------------------------------------------------------------------
# Stage B9: path pipeline
# ---------------------------------------------------------------------------


def test_endpoints_exactly_on_spawn_and_goal():
    """B9.1 - the first/last waypoints are the spawn/goal nodes, not a noisy approximation."""
    prm, path = make_optimal_path()
    start = prm.nodes[prm.cfg.start_idx]
    goal = prm.nodes[prm.cfg.goal_idx]
    # path points are height-map coordinates with z_offset added in z only
    assert path.path[0][0] == pytest.approx(start[0])
    assert path.path[0][1] == pytest.approx(start[1])
    assert path.path[-1][0] == pytest.approx(goal[0])
    assert path.path[-1][1] == pytest.approx(goal[1])
    assert len(path.path) == 15


def test_doorway_segment_stays_passable():
    """B9.3 - the most important regression: a genuine 0.8 m doorway must not be marked invalid."""
    prm, path = make_optimal_path()
    wall_y = prm._height_np.shape[1] // 2
    crossing = [
        i
        for i in range(len(path.path) - 1)
        if (path.path[i][1] - wall_y) * (path.path[i + 1][1] - wall_y) <= 0
    ]
    assert crossing, "expected at least one segment to cross the wall"
    for i in crossing:
        min_clearance = prm.segment_min_clearance(path.path[i], path.path[i + 1])
        assert min_clearance >= 0.3, f"segment {i} does not go through the doorway (clearance {min_clearance})"
        assert path.seg_passable[i], f"doorway segment {i} was marked impassable"
        assert path.pinch_mask[i], f"doorway segment {i} should be a pinch segment"


def test_injected_event_marks_segments_invalid_and_keeps_z():
    """B9.2 + B9.4 - an injected point makes its two segments invalid and never changes z."""
    height_map = np.zeros((200, 200), dtype=np.int16)
    height_map[:, 56:70] = WALL  # a 1.4 m thick wall 0.6 m to the +y side of the test path
    prm = make_prm(height_map)

    gp = GlobalPath(v2_cfg(), env_id=0, prm=prm)
    xs = np.linspace(20, 180, 15)
    path = np.stack([xs, np.full_like(xs, 50.0), np.zeros_like(xs)], axis=1)
    z_before = path[:, 2].copy()

    event = gp.inject_blocking_event(path)
    assert event is not None, "expected a blocking event to be injectable on this geometry"
    i, push_m = event
    assert 2 <= i <= 12
    assert push_m > 0.0
    assert prm.clearance_at(path[i]) == 0.0, "the pushed waypoint must land inside an obstacle"
    np.testing.assert_allclose(path[:, 2], z_before, atol=0.0)

    gp.block_event = event
    gp.compute_segment_masks(path)
    assert not gp.seg_passable[i - 1]
    assert not gp.seg_passable[i]
    assert gp.push_per_seg[i - 1] == pytest.approx(push_m)
    assert gp.push_per_seg[i] == pytest.approx(push_m)


def test_noise_keeps_endpoints_and_avoids_narrow_gaps():
    """B9.1 (noise) - interior points are perturbed, endpoints are exact, doorways stay usable."""
    prm = make_prm(make_height_map())
    gp = GlobalPath(v2_cfg(), env_id=0, prm=prm)
    xs = np.linspace(20, 180, 15)
    path = np.stack([xs, np.full_like(xs, 50.0), np.zeros_like(xs)], axis=1)

    noisy = gp.add_shaped_noise_to_path(path)

    np.testing.assert_allclose(noisy[0], path[0])
    np.testing.assert_allclose(noisy[-1], path[-1])
    np.testing.assert_allclose(noisy[:, 2], path[:, 2])
    # open-floor waypoints are allowed to move by up to `noise_max_m`
    max_shift_m = float(np.max(np.linalg.norm((noisy - path)[:, :2], axis=1)) * H_SCALE)
    assert max_shift_m <= 0.351
    # every noisy segment stays passable for the body
    for i in range(len(noisy) - 1):
        assert prm.segment_passable(noisy[i], noisy[i + 1], 0.31)


def test_out_of_map_points_are_rejected_not_crashing():
    """A point outside the tile must be refused, not raise IndexError.

    The clearance map does not taper at the tile border, so noise (or the straight spawn->goal
    fallback used when A* fails) can produce coordinates outside the height map. `line()` then yields
    cells past the array bounds - this crashed a 1046-environment Go2W run.
    """
    prm = make_prm(make_height_map())  # 200 x 200 cells
    p_in = np.array([10.0, 10.0, 0.0])
    p_out = np.array([10.0, 240.0, 0.0])

    assert not prm.in_bounds(p_out)
    assert prm.segment_passable(p_in, p_out, 0.31) is False
    assert prm.segment_min_clearance(p_in, p_out) >= 0.0  # clipped, does not raise


def test_noise_never_leaves_the_map():
    """Waypoints hugging the tile border keep their coordinates inside the map."""
    prm = make_prm(make_height_map())
    gp = GlobalPath(v2_cfg(), env_id=0, prm=prm)
    xs = np.linspace(20, 180, 15)
    path = np.stack([xs, np.full_like(xs, 3.0), np.zeros_like(xs)], axis=1)  # 0.3 m from the edge

    noisy = gp.add_shaped_noise_to_path(path)

    rows, cols = prm._height_np.shape
    assert np.all((noisy[:, 0] >= 0) & (noisy[:, 0] <= rows - 1))
    assert np.all((noisy[:, 1] >= 0) & (noisy[:, 1] <= cols - 1))


def test_relax_mask_is_not_wrapped():
    """B9.5 - an invalid *last* segment must not relax segment 0 (that would be `torch.roll`)."""
    mp = MultiPath.__new__(MultiPath)
    seg_passable = torch.tensor([[True] * 13 + [False]])
    pinch = torch.zeros(1, 14, dtype=torch.bool)
    relax = mp._compute_relax(seg_passable, pinch)
    assert not relax[0, 0], "segment 0 must not be relaxed by the last segment's state"
    assert relax[0, 12] and relax[0, 13]
    assert not relax[0, 11]

    gate = mp._compute_gates(seg_passable, relax, torch.zeros(1, 14), v2_cfg())
    assert gate[0, 0] == pytest.approx(0.8)  # untouched normal segment
    assert gate[0, 12] == pytest.approx(1.2)  # dilated neighbour of an invalid segment
    assert gate[0, 13] == pytest.approx(1.6)  # genuinely infeasible segment
    assert not relax[0, 0]


def test_gate_grows_with_push_distance():
    """The invalid gate is `max(1.6, push + 0.6)`, not a fixed constant."""
    mp = MultiPath.__new__(MultiPath)
    seg_passable = torch.tensor([[False] * 14])
    relax = torch.zeros(1, 14, dtype=torch.bool)
    push = torch.full((1, 14), 1.3)
    gate = mp._compute_gates(seg_passable, relax, push, v2_cfg())
    assert gate[0, 0] == pytest.approx(1.9)


def test_dense_path_is_equal_arc_length_and_locked():
    prm, path = make_optimal_path()
    dense = path.path_dense
    assert dense.shape[0] == 64
    # Same polyline, so the total arc length must match the 15-point path's (z differs by the
    # constant z_offset only, which cancels out of the differences).
    coarse = np.array([p for p in path.path], dtype=np.float64)
    coarse[:, 2] -= 0.5 / V_SCALE
    coarse_len = np.linalg.norm(np.diff(coarse, axis=0), axis=1).sum()
    dense_len = np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()
    assert dense_len == pytest.approx(coarse_len, rel=1e-3)
    # Equal arc-length sampling: consecutive steps vary only around corners, never more than 25%.
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    assert seg.max() < 1.25 * seg.mean()
    assert dense[0][0] == pytest.approx(path.path[0][0])
    assert dense[-1][0] == pytest.approx(path.path[-1][0])


# ---------------------------------------------------------------------------
# Stage C1: metric progress accounting
# ---------------------------------------------------------------------------


def make_multi_path(points, seg_passable=None, gate=0.8):
    """A `MultiPath` with only the state `_v2_path_metrics` needs."""
    mp = MultiPath.__new__(MultiPath)
    n = len(points)
    mp.num_envs = 1
    mp.device = torch.device("cpu")
    mp.size_path = n
    mp.num_seg = n - 1
    mp.num_dense = 64
    mp.metrics_v2 = True
    mp.paths_unscaled = torch.zeros(1, n, 3)
    mp.seg_passable = (
        torch.ones(1, n - 1, dtype=torch.bool) if seg_passable is None else torch.tensor([seg_passable], dtype=torch.bool)
    )
    mp.seg_gate = torch.full((1, n - 1), float(gate))
    mp.relax_mask = torch.zeros(1, n - 1, dtype=torch.bool)
    mp.s_track = torch.zeros(1)
    mp.s_reward = torch.zeros(1)
    mp._last_idx_glob = None
    mp._last_advance = None
    mp._world_points = np.asarray(points, dtype=np.float64)
    return mp


def place(mp, x, y=0.0):
    """Put the robot at world (x, y): the body-frame path is the world path minus that position."""
    shifted = mp._world_points[:, :2] - np.array([x, y])
    mp.paths_unscaled[0, :, :2] = torch.tensor(shifted, dtype=torch.float32)
    mp.paths_unscaled[0, :, 2] = 0.0


def straight_points(length_m=14.0, step=1.0):
    xs = np.arange(0.0, length_m + 1e-9, step)
    return np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)


def test_u_shape_jump_pays_nothing():
    """C1.1 - a 16 m projection jump across a U-shaped detour pays 0 (the old rule paid ~16 m)."""
    points = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [5.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    mp = make_multi_path(points)
    place(mp, 0.0, 0.0)
    _, paid_start = mp._v2_path_metrics()

    place(mp, 5.0, 1.0)  # teleport onto the opposite arm
    _, paid_jump = mp._v2_path_metrics()
    assert float(paid_start) == pytest.approx(0.0, abs=1e-6)
    assert float(paid_jump) == pytest.approx(0.0, abs=1e-6)


def test_jump_over_invalid_segments_is_capped_at_pay_cap():
    """C1.3 - a 2.6 m projection jump whose interior is entirely invalid pays exactly PAY_CAP."""
    points = straight_points()
    seg_passable = [True, False, False] + [True] * 10
    mp = make_multi_path(points, seg_passable=seg_passable)
    place(mp, 0.2)
    mp._v2_path_metrics()

    place(mp, 3.0)
    _, paid = mp._v2_path_metrics()
    assert float(paid) == pytest.approx(1.5)


def test_walking_along_path_pays_every_metre():
    """C1.4 - 100 steps of 0.12 m along a straight path pays ~12 m."""
    points = straight_points(length_m=20.0)
    mp = make_multi_path(points)
    place(mp, 0.0)
    mp._v2_path_metrics()

    total = 0.0
    x = 0.0
    for _ in range(100):
        x += 0.12
        place(mp, x)
        _, paid = mp._v2_path_metrics()
        total += float(paid)
    assert total == pytest.approx(12.0, abs=0.2)


def test_reversing_does_not_re_pay():
    """C1.5 - forward 1 m, back 1 m, forward 1 m pays 1 m in total."""
    points = straight_points()
    mp = make_multi_path(points)
    place(mp, 0.0)
    mp._v2_path_metrics()

    paid = 0.0
    for x in (0.4, 0.8, 1.0):
        place(mp, x)
        paid += float(mp._v2_path_metrics()[1])
    for x in (0.6, 0.2, 0.0):
        place(mp, x)
        paid += float(mp._v2_path_metrics()[1])
    for x in (0.4, 0.8, 1.0):
        place(mp, x)
        paid += float(mp._v2_path_metrics()[1])

    assert paid == pytest.approx(1.0, abs=0.1)


def test_tracker_always_converges():
    """C2 - `|s_track - s_glob|` decreases monotonically, i.e. the gate cannot deadlock."""
    points = straight_points()
    mp = make_multi_path(points)
    place(mp, 0.0)
    mp._v2_path_metrics()
    place(mp, 9.0)
    gaps = []
    for _ in range(12):
        mp._v2_path_metrics()
        gaps.append(abs(float(mp.s_track) - 9.0))
    assert all(b <= a + 1e-6 for a, b in zip(gaps, gaps[1:]))
    assert gaps[-1] < 1.0


def test_corridor_gate_blocks_payment_off_path():
    """Progress is not paid while the robot is outside the segment's gate."""
    points = straight_points()
    mp = make_multi_path(points, gate=0.8)
    place(mp, 0.0)
    mp._v2_path_metrics()

    place(mp, 0.5, y=1.2)  # 1.2 m off the path, outside the 0.8 m gate
    _, paid = mp._v2_path_metrics()
    assert float(paid) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Stage A4: goal dwell semantics
# ---------------------------------------------------------------------------


class FakeGoalCommand:
    def __init__(self, required_steps: int = 8):
        self.steps_at_goal = torch.zeros(1)
        self.time_at_goal = torch.zeros(1)
        self.required_steps_at_goal = required_steps
        self.goal_dwell_completed = torch.zeros(1, dtype=torch.bool)
        self.goal_reached_buffer = SimpleNamespace(add=lambda *a, **k: None)
        self.command_body = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

    def _get_unscaled_command(self):
        return self.command_body


def dwell_env(distance: float, goal: FakeGoalCommand):
    return SimpleNamespace(
        scene={"robot": object()},
        command_manager=SimpleNamespace(_terms={"robot_goal": goal}, get_term=lambda name: goal),
        step_dt=0.2,
        max_episode_length=450,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
    )


def set_distance(goal: FakeGoalCommand, distance: float):
    goal.command_body = torch.tensor([[distance, 0.0, 0.0, 0.0]])


def test_dwell_requires_consecutive_steps():
    """A4.1-3 - leaving the radius resets the counter; only 8 consecutive steps terminate."""
    from isaaclab_nav_task.navigation.mdp.terminations import at_goal_navigation

    goal = FakeGoalCommand()
    set_distance(goal, 0.4)
    env = dwell_env(0.4, goal)

    for _ in range(5):
        assert not bool(at_goal_navigation(env, reset_on_exit=True).item())
    assert int(goal.steps_at_goal.item()) == 5

    set_distance(goal, 3.0)  # leave the radius
    assert not bool(at_goal_navigation(env, reset_on_exit=True).item())
    assert int(goal.steps_at_goal.item()) == 0, "counter must reset on exit"

    set_distance(goal, 0.4)
    fired = [bool(at_goal_navigation(env, reset_on_exit=True).item()) for _ in range(8)]
    assert fired == [False] * 7 + [True], "termination must happen exactly on the 8th consecutive step"


def test_dwell_pulse_is_single_step():
    from isaaclab_nav_task.navigation.mdp.rewards import goal_reached_bonus
    from isaaclab_nav_task.navigation.mdp.terminations import at_goal_navigation

    goal = FakeGoalCommand()
    set_distance(goal, 0.4)
    env = dwell_env(0.4, goal)

    bonuses = []
    for _ in range(10):
        at_goal_navigation(env, reset_on_exit=True)
        bonuses.append(float(goal_reached_bonus(env)))

    nonzero = [b for b in bonuses if b > 0]
    assert len(nonzero) == 1, f"expected exactly one bonus pulse, got {bonuses}"
    # `weight * value * dt` must equal the configured 25.0 one-shot amount
    assert nonzero[0] * 0.2 * 25.0 == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Stage D4: reward economics
# ---------------------------------------------------------------------------


def test_reward_economics_moving_beats_standing():
    """D4 - completing the path must be worth far more than timing out, and the path must pay more
    than the direct-to-goal term (it paid 22x less before this redesign)."""
    from isaaclab_nav_task.navigation.config.b2w.navigation_env_cfg import B2WRewardsCfg

    cfg = B2WRewardsCfg()
    dt = 0.2
    weights = {name: term.weight for name, term in cfg.__dict__.items() if hasattr(term, "weight")}

    standing = weights["time_out_penalty"] * dt  # timeout on the horizon, nothing else happens
    path_reward = weights["progress_along_path"] * 25.0  # per_metre=True: weight * metres
    direct_reward = weights["progress"] * 20.0
    completing = path_reward + weights["goal_reached_bonus"] + weights["reach_goal"] * dt * 8 + direct_reward

    assert standing == pytest.approx(-10.0)
    assert completing > 50.0, completing
    assert path_reward > 5.0 * direct_reward, "the path term must dominate the direct-goal term"


# ---------------------------------------------------------------------------
# Stage E4: the training forward pass and the JIT export must agree
# ---------------------------------------------------------------------------

def test_jit_export_matches_training_forward():
    """E4 - exporting with 30 path waypoints must not silently re-slice the observation.

    The exporter derives `path_length`/`num_waypoints` from the module, so a mismatch can only come
    from the config (60 vs 120) or from the observation layout - exactly what this pins down.
    """
    import os
    import tempfile

    from rsl_rl.modules import ActorCriticSRU

    actor_obs_dim, critic_obs_dim = 2697, 5834  # the B2W v2 dimensions (see the report)
    actor_critic = ActorCriticSRU(
        actor_obs_dim,
        critic_obs_dim,
        3,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
        activation="elu",
        rnn_hidden_size=64,
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        dropout=0.0,
        num_cameras=1,
        image_input_dims=(64, 5, 8),
        height_input_dims=(64, 7, 7),
        path_length=120,
        init_noise_std=1.0,
    )
    actor_critic.eval()
    assert actor_critic.num_waypoints == 30
    assert actor_critic.actor_proprioceptive_input_dim == 17
    assert actor_critic.critic_proprioceptive_input_dim == 17

    # The JIT exporter owns a fixed batch-1 hidden state (it targets single-robot deployment), so the
    # comparison runs one observation at a time - that is a property of the exporter, not of the path
    # observation being checked here.
    obs = torch.randn(1, actor_obs_dim)
    with torch.no_grad():
        reference = actor_critic.act_inference(obs)
        with tempfile.TemporaryDirectory() as tmp:
            actor_critic.export_jit(path=tmp, filename="policy.pt")
            exported = torch.jit.load(os.path.join(tmp, "policy.pt"))
            exported_out = exported(obs)

    torch.testing.assert_close(exported_out, reference, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# The shared path-aware v2 layer: both robots must get exactly the same setup
# ---------------------------------------------------------------------------


def _obs_term_names(group):
    return [name for name in group.__dict__ if name not in (
        "enable_corruption", "concatenate_terms", "history_length",
        "flatten_history_dim", "concatenate_dim",
    )]


def test_b2w_and_go2w_get_identical_v2_layers():
    """Both tasks go through `apply_path_aware_v2`; assert the resulting layers are identical.

    This is the guard against the failure mode this module exists for: a change applied to one
    robot's config file and forgotten in the other's.
    """
    from isaaclab_nav_task.navigation.config.b2w.navigation_env_cfg import B2WNavigationEnvCfg
    from isaaclab_nav_task.navigation.config.go2w.navigation_env_cfg import (
        Go2WNavigationEnvCfg_PathAwareV2,
        Go2WNavigationEnvCfg_PathAwareV2_PLAY,
    )

    b2w = B2WNavigationEnvCfg()
    go2w = Go2WNavigationEnvCfg_PathAwareV2()

    for cfg in (b2w, go2w):
        path_cfg = cfg.commands.robot_goal.path_cfg
        assert path_cfg.path_pipeline == "v2"
        assert path_cfg.category_weights == [0.45, 0.25, 0.20, 0.10]
        assert path_cfg.seg_padding_m == pytest.approx(0.31)
        assert cfg.episode_length_s == pytest.approx(90.0)
        assert cfg.terminations.time_out.params["strict_success"] is True
        assert cfg.terminations.early_termination.params["reset_on_exit"] is True
        # All difficulty rows available from the start (the curriculum manager has no terrain-level
        # term wired), so a run is not silently pinned to the easiest row.
        assert cfg.scene.terrain.terrain_generator.curriculum is False
        assert cfg.scene.terrain.max_init_terrain_level == 10

        # Observation layout: distance_to_path belongs to the proprioceptive block (before
        # target_path), target_path_local extends the path block (after target_path).
        for group in (cfg.observations.policy, cfg.observations.critic):
            names = _obs_term_names(group)
            assert names.index("distance_to_path") == names.index("target_path") - 1
            assert names.index("target_path_local") == names.index("target_path") + 1

        # The path-aware reward set, not each robot's legacy one.
        weights = {n: t.weight for n, t in cfg.rewards.__dict__.items() if hasattr(t, "weight")}
        assert weights["progress_along_path"] == pytest.approx(1.2)
        assert weights["progress"] == pytest.approx(0.15)
        assert weights["goal_reached_bonus"] == pytest.approx(25.0)
        assert weights["path_deviation_penalty"] == pytest.approx(-0.15)
        assert weights["time_out_penalty"] == pytest.approx(-50.0)

        # `joint_acc_l2_joint` is inherited from the shared base `RewardsCfg`, whose regexes are the
        # AoW-D ones; every task that adopts the path-aware reward set must re-point them, otherwise
        # env construction dies with "Not all regular expressions are matched".
        joint_names = cfg.rewards.joint_acc_l2_joint.params["asset_cfg"].joint_names
        assert ".*HAA" not in joint_names and ".*WHEEL" not in joint_names
        assert ".*hip_joint" in joint_names and ".*foot_joint" in joint_names

    # Playback keeps the 60 s protocol on both robots.
    from isaaclab_nav_task.navigation.config.b2w.navigation_env_cfg import B2WNavigationEnvCfg_PLAY

    assert B2WNavigationEnvCfg_PLAY().episode_length_s == pytest.approx(60.0)
    assert Go2WNavigationEnvCfg_PathAwareV2_PLAY().episode_length_s == pytest.approx(60.0)


def test_go2w_v2_runners_match_the_b2w_schedule():
    from isaaclab_nav_task.navigation.config.b2w.agents.rsl_rl_cfg import (
        B2WNavMDPORunnerCfg,
        B2WNavPPORunnerCfg,
    )
    from isaaclab_nav_task.navigation.config.go2w.agents.rsl_rl_cfg import (
        Go2WNavMDPORunnerPathAwareV2Cfg,
        Go2WNavPPORunnerPathAwareV2Cfg,
    )

    b2w_mdpo, go2w_mdpo = B2WNavMDPORunnerCfg(), Go2WNavMDPORunnerPathAwareV2Cfg()
    assert go2w_mdpo.policy.path_length == 120 == b2w_mdpo.policy.path_length
    for field in (
        "schedule", "min_learning_rate", "entropy_schedule", "min_entropy_coef",
        "entropy_hold_iters", "entropy_decay_span", "log_std_schedule",
        "log_std_anneal_start", "log_std_anneal_mid", "log_std_anneal_end",
        "log_std_max_initial", "log_std_max_mid", "log_std_max_final",
    ):
        assert getattr(go2w_mdpo.algorithm, field) == getattr(b2w_mdpo.algorithm, field), field
    assert go2w_mdpo.algorithm.schedule != "adaptive", "MDPO has no adaptive schedule"

    b2w_ppo, go2w_ppo = B2WNavPPORunnerCfg(), Go2WNavPPORunnerPathAwareV2Cfg()
    assert go2w_ppo.policy.path_length == 120 == b2w_ppo.policy.path_length
    assert go2w_ppo.policy.log_std_max == pytest.approx(b2w_ppo.policy.log_std_max)


# ---------------------------------------------------------------------------
# v1.1 (P1): goal-hotel removal, heading potential, stair free mask, relax semantics
# ---------------------------------------------------------------------------


def test_v11_proximity_rewards_are_zeroed():
    """U1 - no reward may be paid for *occupying* the goal radius any more."""
    from isaaclab_nav_task.navigation.config.b2w.navigation_env_cfg import B2WRewardsCfg

    cfg = B2WRewardsCfg()
    assert cfg.reach_goal.weight == 0.0
    assert cfg.reach_goal_xy_tight.weight == 0.0
    assert cfg.reach_goal_xy_soft.weight == 0.0
    # the dense brake gradient and the completion bonus are what is left near the goal
    assert cfg.velocity_at_goal.weight == pytest.approx(-0.2)
    assert cfg.goal_reached_bonus.weight == pytest.approx(25.0)
    assert cfg.heading_progress_to_path.weight == pytest.approx(1.0)


class FakeRobot:
    def __init__(self, lin=(1.0, 0.0, 0.0), ang=(0.0, 0.0, 0.5)):
        self.data = SimpleNamespace(
            root_lin_vel_b=torch.tensor([list(lin)]),
            root_ang_vel_b=torch.tensor([list(ang)]),
            root_lin_vel_w=torch.tensor([[*lin]]),
        )


def test_v11_velocity_at_goal_penalty_only_inside_the_radius():
    from isaaclab_nav_task.navigation.mdp.rewards import velocity_at_goal_penalty

    goal = FakeGoalCommand()
    env = dwell_env(0.3, goal)
    env.scene = {"robot": FakeRobot(lin=(1.0, 0.0, 0.0), ang=(0.0, 0.0, 0.4))}

    set_distance(goal, 0.3)
    inside = float(velocity_at_goal_penalty(env))
    set_distance(goal, 2.0)
    outside = float(velocity_at_goal_penalty(env))

    assert inside == pytest.approx(1.0 + 0.5 * 0.4), "lin + 0.5*|wz| inside the radius"
    assert outside == 0.0


def test_v11_timeout_never_counts_as_success():
    """U2 - under the dwell rule a completion ends the episode first, so a timeout is a failure."""
    from isaaclab_nav_task.navigation.mdp.terminations import time_out_navigation

    added = []

    goal = FakeGoalCommand()
    goal.was_in_goal_at_timeout = torch.zeros(1)
    goal.goal_reached_buffer = SimpleNamespace(add=lambda value, ids: added.append(value.clone()))
    set_distance(goal, 0.3)  # inside the radius on the timeout step
    env = dwell_env(0.3, goal)
    env.episode_length_buf = torch.tensor([450])  # at the horizon
    env.max_episode_length = 450

    terminated = time_out_navigation(env, strict_success=True)
    assert bool(terminated.item())
    assert added and float(added[-1].sum()) == 0.0, "a timeout must never be credited as success"
    assert float(goal.was_in_goal_at_timeout.item()) == 1.0, "but it is still counted as a metric"


def test_v11_relax_always_includes_invalid_segments():
    """U3 - an invalid segment is infeasible by construction: leaving the path there must be free."""
    mp = MultiPath.__new__(MultiPath)
    # segments 3 and 4 invalid, and the whole neighbourhood 1..6 is a narrow doorway (pinch)
    seg_passable = torch.ones(1, 14, dtype=torch.bool)
    seg_passable[0, 3] = False
    seg_passable[0, 4] = False
    pinch = torch.zeros(1, 14, dtype=torch.bool)
    pinch[0, 1:7] = True

    relax = mp._compute_relax(seg_passable, pinch)
    assert bool(relax[0, 3]) and bool(relax[0, 4]), "invalid segments must be relaxed even inside a pinch"
    # the pinch neighbourhood of those segments is *not* relaxed: a doorway next to a blocked
    # segment must stay strict, otherwise the exemption spills onto the door frame
    assert not bool(relax[0, 2]) and not bool(relax[0, 5])
    assert not bool(relax[0, 0]), "segment 0 must not be relaxed (no wraparound)"

    # outside a pinch, the neighbours of an invalid segment are relaxed (two-sided dilation)
    pinch_free = torch.zeros(1, 14, dtype=torch.bool)
    relax2 = mp._compute_relax(seg_passable, pinch_free)
    assert bool(relax2[0, 2]) and bool(relax2[0, 5])


def make_prm_from_height_map(height_map: np.ndarray) -> Prm:
    """A `Prm` built by the real mask code (not the hand-set masks the other tests use)."""
    prm = Prm.__new__(Prm)
    prm.cfg = PrmConfig(fpr=0.0, fnr=0.0, padding=4, max_height_diff=50.0, N=10)
    prm.height_map = torch.from_numpy(height_map.astype(np.int16))
    prm.compute_free_mask()
    prm.compute_padding_mask()
    prm._height_np = prm.height_map.numpy()
    prm._free_np = prm.free_mask.numpy()
    prm._padding_np = prm.padding_mask.numpy()
    prm.h_scale = H_SCALE
    prm.v_scale = V_SCALE
    prm.size = (height_map.shape[0] * H_SCALE, height_map.shape[1] * H_SCALE)
    prm.terrain_origin = torch.zeros(3)
    prm.compute_clearance_map()
    return prm


def stair_height_map(n: int = 200, x0: int = 90, y0: int = 40, y1: int = 160) -> np.ndarray:
    """5 rises of 0.2 m (40 units) on 0.4 m treads, inside a corridor."""
    h = np.zeros((n, n), dtype=np.int16)
    h[:, :y0] = WALL
    h[:, y1:] = WALL
    for k in range(5):
        h[x0 + 4 * k : x0 + 4 * (k + 1), :] = 40 * (k + 1)
    h[x0 + 20 :, :] = 200
    return h


def test_v11_stairs_are_in_the_free_mask():
    """U4 - the staircase treads must be free, or the clearance map turns the whole flight into an
    obstacle and every path across it is flagged impassable (D3: 8.1/14 segments on stairs tiles)."""
    prm = make_prm_from_height_map(stair_height_map())
    tread = (90 + 8, 100)  # a middle tread, height 80 units
    assert prm._free_np[tread], "stair treads must be free"
    assert prm.clearance_at(tread) > 0.3, f"clearance on a tread was {prm.clearance_at(tread):.2f} m"

    a = np.array([60.0, 100.0, 0.0])
    b = np.array([140.0, 100.0, 0.0])
    assert prm.segment_passable(a, b, 0.31), "a chord across the staircase must be passable"


class FakePathGoal(FakeGoalCommand):
    """A goal command whose local path points in a controllable direction."""

    def __init__(self, direction=(1.0, 0.0)):
        super().__init__()
        self.direction = direction

    def get_path_local(self, horizon_m: float = 1.5):
        d = torch.tensor([[*self.direction, 0.0, 0.0]], dtype=torch.float32)
        return d.unsqueeze(1).repeat(1, 15, 1)


def heading_env(distance: float = 5.0) -> tuple:
    from isaaclab_nav_task.navigation.mdp.rewards import heading_progress_to_path

    goal = FakePathGoal()
    set_distance(goal, distance)
    env = dwell_env(distance, goal)
    env.episode_length_buf = torch.tensor([10], dtype=torch.long)
    return env, goal, heading_progress_to_path


def test_v11_heading_potential_totals_the_initial_error():
    """U6 - turning 180 deg pays pi in total; oscillating pays nothing."""
    import math

    env, goal, reward_fn = heading_env()
    steps = 16
    total = 0.0
    for k in range(steps):
        angle = math.pi * (1.0 - k / steps)  # 180 deg -> 0 over 16 steps
        goal.direction = (math.cos(angle), math.sin(angle))
        total += float(reward_fn(env)) * 0.2  # reward manager multiplies by dt
    assert total == pytest.approx(math.pi, rel=0.1), f"180 deg turn paid {total:.2f}, expected ~pi"

    # oscillation: the same error swinging back and forth must net ~0
    env2, goal2, reward_fn2 = heading_env()
    total2 = 0.0
    for k in range(40):
        angle = 0.4 if k % 2 == 0 else 0.2
        goal2.direction = (math.cos(angle), math.sin(angle))
        total2 += float(reward_fn2(env2)) * 0.2
    assert abs(total2) < 0.25, f"oscillation paid {total2:.2f}, expected ~0"


def test_v11_heading_potential_has_no_spike_at_a_corner():
    """U6b - a 90 deg corner swings the carrot; the per-step payment must stay clamped."""
    import math

    env, goal, reward_fn = heading_env()
    goal.direction = (math.cos(math.pi / 2), math.sin(math.pi / 2))
    reward_fn(env)  # seed
    goal.direction = (1.0, 0.0)  # the carrot jumped by 90 deg in one step
    step_reward = float(reward_fn(env)) * 0.2
    assert step_reward <= 0.2 + 1e-6, f"corner produced a spike of {step_reward:.3f}"


def test_v11_heading_error_observation_uses_the_lookahead_point():
    """U7 - sample 0 of the local path sits on the robot (degenerate); the bearing must come from
    the last sample."""
    import math

    from isaaclab_nav_task.navigation.mdp.observations import path_heading_error

    goal = FakePathGoal(direction=(0.0, 1.0))
    env = dwell_env(5.0, goal)
    env.episode_length_buf = torch.tensor([10], dtype=torch.long)
    obs = path_heading_error(env)
    assert obs.shape == (1, 2)
    assert float(obs[0, 0]) == pytest.approx(math.cos(math.pi / 2), abs=1e-5)
    assert float(obs[0, 1]) == pytest.approx(1.0, abs=1e-5)


def test_v11_log_std_schedule_reaches_the_final_wall_by_3600():
    """U8 - the compressed schedule must actually reach the final wall inside a 4000-iteration run."""
    from types import SimpleNamespace as NS

    from rsl_rl.algorithms.mdpo import MDPO

    fake = NS(
        log_std_schedule="cosine",
        log_std_anneal_start=600,
        log_std_anneal_mid=2200,
        log_std_anneal_end=3600,
        log_std_max_initial=0.1,
        log_std_max_mid=-0.6,
        log_std_max_final=-1.0,
        actor_critic_1=NS(),
        actor_critic_2=NS(),
    )
    MDPO._update_log_std_bounds(fake, 600)
    assert fake.current_log_std_max == pytest.approx(0.1)
    MDPO._update_log_std_bounds(fake, 2200)
    assert fake.current_log_std_max == pytest.approx(-0.6)
    MDPO._update_log_std_bounds(fake, 3600)
    assert fake.current_log_std_max == pytest.approx(-1.0)
    MDPO._update_log_std_bounds(fake, 4000)
    assert fake.current_log_std_max == pytest.approx(-1.0)
