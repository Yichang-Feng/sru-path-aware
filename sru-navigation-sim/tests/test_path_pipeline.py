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
