import argparse
import sys
import types
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

isaacgym_module = types.ModuleType("isaacgym")
isaacgym_module.gymapi = object()
sys.modules.setdefault("isaacgym", isaacgym_module)

import collect_dataset_pick_place_release as collector
import torch


def _args(**overrides):
    defaults = {
        "table_x_half_extent": 0.30,
        "table_x_inset_margin": 0.06,
        "table_y_half_extent": 0.20,
        "table_y_inset_margin": 0.04,
        "place_goal_x_margin": 0.10,
        "place_goal_y_margin": 0.06,
        "xy_range": 0.10,
        "min_effective_transport": 0.05,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _EnvStub:
    def __init__(self):
        self.table_init_state = torch.tensor([[0.0, 0.0, collector.TABLE_Z]], dtype=torch.float32)


def test_sampled_start_and_goal_stay_in_long_table_bands():
    args = _args()
    rng = np.random.default_rng(0)
    nominal_start_pose = [0.09, 0.05, 0.55, 0.0, 0.0, 0.0, 1.0]
    start_band, goal_band = collector._start_and_goal_x_bands(args)
    env = _EnvStub()

    for _ in range(256):
        start_pose, goal_x, _ = collector._sample_end_to_end_start_and_goal(
            rng,
            nominal_start_pose,
            args,
        )
        goals = collector._build_pick_place_goals(
            start_pose,
            goal_x=goal_x,
            lift_height=0.20,
            place_height=0.02,
            place_hold_goals=10,
        )
        place_goal = goals[-1]

        assert start_band[0] <= start_pose[0] <= start_band[1]
        assert goal_band[0] <= goal_x <= goal_band[1]
        assert goal_x > start_pose[0]
        assert -0.14 <= start_pose[1] <= 0.14
        assert place_goal[0] == goal_x
        assert place_goal[1] == start_pose[1]
        assert collector._place_goal_in_safe_zone(
            env,
            place_goal,
            table_x_half_extent=args.table_x_half_extent,
            table_x_inset_margin=args.place_goal_x_margin,
            table_y_half_extent=args.table_y_half_extent,
            table_y_inset_margin=args.place_goal_y_margin,
        )


def test_sampled_y_is_clamped_to_table_bounds():
    args = _args(xy_range=0.20)
    rng = np.random.default_rng(1)
    nominal_start_pose = [0.09, 0.13, 0.55, 0.0, 0.0, 0.0, 1.0]

    for _ in range(64):
        start_pose, _, _ = collector._sample_end_to_end_start_and_goal(
            rng,
            nominal_start_pose,
            args,
        )
        assert -0.14 <= start_pose[1] <= 0.14


def test_release_outcome_marks_drop_reattempt_as_failure():
    (
        pick_place_success,
        release_goal_success,
        release_stable,
        release_success,
        failure_stage,
    ) = collector._classify_release_outcome(
        max_successes_seen=6,
        release_start_goal_idx=3,
        total_goals=6,
        entered_release_phase=True,
        final_object_on_table=True,
        final_place_xy_error_m=0.01,
        final_place_z_error_m=0.01,
        final_object_speed_mps=0.02,
        release_xy_tolerance=0.05,
        release_z_tolerance=0.04,
        release_speed_tolerance=0.25,
        reattempted_after_drop=True,
    )

    assert pick_place_success
    assert release_goal_success
    assert release_stable
    assert not release_success
    assert failure_stage == "drop_reattempt"


def test_release_outcome_stays_successful_without_drop_reattempt():
    (
        pick_place_success,
        release_goal_success,
        release_stable,
        release_success,
        failure_stage,
    ) = collector._classify_release_outcome(
        max_successes_seen=6,
        release_start_goal_idx=3,
        total_goals=6,
        entered_release_phase=True,
        final_object_on_table=True,
        final_place_xy_error_m=0.01,
        final_place_z_error_m=0.01,
        final_object_speed_mps=0.02,
        release_xy_tolerance=0.05,
        release_z_tolerance=0.04,
        release_speed_tolerance=0.25,
        reattempted_after_drop=False,
    )

    assert pick_place_success
    assert release_goal_success
    assert release_stable
    assert release_success
    assert failure_stage is None


def test_drop_detected_after_partial_pickup_attempt_before_release():
    assert collector._drop_detected_after_pickup_attempt(
        object_height_above_init_m=0.005,
        max_object_height_above_init_m=0.03,
        lifted_object=False,
        in_release_phase=False,
    )


def test_drop_not_detected_for_small_height_jitter():
    assert not collector._drop_detected_after_pickup_attempt(
        object_height_above_init_m=0.005,
        max_object_height_above_init_m=0.01,
        lifted_object=False,
        in_release_phase=False,
    )


def test_drop_not_detected_for_release_phase_settle():
    assert not collector._drop_detected_after_pickup_attempt(
        object_height_above_init_m=0.005,
        max_object_height_above_init_m=0.03,
        lifted_object=False,
        in_release_phase=True,
    )
