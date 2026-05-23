#!/usr/bin/env python3
"""Stage 5: collect pick-place data on a modestly longer end-to-end table.

This collector owns the `pick_place` and `pick_place_release` modes. Pickup-only
collection remains in `stage5_collect_dataset.py`.
"""

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Isaac Gym import order matters: import isaacgym before torch.
from isaacgym import gymapi  # noqa: F401
import torch

from stage5_collect_common import (
    CHECKPOINT_PATH,
    COLLECTION_TYPES,
    CONFIG_PATH,
    DATASET_CAMERA_HEIGHT,
    DATASET_CAMERA_HORIZONTAL_FOV,
    DATASET_CAMERA_POSITION,
    DATASET_CAMERA_TARGET,
    DATASET_CAMERA_WIDTH,
    DEFAULT_COLLECTION_TYPE,
    DEFAULT_HORIZON,
    DEFAULT_HORIZON_WITH_RELEASE,
    DEFAULT_OBJECT_CATEGORY,
    DEFAULT_OBJECT_NAME,
    DEFAULT_RELEASE_STEPS,
    DEFAULT_START_Z_OFFSET,
    DEFAULT_TASK_NAME,
    DRY_RUN_ROLLOUTS,
    GOAL_STAGE_NAMES,
    LATERAL_OFFSET_RANGE_M,
    LIFT_HEIGHT_M,
    N_ACT,
    N_OBS,
    PLACE_GOAL_X_MARGIN_M,
    PLACE_GOAL_Y_MARGIN_M,
    PLACE_HEIGHT_M,
    PLACE_HOLD_GOALS,
    PICKUP_SUCCESS_HOLD_STEPS,
    RELEASE_GOAL_STAGE_NAMES,
    RELEASE_SPEED_TOLERANCE_MPS,
    RELEASE_XY_TOLERANCE_M,
    RELEASE_Z_TOLERANCE_M,
    ReleaseRolloutResult,
    RolloutResult,
    TABLE_Y_HALF_EXTENT_M,
    TABLE_Y_INSET_MARGIN_M,
    TABLE_Z,
    _annotate_rollout_frames,
    _append_episode,
    _current_counts,
    _destroy_env,
    _jsonable,
    _load_nominal_start_pose,
    _open_or_create_zarr,
    _resolve_noise_args,
    _sample_group_noise,
    _update_noisy_metric_attrs,
    _update_object_registry,
    _write_rollout_video,
)
from stage5_anchored_recovery import (
    AnchoredBranchSnapshot,
    AnchoredRecoveryConfig,
    build_anchored_recovery_config,
    maybe_fork_rng,
    run_anchored_branch,
    sample_branch_trigger_steps,
    update_anchored_root_attrs,
    validate_anchored_args,
    validate_anchored_resume_attrs,
)


LONG_TABLE_URDF = "urdf/table_pick_place_release.urdf"
LONG_TABLE_X_HALF_EXTENT_M = 0.60 / 2.0
LONG_TABLE_X_INSET_MARGIN_M = 0.06
END_BAND_FRACTION = 0.30
FALLBACK_GOAL_UPPER_FRACTION = 0.35
MIN_EFFECTIVE_TRANSPORT_M = 0.05
DROP_RETRY_MIN_RISE_M = 0.02
DROP_RETRY_RETURN_TO_START_TOLERANCE_M = 0.01


def _goal_stage_name(goal_idx: int) -> str:
    if goal_idx <= 0:
        return GOAL_STAGE_NAMES[0]
    if goal_idx == 1:
        return GOAL_STAGE_NAMES[1]
    return GOAL_STAGE_NAMES[2]


def _failure_stage_from_successes(
    max_successes_seen: int,
    total_goals: int,
) -> Optional[str]:
    if max_successes_seen >= total_goals:
        return None
    return _goal_stage_name(max_successes_seen)


def _release_goal_stage_name(goal_idx: int, release_start_goal_idx: int) -> str:
    if goal_idx <= 0:
        return RELEASE_GOAL_STAGE_NAMES[0]
    if goal_idx == 1:
        return RELEASE_GOAL_STAGE_NAMES[1]
    if goal_idx < release_start_goal_idx:
        return RELEASE_GOAL_STAGE_NAMES[2]
    return RELEASE_GOAL_STAGE_NAMES[3]


def _release_failure_stage_from_successes(
    max_successes_seen: int,
    release_start_goal_idx: int,
    total_goals: int,
) -> Optional[str]:
    if max_successes_seen >= total_goals:
        return None
    if max_successes_seen >= release_start_goal_idx:
        return "release"
    return _release_goal_stage_name(max_successes_seen, release_start_goal_idx)


def _classify_release_outcome(
    *,
    max_successes_seen: int,
    release_start_goal_idx: int,
    total_goals: int,
    entered_release_phase: bool,
    final_object_on_table: bool,
    final_place_xy_error_m: float,
    final_place_z_error_m: float,
    final_object_speed_mps: float,
    release_xy_tolerance: float,
    release_z_tolerance: float,
    release_speed_tolerance: float,
    reattempted_after_drop: bool,
) -> Tuple[bool, bool, bool, bool, Optional[str]]:
    pick_place_success = max_successes_seen >= release_start_goal_idx
    release_goal_success = max_successes_seen >= total_goals
    release_stable = (
        entered_release_phase
        and final_object_on_table
        and final_place_xy_error_m <= release_xy_tolerance
        and final_place_z_error_m <= release_z_tolerance
        and final_object_speed_mps <= release_speed_tolerance
    )
    release_success = pick_place_success and release_stable and not reattempted_after_drop
    failure_stage = _release_failure_stage_from_successes(
        max_successes_seen=max_successes_seen,
        release_start_goal_idx=release_start_goal_idx,
        total_goals=total_goals,
    )
    if reattempted_after_drop:
        failure_stage = "drop_reattempt"
    return (
        pick_place_success,
        release_goal_success,
        release_stable,
        release_success,
        failure_stage,
    )


def _drop_detected_after_pickup_attempt(
    *,
    object_height_above_init_m: float,
    max_object_height_above_init_m: float,
    lifted_object: bool,
    in_release_phase: bool,
) -> bool:
    # `lifted_object` only flips after a fairly large rise, so also treat
    # "rose meaningfully from the table, then fell back near start height"
    # as a dropped pickup attempt before the release phase.
    dropped_after_full_lift = lifted_object and object_height_above_init_m < 0.0
    dropped_after_retryable_pickup = (
        not in_release_phase
        and max_object_height_above_init_m >= DROP_RETRY_MIN_RISE_M
        and object_height_above_init_m <= DROP_RETRY_RETURN_TO_START_TOLERANCE_M
    )
    return dropped_after_full_lift or dropped_after_retryable_pickup


def _annotate_release_rollout_frames(
    frames: np.ndarray,
    *,
    rollout_idx: int,
    stage_names: List[str],
    goal_dists: List[float],
    successes_per_step: List[int],
    release_phase_per_step: List[bool],
    release_start_step: Optional[int],
) -> np.ndarray:
    from PIL import Image, ImageDraw

    annotated_frames: List[np.ndarray] = []
    n_steps = min(
        len(frames),
        len(stage_names),
        len(goal_dists),
        len(successes_per_step),
        len(release_phase_per_step),
    )
    for step in range(n_steps):
        image = Image.fromarray(frames[step])
        draw = ImageDraw.Draw(image)
        lines = [
            f"rollout {rollout_idx:04d}",
            f"step {step:03d}",
            f"stage {stage_names[step]}",
            f"goal_dist {goal_dists[step]:.4f}",
            f"successes {successes_per_step[step]}",
            f"release_phase {int(release_phase_per_step[step])}",
        ]
        if release_start_step is not None:
            lines.append(f"release_start {release_start_step:03d}")
        y = 12
        for line in lines:
            bbox = draw.textbbox((12, y), line)
            draw.rectangle(
                (bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
                fill=(0, 0, 0),
            )
            draw.text((12, y), line, fill=(255, 255, 255))
            y += 22
        annotated_frames.append(np.asarray(image, dtype=np.uint8))
    return np.stack(annotated_frames, axis=0)


def _make_long_table_env(
    *,
    num_envs: int,
    nominal_start_pose: List[float],
    horizon: int,
    headless: bool,
    device: str,
    seed: int,
    object_name: str,
    goal_poses: Optional[List[List[float]]] = None,
):
    from deployment.isaac.isaac_env import create_env

    if goal_poses is None:
        raise ValueError("goal_poses must be provided")

    overrides = {
        "seed": seed,
        "task.env.numEnvs": num_envs,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        "task.env.enableCameraSensors": True,
        "task.env.enableDatasetCameras": True,
        "task.env.objectName": object_name,
        "task.env.useFixedGoalStates": True,
        "task.env.fixedGoalStates": goal_poses,
        "task.env.showGoalObjectVisual": False,
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": nominal_start_pose,
        "task.env.startArmHigher": True,
        "task.env.asset.table": LONG_TABLE_URDF,
        "task.env.tableResetZ": TABLE_Z,
        "task.env.tableObjectZOffset": float(nominal_start_pose[2] - TABLE_Z),
        "task.env.resetPositionNoiseX": 0.0,
        "task.env.resetPositionNoiseY": 0.0,
        "task.env.resetPositionNoiseZ": 0.0,
        "task.env.randomizeObjectRotation": False,
        "task.env.resetDofPosRandomIntervalFingers": 0.0,
        "task.env.resetDofPosRandomIntervalArm": 0.0,
        "task.env.resetDofVelRandomInterval": 0.0,
        "task.env.tableResetZRange": 0.0,
        "task.env.useActionDelay": False,
        "task.env.useObsDelay": False,
        "task.env.useObjectStateDelayNoise": False,
        "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
        "task.env.resetWhenDropped": False,
        "task.env.armMovingAverage": 0.1,
        "task.env.evalSuccessTolerance": 0.01,
        "task.env.successSteps": 1,
        "task.env.fixedSizeKeypointReward": True,
        "task.env.forceScale": 0.0,
        "task.env.torqueScale": 0.0,
        "task.env.linVelImpulseScale": 0.0,
        "task.env.angVelImpulseScale": 0.0,
        "task.env.datasetCameraPosition": DATASET_CAMERA_POSITION,
        "task.env.datasetCameraTarget": DATASET_CAMERA_TARGET,
        "task.env.datasetCameraHorizontalFov": DATASET_CAMERA_HORIZONTAL_FOV,
        "task.env.datasetCameraWidth": DATASET_CAMERA_WIDTH,
        "task.env.datasetCameraHeight": DATASET_CAMERA_HEIGHT,
    }

    return create_env(
        config_path=str(CONFIG_PATH),
        headless=headless,
        device=device,
        enable_viewer_sync_at_start=False,
        episode_length=horizon + 2,
        overrides=overrides,
    )


def _table_bounds(args: argparse.Namespace) -> Tuple[float, float, float, float]:
    x_min = -float(args.table_x_half_extent) + float(args.table_x_inset_margin)
    x_max = float(args.table_x_half_extent) - float(args.table_x_inset_margin)
    y_min = -float(args.table_y_half_extent) + float(args.table_y_inset_margin)
    y_max = float(args.table_y_half_extent) - float(args.table_y_inset_margin)
    return x_min, x_max, y_min, y_max


def _goal_safe_x_max(args: argparse.Namespace) -> float:
    return float(args.table_x_half_extent) - max(
        float(args.table_x_inset_margin),
        float(args.place_goal_x_margin),
    ) - 1e-6


def _goal_safe_y_bounds(args: argparse.Namespace) -> Tuple[float, float]:
    half_extent = float(args.table_y_half_extent)
    inset = max(float(args.table_y_inset_margin), float(args.place_goal_y_margin))
    return -half_extent + inset + 1e-6, half_extent - inset - 1e-6


def _start_and_goal_x_bands(args: argparse.Namespace) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_min, x_max, _, _ = _table_bounds(args)
    usable_span = x_max - x_min
    band_span = usable_span * END_BAND_FRACTION
    start_band = (x_min, x_min + band_span)
    goal_band = (x_max - band_span, min(x_max, _goal_safe_x_max(args)))
    if start_band[0] >= start_band[1]:
        raise ValueError("Invalid start band for long-table collector.")
    if goal_band[0] >= goal_band[1]:
        raise ValueError("Invalid goal band for long-table collector.")
    return start_band, goal_band


def _default_goal_x(args: argparse.Namespace) -> float:
    _, goal_band = _start_and_goal_x_bands(args)
    return 0.5 * (goal_band[0] + goal_band[1])


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _sample_end_to_end_start_and_goal(
    rng: np.random.Generator,
    nominal_start_pose: List[float],
    args: argparse.Namespace,
) -> Tuple[List[float], float, float]:
    start_band, goal_band = _start_and_goal_x_bands(args)
    y_min, y_max = _goal_safe_y_bounds(args)
    start_pose = list(nominal_start_pose)
    start_pose[0] = float(rng.uniform(start_band[0], start_band[1]))
    nominal_y = float(nominal_start_pose[1])
    start_pose[1] = _clamp(
        float(nominal_y + rng.uniform(-args.xy_range, args.xy_range)),
        y_min,
        y_max,
    )
    goal_x = float(rng.uniform(goal_band[0], goal_band[1]))
    if goal_x <= start_pose[0]:
        goal_x = max(goal_band[0], start_pose[0] + float(args.min_effective_transport))
    if goal_x <= start_pose[0]:
        raise ValueError("Sampled goal_x must be greater than start_x.")
    return start_pose, goal_x, float(goal_x - start_pose[0])


def _quat_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    norm = float(np.linalg.norm(quat_xyzw))
    if norm <= 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = quat_xyzw / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _object_xy_offset_bounds_for_quat(
    env,
    quat_xyzw: np.ndarray,
) -> Tuple[float, float, float, float]:
    keypoint_offsets = getattr(env, "object_keypoint_offsets_fixed_size", None)
    if keypoint_offsets is None:
        return 0.0, 0.0, 0.0, 0.0

    keypoint_offsets_np = keypoint_offsets[0].detach().cpu().numpy()
    if keypoint_offsets_np.ndim != 2 or keypoint_offsets_np.shape[1] != 3 or keypoint_offsets_np.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0

    rot_m = _quat_xyzw_to_rotation_matrix(quat_xyzw)
    rotated_offsets = keypoint_offsets_np @ rot_m.T
    return (
        float(rotated_offsets[:, 0].min()),
        float(rotated_offsets[:, 0].max()),
        float(rotated_offsets[:, 1].min()),
        float(rotated_offsets[:, 1].max()),
    )


def _release_goal_safe_center_bounds(
    env,
    object_quat_xyzw: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[float, float, float, float]:
    table_center_np = env.table_init_state[0, 0:3].detach().cpu().numpy()
    usable_x_half_extent = max(float(args.table_x_half_extent) - float(args.place_goal_x_margin), 0.0)
    usable_y_half_extent = max(float(args.table_y_half_extent) - float(args.place_goal_y_margin), 0.0)
    table_x_min = float(table_center_np[0]) - usable_x_half_extent
    table_x_max = float(table_center_np[0]) + usable_x_half_extent
    table_y_min = float(table_center_np[1]) - usable_y_half_extent
    table_y_max = float(table_center_np[1]) + usable_y_half_extent
    offset_x_min, offset_x_max, offset_y_min, offset_y_max = _object_xy_offset_bounds_for_quat(
        env,
        object_quat_xyzw,
    )
    return (
        table_x_min - offset_x_min,
        table_x_max - offset_x_max,
        table_y_min - offset_y_min,
        table_y_max - offset_y_max,
    )


def _sample_end_to_end_start_and_goal_release(
    env,
    rng: np.random.Generator,
    nominal_start_pose: List[float],
    args: argparse.Namespace,
) -> Tuple[List[float], float, float]:
    start_band, goal_band = _start_and_goal_x_bands(args)
    goal_x_min_safe, goal_x_max_safe, goal_y_min_safe, goal_y_max_safe = _release_goal_safe_center_bounds(
        env,
        np.asarray(nominal_start_pose[3:7], dtype=np.float32),
        args,
    )
    min_transport = float(args.min_effective_transport)
    y_low = max(goal_y_min_safe, float(nominal_start_pose[1]) - float(args.xy_range))
    y_high = min(goal_y_max_safe, float(nominal_start_pose[1]) + float(args.xy_range))
    if y_low >= y_high:
        raise ValueError(
            f"No valid footprint-safe release Y range: [{y_low:.4f}, {y_high:.4f}]"
        )

    preferred_goal_x_floor = max(goal_band[0], goal_x_min_safe)
    preferred_goal_x_ceiling = min(goal_band[1], goal_x_max_safe)
    fallback_goal_x_floor = goal_x_min_safe
    fallback_goal_x_ceiling = goal_x_max_safe
    fallback_goal_x_span = max(fallback_goal_x_ceiling - fallback_goal_x_floor, 0.0)
    fallback_goal_x_floor = max(
        fallback_goal_x_floor,
        fallback_goal_x_ceiling - fallback_goal_x_span * FALLBACK_GOAL_UPPER_FRACTION,
    )
    using_goal_band_fallback = preferred_goal_x_floor >= preferred_goal_x_ceiling
    goal_x_floor = preferred_goal_x_floor
    goal_x_ceiling = preferred_goal_x_ceiling
    if using_goal_band_fallback:
        goal_x_floor = fallback_goal_x_floor
        goal_x_ceiling = fallback_goal_x_ceiling
    if goal_x_floor >= goal_x_ceiling:
        raise ValueError(
            f"No valid footprint-safe release goal X range: [{goal_x_floor:.4f}, {goal_x_ceiling:.4f}]"
        )

    start_x_low = start_band[0]
    start_x_high = min(start_band[1], goal_x_ceiling - min_transport)
    if start_x_low >= start_x_high:
        raise ValueError(
            f"No valid release start X range: [{start_x_low:.4f}, {start_x_high:.4f}] "
            f"given goal ceiling={goal_x_ceiling:.4f} and min_transport={min_transport:.4f}"
        )

    start_pose = list(nominal_start_pose)
    start_pose[0] = float(rng.uniform(start_x_low, start_x_high))
    start_pose[1] = float(rng.uniform(y_low, y_high))

    goal_x_low = max(goal_x_floor, start_pose[0] + min_transport)
    goal_x_high = goal_x_ceiling
    if goal_x_low >= goal_x_high:
        raise ValueError(
            f"No valid footprint-safe release X range: [{goal_x_low:.4f}, {goal_x_high:.4f}] "
            f"for start_x={start_pose[0]:.4f}"
        )

    goal_x = float(rng.uniform(goal_x_low, goal_x_high))
    if using_goal_band_fallback:
        print(
            "[stage5-release-long-table] fallback_goal_x_band "
            f"preferred=[{preferred_goal_x_floor:.4f}, {preferred_goal_x_ceiling:.4f}] "
            f"fallback=[{fallback_goal_x_floor:.4f}, {fallback_goal_x_ceiling:.4f}]",
            flush=True,
        )
    return start_pose, goal_x, float(goal_x - start_pose[0])


def _build_pick_place_goals(
    start_pose: List[float],
    *,
    goal_x: float,
    lift_height: float,
    place_height: float,
    place_hold_goals: int,
) -> List[List[float]]:
    x0, y0, z0, qx, qy, qz, qw = start_pose
    lift_goal = [x0, y0, z0 + lift_height, qx, qy, qz, qw]
    transport_goal = [goal_x, y0, z0 + lift_height, qx, qy, qz, qw]
    place_goal = [goal_x, y0, z0 + place_height, qx, qy, qz, qw]
    return [lift_goal, transport_goal] + [list(place_goal) for _ in range(max(place_hold_goals, 1))]


def _build_pick_place_release_goals(
    start_pose: List[float],
    *,
    goal_x: float,
    lift_height: float,
    place_height: float,
    place_hold_goals: int,
    release_hold_goals: int,
) -> Tuple[List[List[float]], int, List[float]]:
    base_goals = _build_pick_place_goals(
        start_pose,
        goal_x=goal_x,
        lift_height=lift_height,
        place_height=place_height,
        place_hold_goals=place_hold_goals,
    )
    place_goal = list(base_goals[-1])
    release_start_goal_idx = len(base_goals)
    if release_hold_goals <= 0:
        return base_goals, release_start_goal_idx, place_goal
    release_goals = [list(place_goal) for _ in range(release_hold_goals)]
    return base_goals + release_goals, release_start_goal_idx, place_goal


def _record_geometry_attrs(root, args: argparse.Namespace) -> None:
    root.attrs["table_urdf"] = LONG_TABLE_URDF
    root.attrs["table_x_half_extent"] = float(args.table_x_half_extent)
    root.attrs["table_y_half_extent"] = float(args.table_y_half_extent)
    root.attrs["table_x_inset_margin"] = float(args.table_x_inset_margin)
    root.attrs["table_y_inset_margin"] = float(args.table_y_inset_margin)


def _zero_noise_metrics() -> Dict[str, float]:
    return {"l2_sum": 0.0, "l2_sq_sum": 0.0, "linf_sum": 0.0, "count": 0}


def _accumulate_noise_metrics(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for key in ("l2_sum", "l2_sq_sum", "linf_sum"):
        dst[key] += float(src.get(key, 0.0))
    dst["count"] += int(src.get("count", 0))


def _build_group_noise_sampler(args: argparse.Namespace):
    def _sample(clean_action_t: torch.Tensor) -> torch.Tensor:
        return _sample_group_noise(
            clean_action_t,
            arm_base_noise=args.arm_base_noise,
            arm_wrist_noise=args.arm_wrist_noise,
            thumb_noise=args.thumb_noise,
            index_noise=args.index_noise,
            middle_noise=args.middle_noise,
            ring_noise=args.ring_noise,
            pinky_noise=args.pinky_noise,
        )

    return _sample


def _capture_branch_snapshot(env, policy, obs: torch.Tensor) -> AnchoredBranchSnapshot:
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    return AnchoredBranchSnapshot(
        env_state=env.capture_branch_state(env_ids),
        policy_state=policy.get_internal_state(),
        obs=obs.clone(),
    )


def _restore_branch_snapshot(
    env,
    policy,
    snapshot: AnchoredBranchSnapshot,
) -> torch.Tensor:
    env.restore_branch_state(snapshot.env_state, snapshot.env_state["env_ids"])
    policy.set_internal_state(snapshot.policy_state)
    return snapshot.obs.clone()


def _run_anchored_pick_place_rollout(
    *,
    env,
    args: argparse.Namespace,
    device: str,
    policy,
    start_pose: List[float],
    goals: List[List[float]],
    rollout_idx: int,
    verbose_steps: bool,
    rng: np.random.Generator,
    anchored_config: AnchoredRecoveryConfig,
):
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    goals_t = torch.tensor(goals, device=env.device, dtype=env.trajectory_states.dtype)
    env.trajectory_states = goals_t
    env.max_consecutive_successes = len(goals)
    env.object_init_state[env_ids, 0:7] = torch.tensor(
        [start_pose], device=env.device, dtype=env.object_init_state.dtype
    )
    env.cfg["env"]["tableObjectZOffset"] = float(start_pose[2] - TABLE_Z)
    env.reset_idx(env_ids, tensor_reset=True)
    policy.reset()

    zero_action = torch.zeros((env.num_envs, N_ACT), device=device)
    obs_dict, _, _, _ = env.step(zero_action)
    obs = obs_dict["obs"]
    total_env_steps = 1
    active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)
    noise_sampler = _build_group_noise_sampler(args)
    trigger_steps = set(
        sample_branch_trigger_steps(
            rng=rng,
            branch_min_step=anchored_config.branch_min_step,
            branch_max_step=anchored_config.branch_max_step,
            branches_per_rollout=anchored_config.branches_per_rollout,
            min_gap=anchored_config.branch_gap,
        )
    )

    branch_buffers: List = []
    branch_noise_metrics = _zero_noise_metrics()
    branch_stats = {"attempted": 0, "aborted": 0}

    stage_names: List[str] = []
    goal_dists: List[float] = []
    successes_per_step: List[int] = []
    max_successes_seen = 0
    final_successes = 0
    viewer_closed = False
    prev_stage_idx = -1

    for step in range(args.horizon):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            viewer_closed = True
            break

        if step in trigger_steps:
            branch_stats["attempted"] += 1
            snapshot = _capture_branch_snapshot(env, policy, obs)
            with maybe_fork_rng(device):
                branch_result = run_anchored_branch(
                    env=env,
                    device=device,
                    obs=obs.clone(),
                    config=anchored_config,
                    active_envs=active_envs,
                    build_clean_action=lambda current_obs: policy.get_normalized_action(
                        current_obs, deterministic_actions=True
                    ),
                    sample_group_noise=noise_sampler,
                )
            obs = _restore_branch_snapshot(env, policy, snapshot)
            total_env_steps += branch_result.steps_executed
            if branch_result.aborted_reason == "viewer_closed":
                viewer_closed = True
                break
            if branch_result.success:
                if branch_result.saved_transitions > 0:
                    branch_buffers.append(branch_result.buffer)
                    _accumulate_noise_metrics(
                        branch_noise_metrics,
                        branch_result.noise_metrics,
                    )
            else:
                branch_stats["aborted"] += 1

        action_t = policy.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, _, done, _ = env.step(action_t)
        obs = obs_dict["obs"]
        total_env_steps += 1

        final_successes = int(env.successes[0].item())
        max_successes_seen = max(max_successes_seen, final_successes)
        current_goal_idx = min(final_successes, len(goals) - 1)
        stage_idx = min(current_goal_idx, len(GOAL_STAGE_NAMES) - 1)
        stage_name = _goal_stage_name(current_goal_idx)
        goal_dist = float(env.keypoints_max_dist[0].item())
        stage_names.append(stage_name)
        goal_dists.append(goal_dist)
        successes_per_step.append(final_successes)

        if verbose_steps:
            print(
                f"[stage5-long-table] rollout={rollout_idx:04d} step={step:03d} "
                f"subgoal_idx={stage_idx} goal_idx={current_goal_idx} stage={stage_name} "
                f"distance={goal_dist:.4f} successes={final_successes}/{len(goals)}",
                flush=True,
            )
        elif stage_idx != prev_stage_idx:
            print(
                f"[stage5-long-table] rollout={rollout_idx:04d} advanced_to={stage_name} "
                f"(subgoal_idx={stage_idx}, goal_idx={current_goal_idx}, "
                f"distance={goal_dist:.4f}, successes={final_successes}/{len(goals)})",
                flush=True,
            )
        prev_stage_idx = stage_idx

        if bool(done[0].item()):
            break

    success = max_successes_seen >= len(goals)
    failure_stage = _failure_stage_from_successes(max_successes_seen, len(goals))
    return (
        RolloutResult(
            success=success,
            viewer_closed=viewer_closed,
            steps=total_env_steps,
            max_successes_seen=max_successes_seen,
            final_successes=final_successes,
            failure_stage=failure_stage,
            img=None,
            state=None,
            action=None,
            stage_names=stage_names,
            goal_dists=goal_dists,
            successes_per_step=successes_per_step,
        ),
        branch_buffers,
        branch_noise_metrics,
        branch_stats,
    )


def _run_anchored_release_rollout(
    *,
    env,
    args: argparse.Namespace,
    device: str,
    policy,
    start_pose: List[float],
    goals: List[List[float]],
    release_start_goal_idx: int,
    place_goal: List[float],
    rollout_idx: int,
    verbose_steps: bool,
    rng: np.random.Generator,
    anchored_config: AnchoredRecoveryConfig,
):
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    goals_t = torch.tensor(goals, device=env.device, dtype=env.trajectory_states.dtype)
    env.trajectory_states = goals_t
    env.max_consecutive_successes = len(goals)
    env.object_init_state[env_ids, 0:7] = torch.tensor(
        [start_pose], device=env.device, dtype=env.object_init_state.dtype
    )
    env.cfg["env"]["tableObjectZOffset"] = float(start_pose[2] - TABLE_Z)
    env.reset_idx(env_ids, tensor_reset=True)
    policy.reset()

    zero_action = torch.zeros((env.num_envs, N_ACT), device=device)
    obs_dict, _, _, _ = env.step(zero_action)
    obs = obs_dict["obs"]
    total_env_steps = 1
    active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)
    open_hand_action = _compute_open_hand_action(env)
    noise_sampler = _build_group_noise_sampler(args)
    trigger_steps = set(
        sample_branch_trigger_steps(
            rng=rng,
            branch_min_step=anchored_config.branch_min_step,
            branch_max_step=anchored_config.branch_max_step,
            branches_per_rollout=anchored_config.branches_per_rollout,
            min_gap=anchored_config.branch_gap,
        )
    )

    branch_buffers: List = []
    branch_noise_metrics = _zero_noise_metrics()
    branch_stats = {"attempted": 0, "aborted": 0}
    stage_names: List[str] = []
    goal_dists: List[float] = []
    successes_per_step: List[int] = []
    release_phase_per_step: List[bool] = []

    max_successes_seen = 0
    final_successes = 0
    viewer_closed = False
    prev_stage_name: Optional[str] = None
    release_start_step: Optional[int] = None
    dropped_after_lift = False
    drop_step: Optional[int] = None
    drop_successes_before: Optional[int] = None
    reattempted_after_drop = False
    max_object_height_above_init_m = 0.0

    def _build_clean_action(current_obs: torch.Tensor) -> torch.Tensor:
        current_action_t = policy.get_normalized_action(
            current_obs,
            deterministic_actions=True,
        )
        if (
            args.release_steps > 0
            and int(env.successes[0].item()) >= release_start_goal_idx
        ):
            current_action_t = _build_release_action(
                policy_action_t=current_action_t,
                open_hand_action=open_hand_action,
                arm_mode=args.release_arm_mode,
                hand_blend=args.release_hand_blend,
            )
        return current_action_t

    for step in range(args.horizon):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            viewer_closed = True
            break

        current_successes = int(env.successes[0].item())
        in_release_phase = (
            args.release_steps > 0 and current_successes >= release_start_goal_idx
        )
        if in_release_phase and release_start_step is None:
            release_start_step = step

        if not in_release_phase and step in trigger_steps:
            branch_stats["attempted"] += 1
            snapshot = _capture_branch_snapshot(env, policy, obs)
            with maybe_fork_rng(device):
                branch_result = run_anchored_branch(
                    env=env,
                    device=device,
                    obs=obs.clone(),
                    config=anchored_config,
                    active_envs=active_envs,
                    build_clean_action=_build_clean_action,
                    sample_group_noise=noise_sampler,
                )
            obs = _restore_branch_snapshot(env, policy, snapshot)
            total_env_steps += branch_result.steps_executed
            if branch_result.aborted_reason == "viewer_closed":
                viewer_closed = True
                break
            if branch_result.success:
                if branch_result.saved_transitions > 0:
                    branch_buffers.append(branch_result.buffer)
                    _accumulate_noise_metrics(
                        branch_noise_metrics,
                        branch_result.noise_metrics,
                    )
            else:
                branch_stats["aborted"] += 1

        clean_action_t = _build_clean_action(obs)
        obs_dict, _, done, _ = env.step(clean_action_t)
        obs = obs_dict["obs"]
        total_env_steps += 1

        final_successes = int(env.successes[0].item())
        max_successes_seen = max(max_successes_seen, final_successes)
        goal_dist = float(env.keypoints_max_dist[0].item())
        object_height_above_init_m = float(
            env.object_pose[0, 2].item() - env.object_init_state[0, 2].item()
        )
        max_object_height_above_init_m = max(
            max_object_height_above_init_m,
            object_height_above_init_m,
        )
        dropped_now = _drop_detected_after_pickup_attempt(
            object_height_above_init_m=object_height_above_init_m,
            max_object_height_above_init_m=max_object_height_above_init_m,
            lifted_object=bool(env.lifted_object[0].item()),
            in_release_phase=in_release_phase,
        )
        if dropped_now and not dropped_after_lift:
            dropped_after_lift = True
            drop_step = step
            drop_successes_before = final_successes
        elif (
            dropped_after_lift
            and not reattempted_after_drop
            and drop_successes_before is not None
            and final_successes > drop_successes_before
        ):
            reattempted_after_drop = True

        stage_name = (
            "release"
            if in_release_phase
            else _release_goal_stage_name(current_successes, release_start_goal_idx)
        )
        stage_names.append(stage_name)
        goal_dists.append(goal_dist)
        successes_per_step.append(final_successes)
        release_phase_per_step.append(bool(in_release_phase))

        if verbose_steps:
            print(
                f"[stage5-release-long-table] rollout={rollout_idx:04d} step={step:03d} "
                f"stage={stage_name} successes={final_successes}/{len(goals)} "
                f"goal_dist={goal_dist:.4f}",
                flush=True,
            )
        elif stage_name != prev_stage_name:
            print(
                f"[stage5-release-long-table] rollout={rollout_idx:04d} advanced_to={stage_name} "
                f"(goal_dist={goal_dist:.4f}, successes={final_successes}/{len(goals)})",
                flush=True,
            )
        prev_stage_name = stage_name

        if final_successes >= len(goals):
            print(
                f"[stage5-release-long-table] rollout={rollout_idx:04d} completed_release_goals "
                f"step={step:03d}",
                flush=True,
            )
            break
        if bool(done[0].item()):
            break

    final_object_pose_np = env.object_pose[0, 0:7].detach().cpu().numpy()
    final_object_linvel_np = env.object_linvel[0].detach().cpu().numpy()
    place_goal_np = np.asarray(place_goal, dtype=np.float32)
    final_place_xy_error_m = float(
        np.linalg.norm(final_object_pose_np[0:2] - place_goal_np[0:2])
    )
    final_place_z_error_m = float(abs(final_object_pose_np[2] - place_goal_np[2]))
    final_place_pos_error_m = float(
        np.linalg.norm(final_object_pose_np[0:3] - place_goal_np[0:3])
    )
    final_object_speed_mps = float(np.linalg.norm(final_object_linvel_np))
    entered_release_phase = release_start_step is not None
    release_steps_executed = sum(1 for value in release_phase_per_step if value)
    final_object_on_table = _final_object_on_table(
        env,
        final_object_pose_np,
        table_x_half_extent=args.table_x_half_extent,
        table_x_inset_margin=args.table_x_inset_margin,
        table_y_half_extent=args.table_y_half_extent,
        table_y_inset_margin=args.table_y_inset_margin,
    )
    (
        pick_place_success,
        release_goal_success,
        release_stable,
        release_success,
        failure_stage,
    ) = _classify_release_outcome(
        max_successes_seen=max_successes_seen,
        release_start_goal_idx=release_start_goal_idx,
        total_goals=len(goals),
        entered_release_phase=entered_release_phase,
        final_object_on_table=final_object_on_table,
        final_place_xy_error_m=final_place_xy_error_m,
        final_place_z_error_m=final_place_z_error_m,
        final_object_speed_mps=final_object_speed_mps,
        release_xy_tolerance=args.release_xy_tolerance,
        release_z_tolerance=args.release_z_tolerance,
        release_speed_tolerance=args.release_speed_tolerance,
        reattempted_after_drop=reattempted_after_drop,
    )

    return (
        ReleaseRolloutResult(
            success=release_success,
            viewer_closed=viewer_closed,
            steps=total_env_steps,
            pick_place_success=pick_place_success,
            release_goal_success=release_goal_success,
            release_success=release_success,
            release_stable=release_stable,
            final_object_on_table=final_object_on_table,
            entered_release_phase=entered_release_phase,
            release_start_step=release_start_step,
            release_steps_executed=release_steps_executed,
            max_successes_seen=max_successes_seen,
            final_successes=final_successes,
            failure_stage=failure_stage,
            dropped_after_lift=dropped_after_lift,
            drop_step=drop_step,
            drop_successes_before=drop_successes_before,
            reattempted_after_drop=reattempted_after_drop,
            final_object_pose=np.round(final_object_pose_np, 5).tolist(),
            final_object_linvel=np.round(final_object_linvel_np, 5).tolist(),
            final_place_pos_error_m=final_place_pos_error_m,
            final_place_xy_error_m=final_place_xy_error_m,
            final_place_z_error_m=final_place_z_error_m,
            final_object_speed_mps=final_object_speed_mps,
            img=None,
            state=None,
            action=None,
            stage_names=stage_names,
            goal_dists=goal_dists,
            successes_per_step=successes_per_step,
            release_phase_per_step=release_phase_per_step,
            noise_action_delta_l2_sum=branch_noise_metrics["l2_sum"],
            noise_action_delta_l2_sq_sum=branch_noise_metrics["l2_sq_sum"],
            noise_action_delta_linf_sum=branch_noise_metrics["linf_sum"],
            noise_action_delta_count=branch_noise_metrics["count"],
        ),
        branch_buffers,
        branch_noise_metrics,
        branch_stats,
    )


def _run_clean_rollout(
    *,
    env,
    args: argparse.Namespace,
    device: str,
    policy,
    start_pose: List[float],
    goals: List[List[float]],
    rollout_idx: int,
    save_data: bool,
    verbose_steps: bool,
) -> RolloutResult:
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    goals_t = torch.tensor(goals, device=env.device, dtype=env.trajectory_states.dtype)
    env.trajectory_states = goals_t
    env.max_consecutive_successes = len(goals)
    env.object_init_state[env_ids, 0:7] = torch.tensor(
        [start_pose], device=env.device, dtype=env.object_init_state.dtype
    )
    env.cfg["env"]["tableObjectZOffset"] = float(start_pose[2] - TABLE_Z)
    env.reset_idx(env_ids, tensor_reset=True)
    policy.reset()

    zero_action = torch.zeros((env.num_envs, N_ACT), device=device)
    obs_dict, _, _, _ = env.step(zero_action)
    obs = obs_dict["obs"]
    active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)

    images: List[np.ndarray] = []
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    stage_names: List[str] = []
    goal_dists: List[float] = []
    successes_per_step: List[int] = []

    max_successes_seen = 0
    final_successes = 0
    viewer_closed = False
    prev_stage_idx = -1
    steps_executed = 0

    for step in range(args.horizon):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            print("[stage5-long-table] viewer closed; stopping", flush=True)
            viewer_closed = True
            break

        image_t = env.render_dataset_camera_rgb(active_envs)
        action_t = policy.get_normalized_action(obs, deterministic_actions=True)

        if save_data:
            images.append(image_t[0].detach().cpu().numpy().astype(np.uint8))
            states.append(obs[0].detach().cpu().numpy().astype(np.float32))
            actions.append(action_t[0].detach().cpu().numpy().astype(np.float32))

        obs_dict, _, done, _ = env.step(action_t)
        obs = obs_dict["obs"]
        steps_executed += 1

        final_successes = int(env.successes[0].item())
        max_successes_seen = max(max_successes_seen, final_successes)
        current_goal_idx = min(final_successes, len(goals) - 1)
        stage_idx = min(current_goal_idx, len(GOAL_STAGE_NAMES) - 1)
        stage_name = _goal_stage_name(current_goal_idx)
        goal_dist = float(env.keypoints_max_dist[0].item())
        stage_names.append(stage_name)
        goal_dists.append(goal_dist)
        successes_per_step.append(final_successes)

        if verbose_steps:
            print(
                f"[stage5-long-table] rollout={rollout_idx:04d} step={step:03d} "
                f"subgoal_idx={stage_idx} goal_idx={current_goal_idx} stage={stage_name} "
                f"distance={goal_dist:.4f} successes={final_successes}/{len(goals)}",
                flush=True,
            )
        elif stage_idx != prev_stage_idx:
            print(
                f"[stage5-long-table] rollout={rollout_idx:04d} advanced_to={stage_name} "
                f"(subgoal_idx={stage_idx}, goal_idx={current_goal_idx}, "
                f"distance={goal_dist:.4f}, successes={final_successes}/{len(goals)})",
                flush=True,
            )
        prev_stage_idx = stage_idx

        if bool(done[0].item()):
            break

    success = max_successes_seen >= len(goals)
    failure_stage = _failure_stage_from_successes(max_successes_seen, len(goals))
    img = np.stack(images, axis=0).astype(np.uint8) if save_data and images else None
    state = np.stack(states, axis=0).astype(np.float32) if save_data and states else None
    action = np.stack(actions, axis=0).astype(np.float32) if save_data and actions else None
    return RolloutResult(
        success=success,
        viewer_closed=viewer_closed,
        steps=steps_executed,
        max_successes_seen=max_successes_seen,
        final_successes=final_successes,
        failure_stage=failure_stage,
        img=img,
        state=state,
        action=action,
        stage_names=stage_names,
        goal_dists=goal_dists,
        successes_per_step=successes_per_step,
    )


def _build_noisy_worker_cmd(
    args: argparse.Namespace,
    *,
    batch_idx: int,
    start_x: float,
    start_y: float,
    goal_x: float,
) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--collection-type",
        "pick_place",
        "--object-category",
        args.object_category,
        "--object-name",
        args.object_name,
        "--task-name",
        args.task_name,
        "--object-id",
        str(args.object_id),
        "--category-id",
        str(args.category_id),
        "--num-envs",
        str(args.num_envs),
        "--target-transitions",
        str(args.target_transitions),
        "--xy-range",
        str(args.xy_range),
        "--start-z-offset",
        str(args.start_z_offset),
        "--seed",
        str(args.seed),
        "--horizon",
        str(args.horizon),
        "--lift-height",
        str(args.lift_height),
        "--lateral-offset-range",
        str(args.lateral_offset_range),
        "--place-height",
        str(args.place_height),
        "--place-hold-goals",
        str(args.place_hold_goals),
        "--table-x-half-extent",
        str(args.table_x_half_extent),
        "--table-x-inset-margin",
        str(args.table_x_inset_margin),
        "--table-y-half-extent",
        str(args.table_y_half_extent),
        "--table-y-inset-margin",
        str(args.table_y_inset_margin),
        "--place-goal-x-margin",
        str(args.place_goal_x_margin),
        "--place-goal-y-margin",
        str(args.place_goal_y_margin),
        "--min-effective-transport",
        str(args.min_effective_transport),
        "--output-zarr",
        str(args.output_zarr),
        "--gif-fps",
        str(args.gif_fps),
        "--save-preview-every",
        str(args.save_preview_every),
        "--variant",
        args.variant,
        "--noise-scale",
        str(args.noise_scale),
        "--arm-base-noise",
        str(args.arm_base_noise),
        "--arm-wrist-noise",
        str(args.arm_wrist_noise),
        "--thumb-noise",
        str(args.thumb_noise),
        "--index-noise",
        str(args.index_noise),
        "--middle-noise",
        str(args.middle_noise),
        "--ring-noise",
        str(args.ring_noise),
        "--pinky-noise",
        str(args.pinky_noise),
        "--ou-theta",
        str(args.ou_theta),
        "--ou-mu",
        str(args.ou_mu),
        "--ou-dt",
        str(args.ou_dt),
        "--resume",
        "--noisy-worker",
        "--worker-batch-idx",
        str(batch_idx),
        "--worker-start-x",
        str(start_x),
        "--worker-start-y",
        str(start_y),
        "--worker-goal-x",
        str(goal_x),
    ]
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])
    if args.viewer:
        cmd.append("--viewer")
    return cmd


def _collect_clean(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_envs != 1:
        print(
            "[stage5-long-table] forcing --num-envs=1 for clean pick-and-place collection",
            flush=True,
        )
        args.num_envs = 1
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    root = None
    n0 = 0
    e0 = 0
    if not args.dry_run:
        root = _open_or_create_zarr(
            args.output_zarr,
            img_h=DATASET_CAMERA_HEIGHT,
            img_w=DATASET_CAMERA_WIDTH,
            resume=args.resume,
        )
        n0, e0 = _current_counts(root)
        _update_object_registry(root, args)
        _record_geometry_attrs(root, args)

    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )
    bootstrap_goal_x = _default_goal_x(args)
    bootstrap_goals = _build_pick_place_goals(
        nominal_start_pose,
        goal_x=bootstrap_goal_x,
        lift_height=args.lift_height,
        place_height=args.place_height,
        place_hold_goals=args.place_hold_goals,
    )

    print(f"[stage5-long-table] output_zarr={args.output_zarr}", flush=True)
    print(f"[stage5-long-table] variant={args.variant}", flush=True)
    print(f"[stage5-long-table] dry_run={args.dry_run}", flush=True)
    if args.dry_run:
        print(f"[stage5-long-table] dry_run_video_dir={args.dry_run_video_dir}", flush=True)
    print(
        f"[stage5-long-table] starting counts: transitions={n0}, episodes={e0}",
        flush=True,
    )
    print(
        f"[stage5-long-table] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} object_id={args.object_id} category_id={args.category_id}",
        flush=True,
    )
    print(
        f"[stage5-long-table] target_transitions={args.target_transitions} "
        f"xy_range=+/-{args.xy_range:.3f}m "
        f"lift_height={args.lift_height:.3f}m place_height={args.place_height:.3f}m "
        f"place_hold_goals={args.place_hold_goals} horizon={args.horizon}",
        flush=True,
    )

    env = _make_long_table_env(
        num_envs=1,
        nominal_start_pose=nominal_start_pose,
        goal_poses=bootstrap_goals,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
        seed=args.seed,
        object_name=args.object_name,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    env.set_env_state(checkpoint[0]["env_state"])
    print(
        f"[stage5-long-table] camera pos={_jsonable(env.cfg['env']['datasetCameraPosition'])} "
        f"target={_jsonable(env.cfg['env']['datasetCameraTarget'])}",
        flush=True,
    )
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
        num_envs=1,
    )

    attempted_rollouts = 0
    successful_completions = 0
    written_episodes = 0
    discarded_unsuccessful = 0
    discarded_invalid = 0
    failure_breakdown: Dict[str, int] = {name: 0 for name in GOAL_STAGE_NAMES}
    total_env_steps = 0
    t_start = time.time()
    max_rollouts = args.dry_run_rollouts if args.dry_run else None

    try:
        while True:
            n_transitions = _current_counts(root)[0] if root is not None else 0
            if not args.dry_run and n_transitions >= args.target_transitions:
                break
            if max_rollouts is not None and attempted_rollouts >= max_rollouts:
                break
            if args.max_attempted_episodes is not None and attempted_rollouts >= args.max_attempted_episodes:
                break
            if args.max_steps is not None and total_env_steps >= args.max_steps:
                break

            rollout_idx = attempted_rollouts
            start_pose, goal_x, lateral_offset = _sample_end_to_end_start_and_goal(
                rng,
                nominal_start_pose,
                args,
            )
            goals = _build_pick_place_goals(
                start_pose,
                goal_x=goal_x,
                lift_height=args.lift_height,
                place_height=args.place_height,
                place_hold_goals=args.place_hold_goals,
            )

            print(
                f"\n[stage5-long-table] rollout={rollout_idx:04d} "
                f"start_pose={np.array(start_pose).round(4).tolist()} "
                f"goal_x={goal_x:+.3f} lateral_offset={lateral_offset:+.3f}",
                flush=True,
            )
            if args.dry_run:
                print(f"[stage5-long-table] rollout={rollout_idx:04d} goals={goals}", flush=True)

            result = _run_clean_rollout(
                env=env,
                args=args,
                device=device,
                policy=policy,
                start_pose=start_pose,
                goals=goals,
                rollout_idx=rollout_idx,
                save_data=True,
                verbose_steps=args.dry_run or args.log_every_step,
            )
            attempted_rollouts += 1
            total_env_steps += result.steps

            if result.viewer_closed:
                break

            if result.success:
                successful_completions += 1
                if root is not None and result.img is not None and result.state is not None and result.action is not None:
                    appended = _append_episode(
                        root,
                        result.img,
                        result.state,
                        result.action,
                        object_id=args.object_id,
                        category_id=args.category_id,
                    )
                    if appended:
                        written_episodes += 1
                    else:
                        discarded_invalid += 1
            else:
                discarded_unsuccessful += 1
                if result.failure_stage is not None:
                    failure_breakdown[result.failure_stage] += 1

            if args.dry_run and result.img is not None:
                status = "success" if result.success else f"failed_{result.failure_stage}"
                video_path = args.dry_run_video_dir / f"rollout_{rollout_idx:04d}_{status}.mp4"
                annotated_frames = _annotate_rollout_frames(
                    result.img,
                    rollout_idx=rollout_idx,
                    stage_names=result.stage_names or [],
                    goal_dists=result.goal_dists or [],
                    successes_per_step=result.successes_per_step or [],
                )
                _write_rollout_video(video_path, annotated_frames, args.dry_run_video_fps)

            if root is not None:
                n_transitions, n_episodes = _current_counts(root)
                elapsed = max(time.time() - t_start, 1e-6)
                rate = (n_transitions - n0) / elapsed
                remaining = max(args.target_transitions - n_transitions, 0)
                eta_min = remaining / max(rate, 1e-6) / 60.0
                root.attrs["attempted_rollouts"] = attempted_rollouts
                root.attrs["successful_completions"] = successful_completions
                root.attrs["written_episodes"] = written_episodes
                root.attrs["discarded_unsuccessful"] = discarded_unsuccessful
                root.attrs["discarded_invalid"] = discarded_invalid
                root.attrs["failure_lift"] = failure_breakdown["lift"]
                root.attrs["failure_transport"] = failure_breakdown["transport"]
                root.attrs["failure_place"] = failure_breakdown["place"]
                root.attrs["xy_range"] = args.xy_range
                root.attrs["horizon"] = args.horizon
                root.attrs["lift_height_m"] = args.lift_height
                root.attrs["lateral_offset_range_m"] = args.lateral_offset_range
                root.attrs["place_height_m"] = args.place_height
                root.attrs["start_z_offset"] = args.start_z_offset
                root.attrs["variant"] = args.variant
                root.attrs["noise_strategy"] = args.noise_strategy
                root.attrs["executed_action_clipped"] = args.variant != "clean"
                root.attrs["collection_task"] = "pick_place"
                print(
                    f"[stage5-long-table] transitions={n_transitions}/{args.target_transitions} "
                    f"episodes={n_episodes} attempted={attempted_rollouts} "
                    f"completed={successful_completions} written={written_episodes} "
                    f"rate={rate:.1f} trans/sec eta={eta_min:.1f}min",
                    flush=True,
                )
    finally:
        _destroy_env(env)


def _collect_anchored_pick_place(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    anchored_config = build_anchored_recovery_config(args)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    root = None
    n0 = 0
    e0 = 0
    if not args.dry_run:
        root = _open_or_create_zarr(
            args.output_zarr,
            img_h=DATASET_CAMERA_HEIGHT,
            img_w=DATASET_CAMERA_WIDTH,
            resume=args.resume,
        )
        n0, e0 = _current_counts(root)
        if args.resume:
            validate_anchored_resume_attrs(
                root,
                variant=args.variant,
                config=anchored_config,
            )
        _update_object_registry(root, args)
        _record_geometry_attrs(root, args)
        root.attrs["collection_task"] = "pick_place"
        update_anchored_root_attrs(
            root,
            variant=args.variant,
            config=anchored_config,
            base_rollouts_attempted=0,
            base_rollouts_succeeded=0,
            branches_attempted=0,
            branches_written=0,
            branches_aborted=0,
            saved_transitions=0,
        )

    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )
    bootstrap_goal_x = _default_goal_x(args)
    bootstrap_goals = _build_pick_place_goals(
        nominal_start_pose,
        goal_x=bootstrap_goal_x,
        lift_height=args.lift_height,
        place_height=args.place_height,
        place_hold_goals=args.place_hold_goals,
    )

    print(f"[stage5-long-table-anchored] output_zarr={args.output_zarr}", flush=True)
    print(
        f"[stage5-long-table-anchored] starting counts: transitions={n0}, episodes={e0}",
        flush=True,
    )
    print(
        f"[stage5-long-table-anchored] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} object_id={args.object_id} category_id={args.category_id}",
        flush=True,
    )
    print(
        f"[stage5-long-table-anchored] target_transitions={args.target_transitions} "
        f"branches_per_rollout={anchored_config.branches_per_rollout} "
        f"perturb_steps={anchored_config.perturb_steps} "
        f"recovery_steps={anchored_config.recovery_steps} "
        f"branch_window=[{anchored_config.branch_min_step}, {anchored_config.branch_max_step}]",
        flush=True,
    )

    env = _make_long_table_env(
        num_envs=1,
        nominal_start_pose=nominal_start_pose,
        goal_poses=bootstrap_goals,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
        seed=args.seed,
        object_name=args.object_name,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    env.set_env_state(checkpoint[0]["env_state"])
    print(
        f"[stage5-long-table-anchored] camera pos={_jsonable(env.cfg['env']['datasetCameraPosition'])} "
        f"target={_jsonable(env.cfg['env']['datasetCameraTarget'])}",
        flush=True,
    )
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
        num_envs=1,
    )

    attempted_rollouts = 0
    successful_completions = 0
    written_episodes = 0
    discarded_unsuccessful = 0
    discarded_invalid = 0
    failure_breakdown: Dict[str, int] = {name: 0 for name in GOAL_STAGE_NAMES}
    anchored_base_rollouts_attempted = 0
    anchored_base_rollouts_succeeded = 0
    anchored_branches_attempted = 0
    anchored_branches_written = 0
    anchored_branches_aborted = 0
    anchored_saved_transitions = 0
    total_env_steps = 0
    t_start = time.time()
    max_rollouts = args.dry_run_rollouts if args.dry_run else None

    try:
        while True:
            n_transitions = _current_counts(root)[0] if root is not None else 0
            if not args.dry_run and n_transitions >= args.target_transitions:
                break
            if max_rollouts is not None and attempted_rollouts >= max_rollouts:
                break
            if (
                args.max_attempted_episodes is not None
                and attempted_rollouts >= args.max_attempted_episodes
            ):
                break
            if args.max_steps is not None and total_env_steps >= args.max_steps:
                break

            rollout_idx = attempted_rollouts
            start_pose, goal_x, lateral_offset = _sample_end_to_end_start_and_goal(
                rng,
                nominal_start_pose,
                args,
            )
            goals = _build_pick_place_goals(
                start_pose,
                goal_x=goal_x,
                lift_height=args.lift_height,
                place_height=args.place_height,
                place_hold_goals=args.place_hold_goals,
            )

            print(
                f"\n[stage5-long-table-anchored] rollout={rollout_idx:04d} "
                f"start_pose={np.array(start_pose).round(4).tolist()} "
                f"goal_x={goal_x:+.3f} lateral_offset={lateral_offset:+.3f}",
                flush=True,
            )

            result, branch_buffers, branch_noise_metrics, branch_stats = (
                _run_anchored_pick_place_rollout(
                    env=env,
                    args=args,
                    device=device,
                    policy=policy,
                    start_pose=start_pose,
                    goals=goals,
                    rollout_idx=rollout_idx,
                    verbose_steps=args.log_every_step,
                    rng=rng,
                    anchored_config=anchored_config,
                )
            )
            attempted_rollouts += 1
            anchored_base_rollouts_attempted += 1
            total_env_steps += result.steps
            anchored_branches_attempted += branch_stats["attempted"]
            anchored_branches_aborted += branch_stats["aborted"]

            if result.viewer_closed:
                break

            if result.success:
                successful_completions += 1
                anchored_base_rollouts_succeeded += 1
                if root is not None:
                    wrote_branch = False
                    for branch_buffer in branch_buffers:
                        img_ep, state_ep, action_ep = branch_buffer.as_episode()
                        appended = _append_episode(
                            root,
                            img_ep,
                            state_ep,
                            action_ep,
                            object_id=args.object_id,
                            category_id=args.category_id,
                        )
                        if appended:
                            wrote_branch = True
                            written_episodes += 1
                            anchored_branches_written += 1
                            anchored_saved_transitions += int(img_ep.shape[0])
                        else:
                            discarded_invalid += 1
                    if wrote_branch and branch_noise_metrics["count"] > 0:
                        _update_noisy_metric_attrs(root, branch_noise_metrics)
            else:
                discarded_unsuccessful += 1
                if result.failure_stage is not None:
                    failure_breakdown[result.failure_stage] += 1

            if root is not None:
                n_transitions, n_episodes = _current_counts(root)
                elapsed = max(time.time() - t_start, 1e-6)
                rate = (n_transitions - n0) / elapsed
                remaining = max(args.target_transitions - n_transitions, 0)
                eta_min = remaining / max(rate, 1e-6) / 60.0
                root.attrs["attempted_rollouts"] = attempted_rollouts
                root.attrs["successful_completions"] = successful_completions
                root.attrs["written_episodes"] = written_episodes
                root.attrs["discarded_unsuccessful"] = discarded_unsuccessful
                root.attrs["discarded_invalid"] = discarded_invalid
                root.attrs["failure_lift"] = failure_breakdown["lift"]
                root.attrs["failure_transport"] = failure_breakdown["transport"]
                root.attrs["failure_place"] = failure_breakdown["place"]
                root.attrs["xy_range"] = args.xy_range
                root.attrs["horizon"] = args.horizon
                root.attrs["lift_height_m"] = args.lift_height
                root.attrs["lateral_offset_range_m"] = args.lateral_offset_range
                root.attrs["place_height_m"] = args.place_height
                root.attrs["start_z_offset"] = args.start_z_offset
                root.attrs["collection_task"] = "pick_place"
                update_anchored_root_attrs(
                    root,
                    variant=args.variant,
                    config=anchored_config,
                    base_rollouts_attempted=anchored_base_rollouts_attempted,
                    base_rollouts_succeeded=anchored_base_rollouts_succeeded,
                    branches_attempted=anchored_branches_attempted,
                    branches_written=anchored_branches_written,
                    branches_aborted=anchored_branches_aborted,
                    saved_transitions=anchored_saved_transitions,
                )
                print(
                    f"[stage5-long-table-anchored] transitions={n_transitions}/{args.target_transitions} "
                    f"episodes={n_episodes} rollouts={attempted_rollouts} "
                    f"base_successes={successful_completions} branches_written={anchored_branches_written} "
                    f"branches_aborted={anchored_branches_aborted} rate={rate:.1f} trans/sec "
                    f"eta={eta_min:.1f}min",
                    flush=True,
                )
    finally:
        _destroy_env(env)


def _collect_noisy_worker(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert args.worker_batch_idx is not None
    assert args.worker_start_x is not None
    assert args.worker_start_y is not None
    assert args.worker_goal_x is not None

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = _open_or_create_zarr(
        args.output_zarr,
        img_h=DATASET_CAMERA_HEIGHT,
        img_w=DATASET_CAMERA_WIDTH,
        resume=True,
    )
    _update_object_registry(root, args)
    _record_geometry_attrs(root, args)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )
    start_pose = list(nominal_start_pose)
    start_pose[0] = float(args.worker_start_x)
    start_pose[1] = float(args.worker_start_y)
    goal_x = float(args.worker_goal_x)
    goals = _build_pick_place_goals(
        start_pose,
        goal_x=goal_x,
        lift_height=args.lift_height,
        place_height=args.place_height,
        place_hold_goals=args.place_hold_goals,
    )

    env = _make_long_table_env(
        num_envs=args.num_envs,
        nominal_start_pose=start_pose,
        goal_poses=goals,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
        seed=args.seed + int(args.worker_batch_idx),
        object_name=args.object_name,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.set_env_state(checkpoint[0]["env_state"])
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
        num_envs=env.num_envs,
    )
    policy.reset()

    zero_action = torch.zeros((env.num_envs, N_ACT), device=device)
    obs_dict, _, _, _ = env.step(zero_action)
    obs = obs_dict["obs"]
    noise_state = torch.zeros((env.num_envs, N_ACT), device=device)
    sqrt_dt = math.sqrt(args.ou_dt)

    per_env_imgs = [[] for _ in range(env.num_envs)]
    per_env_states = [[] for _ in range(env.num_envs)]
    per_env_actions = [[] for _ in range(env.num_envs)]
    per_env_max_successes = [0 for _ in range(env.num_envs)]
    batch_noise_metrics = {"l2_sum": 0.0, "l2_sq_sum": 0.0, "linf_sum": 0.0, "count": 0}
    active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)

    for _ in range(args.horizon):
        image_t = env.render_dataset_camera_rgb(active_envs)
        clean_action_t = policy.get_normalized_action(obs, deterministic_actions=True)
        sigma_noise = _sample_group_noise(
            clean_action_t,
            arm_base_noise=args.arm_base_noise,
            arm_wrist_noise=args.arm_wrist_noise,
            thumb_noise=args.thumb_noise,
            index_noise=args.index_noise,
            middle_noise=args.middle_noise,
            ring_noise=args.ring_noise,
            pinky_noise=args.pinky_noise,
        )
        noise_state = (
            noise_state
            + args.ou_theta * (args.ou_mu - noise_state) * args.ou_dt
            + sigma_noise * sqrt_dt
        )
        executed_action_t = torch.clamp(clean_action_t + noise_state, -1.0, 1.0)
        action_to_save_t = executed_action_t if args.variant == "noisy_noisy" else clean_action_t

        delta_t = executed_action_t - clean_action_t
        delta_l2_t = torch.linalg.vector_norm(delta_t, dim=1)
        delta_linf_t = torch.abs(delta_t).amax(dim=1)
        batch_noise_metrics["l2_sum"] += float(delta_l2_t.sum().item())
        batch_noise_metrics["l2_sq_sum"] += float(torch.square(delta_l2_t).sum().item())
        batch_noise_metrics["linf_sum"] += float(delta_linf_t.sum().item())
        batch_noise_metrics["count"] += int(delta_l2_t.numel())

        image_np = image_t.detach().cpu().numpy().astype(np.uint8)
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        action_np = action_to_save_t.detach().cpu().numpy().astype(np.float32)
        for env_i in range(env.num_envs):
            per_env_imgs[env_i].append(image_np[env_i])
            per_env_states[env_i].append(obs_np[env_i])
            per_env_actions[env_i].append(action_np[env_i])

        obs_dict, _, _, _ = env.step(executed_action_t)
        obs = obs_dict["obs"]
        successes_np = env.successes.detach().cpu().numpy().astype(int)
        for env_i in range(env.num_envs):
            per_env_max_successes[env_i] = max(per_env_max_successes[env_i], int(successes_np[env_i]))

    for env_i in range(env.num_envs):
        if per_env_max_successes[env_i] < len(goals):
            continue
        _append_episode(
            root,
            np.stack(per_env_imgs[env_i], axis=0).astype(np.uint8),
            np.stack(per_env_states[env_i], axis=0).astype(np.float32),
            np.stack(per_env_actions[env_i], axis=0).astype(np.float32),
            object_id=args.object_id,
            category_id=args.category_id,
        )

    _update_noisy_metric_attrs(root, batch_noise_metrics)
    _destroy_env(env)


def _collect_noisy_parent(args: argparse.Namespace) -> None:
    root = _open_or_create_zarr(
        args.output_zarr,
        img_h=DATASET_CAMERA_HEIGHT,
        img_w=DATASET_CAMERA_WIDTH,
        resume=args.resume,
    )
    _update_object_registry(root, args)
    _record_geometry_attrs(root, args)

    attempted_episodes = int(root.attrs.get("attempted_episodes", 0))
    written_episodes = int(root.attrs.get("written_episodes", 0))
    batch_idx = int(root.attrs.get("last_batch_idx", -1)) + 1
    rng = np.random.default_rng(args.seed)
    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )
    for _ in range(batch_idx):
        _sample_end_to_end_start_and_goal(rng, nominal_start_pose, args)

    t_start = time.time()
    while True:
        n_before, e_before = _current_counts(root)
        if n_before >= args.target_transitions:
            break
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        if args.max_attempted_episodes is not None and attempted_episodes >= args.max_attempted_episodes:
            break

        start_pose, goal_x, _ = _sample_end_to_end_start_and_goal(rng, nominal_start_pose, args)
        subprocess.run(
            _build_noisy_worker_cmd(
                args,
                batch_idx=batch_idx,
                start_x=float(start_pose[0]),
                start_y=float(start_pose[1]),
                goal_x=goal_x,
            ),
            check=True,
        )

        n_after, e_after = _current_counts(root)
        attempted_episodes += args.num_envs
        written_episodes += max(e_after - e_before, 0)
        elapsed = max(time.time() - t_start, 1e-6)
        rate = (n_after - n_before) / elapsed
        remaining = max(args.target_transitions - n_after, 0)
        eta_min = remaining / max(rate, 1e-6) / 60.0
        root.attrs["last_batch_idx"] = batch_idx
        root.attrs["attempted_episodes"] = attempted_episodes
        root.attrs["written_episodes"] = written_episodes
        root.attrs["xy_range"] = args.xy_range
        root.attrs["horizon"] = args.horizon
        root.attrs["lift_height_m"] = args.lift_height
        root.attrs["lateral_offset_range_m"] = args.lateral_offset_range
        root.attrs["place_height_m"] = args.place_height
        root.attrs["start_z_offset"] = args.start_z_offset
        root.attrs["variant"] = args.variant
        root.attrs["noise_strategy"] = args.noise_strategy
        root.attrs["executed_action_clipped"] = True
        root.attrs["collection_task"] = "pick_place"
        print(
            f"[stage5-long-table-noisy] batch={batch_idx:04d} total_transitions={n_after}/{args.target_transitions} "
            f"written={written_episodes} attempted={attempted_episodes} "
            f"rate={rate:.1f} transitions/sec eta={eta_min:.1f} min",
            flush=True,
        )
        batch_idx += 1


def _final_object_on_table(
    env,
    object_pose_np: np.ndarray,
    *,
    table_x_half_extent: float,
    table_x_inset_margin: float,
    table_y_half_extent: float,
    table_y_inset_margin: float,
) -> bool:
    table_center_np = env.table_init_state[0, 0:3].detach().cpu().numpy()
    usable_x_half_extent = max(float(table_x_half_extent) - float(table_x_inset_margin), 0.0)
    usable_y_half_extent = max(float(table_y_half_extent) - float(table_y_inset_margin), 0.0)
    table_x_min = float(table_center_np[0]) - usable_x_half_extent
    table_x_max = float(table_center_np[0]) + usable_x_half_extent
    table_y_min = float(table_center_np[1]) - usable_y_half_extent
    table_y_max = float(table_center_np[1]) + usable_y_half_extent

    offset_x_min, offset_x_max, offset_y_min, offset_y_max = _object_xy_offset_bounds_for_quat(
        env,
        np.asarray(object_pose_np[3:7], dtype=np.float32),
    )
    x_min = float(object_pose_np[0]) + offset_x_min
    x_max = float(object_pose_np[0]) + offset_x_max
    y_min = float(object_pose_np[1]) + offset_y_min
    y_max = float(object_pose_np[1]) + offset_y_max

    return (
        x_min >= table_x_min
        and x_max <= table_x_max
        and y_min >= table_y_min
        and y_max <= table_y_max
    )


def _place_goal_in_safe_zone(
    env,
    place_goal: List[float],
    *,
    table_x_half_extent: float,
    table_x_inset_margin: float,
    table_y_half_extent: float,
    table_y_inset_margin: float,
) -> bool:
    place_goal_np = np.asarray(place_goal, dtype=np.float32)
    return _final_object_on_table(
        env,
        place_goal_np,
        table_x_half_extent=table_x_half_extent,
        table_x_inset_margin=table_x_inset_margin,
        table_y_half_extent=table_y_half_extent,
        table_y_inset_margin=table_y_inset_margin,
    )


def _compute_open_hand_action(env) -> torch.Tensor:
    desired_hand_targets = env.hand_arm_default_dof_pos[7 : env.num_hand_arm_dofs]
    lower = env.arm_hand_dof_lower_limits[7 : env.num_hand_arm_dofs]
    upper = env.arm_hand_dof_upper_limits[7 : env.num_hand_arm_dofs]
    normalized = (2.0 * desired_hand_targets - upper - lower) / (upper - lower)
    return torch.clamp(normalized, -1.0, 1.0)


def _build_release_action(
    *,
    policy_action_t: torch.Tensor,
    open_hand_action: torch.Tensor,
    arm_mode: str,
    hand_blend: float,
) -> torch.Tensor:
    action_t = policy_action_t.clone()
    if arm_mode == "hold":
        action_t[:, :7] = 0.0
    elif arm_mode != "policy":
        raise ValueError(f"Unsupported --release-arm-mode={arm_mode}")
    open_hand_action = open_hand_action.unsqueeze(0).expand(action_t.shape[0], -1)
    action_t[:, 7:] = (1.0 - hand_blend) * action_t[:, 7:] + hand_blend * open_hand_action
    return torch.clamp(action_t, -1.0, 1.0)


def _run_release_rollout(
    *,
    env,
    args: argparse.Namespace,
    device: str,
    policy,
    start_pose: List[float],
    goals: List[List[float]],
    release_start_goal_idx: int,
    place_goal: List[float],
    rollout_idx: int,
    save_data: bool,
    verbose_steps: bool,
) -> ReleaseRolloutResult:
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    goals_t = torch.tensor(goals, device=env.device, dtype=env.trajectory_states.dtype)
    env.trajectory_states = goals_t
    env.max_consecutive_successes = len(goals)
    env.object_init_state[env_ids, 0:7] = torch.tensor(
        [start_pose], device=env.device, dtype=env.object_init_state.dtype
    )
    env.cfg["env"]["tableObjectZOffset"] = float(start_pose[2] - TABLE_Z)
    env.reset_idx(env_ids, tensor_reset=True)
    policy.reset()

    zero_action = torch.zeros((env.num_envs, N_ACT), device=device)
    obs_dict, _, _, _ = env.step(zero_action)
    obs = obs_dict["obs"]
    active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)
    open_hand_action = _compute_open_hand_action(env)
    noise_state = torch.zeros((env.num_envs, N_ACT), device=device)
    sqrt_dt = math.sqrt(args.ou_dt)
    noise_metrics = {"l2_sum": 0.0, "l2_sq_sum": 0.0, "linf_sum": 0.0, "count": 0}

    images: List[np.ndarray] = []
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    stage_names: List[str] = []
    goal_dists: List[float] = []
    successes_per_step: List[int] = []
    release_phase_per_step: List[bool] = []

    max_successes_seen = 0
    final_successes = 0
    viewer_closed = False
    prev_stage_name: Optional[str] = None
    release_start_step: Optional[int] = None
    steps_executed = 0
    dropped_after_lift = False
    drop_step: Optional[int] = None
    drop_successes_before: Optional[int] = None
    reattempted_after_drop = False
    max_object_height_above_init_m = 0.0

    for step in range(args.horizon):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            viewer_closed = True
            break

        current_successes = int(env.successes[0].item())
        in_release_phase = args.release_steps > 0 and current_successes >= release_start_goal_idx
        if in_release_phase and release_start_step is None:
            release_start_step = step

        image_t = env.render_dataset_camera_rgb(active_envs)
        clean_action_t = policy.get_normalized_action(obs, deterministic_actions=True)
        if in_release_phase:
            clean_action_t = _build_release_action(
                policy_action_t=clean_action_t,
                open_hand_action=open_hand_action,
                arm_mode=args.release_arm_mode,
                hand_blend=args.release_hand_blend,
            )

        if args.variant == "clean":
            executed_action_t = clean_action_t
            action_to_save_t = clean_action_t
        else:
            sigma_noise = _sample_group_noise(
                clean_action_t,
                arm_base_noise=args.arm_base_noise,
                arm_wrist_noise=args.arm_wrist_noise,
                thumb_noise=args.thumb_noise,
                index_noise=args.index_noise,
                middle_noise=args.middle_noise,
                ring_noise=args.ring_noise,
                pinky_noise=args.pinky_noise,
            )
            noise_state = (
                noise_state
                + args.ou_theta * (args.ou_mu - noise_state) * args.ou_dt
                + sigma_noise * sqrt_dt
            )
            executed_action_t = torch.clamp(clean_action_t + noise_state, -1.0, 1.0)
            action_to_save_t = executed_action_t if args.variant == "noisy_noisy" else clean_action_t

            delta_t = executed_action_t - clean_action_t
            delta_l2_t = torch.linalg.vector_norm(delta_t, dim=1)
            delta_linf_t = torch.abs(delta_t).amax(dim=1)
            noise_metrics["l2_sum"] += float(delta_l2_t.sum().item())
            noise_metrics["l2_sq_sum"] += float(torch.square(delta_l2_t).sum().item())
            noise_metrics["linf_sum"] += float(delta_linf_t.sum().item())
            noise_metrics["count"] += int(delta_l2_t.numel())

        if save_data:
            images.append(image_t[0].detach().cpu().numpy().astype(np.uint8))
            states.append(obs[0].detach().cpu().numpy().astype(np.float32))
            actions.append(action_to_save_t[0].detach().cpu().numpy().astype(np.float32))

        obs_dict, _, done, _ = env.step(executed_action_t)
        obs = obs_dict["obs"]
        steps_executed += 1

        final_successes = int(env.successes[0].item())
        max_successes_seen = max(max_successes_seen, final_successes)
        goal_dist = float(env.keypoints_max_dist[0].item())
        object_height_above_init_m = float(
            env.object_pose[0, 2].item() - env.object_init_state[0, 2].item()
        )
        max_object_height_above_init_m = max(
            max_object_height_above_init_m,
            object_height_above_init_m,
        )
        dropped_now = _drop_detected_after_pickup_attempt(
            object_height_above_init_m=object_height_above_init_m,
            max_object_height_above_init_m=max_object_height_above_init_m,
            lifted_object=bool(env.lifted_object[0].item()),
            in_release_phase=in_release_phase,
        )
        if dropped_now and not dropped_after_lift:
            dropped_after_lift = True
            drop_step = step
            drop_successes_before = final_successes
        elif (
            dropped_after_lift
            and not reattempted_after_drop
            and drop_successes_before is not None
            and final_successes > drop_successes_before
        ):
            reattempted_after_drop = True
        stage_name = (
            "release" if in_release_phase else _release_goal_stage_name(current_successes, release_start_goal_idx)
        )
        stage_names.append(stage_name)
        goal_dists.append(goal_dist)
        successes_per_step.append(final_successes)
        release_phase_per_step.append(bool(in_release_phase))

        if verbose_steps:
            print(
                f"[stage5-release-long-table] rollout={rollout_idx:04d} step={step:03d} "
                f"stage={stage_name} successes={final_successes}/{len(goals)} goal_dist={goal_dist:.4f}",
                flush=True,
            )
        elif stage_name != prev_stage_name:
            print(
                f"[stage5-release-long-table] rollout={rollout_idx:04d} advanced_to={stage_name} "
                f"(goal_dist={goal_dist:.4f}, successes={final_successes}/{len(goals)})",
                flush=True,
            )
        prev_stage_name = stage_name

        if final_successes >= len(goals):
            print(
                f"[stage5-release-long-table] rollout={rollout_idx:04d} completed_release_goals "
                f"step={step:03d}",
                flush=True,
            )
            break

        if bool(done[0].item()):
            break

    final_object_pose_np = env.object_pose[0, 0:7].detach().cpu().numpy()
    final_object_linvel_np = env.object_linvel[0].detach().cpu().numpy()
    place_goal_np = np.asarray(place_goal, dtype=np.float32)
    final_place_xy_error_m = float(np.linalg.norm(final_object_pose_np[0:2] - place_goal_np[0:2]))
    final_place_z_error_m = float(abs(final_object_pose_np[2] - place_goal_np[2]))
    final_place_pos_error_m = float(np.linalg.norm(final_object_pose_np[0:3] - place_goal_np[0:3]))
    final_object_speed_mps = float(np.linalg.norm(final_object_linvel_np))

    entered_release_phase = release_start_step is not None
    release_steps_executed = sum(1 for value in release_phase_per_step if value)
    final_object_on_table = _final_object_on_table(
        env,
        final_object_pose_np,
        table_x_half_extent=args.table_x_half_extent,
        table_x_inset_margin=args.table_x_inset_margin,
        table_y_half_extent=args.table_y_half_extent,
        table_y_inset_margin=args.table_y_inset_margin,
    )
    (
        pick_place_success,
        release_goal_success,
        release_stable,
        release_success,
        failure_stage,
    ) = _classify_release_outcome(
        max_successes_seen=max_successes_seen,
        release_start_goal_idx=release_start_goal_idx,
        total_goals=len(goals),
        entered_release_phase=entered_release_phase,
        final_object_on_table=final_object_on_table,
        final_place_xy_error_m=final_place_xy_error_m,
        final_place_z_error_m=final_place_z_error_m,
        final_object_speed_mps=final_object_speed_mps,
        release_xy_tolerance=args.release_xy_tolerance,
        release_z_tolerance=args.release_z_tolerance,
        release_speed_tolerance=args.release_speed_tolerance,
        reattempted_after_drop=reattempted_after_drop,
    )

    img = np.stack(images, axis=0).astype(np.uint8) if save_data and images else None
    state = np.stack(states, axis=0).astype(np.float32) if save_data and states else None
    action = np.stack(actions, axis=0).astype(np.float32) if save_data and actions else None
    return ReleaseRolloutResult(
        success=release_success,
        viewer_closed=viewer_closed,
        steps=steps_executed,
        pick_place_success=pick_place_success,
        release_goal_success=release_goal_success,
        release_success=release_success,
        release_stable=release_stable,
        final_object_on_table=final_object_on_table,
        entered_release_phase=entered_release_phase,
        release_start_step=release_start_step,
        release_steps_executed=release_steps_executed,
        max_successes_seen=max_successes_seen,
        final_successes=final_successes,
        failure_stage=failure_stage,
        dropped_after_lift=dropped_after_lift,
        drop_step=drop_step,
        drop_successes_before=drop_successes_before,
        reattempted_after_drop=reattempted_after_drop,
        final_object_pose=np.round(final_object_pose_np, 5).tolist(),
        final_object_linvel=np.round(final_object_linvel_np, 5).tolist(),
        final_place_pos_error_m=final_place_pos_error_m,
        final_place_xy_error_m=final_place_xy_error_m,
        final_place_z_error_m=final_place_z_error_m,
        final_object_speed_mps=final_object_speed_mps,
        img=img,
        state=state,
        action=action,
        stage_names=stage_names,
        goal_dists=goal_dists,
        successes_per_step=successes_per_step,
        release_phase_per_step=release_phase_per_step,
        noise_action_delta_l2_sum=noise_metrics["l2_sum"],
        noise_action_delta_l2_sq_sum=noise_metrics["l2_sq_sum"],
        noise_action_delta_linf_sum=noise_metrics["linf_sum"],
        noise_action_delta_count=noise_metrics["count"],
    )


def _collect_pick_place_release(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    if args.variant != "clean":
        args = _resolve_noise_args(args)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_envs != 1:
        print("[stage5-release-long-table] forcing --num-envs=1", flush=True)
        args.num_envs = 1

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    root = None
    n0 = 0
    e0 = 0
    if not args.dry_run:
        root = _open_or_create_zarr(
            args.output_zarr,
            img_h=DATASET_CAMERA_HEIGHT,
            img_w=DATASET_CAMERA_WIDTH,
            resume=args.resume,
        )
        n0, e0 = _current_counts(root)
        _update_object_registry(root, args)
        _record_geometry_attrs(root, args)

    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )
    bootstrap_goal_x = _default_goal_x(args)
    bootstrap_goals, _, _ = _build_pick_place_release_goals(
        nominal_start_pose,
        goal_x=bootstrap_goal_x,
        lift_height=args.lift_height,
        place_height=args.place_height,
        place_hold_goals=args.place_hold_goals,
        release_hold_goals=args.release_steps,
    )

    print(f"[stage5-release-long-table] output_zarr={args.output_zarr}", flush=True)
    print(f"[stage5-release-long-table] dry_run={args.dry_run}", flush=True)
    print(
        f"[stage5-release-long-table] starting counts: transitions={n0}, episodes={e0}",
        flush=True,
    )
    if args.horizon < DEFAULT_HORIZON_WITH_RELEASE:
        print(
            f"[stage5-release-long-table] warning: --horizon={args.horizon} is the total "
            f"budget for lift+transport+place+release. The release-aware default is "
            f"{DEFAULT_HORIZON_WITH_RELEASE}, so shorter horizons can enter release late "
            f"and then time out.",
            flush=True,
        )

    env = _make_long_table_env(
        num_envs=1,
        nominal_start_pose=nominal_start_pose,
        goal_poses=bootstrap_goals,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
        seed=args.seed,
        object_name=args.object_name,
    )
    print("[stage5-release-long-table] env constructed; refreshing root state tensor", flush=True)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    print(f"[stage5-release-long-table] loading env checkpoint from {CHECKPOINT_PATH}", flush=True)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    print("[stage5-release-long-table] env checkpoint loaded; restoring env state", flush=True)
    env.set_env_state(checkpoint[0]["env_state"])
    print("[stage5-release-long-table] env state restored; creating RL player", flush=True)
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
        num_envs=1,
    )
    print("[stage5-release-long-table] RL player created", flush=True)

    attempted_rollouts = 0
    successful_completions = 0
    release_stable_count = 0
    written_episodes = 0
    discarded_unsuccessful = 0
    discarded_invalid = 0
    discarded_unsafe_place_goals = 0
    failure_breakdown: Dict[str, int] = {name: 0 for name in RELEASE_GOAL_STAGE_NAMES}
    failure_breakdown["drop_reattempt"] = 0
    total_env_steps = 0
    t_start = time.time()
    max_rollouts = args.dry_run_rollouts if args.dry_run else None

    try:
        while True:
            n_transitions = _current_counts(root)[0] if root is not None else 0
            if not args.dry_run and n_transitions >= args.target_transitions:
                break
            if max_rollouts is not None and attempted_rollouts >= max_rollouts:
                break
            if args.max_attempted_episodes is not None and attempted_rollouts >= args.max_attempted_episodes:
                break
            if args.max_steps is not None and total_env_steps >= args.max_steps:
                break

            rollout_idx = attempted_rollouts
            start_pose, goal_x, lateral_offset = _sample_end_to_end_start_and_goal_release(
                env,
                rng,
                nominal_start_pose,
                args,
            )
            goals, release_start_goal_idx, place_goal = _build_pick_place_release_goals(
                start_pose,
                goal_x=goal_x,
                lift_height=args.lift_height,
                place_height=args.place_height,
                place_hold_goals=args.place_hold_goals,
                release_hold_goals=args.release_steps,
            )

            if not _place_goal_in_safe_zone(
                env,
                place_goal,
                table_x_half_extent=args.table_x_half_extent,
                table_x_inset_margin=args.place_goal_x_margin,
                table_y_half_extent=args.table_y_half_extent,
                table_y_inset_margin=args.place_goal_y_margin,
            ):
                discarded_invalid += 1
                discarded_unsafe_place_goals += 1
                print(
                    f"[stage5-release-long-table] rejected_unsafe_place_goal "
                    f"rollout={rollout_idx:04d} "
                    f"start_pose={np.array(start_pose).round(4).tolist()} "
                    f"place_goal={np.array(place_goal).round(4).tolist()}",
                    flush=True,
                )
                continue

            print(
                f"\n[stage5-release-long-table] rollout={rollout_idx:04d} "
                f"start_pose={np.array(start_pose).round(4).tolist()} "
                f"goal_x={goal_x:+.3f} lateral_offset={lateral_offset:+.3f}",
                flush=True,
            )

            result = _run_release_rollout(
                env=env,
                args=args,
                device=device,
                policy=policy,
                start_pose=start_pose,
                goals=goals,
                release_start_goal_idx=release_start_goal_idx,
                place_goal=place_goal,
                rollout_idx=rollout_idx,
                save_data=True,
                verbose_steps=args.dry_run or args.log_every_step,
            )
            attempted_rollouts += 1
            total_env_steps += result.steps
            if result.entered_release_phase and result.release_steps_executed < args.release_steps:
                print(
                    f"[stage5-release-long-table] rollout={rollout_idx:04d} truncated_release "
                    f"release_start_step={result.release_start_step} "
                    f"release_steps_executed={result.release_steps_executed}/{args.release_steps} "
                    f"horizon={args.horizon}",
                    flush=True,
                )

            if result.viewer_closed:
                break

            if result.release_success:
                successful_completions += 1
            else:
                discarded_unsuccessful += 1
                if result.failure_stage is not None:
                    failure_breakdown[result.failure_stage] += 1
                if result.reattempted_after_drop:
                    print(
                        f"[stage5-release-long-table] rollout={rollout_idx:04d} drop_reattempt "
                        f"drop_step={result.drop_step} drop_successes_before={result.drop_successes_before} "
                        f"final_successes={result.final_successes}",
                        flush=True,
                    )
            if result.release_stable:
                release_stable_count += 1

            if root is not None and result.success and result.img is not None and result.state is not None and result.action is not None:
                appended = _append_episode(
                    root,
                    result.img,
                    result.state,
                    result.action,
                    object_id=args.object_id,
                    category_id=args.category_id,
                )
                if appended:
                    written_episodes += 1
                else:
                    discarded_invalid += 1

            if args.dry_run and result.img is not None:
                status = "release_success" if result.release_success else "release_failed"
                video_path = args.dry_run_video_dir / f"rollout_{rollout_idx:04d}_{status}.mp4"
                annotated_frames = _annotate_release_rollout_frames(
                    result.img,
                    rollout_idx=rollout_idx,
                    stage_names=result.stage_names or [],
                    goal_dists=result.goal_dists or [],
                    successes_per_step=result.successes_per_step or [],
                    release_phase_per_step=result.release_phase_per_step or [],
                    release_start_step=result.release_start_step,
                )
                _write_rollout_video(video_path, annotated_frames, args.dry_run_video_fps)

            if root is not None:
                if result.noise_action_delta_count > 0:
                    _update_noisy_metric_attrs(
                        root,
                        {
                            "l2_sum": result.noise_action_delta_l2_sum,
                            "l2_sq_sum": result.noise_action_delta_l2_sq_sum,
                            "linf_sum": result.noise_action_delta_linf_sum,
                            "count": result.noise_action_delta_count,
                        },
                    )
                n_transitions, n_episodes = _current_counts(root)
                elapsed = max(time.time() - t_start, 1e-6)
                rate = (n_transitions - n0) / elapsed
                remaining = max(args.target_transitions - n_transitions, 0)
                eta_min = remaining / max(rate, 1e-6) / 60.0
                root.attrs["attempted_rollouts"] = attempted_rollouts
                root.attrs["successful_completions"] = successful_completions
                root.attrs["release_stable_rollouts"] = release_stable_count
                root.attrs["written_episodes"] = written_episodes
                root.attrs["discarded_unsuccessful"] = discarded_unsuccessful
                root.attrs["discarded_invalid"] = discarded_invalid
                root.attrs["discarded_unsafe_place_goals"] = discarded_unsafe_place_goals
                root.attrs["failure_lift"] = failure_breakdown["lift"]
                root.attrs["failure_transport"] = failure_breakdown["transport"]
                root.attrs["failure_place"] = failure_breakdown["place"]
                root.attrs["failure_release"] = failure_breakdown["release"]
                root.attrs["failure_drop_reattempt"] = failure_breakdown["drop_reattempt"]
                root.attrs["xy_range"] = args.xy_range
                root.attrs["horizon"] = args.horizon
                root.attrs["lift_height_m"] = args.lift_height
                root.attrs["lateral_offset_range_m"] = args.lateral_offset_range
                root.attrs["place_height_m"] = args.place_height
                root.attrs["start_z_offset"] = args.start_z_offset
                root.attrs["variant"] = args.variant
                root.attrs["noise_strategy"] = args.noise_strategy
                root.attrs["executed_action_clipped"] = args.variant != "clean"
                root.attrs["release_steps"] = args.release_steps
                root.attrs["release_arm_mode"] = args.release_arm_mode
                root.attrs["release_hand_blend"] = args.release_hand_blend
                root.attrs["place_goal_x_margin_m"] = args.place_goal_x_margin
                root.attrs["place_goal_y_margin_m"] = args.place_goal_y_margin
                root.attrs["collection_task"] = "pick_place_release_experimental"
                print(
                    f"[stage5-release-long-table] transitions={n_transitions}/{args.target_transitions} "
                    f"episodes={n_episodes} attempted={attempted_rollouts} "
                    f"stable={release_stable_count} written={written_episodes} "
                    f"rate={rate:.1f} trans/sec eta={eta_min:.1f}min",
                    flush=True,
                )
    finally:
        _destroy_env(env)


def _collect_anchored_pick_place_release(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    anchored_config = build_anchored_recovery_config(args)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    root = None
    n0 = 0
    e0 = 0
    if not args.dry_run:
        root = _open_or_create_zarr(
            args.output_zarr,
            img_h=DATASET_CAMERA_HEIGHT,
            img_w=DATASET_CAMERA_WIDTH,
            resume=args.resume,
        )
        n0, e0 = _current_counts(root)
        if args.resume:
            validate_anchored_resume_attrs(
                root,
                variant=args.variant,
                config=anchored_config,
            )
        _update_object_registry(root, args)
        _record_geometry_attrs(root, args)
        root.attrs["collection_task"] = "pick_place_release_experimental"
        update_anchored_root_attrs(
            root,
            variant=args.variant,
            config=anchored_config,
            base_rollouts_attempted=0,
            base_rollouts_succeeded=0,
            branches_attempted=0,
            branches_written=0,
            branches_aborted=0,
            saved_transitions=0,
        )

    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )
    bootstrap_goal_x = _default_goal_x(args)
    bootstrap_goals, _, _ = _build_pick_place_release_goals(
        nominal_start_pose,
        goal_x=bootstrap_goal_x,
        lift_height=args.lift_height,
        place_height=args.place_height,
        place_hold_goals=args.place_hold_goals,
        release_hold_goals=args.release_steps,
    )

    print(
        f"[stage5-release-long-table-anchored] output_zarr={args.output_zarr}",
        flush=True,
    )
    print(
        "[stage5-release-long-table-anchored] "
        f"branches_per_rollout={anchored_config.branches_per_rollout} "
        f"perturb_steps={anchored_config.perturb_steps} "
        f"recovery_steps={anchored_config.recovery_steps} "
        f"branch_window=[{anchored_config.branch_min_step}, {anchored_config.branch_max_step}]",
        flush=True,
    )

    env = _make_long_table_env(
        num_envs=1,
        nominal_start_pose=nominal_start_pose,
        goal_poses=bootstrap_goals,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
        seed=args.seed,
        object_name=args.object_name,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    env.set_env_state(checkpoint[0]["env_state"])
    print(
        f"[stage5-release-long-table-anchored] camera pos={_jsonable(env.cfg['env']['datasetCameraPosition'])} "
        f"target={_jsonable(env.cfg['env']['datasetCameraTarget'])}",
        flush=True,
    )
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
        num_envs=1,
    )

    attempted_rollouts = 0
    successful_completions = 0
    release_stable_count = 0
    written_episodes = 0
    discarded_unsuccessful = 0
    discarded_invalid = 0
    discarded_unsafe_place_goals = 0
    failure_breakdown: Dict[str, int] = {name: 0 for name in RELEASE_GOAL_STAGE_NAMES}
    failure_breakdown["drop_reattempt"] = 0
    anchored_base_rollouts_attempted = 0
    anchored_base_rollouts_succeeded = 0
    anchored_branches_attempted = 0
    anchored_branches_written = 0
    anchored_branches_aborted = 0
    anchored_saved_transitions = 0
    total_env_steps = 0
    t_start = time.time()
    max_rollouts = args.dry_run_rollouts if args.dry_run else None

    try:
        while True:
            n_transitions = _current_counts(root)[0] if root is not None else 0
            if not args.dry_run and n_transitions >= args.target_transitions:
                break
            if max_rollouts is not None and attempted_rollouts >= max_rollouts:
                break
            if (
                args.max_attempted_episodes is not None
                and attempted_rollouts >= args.max_attempted_episodes
            ):
                break
            if args.max_steps is not None and total_env_steps >= args.max_steps:
                break

            rollout_idx = attempted_rollouts
            start_pose, goal_x, lateral_offset = _sample_end_to_end_start_and_goal_release(
                env,
                rng,
                nominal_start_pose,
                args,
            )
            goals, release_start_goal_idx, place_goal = _build_pick_place_release_goals(
                start_pose,
                goal_x=goal_x,
                lift_height=args.lift_height,
                place_height=args.place_height,
                place_hold_goals=args.place_hold_goals,
                release_hold_goals=args.release_steps,
            )

            if not _place_goal_in_safe_zone(
                env,
                place_goal,
                table_x_half_extent=args.table_x_half_extent,
                table_x_inset_margin=args.place_goal_x_margin,
                table_y_half_extent=args.table_y_half_extent,
                table_y_inset_margin=args.place_goal_y_margin,
            ):
                discarded_invalid += 1
                discarded_unsafe_place_goals += 1
                print(
                    f"[stage5-release-long-table-anchored] rejected_unsafe_place_goal "
                    f"rollout={rollout_idx:04d} start_pose={np.array(start_pose).round(4).tolist()} "
                    f"place_goal={np.array(place_goal).round(4).tolist()}",
                    flush=True,
                )
                continue

            print(
                f"\n[stage5-release-long-table-anchored] rollout={rollout_idx:04d} "
                f"start_pose={np.array(start_pose).round(4).tolist()} "
                f"goal_x={goal_x:+.3f} lateral_offset={lateral_offset:+.3f}",
                flush=True,
            )

            result, branch_buffers, branch_noise_metrics, branch_stats = (
                _run_anchored_release_rollout(
                    env=env,
                    args=args,
                    device=device,
                    policy=policy,
                    start_pose=start_pose,
                    goals=goals,
                    release_start_goal_idx=release_start_goal_idx,
                    place_goal=place_goal,
                    rollout_idx=rollout_idx,
                    verbose_steps=args.log_every_step,
                    rng=rng,
                    anchored_config=anchored_config,
                )
            )
            attempted_rollouts += 1
            anchored_base_rollouts_attempted += 1
            total_env_steps += result.steps
            anchored_branches_attempted += branch_stats["attempted"]
            anchored_branches_aborted += branch_stats["aborted"]

            if result.entered_release_phase and result.release_steps_executed < args.release_steps:
                print(
                    f"[stage5-release-long-table-anchored] rollout={rollout_idx:04d} truncated_release "
                    f"release_start_step={result.release_start_step} "
                    f"release_steps_executed={result.release_steps_executed}/{args.release_steps} "
                    f"horizon={args.horizon}",
                    flush=True,
                )

            if result.viewer_closed:
                break

            if result.release_success:
                successful_completions += 1
                anchored_base_rollouts_succeeded += 1
            else:
                discarded_unsuccessful += 1
                if result.failure_stage is not None:
                    failure_breakdown[result.failure_stage] += 1
                if result.reattempted_after_drop:
                    print(
                        f"[stage5-release-long-table-anchored] rollout={rollout_idx:04d} drop_reattempt "
                        f"drop_step={result.drop_step} drop_successes_before={result.drop_successes_before} "
                        f"final_successes={result.final_successes}",
                        flush=True,
                    )

            if result.release_stable:
                release_stable_count += 1

            if root is not None and result.success:
                wrote_branch = False
                for branch_buffer in branch_buffers:
                    img_ep, state_ep, action_ep = branch_buffer.as_episode()
                    appended = _append_episode(
                        root,
                        img_ep,
                        state_ep,
                        action_ep,
                        object_id=args.object_id,
                        category_id=args.category_id,
                    )
                    if appended:
                        wrote_branch = True
                        written_episodes += 1
                        anchored_branches_written += 1
                        anchored_saved_transitions += int(img_ep.shape[0])
                    else:
                        discarded_invalid += 1
                if wrote_branch and branch_noise_metrics["count"] > 0:
                    _update_noisy_metric_attrs(root, branch_noise_metrics)

            if root is not None:
                n_transitions, n_episodes = _current_counts(root)
                elapsed = max(time.time() - t_start, 1e-6)
                rate = (n_transitions - n0) / elapsed
                remaining = max(args.target_transitions - n_transitions, 0)
                eta_min = remaining / max(rate, 1e-6) / 60.0
                root.attrs["attempted_rollouts"] = attempted_rollouts
                root.attrs["successful_completions"] = successful_completions
                root.attrs["release_stable_rollouts"] = release_stable_count
                root.attrs["written_episodes"] = written_episodes
                root.attrs["discarded_unsuccessful"] = discarded_unsuccessful
                root.attrs["discarded_invalid"] = discarded_invalid
                root.attrs["discarded_unsafe_place_goals"] = discarded_unsafe_place_goals
                root.attrs["failure_lift"] = failure_breakdown["lift"]
                root.attrs["failure_transport"] = failure_breakdown["transport"]
                root.attrs["failure_place"] = failure_breakdown["place"]
                root.attrs["failure_release"] = failure_breakdown["release"]
                root.attrs["failure_drop_reattempt"] = failure_breakdown["drop_reattempt"]
                root.attrs["xy_range"] = args.xy_range
                root.attrs["horizon"] = args.horizon
                root.attrs["lift_height_m"] = args.lift_height
                root.attrs["lateral_offset_range_m"] = args.lateral_offset_range
                root.attrs["place_height_m"] = args.place_height
                root.attrs["start_z_offset"] = args.start_z_offset
                root.attrs["release_steps"] = args.release_steps
                root.attrs["release_arm_mode"] = args.release_arm_mode
                root.attrs["release_hand_blend"] = args.release_hand_blend
                root.attrs["place_goal_x_margin_m"] = args.place_goal_x_margin
                root.attrs["place_goal_y_margin_m"] = args.place_goal_y_margin
                root.attrs["collection_task"] = "pick_place_release_experimental"
                update_anchored_root_attrs(
                    root,
                    variant=args.variant,
                    config=anchored_config,
                    base_rollouts_attempted=anchored_base_rollouts_attempted,
                    base_rollouts_succeeded=anchored_base_rollouts_succeeded,
                    branches_attempted=anchored_branches_attempted,
                    branches_written=anchored_branches_written,
                    branches_aborted=anchored_branches_aborted,
                    saved_transitions=anchored_saved_transitions,
                )
                print(
                    f"[stage5-release-long-table-anchored] transitions={n_transitions}/{args.target_transitions} "
                    f"episodes={n_episodes} attempted={attempted_rollouts} "
                    f"release_successes={successful_completions} stable={release_stable_count} "
                    f"branches_written={anchored_branches_written} branches_aborted={anchored_branches_aborted} "
                    f"rate={rate:.1f} trans/sec eta={eta_min:.1f}min",
                    flush=True,
                )
    finally:
        _destroy_env(env)


def _normalize_collection_type_args(
    args: argparse.Namespace,
    cli_args: List[str],
) -> argparse.Namespace:
    if "--output-zarr" not in cli_args:
        if args.collection_type == "pick_place_release":
            args.output_zarr = Path("data/stage5_claw_hammer_pick_place_release_long_table.zarr")
        else:
            args.output_zarr = Path("data/stage5_claw_hammer_pick_place_long_table.zarr")

    if "--horizon" not in cli_args:
        if args.collection_type == "pick_place_release":
            args.horizon = DEFAULT_HORIZON_WITH_RELEASE

    if "--dry-run-video-dir" not in cli_args:
        if args.collection_type == "pick_place_release":
            args.dry_run_video_dir = Path("data/stage5_pick_place_release_long_table_dry_run_videos")
        else:
            args.dry_run_video_dir = Path("data/stage5_pick_place_long_table_dry_run_videos")

    return validate_anchored_args(args)


def collect(args: argparse.Namespace) -> None:
    if args.collection_type == "pick_place_release":
        args.release_hand_blend = float(np.clip(args.release_hand_blend, 0.0, 1.0))
        if args.noise_strategy == "anchored_recovery":
            args = _resolve_noise_args(args)
            _collect_anchored_pick_place_release(args)
            return
        _collect_pick_place_release(args)
        return
    if args.noise_strategy == "anchored_recovery":
        args = _resolve_noise_args(args)
        _collect_anchored_pick_place(args)
        return
    if args.variant != "clean":
        args = _resolve_noise_args(args)
    if args.variant == "clean":
        _collect_clean(args)
    elif args.noisy_worker:
        _collect_noisy_worker(args)
    else:
        _collect_noisy_parent(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-type",
        choices=("pick_place", "pick_place_release"),
        default=DEFAULT_COLLECTION_TYPE,
    )
    parser.add_argument("--object-category", type=str, default=DEFAULT_OBJECT_CATEGORY)
    parser.add_argument("--object-name", type=str, default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK_NAME)
    parser.add_argument("--object-id", type=int, default=0)
    parser.add_argument("--category-id", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--target-transitions", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-attempted-episodes", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--xy-range", type=float, default=0.10)
    parser.add_argument("--start-z-offset", type=float, default=DEFAULT_START_Z_OFFSET)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--lift-height", type=float, default=LIFT_HEIGHT_M)
    parser.add_argument("--lateral-offset-range", type=float, default=LATERAL_OFFSET_RANGE_M)
    parser.add_argument("--place-height", type=float, default=PLACE_HEIGHT_M)
    parser.add_argument("--place-hold-goals", type=int, default=PLACE_HOLD_GOALS)
    parser.add_argument("--table-x-half-extent", type=float, default=LONG_TABLE_X_HALF_EXTENT_M)
    parser.add_argument("--table-x-inset-margin", type=float, default=LONG_TABLE_X_INSET_MARGIN_M)
    parser.add_argument("--table-y-half-extent", type=float, default=TABLE_Y_HALF_EXTENT_M)
    parser.add_argument("--table-y-inset-margin", type=float, default=TABLE_Y_INSET_MARGIN_M)
    parser.add_argument("--place-goal-x-margin", type=float, default=PLACE_GOAL_X_MARGIN_M)
    parser.add_argument("--place-goal-y-margin", type=float, default=PLACE_GOAL_Y_MARGIN_M)
    parser.add_argument("--min-effective-transport", type=float, default=MIN_EFFECTIVE_TRANSPORT_M)
    parser.add_argument("--output-zarr", type=Path, default=Path("data/stage5_claw_hammer_pick_place_long_table.zarr"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-rollouts", type=int, default=DRY_RUN_ROLLOUTS)
    parser.add_argument("--dry-run-video-dir", type=Path, default=Path("data/stage5_pick_place_long_table_dry_run_videos"))
    parser.add_argument("--dry-run-video-fps", type=int, default=15)
    parser.add_argument("--log-every-step", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--save-preview-every", type=int, default=20)
    parser.add_argument("--pickup-success-hold-steps", type=int, default=PICKUP_SUCCESS_HOLD_STEPS)
    parser.add_argument(
        "--variant",
        choices=["clean", "noisy_clean", "noisy_noisy"],
        default="clean",
    )
    parser.add_argument(
        "--noise-strategy",
        choices=["continuous_ou", "anchored_recovery"],
        default="continuous_ou",
    )
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--arm-base-noise", type=float, default=None)
    parser.add_argument("--arm-wrist-noise", type=float, default=None)
    parser.add_argument("--thumb-noise", type=float, default=None)
    parser.add_argument("--index-noise", type=float, default=None)
    parser.add_argument("--middle-noise", type=float, default=None)
    parser.add_argument("--ring-noise", type=float, default=None)
    parser.add_argument("--pinky-noise", type=float, default=None)
    parser.add_argument("--ou-theta", type=float, default=0.15)
    parser.add_argument("--ou-mu", type=float, default=0.0)
    parser.add_argument("--ou-dt", type=float, default=1.0)
    parser.add_argument("--anchored-branches-per-rollout", type=int, default=3)
    parser.add_argument("--anchored-branch-min-step", type=int, default=10)
    parser.add_argument("--anchored-branch-max-step", type=str, default="auto")
    parser.add_argument("--anchored-perturb-steps", type=int, default=3)
    parser.add_argument("--anchored-recovery-steps", type=int, default=15)
    parser.add_argument("--release-steps", type=int, default=DEFAULT_RELEASE_STEPS)
    parser.add_argument("--release-arm-mode", choices=["hold", "policy"], default="hold")
    parser.add_argument("--release-hand-blend", type=float, default=1.0)
    parser.add_argument("--release-xy-tolerance", type=float, default=RELEASE_XY_TOLERANCE_M)
    parser.add_argument("--release-z-tolerance", type=float, default=RELEASE_Z_TOLERANCE_M)
    parser.add_argument("--release-speed-tolerance", type=float, default=RELEASE_SPEED_TOLERANCE_MPS)
    parser.add_argument("--noisy-worker", action="store_true")
    parser.add_argument("--worker-batch-idx", type=int, default=None)
    parser.add_argument("--worker-start-x", type=float, default=None)
    parser.add_argument("--worker-start-y", type=float, default=None)
    parser.add_argument("--worker-goal-x", type=float, default=None)
    args = parser.parse_args()
    return _normalize_collection_type_args(args, sys.argv[1:])


if __name__ == "__main__":
    collect(parse_args())
