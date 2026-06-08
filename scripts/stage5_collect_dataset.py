#!/usr/bin/env python3
"""Stage 5: collect DexToolBench pickup rollouts into Diffusion Policy Zarr.

This file remains the pickup-focused collector. `pick_place` and `pick_place_release` are delegated to
`collect_dataset_pick_place_release.py` for the longer-table setup.

Architecture:
- clean:
  - ONE persistent env, created once. No teardown/recreation loop.
  - Per-env rolling buffers. When an env finishes an episode (done == True),
    we gate on pickup success and either append to Zarr or discard.
- noisy_clean / noisy_noisy:
  - Recreate the env per rollout batch with a single randomized start pose.
  - Execute OU-correlated noisy actions.
  - Save either clean expert labels (`noisy_clean`) or noisy executed labels
    (`noisy_noisy`).
- All variants preserve Stage 4 ordering:
  render image -> query policy -> save (img, obs, action) -> step env.

Smoke test:
    python stage5_collect_dataset.py \
      --num-envs 4 \
      --target-transitions 1000 \
      --output-zarr data/stage5_smoke.zarr

Full run:
    python stage5_collect_dataset.py \
      --num-envs 16 \
      --target-transitions 50000 \
      --output-zarr data/stage5_claw_hammer_v1.zarr
"""

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

# Isaac Gym import order matters: import isaacgym before torch.
from isaacgym import gymapi  # noqa: F401
import torch

import zarr
from numcodecs import Blosc

from stage5_anchored_recovery import (
    AnchoredBranchSnapshot,
    build_anchored_recovery_config,
    maybe_fork_rng,
    run_anchored_branch,
    sample_branch_trigger_steps,
    update_anchored_root_attrs,
    validate_anchored_args,
    validate_anchored_resume_attrs,
)
from stage5_noise_config import resolve_noise_config


N_OBS = 140
N_ACT = 29
# Horizon needs headroom over the policy's worst-case rollout length on
# randomized starts. At xy_range=0.10 we observed pickups completing in 124-167
# steps (stage 4 horizon=250 sweep); 150 truncates ~half of them.
DEFAULT_HORIZON = 300
MIN_EPISODE_LEN = 30  # discard implausibly short episodes

DEFAULT_OBJECT_CATEGORY = "hammer"
DEFAULT_OBJECT_NAME = "claw_hammer"
DEFAULT_TASK_NAME = "swing_down"
DEFAULT_COLLECTION_TYPE = "pick_place"
COLLECTION_TYPES = ("pickup", "pick_place", "pick_place_release")

CONFIG_PATH = Path("pretrained_policy/config.yaml")
CHECKPOINT_PATH = Path("pretrained_policy/model.pth")

TABLE_Z = 0.38
TABLE_URDF = "urdf/table_narrow.urdf"
# Spawn directly on the table by default for diffusion-policy data/eval.
# Raising this above zero makes the tool drop onto the table before the episode.
DEFAULT_START_Z_OFFSET = 0.0
LIFT_HEIGHT_M = 0.20
LATERAL_OFFSET_RANGE_M = 0.15
PLACE_HEIGHT_M = 0.02
PLACE_HOLD_GOALS = 10
TABLE_X_HALF_EXTENT_M = 0.475 / 2.0
TABLE_X_INSET_MARGIN_M = 0.06
MIN_EFFECTIVE_TRANSPORT_M = 0.05
DRY_RUN_ROLLOUTS = 5
GOAL_STAGE_NAMES = ("lift", "transport", "place")
PICKUP_ONLY_START_Z_OFFSET = 0.03
PICKUP_ONLY_TABLE_URDF = "urdf/table_narrow_nail.urdf"
DEFAULT_RELEASE_STEPS = 45
DEFAULT_HORIZON_WITH_RELEASE = 325
RELEASE_GOAL_STAGE_NAMES = ("lift", "transport", "place", "release")
RELEASE_XY_TOLERANCE_M = 0.05
RELEASE_Z_TOLERANCE_M = 0.04
RELEASE_SPEED_TOLERANCE_MPS = 0.25
TABLE_Y_HALF_EXTENT_M = 0.4 / 2.0
TABLE_Y_INSET_MARGIN_M = 0.04
PLACE_GOAL_X_MARGIN_M = 0.10
PLACE_GOAL_Y_MARGIN_M = 0.06

PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M = 0.02
PICKUP_SUCCESS_MIN_LIFT_M = 0.12
PICKUP_SUCCESS_HOLD_STEPS = 5

DATASET_CAMERA_POSITION = [0.55, -1.35, 1.10]
DATASET_CAMERA_TARGET = [-0.1, 0.35, 0.60]
DATASET_CAMERA_HORIZONTAL_FOV = 30.0
DATASET_CAMERA_WIDTH = 512
# Height divisible by 16 (and 32) so MP4 encoders don't pad and image encoders
# with strided convs don't hit weird off-by-one shapes.
DATASET_CAMERA_HEIGHT = 384

# Manipulation-specific action groups for the 29-D IIWA + Sharpa action space.
ARM_BASE_IDXS: List[int] = [0, 1, 2]
ARM_WRIST_IDXS: List[int] = [3, 4, 5, 6]
THUMB_IDXS: List[int] = [7, 8, 9, 10, 11]
INDEX_IDXS: List[int] = [12, 13, 14, 15]
MIDDLE_IDXS: List[int] = [16, 17, 18, 19]
RING_IDXS: List[int] = [20, 21, 22, 23]
PINKY_IDXS: List[int] = [24, 25, 26, 27, 28]


def _jsonable(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


@dataclass
class RolloutResult:
    success: bool
    viewer_closed: bool
    steps: int
    max_successes_seen: int
    final_successes: int
    failure_stage: Optional[str]
    img: Optional[np.ndarray]
    state: Optional[np.ndarray]
    action: Optional[np.ndarray]
    stage_names: Optional[List[str]]
    goal_dists: Optional[List[float]]
    successes_per_step: Optional[List[int]]


@dataclass
class ReleaseRolloutResult:
    success: bool
    viewer_closed: bool
    steps: int
    pick_place_success: bool
    release_goal_success: bool
    release_success: bool
    release_stable: bool
    final_object_on_table: bool
    entered_release_phase: bool
    release_start_step: Optional[int]
    release_steps_executed: int
    max_successes_seen: int
    final_successes: int
    failure_stage: Optional[str]
    final_object_pose: List[float]
    final_object_linvel: List[float]
    final_place_pos_error_m: float
    final_place_xy_error_m: float
    final_place_z_error_m: float
    final_object_speed_mps: float
    img: Optional[np.ndarray]
    state: Optional[np.ndarray]
    action: Optional[np.ndarray]
    stage_names: Optional[List[str]]
    goal_dists: Optional[List[float]]
    successes_per_step: Optional[List[int]]
    release_phase_per_step: Optional[List[bool]]
    noise_action_delta_l2_sum: float
    noise_action_delta_l2_sq_sum: float
    noise_action_delta_linf_sum: float
    noise_action_delta_count: int


def _sample_start_pose(
    rng: np.random.Generator,
    nominal_start_pose: List[float],
    xy_range: float,
) -> List[float]:
    start_pose = list(nominal_start_pose)
    dx, dy = rng.uniform(-xy_range, xy_range, size=2)
    start_pose[0] += float(dx)
    start_pose[1] += float(dy)
    return start_pose


def _sample_transport_offset(
    rng: np.random.Generator,
    *,
    start_x: float,
    lateral_offset_range: float,
    table_x_min: float,
    table_x_max: float,
    min_effective_transport: float,
) -> float:
    if lateral_offset_range <= 0.0:
        return 0.0

    feasible_offsets: List[float] = []
    max_left = max(start_x - table_x_min, 0.0)
    max_right = max(table_x_max - start_x, 0.0)

    left_offset = -min(float(lateral_offset_range), max_left)
    right_offset = min(float(lateral_offset_range), max_right)

    if abs(left_offset) >= min_effective_transport:
        feasible_offsets.append(left_offset)
    if abs(right_offset) >= min_effective_transport:
        feasible_offsets.append(right_offset)

    if not feasible_offsets:
        if max_right >= max_left:
            return right_offset
        return left_offset

    return float(feasible_offsets[int(rng.integers(0, len(feasible_offsets)))])


def _build_pick_place_goals(
    start_pose: List[float],
    *,
    lift_height: float,
    lateral_offset: float,
    place_height: float,
    place_hold_goals: int,
) -> List[List[float]]:
    x0, y0, z0, qx, qy, qz, qw = start_pose
    lift_goal = [x0, y0, z0 + lift_height, qx, qy, qz, qw]
    transport_goal = [x0 + lateral_offset, y0, z0 + lift_height, qx, qy, qz, qw]
    place_goal = [x0 + lateral_offset, y0, z0 + place_height, qx, qy, qz, qw]
    return [lift_goal, transport_goal] + [list(place_goal) for _ in range(max(place_hold_goals, 1))]


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


def _annotate_rollout_frames(
    frames: np.ndarray,
    *,
    rollout_idx: int,
    stage_names: List[str],
    goal_dists: List[float],
    successes_per_step: List[int],
) -> np.ndarray:
    from PIL import Image, ImageDraw

    annotated_frames: List[np.ndarray] = []
    n_steps = min(len(frames), len(stage_names), len(goal_dists), len(successes_per_step))
    for step in range(n_steps):
        image = Image.fromarray(frames[step])
        draw = ImageDraw.Draw(image)
        lines = [
            f"rollout {rollout_idx:04d}",
            f"step {step:03d}",
            f"stage {stage_names[step]}",
            f"successes {successes_per_step[step]}/3",
            f"dist {goal_dists[step]:.4f}",
        ]
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


def _write_rollout_video(video_path: Path, frames: np.ndarray, fps: int) -> None:
    import imageio.v2 as imageio

    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(video_path), frames, fps=fps)


def _sample_group_noise(
    clean_action_t: torch.Tensor,
    *,
    arm_base_noise: float,
    arm_wrist_noise: float,
    thumb_noise: float,
    index_noise: float,
    middle_noise: float,
    ring_noise: float,
    pinky_noise: float,
) -> torch.Tensor:
    sigma_noise = torch.zeros_like(clean_action_t)
    group_specs = [
        (ARM_BASE_IDXS, arm_base_noise),
        (ARM_WRIST_IDXS, arm_wrist_noise),
        (THUMB_IDXS, thumb_noise),
        (INDEX_IDXS, index_noise),
        (MIDDLE_IDXS, middle_noise),
        (RING_IDXS, ring_noise),
        (PINKY_IDXS, pinky_noise),
    ]
    for idxs, std in group_specs:
        sigma_noise[:, idxs] = torch.normal(
            mean=0.0,
            std=std,
            size=clean_action_t[:, idxs].shape,
            device=clean_action_t.device,
        )
    return sigma_noise


def _load_nominal_start_pose(
    category: str,
    name: str,
    task: str,
    start_z_offset: float = DEFAULT_START_Z_OFFSET,
) -> List[float]:
    from isaacgymenvs.utils.utils import get_repo_root_dir

    trajectory_path = (
        get_repo_root_dir()
        / "dextoolbench"
        / "trajectories"
        / category
        / name
        / f"{task}.json"
    )
    assert trajectory_path.exists(), f"Missing trajectory: {trajectory_path}"
    with open(trajectory_path, "r") as f:
        traj_data = json.load(f)

    object_start_pose = list(traj_data["start_pose"])
    object_start_pose[2] += float(start_z_offset)
    return object_start_pose


def _make_env(
    *,
    num_envs: int,
    nominal_start_pose: List[float],
    xy_range: float,
    horizon: int,
    headless: bool,
    device: str,
    seed: int,
    object_name: str,
    goal_poses: Optional[List[List[float]]] = None,
    nominal_goal_pose: Optional[List[float]] = None,
):
    """Create an env with a fixed start pose and fixed goal states.

    `goal_poses` is the current interface for multi-stage trajectories.
    `nominal_goal_pose` is kept for older single-goal callers such as eval.
    """
    from deployment.isaac.isaac_env import create_env

    if goal_poses is None:
        if nominal_goal_pose is None:
            raise ValueError("Either goal_poses or nominal_goal_pose must be provided.")
        goal_poses = [nominal_goal_pose]
    elif nominal_goal_pose is not None:
        raise ValueError("Provide only one of goal_poses or nominal_goal_pose.")

    overrides = {
        "seed": seed,
        "task.env.numEnvs": num_envs,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        # Cameras MUST be on; we read RGB from the dataset camera every step.
        "task.env.enableCameraSensors": True,
        "task.env.enableDatasetCameras": True,
        "task.env.objectName": object_name,
        "task.env.useFixedGoalStates": True,
        "task.env.fixedGoalStates": goal_poses,
        "task.env.showGoalObjectVisual": False,
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": nominal_start_pose,
        "task.env.startArmHigher": True,
        "task.env.asset.table": TABLE_URDF,
        "task.env.tableResetZ": TABLE_Z,
        # Keep env resets at the requested start pose instead of reverting to
        # the default "drop from above the table" reset behavior.
        "task.env.tableObjectZOffset": float(nominal_start_pose[2] - TABLE_Z),
        # Env-side XY noise is a no-op when useFixedInitObjectPose=True (the env
        # zeroes the random offsets). Set to 0 to make that explicit; actual
        # randomization is applied by mutating env.object_init_state in collect().
        "task.env.resetPositionNoiseX": 0.0,
        "task.env.resetPositionNoiseY": 0.0,
        "task.env.resetPositionNoiseZ": 0.0,
        "task.env.randomizeObjectRotation": False,
        # Keep DOF/vel resets deterministic so the expert sees the distribution
        # it was trained on.
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


def _open_or_create_zarr(path: Path, img_h: int, img_w: int, resume: bool):
    if path.exists() and not resume:
        raise FileExistsError(
            f"{path} already exists. Use --resume to append or remove it first."
        )

    root = zarr.open(str(path), mode="a")
    data = root.require_group("data")
    meta = root.require_group("meta")

    img_compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    small_compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)

    if "img" not in data:
        data.create_dataset(
            "img",
            shape=(0, img_h, img_w, 3),
            chunks=(16, img_h, img_w, 3),
            dtype="uint8",
            compressor=img_compressor,
        )
        data.create_dataset(
            "state",
            shape=(0, N_OBS),
            chunks=(1024, N_OBS),
            dtype="float32",
            compressor=small_compressor,
        )
        data.create_dataset(
            "action",
            shape=(0, N_ACT),
            chunks=(1024, N_ACT),
            dtype="float32",
            compressor=small_compressor,
        )
        data.create_dataset(
            "object_id",
            shape=(0,),
            chunks=(8192,),
            dtype="uint8",
            compressor=small_compressor,
        )
        data.create_dataset(
            "category_id",
            shape=(0,),
            chunks=(8192,),
            dtype="uint8",
            compressor=small_compressor,
        )
        meta.create_dataset(
            "episode_ends",
            shape=(0,),
            chunks=(1024,),
            dtype="int64",
            compressor=small_compressor,
        )
    else:
        assert data["img"].shape[1:] == (img_h, img_w, 3), data["img"].shape
        assert data["state"].shape[1] == N_OBS, data["state"].shape
        assert data["action"].shape[1] == N_ACT, data["action"].shape
        # Backfill new columns when resuming an older zarr that lacks them.
        if "object_id" not in data:
            data.create_dataset(
                "object_id", shape=(0,), chunks=(8192,),
                dtype="uint8", compressor=small_compressor,
            )
        if "category_id" not in data:
            data.create_dataset(
                "category_id", shape=(0,), chunks=(8192,),
                dtype="uint8", compressor=small_compressor,
            )

    root.attrs["schema"] = "diffusion_policy_image_dataset_v2"
    root.attrs["state_dim"] = N_OBS
    root.attrs["action_dim"] = N_ACT
    root.attrs["img_height"] = img_h
    root.attrs["img_width"] = img_w
    return root


def _append_episode(
    root,
    img: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    object_id: int,
    category_id: int,
):
    assert img.dtype == np.uint8
    assert state.dtype == np.float32
    assert action.dtype == np.float32
    assert img.shape[0] == state.shape[0] == action.shape[0]
    assert state.shape[1] == N_OBS
    assert action.shape[1] == N_ACT

    # Sanity gates before committing to disk.
    if np.any(np.isnan(state)) or np.any(np.isnan(action)):
        return False
    if np.any(np.all(state == 0, axis=1)):
        # All-zero observation row indicates a buffer/indexing bug, not real data.
        return False

    data = root["data"]
    meta = root["meta"]

    n = img.shape[0]
    new_n = int(data["img"].shape[0]) + n
    data["img"].append(img, axis=0)
    data["state"].append(state, axis=0)
    data["action"].append(action, axis=0)
    data["object_id"].append(np.full((n,), object_id, dtype=np.uint8), axis=0)
    data["category_id"].append(np.full((n,), category_id, dtype=np.uint8), axis=0)
    meta["episode_ends"].append(np.asarray([new_n], dtype=np.int64), axis=0)
    return True


def _current_counts(root):
    n_transitions = int(root["data"]["img"].shape[0])
    n_episodes = int(root["meta"]["episode_ends"].shape[0])
    return n_transitions, n_episodes


def _episode_pickup_success(
    object_zs: List[float],
    object_start_z: float,
    goal_z: float,
    pickup_gate_history: Optional[List[bool]] = None,
    hold_steps: int = PICKUP_SUCCESS_HOLD_STEPS,
) -> bool:
    if not object_zs:
        return False
    max_object_z = max(object_zs)
    max_lift_m = max_object_z - object_start_z
    z_gate = bool(
        max_object_z >= goal_z - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
        and max_lift_m >= PICKUP_SUCCESS_MIN_LIFT_M
    )
    if not pickup_gate_history:
        return z_gate
    if hold_steps <= 1:
        return z_gate and any(pickup_gate_history)

    consecutive = 0
    for gate in pickup_gate_history:
        consecutive = consecutive + 1 if gate else 0
        if consecutive >= hold_steps:
            return z_gate
    return False


def _first_true_step(values: List[bool]) -> Optional[int]:
    for step, value in enumerate(values):
        if value:
            return step
    return None


def _first_hold_step(values: List[bool], hold_steps: int) -> Optional[int]:
    if hold_steps <= 1:
        return _first_true_step(values)

    consecutive = 0
    for step, value in enumerate(values):
        consecutive = consecutive + 1 if value else 0
        if consecutive >= hold_steps:
            return step
    return None


def _update_object_registry(root, args: argparse.Namespace) -> None:
    object_registry = dict(root.attrs.get("object_id_to_name", {}))
    category_registry = dict(root.attrs.get("category_id_to_name", {}))
    object_registry[str(args.object_id)] = f"{args.object_category}/{args.object_name}"
    category_registry[str(args.category_id)] = args.object_category
    root.attrs["object_id_to_name"] = object_registry
    root.attrs["category_id_to_name"] = category_registry


def _destroy_env(env) -> None:
    # Best-effort cleanup. Some IsaacGym builds are picky, so keep this guarded.
    try:
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
    except Exception:
        pass
    try:
        env.gym.destroy_sim(env.sim)
    except Exception:
        pass
    del env
    torch.cuda.empty_cache()


def _resolve_noise_args(args: argparse.Namespace) -> argparse.Namespace:
    noise_config = resolve_noise_config(
        noise_scale=args.noise_scale,
        arm_base_noise=args.arm_base_noise,
        arm_wrist_noise=args.arm_wrist_noise,
        thumb_noise=args.thumb_noise,
        index_noise=args.index_noise,
        middle_noise=args.middle_noise,
        ring_noise=args.ring_noise,
        pinky_noise=args.pinky_noise,
    )
    args.arm_base_noise = noise_config["arm_base"]
    args.arm_wrist_noise = noise_config["arm_wrist"]
    args.thumb_noise = noise_config["thumb"]
    args.index_noise = noise_config["index"]
    args.middle_noise = noise_config["middle"]
    args.ring_noise = noise_config["ring"]
    args.pinky_noise = noise_config["pinky"]
    return args


def _update_noisy_metric_attrs(root, batch_metrics: dict) -> None:
    total_l2_sum = float(root.attrs.get("noise_action_delta_l2_sum", 0.0)) + float(
        batch_metrics["l2_sum"]
    )
    total_l2_sq_sum = float(
        root.attrs.get("noise_action_delta_l2_sq_sum", 0.0)
    ) + float(batch_metrics["l2_sq_sum"])
    total_linf_sum = float(root.attrs.get("noise_action_delta_linf_sum", 0.0)) + float(
        batch_metrics["linf_sum"]
    )
    total_count = int(root.attrs.get("noise_action_delta_count", 0)) + int(
        batch_metrics["count"]
    )

    root.attrs["noise_action_delta_l2_sum"] = total_l2_sum
    root.attrs["noise_action_delta_l2_sq_sum"] = total_l2_sq_sum
    root.attrs["noise_action_delta_linf_sum"] = total_linf_sum
    root.attrs["noise_action_delta_count"] = total_count

    if total_count <= 0:
        root.attrs["noise_action_delta_l2_mean"] = 0.0
        root.attrs["noise_action_delta_l2_std"] = 0.0
        root.attrs["noise_action_delta_linf_mean"] = 0.0
        return

    mean_l2 = total_l2_sum / total_count
    mean_l2_sq = total_l2_sq_sum / total_count
    std_l2 = math.sqrt(max(mean_l2_sq - mean_l2 * mean_l2, 0.0))
    mean_linf = total_linf_sum / total_count

    root.attrs["noise_action_delta_l2_mean"] = mean_l2
    root.attrs["noise_action_delta_l2_std"] = std_l2
    root.attrs["noise_action_delta_linf_mean"] = mean_linf


def _update_pickup_timing_attrs(root, batch_metrics: dict) -> None:
    total_attempted = int(root.attrs.get("pickup_timing_attempted_episodes", 0)) + int(
        batch_metrics["attempted_episodes"]
    )
    total_gate_count = int(root.attrs.get("pickup_gate_first_step_count", 0)) + int(
        batch_metrics["gate_count"]
    )
    total_gate_sum = float(root.attrs.get("pickup_gate_first_step_sum", 0.0)) + float(
        batch_metrics["gate_step_sum"]
    )
    total_hold_count = int(root.attrs.get("pickup_hold_first_step_count", 0)) + int(
        batch_metrics["hold_count"]
    )
    total_hold_sum = float(root.attrs.get("pickup_hold_first_step_sum", 0.0)) + float(
        batch_metrics["hold_step_sum"]
    )

    root.attrs["pickup_timing_attempted_episodes"] = total_attempted
    root.attrs["pickup_gate_first_step_count"] = total_gate_count
    root.attrs["pickup_gate_first_step_sum"] = total_gate_sum
    root.attrs["pickup_hold_first_step_count"] = total_hold_count
    root.attrs["pickup_hold_first_step_sum"] = total_hold_sum
    root.attrs["pickup_gate_first_step_mean"] = (
        total_gate_sum / total_gate_count if total_gate_count > 0 else -1.0
    )
    root.attrs["pickup_hold_first_step_mean"] = (
        total_hold_sum / total_hold_count if total_hold_count > 0 else -1.0
    )


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
    object_start_zs = env.object_pose[:, 2].detach().cpu().numpy().astype(np.float32)
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
            print("[stage5] viewer closed; stopping", flush=True)
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
                f"[stage5] rollout={rollout_idx:04d} step={step:03d} "
                f"subgoal_idx={stage_idx} goal_idx={current_goal_idx} stage={stage_name} "
                f"distance={goal_dist:.4f} successes={final_successes}/{len(goals)}",
                flush=True,
            )
        elif stage_idx != prev_stage_idx:
            print(
                f"[stage5] rollout={rollout_idx:04d} advanced_to="
                f"{stage_name} (subgoal_idx={stage_idx}, goal_idx={current_goal_idx}, "
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


def _collect_clean(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_envs != 1:
        print(
            "[stage5] forcing --num-envs=1 for clean pick-and-place collection: "
            "fixedGoalStates are shared per env instance, so each rollout needs "
            "its own env to attach rollout-specific subgoals.",
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

    print(f"[stage5] output_zarr={args.output_zarr}", flush=True)
    print(f"[stage5] variant={args.variant}", flush=True)
    print(f"[stage5] dry_run={args.dry_run}", flush=True)
    if args.dry_run:
        print(f"[stage5] dry_run_video_dir={args.dry_run_video_dir}", flush=True)
    print(f"[stage5] starting counts: transitions={n0}, episodes={e0}", flush=True)
    print(
        f"[stage5] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} object_id={args.object_id} category_id={args.category_id}",
        flush=True,
    )
    print(
        f"[stage5] target_transitions={args.target_transitions} xy_range=+/-{args.xy_range:.3f}m "
        f"lift_height={args.lift_height:.3f}m lateral_offset_range=+/-{args.lateral_offset_range:.3f}m "
        f"place_height={args.place_height:.3f}m place_hold_goals={args.place_hold_goals} "
        f"horizon={args.horizon} "
        f"safe_table_x=[{-args.table_x_half_extent + args.table_x_inset_margin:+.3f}, "
        f"{args.table_x_half_extent - args.table_x_inset_margin:+.3f}]",
        flush=True,
    )

    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )
    bootstrap_goals = _build_pick_place_goals(
        nominal_start_pose,
        lift_height=args.lift_height,
        lateral_offset=args.lateral_offset_range,
        place_height=args.place_height,
        place_hold_goals=args.place_hold_goals,
    )
    print("[stage5] creating env (one-time)...", flush=True)
    env = _make_env(
        num_envs=1,
        nominal_start_pose=nominal_start_pose,
        goal_poses=bootstrap_goals,
        xy_range=args.xy_range,
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
        f"[stage5] camera pos={_jsonable(env.cfg['env']['datasetCameraPosition'])} "
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
            if (
                args.max_attempted_episodes is not None
                and attempted_rollouts >= args.max_attempted_episodes
            ):
                print(
                    f"[stage5] hit --max-attempted-episodes={args.max_attempted_episodes}, stopping",
                    flush=True,
                )
                break
            if args.max_steps is not None and total_env_steps >= args.max_steps:
                print(f"[stage5] hit --max-steps={args.max_steps}, stopping", flush=True)
                break

            rollout_idx = attempted_rollouts
            start_pose = _sample_start_pose(rng, nominal_start_pose, args.xy_range)
            lateral_offset = _sample_transport_offset(
                rng,
                start_x=float(start_pose[0]),
                lateral_offset_range=args.lateral_offset_range,
                table_x_min=-args.table_x_half_extent + args.table_x_inset_margin,
                table_x_max=args.table_x_half_extent - args.table_x_inset_margin,
                min_effective_transport=args.min_effective_transport,
            )
            goals = _build_pick_place_goals(
                start_pose,
                lift_height=args.lift_height,
                lateral_offset=lateral_offset,
                place_height=args.place_height,
                place_hold_goals=args.place_hold_goals,
            )

            print(
                f"\n[stage5] rollout={rollout_idx:04d} "
                f"start_pose={np.array(start_pose).round(4).tolist()} "
                f"lateral_offset={lateral_offset:+.3f}m "
                f"direction={'right' if lateral_offset >= 0.0 else 'left'}",
                flush=True,
            )
            if args.dry_run:
                print(f"[stage5] rollout={rollout_idx:04d} goals={goals}", flush=True)

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
                print(
                    f"[stage5] rollout={rollout_idx:04d} success "
                    f"subgoals_reached={result.max_successes_seen}/{len(goals)}",
                    flush=True,
                )
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
                        if args.save_preview_every and written_episodes % args.save_preview_every == 0:
                            import imageio.v2 as imageio

                            preview_dir = args.output_zarr.parent / f"{args.output_zarr.stem}_previews"
                            preview_dir.mkdir(parents=True, exist_ok=True)
                            imageio.mimsave(
                                preview_dir / f"episode_{written_episodes:05d}.gif",
                                result.img,
                                duration=1000.0 / args.gif_fps,
                            )
                    else:
                        discarded_invalid += 1
            else:
                discarded_unsuccessful += 1
                if result.failure_stage is not None:
                    failure_breakdown[result.failure_stage] += 1
                print(
                    f"[stage5] rollout={rollout_idx:04d} failed "
                    f"reached={result.max_successes_seen}/{len(goals)} "
                    f"failure_stage={result.failure_stage}",
                    flush=True,
                )

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
                print(f"[stage5] rollout={rollout_idx:04d} saved_video={video_path}", flush=True)

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
                    f"[stage5] transitions={n_transitions}/{args.target_transitions} "
                    f"episodes={n_episodes} attempted={attempted_rollouts} "
                    f"completed={successful_completions} written={written_episodes} "
                    f"discarded_unsuccessful={discarded_unsuccessful} invalid={discarded_invalid} "
                    f"rate={rate:.1f} trans/sec eta={eta_min:.1f}min",
                    flush=True,
                )
    finally:
        _destroy_env(env)

    elapsed = time.time() - t_start
    n_transitions, n_episodes = _current_counts(root) if root is not None else (0, 0)
    print("\n[stage5] DONE", flush=True)
    print(
        f"[stage5] total_rollouts={attempted_rollouts} successful_completions={successful_completions} "
        f"success_rate={successful_completions / max(attempted_rollouts, 1):.1%}",
        flush=True,
    )
    print(
        f"[stage5] failure_breakdown: lift={failure_breakdown['lift']} "
        f"transport={failure_breakdown['transport']} place={failure_breakdown['place']}",
        flush=True,
    )
    print(
        f"[stage5] discarded_rollouts={discarded_unsuccessful + discarded_invalid} "
        f"(unsuccessful={discarded_unsuccessful}, invalid={discarded_invalid})",
        flush=True,
    )
    if root is not None:
        print(f"[stage5] transitions={n_transitions}", flush=True)
        print(f"[stage5] episodes={n_episodes}", flush=True)
        print(f"[stage5] zarr={args.output_zarr}", flush=True)
    print(f"[stage5] elapsed={elapsed/60:.1f}min", flush=True)


def _build_noisy_worker_cmd(
    args: argparse.Namespace,
    *,
    batch_idx: int,
    dx: float,
    dy: float,
    lateral_offset: float,
) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--object-category", args.object_category,
        "--object-name", args.object_name,
        "--task-name", args.task_name,
        "--object-id", str(args.object_id),
        "--category-id", str(args.category_id),
        "--num-envs", str(args.num_envs),
        "--target-transitions", str(args.target_transitions),
        "--xy-range", str(args.xy_range),
        "--seed", str(args.seed),
        "--horizon", str(args.horizon),
        "--lift-height", str(args.lift_height),
        "--lateral-offset-range", str(args.lateral_offset_range),
        "--place-height", str(args.place_height),
        "--place-hold-goals", str(args.place_hold_goals),
        "--table-x-half-extent", str(args.table_x_half_extent),
        "--table-x-inset-margin", str(args.table_x_inset_margin),
        "--min-effective-transport", str(args.min_effective_transport),
        "--output-zarr", str(args.output_zarr),
        "--gif-fps", str(args.gif_fps),
        "--save-preview-every", str(args.save_preview_every),
        "--pickup-success-hold-steps", str(args.pickup_success_hold_steps),
        "--variant", args.variant,
        "--noise-scale", str(args.noise_scale),
        "--arm-base-noise", str(args.arm_base_noise),
        "--arm-wrist-noise", str(args.arm_wrist_noise),
        "--thumb-noise", str(args.thumb_noise),
        "--index-noise", str(args.index_noise),
        "--middle-noise", str(args.middle_noise),
        "--ring-noise", str(args.ring_noise),
        "--pinky-noise", str(args.pinky_noise),
        "--ou-theta", str(args.ou_theta),
        "--ou-mu", str(args.ou_mu),
        "--ou-dt", str(args.ou_dt),
        "--resume",
        "--noisy-worker",
        "--worker-batch-idx", str(batch_idx),
        "--worker-dx", str(dx),
        "--worker-dy", str(dy),
        "--worker-lateral-offset", str(lateral_offset),
    ]
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])
    if args.viewer:
        cmd.append("--viewer")
    return cmd


def _build_noisy_pickup_worker_cmd(
    args: argparse.Namespace,
    *,
    batch_idx: int,
    dx: float,
    dy: float,
) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--collection-type", "pickup",
        "--object-category", args.object_category,
        "--object-name", args.object_name,
        "--task-name", args.task_name,
        "--object-id", str(args.object_id),
        "--category-id", str(args.category_id),
        "--num-envs", str(args.num_envs),
        "--target-transitions", str(args.target_transitions),
        "--xy-range", str(args.xy_range),
        "--start-z-offset", str(args.start_z_offset),
        "--seed", str(args.seed),
        "--horizon", str(args.horizon),
        "--output-zarr", str(args.output_zarr),
        "--gif-fps", str(args.gif_fps),
        "--save-preview-every", str(args.save_preview_every),
        "--pickup-success-hold-steps", str(args.pickup_success_hold_steps),
        "--variant", args.variant,
        "--noise-scale", str(args.noise_scale),
        "--arm-base-noise", str(args.arm_base_noise),
        "--arm-wrist-noise", str(args.arm_wrist_noise),
        "--thumb-noise", str(args.thumb_noise),
        "--index-noise", str(args.index_noise),
        "--middle-noise", str(args.middle_noise),
        "--ring-noise", str(args.ring_noise),
        "--pinky-noise", str(args.pinky_noise),
        "--ou-theta", str(args.ou_theta),
        "--ou-mu", str(args.ou_mu),
        "--ou-dt", str(args.ou_dt),
        "--resume",
        "--noisy-worker",
        "--worker-batch-idx", str(batch_idx),
        "--worker-dx", str(dx),
        "--worker-dy", str(dy),
    ]
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])
    if args.viewer:
        cmd.append("--viewer")
    return cmd


def _collect_noisy_worker(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert args.worker_batch_idx is not None
    assert args.worker_dx is not None
    assert args.worker_dy is not None
    assert args.worker_lateral_offset is not None
    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = _open_or_create_zarr(
        args.output_zarr,
        img_h=DATASET_CAMERA_HEIGHT,
        img_w=DATASET_CAMERA_WIDTH,
        resume=True,
    )
    n0, e0 = _current_counts(root)
    print(f"[stage5-noisy] output_zarr={args.output_zarr}", flush=True)
    print(
        f"[stage5-noisy] variant={args.variant} noise_scale={args.noise_scale} "
        f"arm_base={args.arm_base_noise} "
        f"arm_wrist={args.arm_wrist_noise} thumb={args.thumb_noise} "
        f"index={args.index_noise} middle={args.middle_noise} "
        f"ring={args.ring_noise} pinky={args.pinky_noise} "
        f"theta={args.ou_theta} dt={args.ou_dt}",
        flush=True,
    )
    print(f"[stage5-noisy] starting counts: transitions={n0}, episodes={e0}", flush=True)
    print(
        f"[stage5-noisy] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} object_id={args.object_id} category_id={args.category_id}",
        flush=True,
    )
    print(
        f"[stage5-noisy] num_envs={args.num_envs}, target_transitions={args.target_transitions}",
        flush=True,
    )

    _update_object_registry(root, args)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )

    dx = float(args.worker_dx)
    dy = float(args.worker_dy)
    lateral_offset = float(args.worker_lateral_offset)
    batch_idx = int(args.worker_batch_idx)
    object_start_pose = list(nominal_start_pose)
    object_start_pose[0] += dx
    object_start_pose[1] += dy

    goals = _build_pick_place_goals(
        object_start_pose,
        lift_height=args.lift_height,
        lateral_offset=lateral_offset,
        place_height=args.place_height,
        place_hold_goals=args.place_hold_goals,
    )

    print(
        f"\n[stage5-noisy] batch={batch_idx:04d} "
        f"dx={dx:+.3f}, dy={dy:+.3f}, lateral_offset={lateral_offset:+.3f}, "
        f"current={n0}/{args.target_transitions}",
        flush=True,
    )
    env = _make_env(
        num_envs=args.num_envs,
        nominal_start_pose=object_start_pose,
        goal_poses=goals,
        xy_range=args.xy_range,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
        seed=args.seed + batch_idx,
        object_name=args.object_name,
    )

    env.gym.refresh_actor_root_state_tensor(env.sim)
    print(
        f"[stage5-noisy] camera pos={_jsonable(env.cfg['env']['datasetCameraPosition'])} "
        f"target={_jsonable(env.cfg['env']['datasetCameraTarget'])}",
        flush=True,
    )

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
    object_start_zs = env.object_pose[:, 2].detach().cpu().numpy().astype(np.float32)

    noise_state = torch.zeros((env.num_envs, N_ACT), device=device)
    sqrt_dt = math.sqrt(args.ou_dt)

    per_env_imgs = [[] for _ in range(env.num_envs)]
    per_env_states = [[] for _ in range(env.num_envs)]
    per_env_actions = [[] for _ in range(env.num_envs)]
    per_env_max_successes = [0 for _ in range(env.num_envs)]
    batch_noise_metrics = {
        "l2_sum": 0.0,
        "l2_sq_sum": 0.0,
        "linf_sum": 0.0,
        "count": 0,
    }

    active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)

    for step in range(args.horizon):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            print("[stage5-noisy] viewer closed; ending collection loop", flush=True)
            break

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

        if step % 30 == 0:
            successes_now = int((env.successes >= env.max_consecutive_successes).sum().item())
            mean_delta_l2 = batch_noise_metrics["l2_sum"] / max(
                batch_noise_metrics["count"], 1
            )
            print(
                f"[stage5-noisy] batch={batch_idx:04d} step={step:03d} "
                f"strict_env_pose_successes_now={successes_now}/{env.num_envs} "
                f"mean_action_delta_l2={mean_delta_l2:.4f}",
                flush=True,
            )

    batch_successes = 0
    for env_i in range(env.num_envs):
        success = per_env_max_successes[env_i] >= len(goals)
        if not success:
            continue

        img_ep = np.stack(per_env_imgs[env_i], axis=0).astype(np.uint8)
        state_ep = np.stack(per_env_states[env_i], axis=0).astype(np.float32)
        action_ep = np.stack(per_env_actions[env_i], axis=0).astype(np.float32)

        appended = _append_episode(
            root,
            img_ep,
            state_ep,
            action_ep,
            object_id=args.object_id,
            category_id=args.category_id,
        )
        if not appended:
            continue

        batch_successes += 1

        if args.save_preview_every and (e0 + batch_successes) % args.save_preview_every == 0:
            import imageio.v2 as imageio

            preview_dir = args.output_zarr.parent / f"{args.output_zarr.stem}_previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                preview_dir / f"episode_{e0 + batch_successes:05d}.gif",
                img_ep,
                duration=1000.0 / args.gif_fps,
            )

    _update_noisy_metric_attrs(root, batch_noise_metrics)
    n_transitions, n_episodes = _current_counts(root)
    batch_mean_delta_l2 = batch_noise_metrics["l2_sum"] / max(batch_noise_metrics["count"], 1)
    batch_mean_delta_linf = batch_noise_metrics["linf_sum"] / max(
        batch_noise_metrics["count"], 1
    )
    print(
        f"[stage5-noisy] batch={batch_idx:04d} successes_written={batch_successes}/{env.num_envs} "
        f"total_transitions={n_transitions}/{args.target_transitions} "
        f"total_episodes={n_episodes} "
        f"mean_action_delta_l2={batch_mean_delta_l2:.4f} "
        f"mean_action_delta_linf={batch_mean_delta_linf:.4f}",
        flush=True,
    )

    _destroy_env(env)


def _collect_noisy_parent(args: argparse.Namespace) -> None:
    root = _open_or_create_zarr(
        args.output_zarr,
        img_h=DATASET_CAMERA_HEIGHT,
        img_w=DATASET_CAMERA_WIDTH,
        resume=args.resume,
    )
    n0, e0 = _current_counts(root)
    print(f"[stage5-noisy] output_zarr={args.output_zarr}", flush=True)
    print(
        f"[stage5-noisy] variant={args.variant} noise_scale={args.noise_scale} "
        f"arm_base={args.arm_base_noise} "
        f"arm_wrist={args.arm_wrist_noise} thumb={args.thumb_noise} "
        f"index={args.index_noise} middle={args.middle_noise} "
        f"ring={args.ring_noise} pinky={args.pinky_noise} "
        f"theta={args.ou_theta} dt={args.ou_dt}",
        flush=True,
    )
    print(f"[stage5-noisy] starting counts: transitions={n0}, episodes={e0}", flush=True)
    print(
        f"[stage5-noisy] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} object_id={args.object_id} category_id={args.category_id}",
        flush=True,
    )
    print(
        f"[stage5-noisy] num_envs={args.num_envs}, target_transitions={args.target_transitions}",
        flush=True,
    )
    _update_object_registry(root, args)

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
        rng.uniform(-args.xy_range, args.xy_range, size=2)
        _sample_transport_offset(
            rng,
            start_x=float(nominal_start_pose[0]),
            lateral_offset_range=args.lateral_offset_range,
            table_x_min=-args.table_x_half_extent + args.table_x_inset_margin,
            table_x_max=args.table_x_half_extent - args.table_x_inset_margin,
            min_effective_transport=args.min_effective_transport,
        )
    t_start = time.time()

    while True:
        n_before, e_before = _current_counts(root)
        if n_before >= args.target_transitions:
            break
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        if args.max_attempted_episodes is not None and attempted_episodes >= args.max_attempted_episodes:
            break

        dx, dy = rng.uniform(-args.xy_range, args.xy_range, size=2)
        start_x = float(nominal_start_pose[0] + dx)
        lateral_offset = _sample_transport_offset(
            rng,
            start_x=start_x,
            lateral_offset_range=args.lateral_offset_range,
            table_x_min=-args.table_x_half_extent + args.table_x_inset_margin,
            table_x_max=args.table_x_half_extent - args.table_x_inset_margin,
            min_effective_transport=args.min_effective_transport,
        )
        cmd = _build_noisy_worker_cmd(
            args,
            batch_idx=batch_idx,
            dx=float(dx),
            dy=float(dy),
            lateral_offset=float(lateral_offset),
        )
        subprocess.run(cmd, check=True)

        root = zarr.open(str(args.output_zarr), mode="a")
        n_after, e_after = _current_counts(root)
        attempted_episodes += args.num_envs
        written_episodes += max(e_after - e_before, 0)

        elapsed = max(time.time() - t_start, 1e-6)
        rate = (n_after - n0) / elapsed
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
        root.attrs["pickup_success_hold_steps"] = args.pickup_success_hold_steps
        root.attrs["variant"] = args.variant
        root.attrs["noise_strategy"] = args.noise_strategy
        root.attrs["executed_action_clipped"] = True
        root.attrs["collection_task"] = "pick_place"
        root.attrs["noise_scale"] = args.noise_scale
        root.attrs["arm_base_noise"] = args.arm_base_noise
        root.attrs["arm_wrist_noise"] = args.arm_wrist_noise
        root.attrs["thumb_noise"] = args.thumb_noise
        root.attrs["index_noise"] = args.index_noise
        root.attrs["middle_noise"] = args.middle_noise
        root.attrs["ring_noise"] = args.ring_noise
        root.attrs["pinky_noise"] = args.pinky_noise
        root.attrs["ou_theta"] = args.ou_theta
        root.attrs["ou_mu"] = args.ou_mu
        root.attrs["ou_dt"] = args.ou_dt

        print(
            f"[stage5-noisy] parent batch={batch_idx:04d} "
            f"total_transitions={n_after}/{args.target_transitions} "
            f"total_episodes={e_after} attempted={attempted_episodes} "
            f"write_success_rate={written_episodes / max(attempted_episodes, 1):.1%} "
            f"mean_action_delta_l2={float(root.attrs.get('noise_action_delta_l2_mean', 0.0)):.4f} "
            f"rate={rate:.1f} transitions/sec eta={eta_min:.1f} min",
            flush=True,
        )
        batch_idx += 1

    n_transitions, n_episodes = _current_counts(root)
    print("\n[stage5-noisy] DONE", flush=True)
    print(f"[stage5-noisy] transitions={n_transitions}", flush=True)
    print(f"[stage5-noisy] episodes={n_episodes}", flush=True)
    print(f"[stage5-noisy] zarr={args.output_zarr}", flush=True)


def _make_pickup_env(
    *,
    num_envs: int,
    nominal_start_pose: List[float],
    nominal_goal_pose: List[float],
    xy_range: float,
    horizon: int,
    headless: bool,
    device: str,
    seed: int,
    object_name: str,
):
    """Create the preserved pickup-only env configuration."""
    from deployment.isaac.isaac_env import create_env

    overrides = {
        "seed": seed,
        "task.env.numEnvs": num_envs,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        "task.env.enableCameraSensors": True,
        "task.env.enableDatasetCameras": True,
        "task.env.objectName": object_name,
        "task.env.useFixedGoalStates": True,
        "task.env.fixedGoalStates": [nominal_goal_pose],
        "task.env.showGoalObjectVisual": False,
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": nominal_start_pose,
        "task.env.startArmHigher": True,
        "task.env.asset.table": PICKUP_ONLY_TABLE_URDF,
        "task.env.tableResetZ": TABLE_Z,
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


def _collect_pickup_noisy_worker(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert args.worker_batch_idx is not None
    assert args.worker_dx is not None
    assert args.worker_dy is not None
    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = _open_or_create_zarr(
        args.output_zarr,
        img_h=DATASET_CAMERA_HEIGHT,
        img_w=DATASET_CAMERA_WIDTH,
        resume=True,
    )
    n0, e0 = _current_counts(root)
    print(f"[stage5-pickup-noisy] output_zarr={args.output_zarr}", flush=True)
    print(
        f"[stage5-pickup-noisy] variant={args.variant} noise_scale={args.noise_scale} "
        f"arm_base={args.arm_base_noise} "
        f"arm_wrist={args.arm_wrist_noise} thumb={args.thumb_noise} "
        f"index={args.index_noise} middle={args.middle_noise} "
        f"ring={args.ring_noise} pinky={args.pinky_noise} "
        f"theta={args.ou_theta} dt={args.ou_dt}",
        flush=True,
    )
    print(
        f"[stage5-pickup-noisy] starting counts: transitions={n0}, episodes={e0}",
        flush=True,
    )
    print(
        f"[stage5-pickup-noisy] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} object_id={args.object_id} category_id={args.category_id}",
        flush=True,
    )

    _update_object_registry(root, args)

    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )

    dx = float(args.worker_dx)
    dy = float(args.worker_dy)
    batch_idx = int(args.worker_batch_idx)
    object_start_pose = list(nominal_start_pose)
    object_start_pose[0] += dx
    object_start_pose[1] += dy
    nominal_goal_pose = list(object_start_pose)
    nominal_goal_pose[2] += LIFT_HEIGHT_M
    goal_z = float(nominal_goal_pose[2])

    print(
        f"\n[stage5-pickup-noisy] batch={batch_idx:04d} "
        f"dx={dx:+.3f}, dy={dy:+.3f}, current={n0}/{args.target_transitions}",
        flush=True,
    )
    env = _make_pickup_env(
        num_envs=args.num_envs,
        nominal_start_pose=object_start_pose,
        nominal_goal_pose=nominal_goal_pose,
        xy_range=args.xy_range,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
        seed=args.seed + batch_idx,
        object_name=args.object_name,
    )

    env.gym.refresh_actor_root_state_tensor(env.sim)
    print(
        f"[stage5-pickup-noisy] camera pos={_jsonable(env.cfg['env']['datasetCameraPosition'])} "
        f"target={_jsonable(env.cfg['env']['datasetCameraTarget'])}",
        flush=True,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
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
    object_start_zs = env.object_pose[:, 2].detach().cpu().numpy().astype(np.float32)

    noise_state = torch.zeros((env.num_envs, N_ACT), device=device)
    sqrt_dt = math.sqrt(args.ou_dt)

    per_env_imgs = [[] for _ in range(env.num_envs)]
    per_env_states = [[] for _ in range(env.num_envs)]
    per_env_actions = [[] for _ in range(env.num_envs)]
    per_env_object_zs = [[] for _ in range(env.num_envs)]
    per_env_pickup_gate_history = [[] for _ in range(env.num_envs)]
    batch_noise_metrics = {
        "l2_sum": 0.0,
        "l2_sq_sum": 0.0,
        "linf_sum": 0.0,
        "count": 0,
    }
    batch_pickup_metrics = {
        "attempted_episodes": env.num_envs,
        "gate_count": 0,
        "gate_step_sum": 0.0,
        "hold_count": 0,
        "hold_step_sum": 0.0,
    }

    active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)
    for step in range(args.horizon):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            print("[stage5-pickup-noisy] viewer closed; ending collection loop", flush=True)
            break

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

        object_pose_np = env.object_pose[:, 0:7].detach().cpu().numpy()
        for env_i in range(env.num_envs):
            object_z = float(object_pose_np[env_i, 2])
            per_env_object_zs[env_i].append(object_z)
            object_start_z = float(object_start_zs[env_i])
            lifted_enough = object_z - object_start_z >= PICKUP_SUCCESS_MIN_LIFT_M
            near_goal = object_z >= goal_z - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
            per_env_pickup_gate_history[env_i].append(bool(lifted_enough and near_goal))

        if step % 30 == 0:
            gates_now = sum(
                1 for history in per_env_pickup_gate_history if history and history[-1]
            )
            mean_delta_l2 = batch_noise_metrics["l2_sum"] / max(
                batch_noise_metrics["count"], 1
            )
            print(
                f"[stage5-pickup-noisy] batch={batch_idx:04d} step={step:03d} "
                f"pickup_gates_now={gates_now}/{env.num_envs} "
                f"mean_action_delta_l2={mean_delta_l2:.4f}",
                flush=True,
            )

    batch_successes = 0
    for env_i in range(env.num_envs):
        gate_step = _first_true_step(per_env_pickup_gate_history[env_i])
        if gate_step is not None:
            batch_pickup_metrics["gate_count"] += 1
            batch_pickup_metrics["gate_step_sum"] += float(gate_step)

        hold_step = _first_hold_step(
            per_env_pickup_gate_history[env_i], args.pickup_success_hold_steps
        )
        if hold_step is not None:
            batch_pickup_metrics["hold_count"] += 1
            batch_pickup_metrics["hold_step_sum"] += float(hold_step)

        success = _episode_pickup_success(
            object_zs=per_env_object_zs[env_i],
            object_start_z=float(object_start_zs[env_i]),
            goal_z=goal_z,
            # Noisy pickup accepts the single-step pickup gate without the
            # consecutive-hold requirement used for timing diagnostics.
            # pickup_gate_history=per_env_pickup_gate_history[env_i],
            # hold_steps=args.pickup_success_hold_steps,
        )
        if not success:
            continue

        img_ep = np.stack(per_env_imgs[env_i], axis=0).astype(np.uint8)
        state_ep = np.stack(per_env_states[env_i], axis=0).astype(np.float32)
        action_ep = np.stack(per_env_actions[env_i], axis=0).astype(np.float32)

        appended = _append_episode(
            root,
            img_ep,
            state_ep,
            action_ep,
            object_id=args.object_id,
            category_id=args.category_id,
        )
        if not appended:
            continue

        batch_successes += 1

        if args.save_preview_every and (e0 + batch_successes) % args.save_preview_every == 0:
            import imageio.v2 as imageio

            preview_dir = args.output_zarr.parent / f"{args.output_zarr.stem}_previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                preview_dir / f"episode_{e0 + batch_successes:05d}.gif",
                img_ep,
                duration=1000.0 / args.gif_fps,
            )

    _update_noisy_metric_attrs(root, batch_noise_metrics)
    _update_pickup_timing_attrs(root, batch_pickup_metrics)
    n_transitions, n_episodes = _current_counts(root)
    batch_mean_delta_l2 = batch_noise_metrics["l2_sum"] / max(batch_noise_metrics["count"], 1)
    batch_mean_delta_linf = batch_noise_metrics["linf_sum"] / max(
        batch_noise_metrics["count"], 1
    )
    print(
        f"[stage5-pickup-noisy] batch={batch_idx:04d} successes_written={batch_successes}/{env.num_envs} "
        f"total_transitions={n_transitions}/{args.target_transitions} "
        f"total_episodes={n_episodes} "
        f"mean_action_delta_l2={batch_mean_delta_l2:.4f} "
        f"mean_action_delta_linf={batch_mean_delta_linf:.4f}",
        flush=True,
    )

    _destroy_env(env)


def _collect_pickup_noisy_parent(args: argparse.Namespace) -> None:
    root = _open_or_create_zarr(
        args.output_zarr,
        img_h=DATASET_CAMERA_HEIGHT,
        img_w=DATASET_CAMERA_WIDTH,
        resume=args.resume,
    )
    n0, e0 = _current_counts(root)
    print(f"[stage5-pickup-noisy] output_zarr={args.output_zarr}", flush=True)
    print(
        f"[stage5-pickup-noisy] variant={args.variant} noise_scale={args.noise_scale} "
        f"arm_base={args.arm_base_noise} "
        f"arm_wrist={args.arm_wrist_noise} thumb={args.thumb_noise} "
        f"index={args.index_noise} middle={args.middle_noise} "
        f"ring={args.ring_noise} pinky={args.pinky_noise} "
        f"theta={args.ou_theta} dt={args.ou_dt}",
        flush=True,
    )
    print(
        f"[stage5-pickup-noisy] starting counts: transitions={n0}, episodes={e0}",
        flush=True,
    )
    print(
        f"[stage5-pickup-noisy] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} object_id={args.object_id} category_id={args.category_id}",
        flush=True,
    )
    print(
        f"[stage5-pickup-noisy] num_envs={args.num_envs}, target_transitions={args.target_transitions}",
        flush=True,
    )
    _update_object_registry(root, args)

    rollouts_per_init = int(getattr(args, "rollouts_per_init", 1))
    attempted_episodes = int(root.attrs.get("attempted_episodes", 0))
    written_episodes = int(root.attrs.get("written_episodes", 0))
    batch_idx = int(root.attrs.get("last_batch_idx", -1)) + 1

    # Advance RNG to the start of the current init group.
    # Each init group covers rollouts_per_init consecutive batches and consumes
    # exactly one (dx, dy) sample; with rollouts_per_init=1 (default) this is
    # identical to the old per-batch sampling behavior.
    init_idx = batch_idx // rollouts_per_init
    rng = np.random.default_rng(args.seed)
    for _ in range(init_idx):
        rng.uniform(-args.xy_range, args.xy_range, size=2)

    # If resuming mid-init, consume that init's (dx, dy) sample now.
    rollout_in_init = batch_idx % rollouts_per_init
    current_dx: float
    current_dy: float
    if rollout_in_init > 0:
        _d = rng.uniform(-args.xy_range, args.xy_range, size=2)
        current_dx, current_dy = float(_d[0]), float(_d[1])
    else:
        current_dx, current_dy = 0.0, 0.0  # will be replaced at loop start

    t_start = time.time()

    while True:
        n_before, e_before = _current_counts(root)
        if n_before >= args.target_transitions:
            break
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        if args.max_attempted_episodes is not None and attempted_episodes >= args.max_attempted_episodes:
            break

        # Sample a new (dx, dy) only at the start of each init group.
        rollout_in_init = batch_idx % rollouts_per_init
        if rollout_in_init == 0:
            _d = rng.uniform(-args.xy_range, args.xy_range, size=2)
            current_dx, current_dy = float(_d[0]), float(_d[1])

        cmd = _build_noisy_pickup_worker_cmd(
            args,
            batch_idx=batch_idx,
            dx=current_dx,
            dy=current_dy,
        )
        subprocess.run(cmd, check=True)

        root = zarr.open(str(args.output_zarr), mode="a")
        n_after, e_after = _current_counts(root)
        attempted_episodes += args.num_envs
        written_episodes += max(e_after - e_before, 0)

        elapsed = max(time.time() - t_start, 1e-6)
        rate = (n_after - n0) / elapsed
        remaining = max(args.target_transitions - n_after, 0)
        eta_min = remaining / max(rate, 1e-6) / 60.0

        root.attrs["last_batch_idx"] = batch_idx
        root.attrs["attempted_episodes"] = attempted_episodes
        root.attrs["written_episodes"] = written_episodes
        root.attrs["xy_range"] = args.xy_range
        root.attrs["horizon"] = args.horizon
        root.attrs["lift_height_m"] = LIFT_HEIGHT_M
        root.attrs["start_z_offset"] = args.start_z_offset
        root.attrs["pickup_success_hold_steps"] = args.pickup_success_hold_steps
        root.attrs["variant"] = args.variant
        root.attrs["noise_strategy"] = args.noise_strategy
        root.attrs["executed_action_clipped"] = True
        root.attrs["collection_task"] = "pickup_only"
        root.attrs["noise_scale"] = args.noise_scale
        root.attrs["arm_base_noise"] = args.arm_base_noise
        root.attrs["arm_wrist_noise"] = args.arm_wrist_noise
        root.attrs["thumb_noise"] = args.thumb_noise
        root.attrs["index_noise"] = args.index_noise
        root.attrs["middle_noise"] = args.middle_noise
        root.attrs["ring_noise"] = args.ring_noise
        root.attrs["pinky_noise"] = args.pinky_noise
        root.attrs["ou_theta"] = args.ou_theta
        root.attrs["ou_mu"] = args.ou_mu
        root.attrs["ou_dt"] = args.ou_dt
        root.attrs["rollouts_per_init"] = rollouts_per_init

        print(
            f"[stage5-pickup-noisy] parent batch={batch_idx:04d} "
            f"init_group={batch_idx // rollouts_per_init} rollout_in_init={rollout_in_init} "
            f"total_transitions={n_after}/{args.target_transitions} "
            f"total_episodes={e_after} attempted={attempted_episodes} "
            f"write_success_rate={written_episodes / max(attempted_episodes, 1):.1%} "
            f"mean_action_delta_l2={float(root.attrs.get('noise_action_delta_l2_mean', 0.0)):.4f} "
            f"rate={rate:.1f} transitions/sec eta={eta_min:.1f} min",
            flush=True,
        )
        batch_idx += 1

    n_transitions, n_episodes = _current_counts(root)
    print("\n[stage5-pickup-noisy] DONE", flush=True)
    print(f"[stage5-pickup-noisy] transitions={n_transitions}", flush=True)
    print(f"[stage5-pickup-noisy] episodes={n_episodes}", flush=True)
    print(f"[stage5-pickup-noisy] zarr={args.output_zarr}", flush=True)


def _collect_pickup_clean(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = _open_or_create_zarr(
        args.output_zarr,
        img_h=DATASET_CAMERA_HEIGHT,
        img_w=DATASET_CAMERA_WIDTH,
        resume=args.resume,
    )
    n0, e0 = _current_counts(root)
    print(f"[stage5-pickup] output_zarr={args.output_zarr}", flush=True)
    print(
        f"[stage5-pickup] starting counts: transitions={n0}, episodes={e0}",
        flush=True,
    )
    print(
        f"[stage5-pickup] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} object_id={args.object_id} category_id={args.category_id}",
        flush=True,
    )
    print(
        f"[stage5-pickup] num_envs={args.num_envs}, "
        f"target_transitions={args.target_transitions}, "
        f"xy_range=+/-{args.xy_range:.3f}m",
        flush=True,
    )

    _update_object_registry(root, args)

    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )
    nominal_goal_pose = list(nominal_start_pose)
    nominal_goal_pose[2] += LIFT_HEIGHT_M
    nominal_start_z = float(nominal_start_pose[2])
    goal_z = float(nominal_goal_pose[2])

    env = _make_pickup_env(
        num_envs=args.num_envs,
        nominal_start_pose=nominal_start_pose,
        nominal_goal_pose=nominal_goal_pose,
        xy_range=args.xy_range,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
        seed=args.seed,
        object_name=args.object_name,
    )

    nominal_init_xy = env.object_init_state[:, 0:2].detach().clone()

    def randomize_object_init_xy(env_id_list: List[int]) -> None:
        if not env_id_list:
            return
        env_ids = torch.tensor(
            env_id_list,
            device=env.object_init_state.device,
            dtype=torch.long,
        )
        deltas = (
            torch.rand(
                (len(env_id_list), 2),
                device=env.object_init_state.device,
                dtype=env.object_init_state.dtype,
            )
            * 2.0
            - 1.0
        ) * float(args.xy_range)
        env.object_init_state[env_ids, 0:2] = nominal_init_xy[env_ids] + deltas

    randomize_object_init_xy(list(range(env.num_envs)))

    env.gym.refresh_actor_root_state_tensor(env.sim)
    print(
        f"[stage5-pickup] camera pos={_jsonable(env.cfg['env']['datasetCameraPosition'])} "
        f"target={_jsonable(env.cfg['env']['datasetCameraTarget'])}",
        flush=True,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
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

    active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)

    per_env_imgs: List[List[np.ndarray]] = [[] for _ in range(env.num_envs)]
    per_env_states: List[List[np.ndarray]] = [[] for _ in range(env.num_envs)]
    per_env_actions: List[List[np.ndarray]] = [[] for _ in range(env.num_envs)]
    per_env_object_zs: List[List[float]] = [[] for _ in range(env.num_envs)]
    per_env_start_z: List[float] = [nominal_start_z] * env.num_envs
    just_reset: List[bool] = [False] * env.num_envs

    attempted_episodes = 0
    written_episodes = 0
    discarded_short = 0
    discarded_unsuccessful = 0
    discarded_invalid = 0
    step_idx = 0
    t_start = time.time()

    while True:
        n_transitions, _ = _current_counts(root)
        if n_transitions >= args.target_transitions:
            break
        if args.max_steps is not None and step_idx >= args.max_steps:
            print(
                f"[stage5-pickup] hit --max-steps={args.max_steps}, stopping",
                flush=True,
            )
            break
        if (
            args.max_attempted_episodes is not None
            and attempted_episodes >= args.max_attempted_episodes
        ):
            print(
                "[stage5-pickup] hit "
                f"--max-attempted-episodes={args.max_attempted_episodes}, stopping",
                flush=True,
            )
            break
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            print("[stage5-pickup] viewer closed; stopping", flush=True)
            break

        image_t = env.render_dataset_camera_rgb(active_envs)
        action_t = policy.get_normalized_action(obs, deterministic_actions=True)

        image_np = image_t.detach().cpu().numpy().astype(np.uint8)
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        action_np = action_t.detach().cpu().numpy().astype(np.float32)

        for env_i in range(env.num_envs):
            if just_reset[env_i]:
                continue
            per_env_imgs[env_i].append(image_np[env_i])
            per_env_states[env_i].append(obs_np[env_i])
            per_env_actions[env_i].append(action_np[env_i])

        obs_dict, _, done, _ = env.step(action_t)
        obs = obs_dict["obs"]

        object_pose_np = env.object_pose[:, 0:7].detach().cpu().numpy()
        for env_i in range(env.num_envs):
            if just_reset[env_i]:
                per_env_start_z[env_i] = float(object_pose_np[env_i, 2])
                just_reset[env_i] = False
                continue
            per_env_object_zs[env_i].append(float(object_pose_np[env_i, 2]))

        done_np = done.detach().cpu().numpy().astype(bool)
        done_indices = np.flatnonzero(done_np)

        for env_i in done_indices.tolist():
            attempted_episodes += 1
            ep_len = len(per_env_imgs[env_i])

            if ep_len < MIN_EPISODE_LEN:
                discarded_short += 1
            else:
                success = _episode_pickup_success(
                    object_zs=per_env_object_zs[env_i],
                    object_start_z=per_env_start_z[env_i],
                    goal_z=goal_z,
                )
                if not success:
                    discarded_unsuccessful += 1
                else:
                    img_ep = np.stack(per_env_imgs[env_i], axis=0).astype(np.uint8)
                    state_ep = np.stack(per_env_states[env_i], axis=0).astype(np.float32)
                    action_ep = np.stack(per_env_actions[env_i], axis=0).astype(np.float32)

                    appended = _append_episode(
                        root,
                        img_ep,
                        state_ep,
                        action_ep,
                        object_id=args.object_id,
                        category_id=args.category_id,
                    )
                    if appended:
                        written_episodes += 1
                        if (
                            args.save_preview_every
                            and written_episodes % args.save_preview_every == 0
                        ):
                            import imageio.v2 as imageio

                            preview_dir = (
                                args.output_zarr.parent
                                / f"{args.output_zarr.stem}_previews"
                            )
                            preview_dir.mkdir(parents=True, exist_ok=True)
                            imageio.mimsave(
                                preview_dir / f"episode_{written_episodes:05d}.gif",
                                img_ep,
                                duration=1000.0 / args.gif_fps,
                            )
                    else:
                        discarded_invalid += 1

            per_env_imgs[env_i].clear()
            per_env_states[env_i].clear()
            per_env_actions[env_i].clear()
            per_env_object_zs[env_i].clear()
            just_reset[env_i] = True

        randomize_object_init_xy(done_indices.tolist())

        step_idx += 1

        if step_idx % 30 == 0:
            n_transitions, n_episodes = _current_counts(root)
            elapsed = max(time.time() - t_start, 1e-6)
            rate = (n_transitions - n0) / elapsed
            remaining = max(args.target_transitions - n_transitions, 0)
            eta_min = remaining / max(rate, 1e-6) / 60.0
            success_rate = written_episodes / max(attempted_episodes, 1)
            print(
                f"[stage5-pickup] step={step_idx} "
                f"transitions={n_transitions}/{args.target_transitions} "
                f"episodes={n_episodes} attempted={attempted_episodes} "
                f"written={written_episodes} short={discarded_short} "
                f"unsuccessful={discarded_unsuccessful} invalid={discarded_invalid} "
                f"success_rate={success_rate:.1%} "
                f"rate={rate:.1f} trans/sec eta={eta_min:.1f}min",
                flush=True,
            )

            root.attrs["last_step_idx"] = step_idx
            root.attrs["attempted_episodes"] = attempted_episodes
            root.attrs["written_episodes"] = written_episodes
            root.attrs["discarded_short"] = discarded_short
            root.attrs["discarded_unsuccessful"] = discarded_unsuccessful
            root.attrs["discarded_invalid"] = discarded_invalid
            root.attrs["xy_range"] = args.xy_range
            root.attrs["horizon"] = args.horizon
            root.attrs["lift_height_m"] = LIFT_HEIGHT_M
            root.attrs["start_z_offset"] = args.start_z_offset
            root.attrs["collection_task"] = "pickup_only"

    n_transitions, n_episodes = _current_counts(root)
    elapsed = time.time() - t_start
    print("\n[stage5-pickup] DONE", flush=True)
    print(f"[stage5-pickup] transitions={n_transitions}", flush=True)
    print(f"[stage5-pickup] episodes={n_episodes}", flush=True)
    print(
        f"[stage5-pickup] attempted={attempted_episodes} "
        f"written={written_episodes}",
        flush=True,
    )
    print(
        f"[stage5-pickup] discarded short={discarded_short} "
        f"unsuccessful={discarded_unsuccessful} invalid={discarded_invalid}",
        flush=True,
    )
    print(f"[stage5-pickup] elapsed={elapsed/60:.1f}min", flush=True)
    print(f"[stage5-pickup] zarr={args.output_zarr}", flush=True)


def _collect_pickup_anchored(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    anchored_config = build_anchored_recovery_config(args)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

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
    root.attrs["collection_task"] = "pickup_only"
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
    nominal_goal_pose = list(nominal_start_pose)
    nominal_goal_pose[2] += LIFT_HEIGHT_M

    print(f"[stage5-pickup-anchored] output_zarr={args.output_zarr}", flush=True)
    print(
        f"[stage5-pickup-anchored] starting counts: transitions={n0}, episodes={e0}",
        flush=True,
    )
    print(
        f"[stage5-pickup-anchored] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} object_id={args.object_id} category_id={args.category_id}",
        flush=True,
    )
    print(
        f"[stage5-pickup-anchored] target_transitions={args.target_transitions} "
        f"branches_per_rollout={anchored_config.branches_per_rollout} "
        f"perturb_steps={anchored_config.perturb_steps} "
        f"recovery_steps={anchored_config.recovery_steps} "
        f"branch_window=[{anchored_config.branch_min_step}, {anchored_config.branch_max_step}]",
        flush=True,
    )

    env = _make_pickup_env(
        num_envs=1,
        nominal_start_pose=nominal_start_pose,
        nominal_goal_pose=nominal_goal_pose,
        xy_range=args.xy_range,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
        seed=args.seed,
        object_name=args.object_name,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    print(
        f"[stage5-pickup-anchored] camera pos={_jsonable(env.cfg['env']['datasetCameraPosition'])} "
        f"target={_jsonable(env.cfg['env']['datasetCameraTarget'])}",
        flush=True,
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    env.set_env_state(checkpoint[0]["env_state"])
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
    anchored_base_rollouts_attempted = 0
    anchored_base_rollouts_succeeded = 0
    anchored_branches_attempted = 0
    anchored_branches_written = 0
    anchored_branches_aborted = 0
    anchored_saved_transitions = 0
    total_env_steps = 0
    t_start = time.time()
    noise_sampler = _build_group_noise_sampler(args)

    try:
        while True:
            n_transitions, _ = _current_counts(root)
            if n_transitions >= args.target_transitions:
                break
            if (
                args.max_attempted_episodes is not None
                and attempted_rollouts >= args.max_attempted_episodes
            ):
                break
            if args.max_steps is not None and total_env_steps >= args.max_steps:
                break
            if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
                break

            rollout_idx = attempted_rollouts
            start_pose = _sample_start_pose(rng, nominal_start_pose, args.xy_range)
            goal_pose = list(start_pose)
            goal_pose[2] += LIFT_HEIGHT_M

            env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
            env.goal_states[env_ids, 0:7] = torch.tensor(
                [goal_pose],
                device=env.device,
                dtype=env.goal_states.dtype,
            )
            env.trajectory_states = torch.tensor(
                [goal_pose],
                device=env.device,
                dtype=env.trajectory_states.dtype,
            )
            env.object_init_state[env_ids, 0:7] = torch.tensor(
                [start_pose],
                device=env.device,
                dtype=env.object_init_state.dtype,
            )
            env.cfg["env"]["tableObjectZOffset"] = float(start_pose[2] - TABLE_Z)
            env.max_consecutive_successes = 1
            env.reset_idx(env_ids, tensor_reset=True)
            policy.reset()

            zero_action = torch.zeros((env.num_envs, N_ACT), device=device)
            obs_dict, _, _, _ = env.step(zero_action)
            obs = obs_dict["obs"]
            rollout_env_steps = 1
            active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)
            object_start_z = float(env.object_pose[0, 2].item())
            goal_z = float(goal_pose[2])
            object_zs: List[float] = []
            branch_buffers: List = []
            branch_noise_metrics = _zero_noise_metrics()
            branch_stats = {"attempted": 0, "aborted": 0}
            trigger_steps = set(
                sample_branch_trigger_steps(
                    rng=rng,
                    branch_min_step=anchored_config.branch_min_step,
                    branch_max_step=anchored_config.branch_max_step,
                    branches_per_rollout=anchored_config.branches_per_rollout,
                    min_gap=anchored_config.branch_gap,
                )
            )
            viewer_closed = False

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
                                current_obs,
                                deterministic_actions=True,
                            ),
                            sample_group_noise=noise_sampler,
                        )
                    obs = _restore_branch_snapshot(env, policy, snapshot)
                    rollout_env_steps += branch_result.steps_executed
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
                rollout_env_steps += 1
                object_zs.append(float(env.object_pose[0, 2].item()))
                if bool(done[0].item()):
                    break

            attempted_rollouts += 1
            anchored_base_rollouts_attempted += 1
            total_env_steps += rollout_env_steps
            anchored_branches_attempted += branch_stats["attempted"]
            anchored_branches_aborted += branch_stats["aborted"]

            if viewer_closed:
                break

            success = _episode_pickup_success(
                object_zs=object_zs,
                object_start_z=object_start_z,
                goal_z=goal_z,
            )
            if success:
                successful_completions += 1
                anchored_base_rollouts_succeeded += 1
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
            root.attrs["xy_range"] = args.xy_range
            root.attrs["horizon"] = args.horizon
            root.attrs["lift_height_m"] = LIFT_HEIGHT_M
            root.attrs["start_z_offset"] = args.start_z_offset
            root.attrs["pickup_success_hold_steps"] = args.pickup_success_hold_steps
            root.attrs["collection_task"] = "pickup_only"
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
                f"[stage5-pickup-anchored] transitions={n_transitions}/{args.target_transitions} "
                f"episodes={n_episodes} rollouts={attempted_rollouts} "
                f"base_successes={successful_completions} branches_written={anchored_branches_written} "
                f"branches_aborted={anchored_branches_aborted} rate={rate:.1f} trans/sec "
                f"eta={eta_min:.1f}min",
                flush=True,
            )
    finally:
        _destroy_env(env)


def _collect_pickup(args: argparse.Namespace) -> None:
    if args.noise_strategy == "anchored_recovery":
        args = _resolve_noise_args(args)
        _collect_pickup_anchored(args)
        return
    if args.variant != "clean":
        args = _resolve_noise_args(args)
    if args.variant == "clean":
        _collect_pickup_clean(args)
    elif args.noisy_worker:
        _collect_pickup_noisy_worker(args)
    else:
        _collect_pickup_noisy_parent(args)


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


def _build_pick_place_release_goals(
    start_pose: List[float],
    *,
    lift_height: float,
    lateral_offset: float,
    place_height: float,
    place_hold_goals: int,
    release_hold_goals: int,
) -> Tuple[List[List[float]], int, List[float]]:
    base_goals = _build_pick_place_goals(
        start_pose,
        lift_height=lift_height,
        lateral_offset=lateral_offset,
        place_height=place_height,
        place_hold_goals=place_hold_goals,
    )
    place_goal = list(base_goals[-1])
    release_start_goal_idx = len(base_goals)
    if release_hold_goals <= 0:
        return base_goals, release_start_goal_idx, place_goal
    release_goals = [list(place_goal) for _ in range(release_hold_goals)]
    return base_goals + release_goals, release_start_goal_idx, place_goal


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
    return (
        abs(float(object_pose_np[0]) - float(table_center_np[0])) <= usable_x_half_extent
        and abs(float(object_pose_np[1]) - float(table_center_np[1])) <= usable_y_half_extent
    )


def _place_goal_in_safe_zone(
    env,
    place_goal: List[float],
    *,
    place_goal_x_margin: float,
    place_goal_y_margin: float,
) -> bool:
    place_goal_np = np.asarray(place_goal, dtype=np.float32)
    return _final_object_on_table(
        env,
        place_goal_np,
        table_x_half_extent=TABLE_X_HALF_EXTENT_M,
        table_x_inset_margin=place_goal_x_margin,
        table_y_half_extent=TABLE_Y_HALF_EXTENT_M,
        table_y_inset_margin=place_goal_y_margin,
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
    action_t[:, 7:] = (
        (1.0 - hand_blend) * action_t[:, 7:] + hand_blend * open_hand_action
    )
    return torch.clamp(action_t, -1.0, 1.0)


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
    noise_metrics = {
        "l2_sum": 0.0,
        "l2_sq_sum": 0.0,
        "linf_sum": 0.0,
        "count": 0,
    }

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

    for step in range(args.horizon):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            print("[stage5-release] viewer closed; stopping", flush=True)
            viewer_closed = True
            break

        current_successes = int(env.successes[0].item())
        in_release_phase = (
            args.release_steps > 0 and current_successes >= release_start_goal_idx
        )
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
            action_to_save_t = (
                executed_action_t if args.variant == "noisy_noisy" else clean_action_t
            )

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
        stage_name = (
            "release"
            if in_release_phase
            else _release_goal_stage_name(current_successes, release_start_goal_idx)
        )
        stage_names.append(stage_name)
        goal_dists.append(goal_dist)
        successes_per_step.append(final_successes)
        release_phase_per_step.append(bool(in_release_phase))

        object_pose_np = env.object_pose[0, 0:7].detach().cpu().numpy()
        place_goal_np = np.asarray(place_goal, dtype=np.float32)
        final_place_xy_error_m = float(
            np.linalg.norm(object_pose_np[0:2] - place_goal_np[0:2])
        )
        final_place_z_error_m = float(abs(object_pose_np[2] - place_goal_np[2]))

        if verbose_steps:
            print(
                f"[stage5-release] rollout={rollout_idx:04d} step={step:03d} "
                f"stage={stage_name} release_phase={int(in_release_phase)} "
                f"successes={final_successes}/{len(goals)} "
                f"goal_dist={goal_dist:.4f} "
                f"place_xy_err={final_place_xy_error_m:.4f} "
                f"place_z_err={final_place_z_error_m:.4f}",
                flush=True,
            )
        elif stage_name != prev_stage_name:
            print(
                f"[stage5-release] rollout={rollout_idx:04d} advanced_to={stage_name} "
                f"(goal_dist={goal_dist:.4f}, successes={final_successes}/{len(goals)})",
                flush=True,
            )
        prev_stage_name = stage_name

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

    pick_place_success = max_successes_seen >= release_start_goal_idx
    release_goal_success = max_successes_seen >= len(goals)
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
    release_stable = (
        entered_release_phase
        and final_object_on_table
        and final_place_xy_error_m <= args.release_xy_tolerance
        and final_place_z_error_m <= args.release_z_tolerance
        and final_object_speed_mps <= args.release_speed_tolerance
    )
    release_success = pick_place_success and release_stable
    success = release_success
    failure_stage = _release_failure_stage_from_successes(
        max_successes_seen=max_successes_seen,
        release_start_goal_idx=release_start_goal_idx,
        total_goals=len(goals),
    )

    img = np.stack(images, axis=0).astype(np.uint8) if save_data and images else None
    state = np.stack(states, axis=0).astype(np.float32) if save_data and states else None
    action = np.stack(actions, axis=0).astype(np.float32) if save_data and actions else None
    return ReleaseRolloutResult(
        success=success,
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
        print(
            "[stage5-release] forcing --num-envs=1: release testing is kept "
            "single-env on purpose so dry-run debugging stays interpretable.",
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

    print(f"[stage5-release] output_zarr={args.output_zarr}", flush=True)
    print(f"[stage5-release] dry_run={args.dry_run}", flush=True)
    if args.dry_run:
        print(
            f"[stage5-release] dry_run_video_dir={args.dry_run_video_dir}",
            flush=True,
        )
    print(
        f"[stage5-release] starting counts: transitions={n0}, episodes={e0}",
        flush=True,
    )
    print(
        f"[stage5-release] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} object_id={args.object_id} category_id={args.category_id}",
        flush=True,
    )
    print(
        f"[stage5-release] target_transitions={args.target_transitions} "
        f"variant={args.variant} "
        f"xy_range=+/-{args.xy_range:.3f}m "
        f"lift_height={args.lift_height:.3f}m "
        f"lateral_offset_range=+/-{args.lateral_offset_range:.3f}m "
        f"place_height={args.place_height:.3f}m "
        f"place_hold_goals={args.place_hold_goals} "
        f"release_steps={args.release_steps} "
        f"release_arm_mode={args.release_arm_mode} "
        f"release_hand_blend={args.release_hand_blend:.2f} "
        f"horizon={args.horizon}",
        flush=True,
    )
    if args.variant != "clean":
        print(
            f"[stage5-release] noise_scale={args.noise_scale} "
            f"arm_base={args.arm_base_noise} arm_wrist={args.arm_wrist_noise} "
            f"thumb={args.thumb_noise} index={args.index_noise} "
            f"middle={args.middle_noise} ring={args.ring_noise} "
            f"pinky={args.pinky_noise} theta={args.ou_theta} dt={args.ou_dt}",
            flush=True,
        )

    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )
    bootstrap_goals, _, _ = _build_pick_place_release_goals(
        nominal_start_pose,
        lift_height=args.lift_height,
        lateral_offset=args.lateral_offset_range,
        place_height=args.place_height,
        place_hold_goals=args.place_hold_goals,
        release_hold_goals=args.release_steps,
    )

    print("[stage5-release] creating env (one-time)...", flush=True)
    env = _make_env(
        num_envs=1,
        nominal_start_pose=nominal_start_pose,
        goal_poses=bootstrap_goals,
        xy_range=args.xy_range,
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
        f"[stage5-release] camera pos={_jsonable(env.cfg['env']['datasetCameraPosition'])} "
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
    failure_breakdown: Dict[str, int] = {
        name: 0 for name in RELEASE_GOAL_STAGE_NAMES
    }
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
                print(
                    f"[stage5-release] hit --max-attempted-episodes="
                    f"{args.max_attempted_episodes}, stopping",
                    flush=True,
                )
                break
            if args.max_steps is not None and total_env_steps >= args.max_steps:
                print(
                    f"[stage5-release] hit --max-steps={args.max_steps}, stopping",
                    flush=True,
                )
                break

            rollout_idx = attempted_rollouts
            start_pose = _sample_start_pose(rng, nominal_start_pose, args.xy_range)
            lateral_offset = _sample_transport_offset(
                rng,
                start_x=float(start_pose[0]),
                lateral_offset_range=args.lateral_offset_range,
                table_x_min=-args.table_x_half_extent + args.table_x_inset_margin,
                table_x_max=args.table_x_half_extent - args.table_x_inset_margin,
                min_effective_transport=args.min_effective_transport,
            )
            goals, release_start_goal_idx, place_goal = _build_pick_place_release_goals(
                start_pose,
                lift_height=args.lift_height,
                lateral_offset=lateral_offset,
                place_height=args.place_height,
                place_hold_goals=args.place_hold_goals,
                release_hold_goals=args.release_steps,
            )

            if not _place_goal_in_safe_zone(
                env,
                place_goal,
                place_goal_x_margin=args.place_goal_x_margin,
                place_goal_y_margin=args.place_goal_y_margin,
            ):
                discarded_invalid += 1
                discarded_unsafe_place_goals += 1
                print(
                    f"[stage5-release] skipped unsafe place goal "
                    f"start_pose={np.array(start_pose).round(4).tolist()} "
                    f"place_goal={np.array(place_goal).round(4).tolist()} "
                    f"x_margin={args.place_goal_x_margin:.3f} "
                    f"y_margin={args.place_goal_y_margin:.3f}",
                    flush=True,
                )
                continue

            print(
                f"\n[stage5-release] rollout={rollout_idx:04d} "
                f"start_pose={np.array(start_pose).round(4).tolist()} "
                f"lateral_offset={lateral_offset:+.3f}m "
                f"release_start_goal_idx={release_start_goal_idx} "
                f"total_goals={len(goals)}",
                flush=True,
            )
            if args.dry_run:
                print(f"[stage5-release] rollout={rollout_idx:04d} goals={goals}", flush=True)

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

            if result.viewer_closed:
                break

            if result.release_success:
                successful_completions += 1
            else:
                discarded_unsuccessful += 1
                if result.failure_stage is not None:
                    failure_breakdown[result.failure_stage] += 1

            if result.release_stable:
                release_stable_count += 1

            print(
                f"[stage5-release] rollout={rollout_idx:04d} "
                f"pick_place_success={result.pick_place_success} "
                f"release_goal_success={result.release_goal_success} "
                f"entered_release={result.entered_release_phase} "
                f"release_success={result.release_success} "
                f"release_stable={result.release_stable} "
                f"on_table={result.final_object_on_table} "
                f"release_steps_executed={result.release_steps_executed} "
                f"final_place_xy_err={result.final_place_xy_error_m:.4f}m "
                f"final_place_z_err={result.final_place_z_error_m:.4f}m "
                f"final_speed={result.final_object_speed_mps:.4f}mps",
                flush=True,
            )

            if (
                root is not None
                and result.success
                and result.steps >= MIN_EPISODE_LEN
                and result.img is not None
                and result.state is not None
                and result.action is not None
            ):
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
                if result.release_success:
                    status = "release_success"
                elif result.pick_place_success:
                    status = "release_failed"
                else:
                    status = "pick_place_failed"
                video_path = (
                    args.dry_run_video_dir / f"rollout_{rollout_idx:04d}_{status}.mp4"
                )
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
                print(
                    f"[stage5-release] rollout={rollout_idx:04d} saved_video={video_path}",
                    flush=True,
                )

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
                if args.variant != "clean":
                    root.attrs["noise_scale"] = args.noise_scale
                    root.attrs["arm_base_noise"] = args.arm_base_noise
                    root.attrs["arm_wrist_noise"] = args.arm_wrist_noise
                    root.attrs["thumb_noise"] = args.thumb_noise
                    root.attrs["index_noise"] = args.index_noise
                    root.attrs["middle_noise"] = args.middle_noise
                    root.attrs["ring_noise"] = args.ring_noise
                    root.attrs["pinky_noise"] = args.pinky_noise
                    root.attrs["ou_theta"] = args.ou_theta
                    root.attrs["ou_mu"] = args.ou_mu
                    root.attrs["ou_dt"] = args.ou_dt
                print(
                    f"[stage5-release] transitions={n_transitions}/{args.target_transitions} "
                    f"episodes={n_episodes} attempted={attempted_rollouts} "
                    f"release_successes={successful_completions} "
                    f"release_stable={release_stable_count} "
                    f"written={written_episodes} invalid_goals={discarded_unsafe_place_goals} "
                    f"mean_action_delta_l2={float(root.attrs.get('noise_action_delta_l2_mean', 0.0)):.4f} "
                    f"rate={rate:.1f} trans/sec "
                    f"eta={eta_min:.1f}min",
                    flush=True,
                )
    finally:
        _destroy_env(env)

    elapsed = time.time() - t_start
    n_transitions, n_episodes = _current_counts(root) if root is not None else (0, 0)
    print("\n[stage5-release] DONE", flush=True)
    print(
        f"[stage5-release] total_rollouts={attempted_rollouts} "
        f"release_successes={successful_completions} "
        f"release_success_rate={successful_completions / max(attempted_rollouts, 1):.1%}",
        flush=True,
    )
    print(
        f"[stage5-release] release_stable_rollouts={release_stable_count} "
        f"stable_rate={release_stable_count / max(attempted_rollouts, 1):.1%}",
        flush=True,
    )
    print(
        f"[stage5-release] failure_breakdown: "
        f"lift={failure_breakdown['lift']} "
        f"transport={failure_breakdown['transport']} "
        f"place={failure_breakdown['place']} "
        f"release={failure_breakdown['release']}",
        flush=True,
    )
    print(
        f"[stage5-release] discarded_unsafe_place_goals={discarded_unsafe_place_goals}",
        flush=True,
    )
    if root is not None:
        print(f"[stage5-release] transitions={n_transitions}", flush=True)
        print(f"[stage5-release] episodes={n_episodes}", flush=True)
        print(f"[stage5-release] zarr={args.output_zarr}", flush=True)
    print(f"[stage5-release] elapsed={elapsed/60:.1f}min", flush=True)


def _collect_pick_place(args: argparse.Namespace) -> None:
    if args.dry_run and args.variant != "clean":
        raise ValueError("--dry-run only supports --variant clean")
    if args.variant != "clean":
        args = _resolve_noise_args(args)
    if args.variant == "clean":
        _collect_clean(args)
    elif args.noisy_worker:
        _collect_noisy_worker(args)
    else:
        _collect_noisy_parent(args)


def _normalize_collection_type_args(
    args: argparse.Namespace,
    cli_args: List[str],
) -> argparse.Namespace:
    if "--output-zarr" not in cli_args:
        if args.collection_type == "pickup":
            args.output_zarr = Path("data/stage5_claw_hammer_pickup_v1.zarr")
        elif args.collection_type == "pick_place_release":
            args.output_zarr = Path(
                "data/stage5_claw_hammer_pick_place_release_experimental.zarr"
            )

    if "--horizon" not in cli_args:
        if args.collection_type == "pickup":
            args.horizon = 250
        elif args.collection_type == "pick_place_release":
            args.horizon = DEFAULT_HORIZON_WITH_RELEASE

    if "--start-z-offset" not in cli_args and args.collection_type == "pickup":
        args.start_z_offset = PICKUP_ONLY_START_Z_OFFSET

    if "--dry-run-video-dir" not in cli_args:
        if args.collection_type == "pick_place_release":
            args.dry_run_video_dir = Path("data/stage5_pick_place_release_dry_run_videos")
        else:
            args.dry_run_video_dir = Path("data/stage5_pick_place_dry_run_videos")

    return validate_anchored_args(args)


def _validate_collection_type_args(args: argparse.Namespace) -> None:
    if args.collection_type == "pickup" and args.dry_run:
        raise ValueError("--dry-run is not supported for --collection-type pickup.")


def collect(args: argparse.Namespace) -> None:
    args = validate_anchored_args(args)
    _validate_collection_type_args(args)
    if args.collection_type == "pickup":
        _collect_pickup(args)
        return
    if args.collection_type == "pick_place_release":
        args.release_hand_blend = float(np.clip(args.release_hand_blend, 0.0, 1.0))
        _collect_pick_place_release(args)
        return
    _collect_pick_place(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-type",
        choices=COLLECTION_TYPES,
        default=DEFAULT_COLLECTION_TYPE,
        help="Which Stage 5 task to collect: pickup, pick_place, or pick_place_release.",
    )
    parser.add_argument("--object-category", type=str, default=DEFAULT_OBJECT_CATEGORY)
    parser.add_argument("--object-name", type=str, default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK_NAME,
                        help="Trajectory file name; only its start_pose is read.")
    parser.add_argument("--object-id", type=int, default=0,
                        help="Stable id written into data/object_id.")
    parser.add_argument("--category-id", type=int, default=0,
                        help="Stable id written into data/category_id.")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--target-transitions", type=int, default=50000)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Hard cap on total env.step() iterations (safety bound).",
    )
    parser.add_argument("--max-attempted-episodes", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--xy-range", type=float, default=0.10)
    parser.add_argument(
        "--start-z-offset",
        type=float,
        default=DEFAULT_START_Z_OFFSET,
        help=(
            "Added to DexToolBench start_pose.z before resets. "
            "Use 0.0 to spawn on the table; positive values drop above it."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--lift-height", type=float, default=LIFT_HEIGHT_M)
    parser.add_argument("--lateral-offset-range", type=float, default=LATERAL_OFFSET_RANGE_M)
    parser.add_argument("--place-height", type=float, default=PLACE_HEIGHT_M)
    parser.add_argument("--place-hold-goals", type=int, default=PLACE_HOLD_GOALS)
    parser.add_argument("--table-x-half-extent", type=float, default=TABLE_X_HALF_EXTENT_M)
    parser.add_argument("--table-x-inset-margin", type=float, default=TABLE_X_INSET_MARGIN_M)
    parser.add_argument("--min-effective-transport", type=float, default=MIN_EFFECTIVE_TRANSPORT_M)
    parser.add_argument(
        "--output-zarr",
        type=Path,
        default=Path("data/stage5_claw_hammer_pick_place_v1.zarr"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-rollouts", type=int, default=DRY_RUN_ROLLOUTS)
    parser.add_argument(
        "--dry-run-video-dir",
        type=Path,
        default=Path("data/stage5_pick_place_dry_run_videos"),
    )
    parser.add_argument("--dry-run-video-fps", type=int, default=15)
    parser.add_argument("--log-every-step", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument(
        "--save-preview-every",
        type=int,
        default=20,
        help="Save a GIF every N written episodes. Use 0 to disable.",
    )
    parser.add_argument(
        "--pickup-success-hold-steps",
        type=int,
        default=PICKUP_SUCCESS_HOLD_STEPS,
    )
    parser.add_argument(
        "--variant",
        choices=["clean", "noisy_clean", "noisy_noisy"],
        default="clean",
        help="clean: execute/save clean actions; noisy_clean: execute noisy but save clean labels; noisy_noisy: execute/save noisy labels.",
    )
    parser.add_argument(
        "--noise-strategy",
        choices=["continuous_ou", "anchored_recovery"],
        default="continuous_ou",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the Stage 5 groupwise base noise defaults.",
    )
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
    parser.add_argument(
        "--release-arm-mode",
        choices=["hold", "policy"],
        default="hold",
        help="During release: hold keeps arm still, policy keeps policy arm actions.",
    )
    parser.add_argument(
        "--release-hand-blend",
        type=float,
        default=1.0,
        help="During release: 1.0 fully overrides hand with open command.",
    )
    parser.add_argument("--release-xy-tolerance", type=float, default=RELEASE_XY_TOLERANCE_M)
    parser.add_argument("--release-z-tolerance", type=float, default=RELEASE_Z_TOLERANCE_M)
    parser.add_argument("--release-speed-tolerance", type=float, default=RELEASE_SPEED_TOLERANCE_MPS)
    parser.add_argument("--table-y-half-extent", type=float, default=TABLE_Y_HALF_EXTENT_M)
    parser.add_argument("--table-y-inset-margin", type=float, default=TABLE_Y_INSET_MARGIN_M)
    parser.add_argument("--place-goal-x-margin", type=float, default=PLACE_GOAL_X_MARGIN_M)
    parser.add_argument("--place-goal-y-margin", type=float, default=PLACE_GOAL_Y_MARGIN_M)
    parser.add_argument(
        "--rollouts-per-init", type=int, default=1,
        help="Number of noisy rollouts per sampled (dx, dy) start position (pickup only).",
    )
    parser.add_argument("--noisy-worker", action="store_true")
    parser.add_argument("--worker-batch-idx", type=int, default=None)
    parser.add_argument("--worker-dx", type=float, default=None)
    parser.add_argument("--worker-dy", type=float, default=None)
    parser.add_argument("--worker-lateral-offset", type=float, default=None)
    args = parser.parse_args()
    return _normalize_collection_type_args(args, sys.argv[1:])


def _delegate_non_pickup_collection() -> None:
    cmd = [sys.executable, "scripts/collect_dataset_pick_place_release.py", *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.collection_type != "pickup":
        _delegate_non_pickup_collection()
    else:
        collect(parsed_args)
