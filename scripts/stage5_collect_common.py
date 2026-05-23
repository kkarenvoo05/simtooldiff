#!/usr/bin/env python3
"""Shared helpers for Stage 5 dataset collectors."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

# Isaac Gym import order matters: import isaacgym before torch.
from isaacgym import gymapi  # noqa: F401
import torch

import zarr
from numcodecs import Blosc

from stage5_noise_config import resolve_noise_config


N_OBS = 140
N_ACT = 29
DEFAULT_HORIZON = 300
MIN_EPISODE_LEN = 30

DEFAULT_OBJECT_CATEGORY = "hammer"
DEFAULT_OBJECT_NAME = "claw_hammer"
DEFAULT_TASK_NAME = "swing_down"
DEFAULT_COLLECTION_TYPE = "pick_place"
COLLECTION_TYPES = ("pickup", "pick_place", "pick_place_release")

CONFIG_PATH = Path("pretrained_policy/config.yaml")
CHECKPOINT_PATH = Path("pretrained_policy/model.pth")

TABLE_Z = 0.38
TABLE_URDF = "urdf/table_narrow.urdf"
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
DATASET_CAMERA_HEIGHT = 384

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
    dropped_after_lift: bool
    drop_step: Optional[int]
    drop_successes_before: Optional[int]
    reattempted_after_drop: bool
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
        if "object_id" not in data:
            data.create_dataset(
                "object_id",
                shape=(0,),
                chunks=(8192,),
                dtype="uint8",
                compressor=small_compressor,
            )
        if "category_id" not in data:
            data.create_dataset(
                "category_id",
                shape=(0,),
                chunks=(8192,),
                dtype="uint8",
                compressor=small_compressor,
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

    if np.any(np.isnan(state)) or np.any(np.isnan(action)):
        return False
    if np.any(np.all(state == 0, axis=1)):
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


def _update_object_registry(root, args: argparse.Namespace) -> None:
    object_registry = dict(root.attrs.get("object_id_to_name", {}))
    category_registry = dict(root.attrs.get("category_id_to_name", {}))
    object_registry[str(args.object_id)] = f"{args.object_category}/{args.object_name}"
    category_registry[str(args.category_id)] = args.object_category
    root.attrs["object_id_to_name"] = object_registry
    root.attrs["category_id_to_name"] = category_registry


def _destroy_env(env) -> None:
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
