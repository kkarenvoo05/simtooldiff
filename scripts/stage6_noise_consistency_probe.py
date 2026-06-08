#!/usr/bin/env python3
"""Stage 6: Per-Step Noise Consistency Probe.

This is a **diagnostic script**, not a data-collection script. It measures
whether the existing continuous per-step OU action-noise scheme (the
``noisy_clean`` collection used in stage5) produces a coherent "noise band"
around the clean expert trajectory, or whether it produces divergent
re-planned trajectories.

The motivating claim being tested:  a goal-conditioned manipulation expert,
when perturbed by continuous per-step action noise, re-plans toward the goal
rather than recovering toward a reference path, so noisy rollouts from a fixed
start *diverge* rather than forming a tube around the clean trajectory.  This
script produces the measurement that confirms or refutes that claim.

**Hard constraints**:

- This script NEVER writes to a training Zarr.  It is read-only with respect
  to training data.
- Output is videos, plots, and a metrics JSON only.
- It does not feed the diffusion policy in any way.

Recommended debug sequence
--------------------------
1. ``--num-initializations 1 --rollouts-per-init 3`` with noise-scale 0 / no
   noise → confirm the 3 clean rollouts are identical (reset fidelity).
2. ``--num-initializations 1 --rollouts-per-init 10 --noise-scale 4`` → watch
   the 10 videos, check the EE-deviation plot.
3. ``--num-initializations 1 --rollouts-per-init 10 --noise-scale 10`` →
   compare; expect more fanning than 4×.
4. Scale to ``--num-initializations 5`` at both scales.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Isaac Gym import order matters: import isaacgym before torch.
from isaacgym import gymapi  # noqa: F401
import torch

# ---------------------------------------------------------------------------
# Shared constants (mirrored from stage5_collect_dataset.py)
# ---------------------------------------------------------------------------

N_OBS = 140
N_ACT = 29
DEFAULT_HORIZON = 300

DEFAULT_OBJECT_CATEGORY = "hammer"
DEFAULT_OBJECT_NAME = "claw_hammer"
DEFAULT_TASK_NAME = "swing_down"

CONFIG_PATH = Path("pretrained_policy/config.yaml")
CHECKPOINT_PATH = Path("pretrained_policy/model.pth")

TABLE_Z = 0.38
LONG_TABLE_URDF = "urdf/table_pick_place_release.urdf"
LONG_TABLE_X_HALF_EXTENT_M = 0.60 / 2.0
LONG_TABLE_X_INSET_MARGIN_M = 0.06
TABLE_Y_HALF_EXTENT_M = 0.4 / 2.0
TABLE_Y_INSET_MARGIN_M = 0.04
END_BAND_FRACTION = 0.30
DEFAULT_START_Z_OFFSET = 0.0
LIFT_HEIGHT_M = 0.20
MIN_EFFECTIVE_TRANSPORT_M = 0.05
PLACE_HEIGHT_M = 0.02
PLACE_HOLD_GOALS = 10

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

PICKUP_SUCCESS_HOLD_STEPS = 5
PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M = 0.02
PICKUP_SUCCESS_MIN_LIFT_M = 0.12

# Band-verdict thresholds (configurable via CLI).
DEFAULT_TUBE_THRESHOLD_M = 0.03       # mean EE dev stays below this → "tube"
DEFAULT_FAN_THRESHOLD_M = 0.10        # mean EE dev exceeds this mid-episode → "fan"

# Reset-fidelity check tolerance (tight).
RESET_FIDELITY_TOLERANCE = 2.5e-2  # 25 mm — accounts for CUDA physics non-determinism between independent resets


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RolloutRecord:
    """Data recorded for a single rollout."""
    rollout_type: str          # "clean_ref", "clean_control", or "noisy"
    rollout_idx: int           # index within init (noisy rollouts 0-indexed)
    init_idx: int
    success: bool
    steps: int
    ee_positions: np.ndarray   # shape (T, 3)
    object_poses: np.ndarray   # shape (T, 7)  pos + quat
    clean_actions: np.ndarray  # shape (T, N_ACT)  deterministic policy output
    exec_actions: np.ndarray   # shape (T, N_ACT)  executed (same as clean for clean rollouts)
    pickup_gate_history: List[bool]
    frames: Optional[np.ndarray]  # shape (T, H, W, 3) uint8, or None if skipped


@dataclass
class InitMetrics:
    init_idx: int
    start_pose: List[float]
    reset_fidelity_passed: bool
    reset_fidelity_max_err: float
    # EE deviation from clean ref, shape (T,)
    ee_dev_mean: List[float]
    ee_dev_std: List[float]
    # EE pairwise spread, shape (T,)
    ee_spread_mean: List[float]
    ee_spread_max: List[float]
    # Final object pose spread
    final_obj_pos_std: List[float]    # [std_x, std_y, std_z]
    # Action-sequence consistency: mean pairwise L2 between clean-action seqs
    action_seq_mean_pairwise_l2: float
    # Pickup timing
    pickup_timing_std: float
    pickup_timing_values: List[Optional[int]]
    # Success rate under noise
    noisy_success_rate: float
    noisy_successes: int
    rollouts_per_init: int
    band_verdict: str   # "tube", "fan", or "intermediate"


# ---------------------------------------------------------------------------
# Noise helpers (copied from stage5_collect_dataset.py)
# ---------------------------------------------------------------------------

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
    """Sample per-group Gaussian noise with the same scheme as stage5."""
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


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def _noise_seed(base_seed: int, init_idx: int, rollout_idx: int) -> int:
    """Return a deterministic noise seed for a given rollout."""
    return base_seed + init_idx * 1000 + rollout_idx


# ---------------------------------------------------------------------------
# Env construction (mirrors _make_env from stage5_collect_dataset.py)
# ---------------------------------------------------------------------------

def _make_env(
    *,
    num_envs: int,
    nominal_start_pose: List[float],
    goal_poses: List[List[float]],
    horizon: int,
    headless: bool,
    device: str,
    seed: int,
    object_name: str,
):
    """Create a single-env with all randomisation zeroed out.

    This mirrors the reset-randomisation-zeroing block from
    stage5_collect_dataset._make_env exactly.
    """
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
        "task.env.fixedGoalStates": goal_poses,
        "task.env.showGoalObjectVisual": False,
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": nominal_start_pose,
        "task.env.startArmHigher": True,
        "task.env.asset.table": LONG_TABLE_URDF,
        "task.env.tableResetZ": TABLE_Z,
        "task.env.tableObjectZOffset": float(nominal_start_pose[2] - TABLE_Z),
        # Zero out ALL reset randomisation — critical for reset fidelity.
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


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

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


def _sample_end_to_end_start_and_goal(
    rng: np.random.Generator,
    nominal_start_pose: List[float],
    xy_range: float,
) -> Tuple[List[float], float, float]:
    """Mirror collect_dataset_pick_place_release._sample_end_to_end_start_and_goal.

    Places the object in the start band (far negative-x end of long table) and
    samples goal_x from the goal band (far positive-x end), so transport always
    goes in one direction and covers a realistic end-to-end distance.
    """
    x_min = -LONG_TABLE_X_HALF_EXTENT_M + LONG_TABLE_X_INSET_MARGIN_M
    x_max = LONG_TABLE_X_HALF_EXTENT_M - LONG_TABLE_X_INSET_MARGIN_M
    band_span = (x_max - x_min) * END_BAND_FRACTION
    start_band = (x_min, x_min + band_span)
    goal_band = (x_max - band_span, x_max)

    y_min = -TABLE_Y_HALF_EXTENT_M + TABLE_Y_INSET_MARGIN_M
    y_max = TABLE_Y_HALF_EXTENT_M - TABLE_Y_INSET_MARGIN_M

    start_pose = list(nominal_start_pose)
    start_pose[0] = float(rng.uniform(start_band[0], start_band[1]))
    start_pose[1] = min(
        max(float(nominal_start_pose[1]) + float(rng.uniform(-xy_range, xy_range)), y_min),
        y_max,
    )

    goal_x = float(rng.uniform(goal_band[0], goal_band[1]))
    if goal_x < start_pose[0] + MIN_EFFECTIVE_TRANSPORT_M:
        goal_x = start_pose[0] + MIN_EFFECTIVE_TRANSPORT_M

    return start_pose, goal_x, float(goal_x - start_pose[0])


def _build_goals(start_pose: List[float], lateral_offset: float) -> List[List[float]]:
    x0, y0, z0, qx, qy, qz, qw = start_pose
    lift_goal = [x0, y0, z0 + LIFT_HEIGHT_M, qx, qy, qz, qw]
    transport_goal = [x0 + lateral_offset, y0, z0 + LIFT_HEIGHT_M, qx, qy, qz, qw]
    place_goal_base = [x0 + lateral_offset, y0, z0 + PLACE_HEIGHT_M, qx, qy, qz, qw]
    return [lift_goal, transport_goal] + [list(place_goal_base) for _ in range(PLACE_HOLD_GOALS)]


# ---------------------------------------------------------------------------
# Rollout procedure
# ---------------------------------------------------------------------------

def _get_ee_pos(env) -> np.ndarray:
    """Return end-effector (palm-center) position for env 0, shape (3,)."""
    return env.palm_center_pos[0].detach().cpu().numpy().astype(np.float32)


def _get_object_pose(env) -> np.ndarray:
    """Return object pose (pos + quat) for env 0, shape (7,)."""
    return env.object_pose[0, :7].detach().cpu().numpy().astype(np.float32)


def _pickup_gate(object_z: float, object_start_z: float, goal_z: float) -> bool:
    max_lift = object_z - object_start_z
    return (
        object_z >= goal_z - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
        and max_lift >= PICKUP_SUCCESS_MIN_LIFT_M
    )


def _first_pickup_hold_step(gate_history: List[bool], hold_steps: int) -> Optional[int]:
    consecutive = 0
    for step, v in enumerate(gate_history):
        consecutive = consecutive + 1 if v else 0
        if consecutive >= hold_steps:
            return step
    return None


def _run_rollout(
    *,
    env,
    policy,
    device: str,
    horizon: int,
    start_pose: List[float],
    goals: List[List[float]],
    noisy: bool,
    args: argparse.Namespace,
    noise_seed: Optional[int],
    rollout_type: str,
    rollout_idx: int,
    init_idx: int,
    save_frames: bool,
) -> RolloutRecord:
    """Execute one rollout with a full deterministic reset, matching Stage 5.

    Uses the same reset sequence as collect_dataset_pick_place_release
    _run_clean_rollout: set goals/state, reset_idx, policy.reset, one
    zero-action warmup step.  This avoids the snapshot-restore path, which
    breaks after an episode ends because IsaacGym's auto-reset fires first
    and leaves PhysX in a state the tensor restore cannot fully undo.
    """
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)

    # Full deterministic reset — identical to Stage 5's _run_clean_rollout.
    goals_t = torch.tensor(goals, device=env.device, dtype=env.trajectory_states.dtype)
    env.trajectory_states = goals_t
    env.max_consecutive_successes = len(goals)
    env.object_init_state[env_ids, 0:7] = torch.tensor(
        [start_pose], device=env.device, dtype=env.object_init_state.dtype
    )
    env.cfg["env"]["tableObjectZOffset"] = float(start_pose[2] - TABLE_Z)
    env.reset_idx(env_ids, tensor_reset=True)
    policy.reset()

    zero_action = torch.zeros((1, N_ACT), device=device)
    obs_dict, _, _, _ = env.step(zero_action)
    obs = obs_dict["obs"]

    object_start_z = float(env.object_pose[0, 2].item())
    goal_z = float(env.trajectory_states[0, 2].item())

    # OU noise state.
    ou_state = torch.zeros((1, N_ACT), device=device)
    sqrt_dt = math.sqrt(args.ou_dt)

    # Set up a per-rollout torch RNG if noisy, so noise draws don't leak.
    rng_ctx = torch.random.fork_rng(enabled=noisy) if noisy else _null_ctx()

    ee_positions: List[np.ndarray] = []
    object_poses: List[np.ndarray] = []
    clean_actions: List[np.ndarray] = []
    exec_actions: List[np.ndarray] = []
    pickup_gate_history: List[bool] = []
    frames: List[np.ndarray] = []

    active_envs = torch.arange(1, device=device, dtype=torch.long)

    with rng_ctx:
        if noisy and noise_seed is not None:
            torch.manual_seed(noise_seed)

        for step in range(horizon):
            if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
                print(f"[stage6] viewer closed at step {step}; stopping", flush=True)
                break

            # Render.
            if save_frames:
                image_t = env.render_dataset_camera_rgb(active_envs)
                frames.append(image_t[0].detach().cpu().numpy().astype(np.uint8))

            # Query expert (deterministic, clean).
            clean_action_t = policy.get_normalized_action(obs, deterministic_actions=True)

            # Add OU noise if requested.
            if noisy:
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
                ou_state = (
                    ou_state
                    + args.ou_theta * (args.ou_mu - ou_state) * args.ou_dt
                    + sigma_noise * sqrt_dt
                )
                exec_action_t = torch.clamp(clean_action_t + ou_state, -1.0, 1.0)
            else:
                exec_action_t = clean_action_t

            # Step env.
            obs_dict, _, done, _ = env.step(exec_action_t)
            obs = obs_dict["obs"]

            # Record.
            ee_positions.append(_get_ee_pos(env))
            object_poses.append(_get_object_pose(env))
            clean_actions.append(clean_action_t[0].detach().cpu().numpy().astype(np.float32))
            exec_actions.append(exec_action_t[0].detach().cpu().numpy().astype(np.float32))

            obj_z = float(env.object_pose[0, 2].item())
            pickup_gate_history.append(_pickup_gate(obj_z, object_start_z, goal_z))

            if bool(done[0].item()):
                break

    # Determine success via env's own success counter.
    n_goals = int(env.max_consecutive_successes)
    success = int(env.successes[0].item()) >= n_goals

    T = len(ee_positions)
    return RolloutRecord(
        rollout_type=rollout_type,
        rollout_idx=rollout_idx,
        init_idx=init_idx,
        success=success,
        steps=T,
        ee_positions=np.stack(ee_positions, axis=0) if ee_positions else np.zeros((0, 3), dtype=np.float32),
        object_poses=np.stack(object_poses, axis=0) if object_poses else np.zeros((0, 7), dtype=np.float32),
        clean_actions=np.stack(clean_actions, axis=0) if clean_actions else np.zeros((0, N_ACT), dtype=np.float32),
        exec_actions=np.stack(exec_actions, axis=0) if exec_actions else np.zeros((0, N_ACT), dtype=np.float32),
        pickup_gate_history=pickup_gate_history,
        frames=np.stack(frames, axis=0) if (save_frames and frames) else None,
    )


class _null_ctx:
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _align_lengths(*arrays: np.ndarray) -> List[np.ndarray]:
    """Truncate all arrays to the minimum length along axis 0."""
    T = min(a.shape[0] for a in arrays)
    return [a[:T] for a in arrays]


def _metric1_ee_dev_from_ref(
    ref: np.ndarray,
    noisy_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean ± std of EE distance from clean reference over time.

    Returns (mean, std) each shape (T,).
    """
    if not noisy_list:
        T = ref.shape[0]
        return np.zeros(T), np.zeros(T)
    T = min(ref.shape[0], min(n.shape[0] for n in noisy_list))
    ref_t = ref[:T]
    devs = np.stack([np.linalg.norm(n[:T] - ref_t, axis=-1) for n in noisy_list], axis=0)
    return devs.mean(axis=0), devs.std(axis=0)


def _metric2_ee_pairwise_spread(
    noisy_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Std and max pairwise distance of EE positions over time.

    Returns (std_per_t, max_pairwise_per_t) each shape (T,).
    """
    if len(noisy_list) < 2:
        T = noisy_list[0].shape[0] if noisy_list else 0
        return np.zeros(T), np.zeros(T)
    T = min(n.shape[0] for n in noisy_list)
    stacked = np.stack([n[:T] for n in noisy_list], axis=0)  # (M, T, 3)
    std_per_t = stacked.std(axis=0).mean(axis=-1)  # (T,)
    # max pairwise: for each t, max L2 between any two rollouts
    max_pair = np.zeros(T, dtype=np.float32)
    M = stacked.shape[0]
    for i in range(M):
        for j in range(i + 1, M):
            dist = np.linalg.norm(stacked[i] - stacked[j], axis=-1)
            max_pair = np.maximum(max_pair, dist)
    return std_per_t, max_pair


def _metric3_final_obj_pos_spread(
    noisy_rollouts: List[RolloutRecord],
) -> np.ndarray:
    """Std of final object xyz across noisy rollouts, shape (3,)."""
    if not noisy_rollouts:
        return np.zeros(3)
    finals = np.stack([r.object_poses[-1, :3] for r in noisy_rollouts if r.object_poses.shape[0] > 0], axis=0)
    if finals.shape[0] == 0:
        return np.zeros(3)
    return finals.std(axis=0)


def _metric4_action_seq_consistency(
    noisy_rollouts: List[RolloutRecord],
) -> float:
    """Mean pairwise L2 distance between clean-action sequences.

    This is the closest proxy for label-consistency — the actual property
    that matters for BC.
    """
    seqs = [r.clean_actions for r in noisy_rollouts if r.clean_actions.shape[0] > 0]
    if len(seqs) < 2:
        return 0.0
    T = min(s.shape[0] for s in seqs)
    seqs_t = np.stack([s[:T] for s in seqs], axis=0)   # (M, T, N_ACT)
    M = seqs_t.shape[0]
    total, count = 0.0, 0
    for i in range(M):
        for j in range(i + 1, M):
            diff = seqs_t[i] - seqs_t[j]
            l2 = np.linalg.norm(diff.reshape(-1))
            total += float(l2)
            count += 1
    return total / count if count > 0 else 0.0


def _metric5_pickup_timing_spread(
    noisy_rollouts: List[RolloutRecord],
) -> Tuple[float, List[Optional[int]]]:
    """Std of first-pickup step across noisy rollouts."""
    values: List[Optional[int]] = []
    for r in noisy_rollouts:
        step = _first_pickup_hold_step(r.pickup_gate_history, PICKUP_SUCCESS_HOLD_STEPS)
        values.append(step)
    numeric = [v for v in values if v is not None]
    std = float(np.std(numeric)) if len(numeric) >= 2 else 0.0
    return std, values


def _band_verdict(
    ee_dev_mean: np.ndarray,
    *,
    tube_threshold: float,
    fan_threshold: float,
) -> str:
    """Classify the deviation curve as 'tube', 'fan', or 'intermediate'."""
    if ee_dev_mean.size == 0:
        return "intermediate"
    T = ee_dev_mean.shape[0]
    mid = T // 2

    if ee_dev_mean.max() < tube_threshold:
        return "tube"
    if ee_dev_mean[mid:].mean() > fan_threshold:
        return "fan"
    return "intermediate"


# ---------------------------------------------------------------------------
# Video / plotting
# ---------------------------------------------------------------------------

def _annotate_frames(
    frames: np.ndarray,
    *,
    init_idx: int,
    rollout_type: str,
    rollout_idx: Optional[int],
    noise_scale: float,
    success: bool,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    annotated = []
    for step, frame in enumerate(frames):
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        lines = [
            f"init {init_idx:02d}",
            f"type {rollout_type}",
            f"step {step:03d}",
            f"noise_scale {noise_scale:.1f}",
            f"success {int(success)}",
        ]
        if rollout_idx is not None:
            lines.insert(2, f"rollout {rollout_idx:02d}")
        y = 12
        for line in lines:
            bbox = draw.textbbox((12, y), line)
            draw.rectangle(
                (bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
                fill=(0, 0, 0),
            )
            draw.text((12, y), line, fill=(255, 255, 255))
            y += 22
        annotated.append(np.asarray(image, dtype=np.uint8))
    return np.stack(annotated, axis=0)


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=fps)


def _save_plots(
    output_dir: Path,
    init_idx: int,
    ee_dev_mean: np.ndarray,
    ee_dev_std: np.ndarray,
    ee_spread_mean: np.ndarray,
    ee_spread_max: np.ndarray,
    noisy_rollouts: List[RolloutRecord],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[stage6] matplotlib not available — skipping plots", flush=True)
        return

    plot_dir = output_dir / "plots" / f"init{init_idx:02d}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(ee_dev_mean.shape[0])

    # Metric 1: EE deviation vs time.
    fig, ax = plt.subplots()
    ax.plot(steps, ee_dev_mean, label="mean EE dev from ref")
    ax.fill_between(steps, ee_dev_mean - ee_dev_std, ee_dev_mean + ee_dev_std, alpha=0.3, label="±std")
    ax.set_xlabel("Step")
    ax.set_ylabel("EE distance from clean ref (m)")
    ax.set_title(f"Init {init_idx}: EE deviation from clean reference")
    ax.legend()
    fig.savefig(plot_dir / "ee_deviation.png", dpi=100)
    plt.close(fig)

    # Metric 2: EE pairwise spread vs time.
    fig, ax = plt.subplots()
    ax.plot(steps[:len(ee_spread_mean)], ee_spread_mean[:len(steps)], label="pairwise std")
    ax.plot(steps[:len(ee_spread_max)], ee_spread_max[:len(steps)], label="max pairwise dist", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("EE spread (m)")
    ax.set_title(f"Init {init_idx}: EE pairwise spread")
    ax.legend()
    fig.savefig(plot_dir / "ee_pairwise_spread.png", dpi=100)
    plt.close(fig)

    # Metric 3: Final object position scatter.
    if noisy_rollouts:
        finals = np.stack([
            r.object_poses[-1, :3] for r in noisy_rollouts if r.object_poses.shape[0] > 0
        ], axis=0)
        if finals.shape[0] > 0:
            fig, ax = plt.subplots()
            ax.scatter(finals[:, 0], finals[:, 1], alpha=0.7, label="noisy rollouts")
            ax.set_xlabel("Object X (m)")
            ax.set_ylabel("Object Y (m)")
            ax.set_title(f"Init {init_idx}: Final object XY positions")
            ax.legend()
            fig.savefig(plot_dir / "final_object_scatter.png", dpi=100)
            plt.close(fig)


def _save_aggregate_plots(
    output_dir: Path,
    all_init_metrics: List[InitMetrics],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir = output_dir / "plots" / "aggregate"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for m in all_init_metrics:
        steps = np.arange(len(m.ee_dev_mean))
        ax.plot(steps, m.ee_dev_mean, alpha=0.6, label=f"init{m.init_idx}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean EE dev from clean ref (m)")
    ax.set_title("Aggregate EE deviation per init")
    ax.legend(fontsize=7)
    fig.savefig(plot_dir / "aggregate_ee_deviation.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Noise config resolution (delegated to stage5_noise_config)
# ---------------------------------------------------------------------------

def _resolve_noise_args(args: argparse.Namespace) -> argparse.Namespace:
    from stage5_noise_config import resolve_noise_config

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


# ---------------------------------------------------------------------------
# Main probe driver
# ---------------------------------------------------------------------------



def _run_probe(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args = _resolve_noise_args(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[stage6] Starting noise consistency probe", flush=True)
    print(
        f"[stage6] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} seed={args.seed}",
        flush=True,
    )
    print(
        f"[stage6] num_initializations={args.num_initializations} "
        f"rollouts_per_init={args.rollouts_per_init} "
        f"noise_scale={args.noise_scale} horizon={args.horizon}",
        flush=True,
    )
    print(
        f"[stage6] noise: arm_base={args.arm_base_noise:.4f} arm_wrist={args.arm_wrist_noise:.4f} "
        f"thumb={args.thumb_noise:.4f} index={args.index_noise:.4f} middle={args.middle_noise:.4f} "
        f"ring={args.ring_noise:.4f} pinky={args.pinky_noise:.4f} "
        f"ou_theta={args.ou_theta} ou_mu={args.ou_mu} ou_dt={args.ou_dt}",
        flush=True,
    )
    print(f"[stage6] output_dir={output_dir}", flush=True)

    nominal_start_pose = _load_nominal_start_pose(
        args.object_category, args.object_name, args.task_name
    )
    # Bootstrap goals use the midpoint of the goal band as a reasonable placeholder,
    # mirroring collect_dataset_pick_place_release._default_goal_x.
    _x_min = -LONG_TABLE_X_HALF_EXTENT_M + LONG_TABLE_X_INSET_MARGIN_M
    _x_max = LONG_TABLE_X_HALF_EXTENT_M - LONG_TABLE_X_INSET_MARGIN_M
    _band_span = (_x_max - _x_min) * END_BAND_FRACTION
    _bootstrap_goal_x = _x_max - _band_span * 0.5
    _bootstrap_lateral_offset = max(_bootstrap_goal_x - float(nominal_start_pose[0]), MIN_EFFECTIVE_TRANSPORT_M)
    bootstrap_goals = _build_goals(nominal_start_pose, _bootstrap_lateral_offset)

    print("[stage6] Creating env (one-time)...", flush=True)
    env = _make_env(
        num_envs=1,
        nominal_start_pose=nominal_start_pose,
        goal_poses=bootstrap_goals,
        horizon=args.horizon,
        headless=not getattr(args, "viewer", False),
        device=device,
        seed=args.seed,
        object_name=args.object_name,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
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

    # Sample N initializations using the same end-to-end band logic as
    # collect_dataset_pick_place_release: start in the far-negative-x band,
    # goal in the far-positive-x band, transport always in one direction.
    rng = np.random.default_rng(args.seed)
    XY_RANGE = 0.10
    init_start_poses = []
    init_lateral_offsets = []
    init_goal_xs = []
    for _ in range(args.num_initializations):
        pose, goal_x, lateral_offset = _sample_end_to_end_start_and_goal(
            rng, nominal_start_pose, XY_RANGE
        )
        init_start_poses.append(pose)
        init_lateral_offsets.append(lateral_offset)
        init_goal_xs.append(goal_x)

    all_init_metrics: List[InitMetrics] = []
    t_start = time.time()

    try:
        for init_idx in range(args.num_initializations):
            start_pose = init_start_poses[init_idx]
            lateral_offset = init_lateral_offsets[init_idx]
            goal_x = init_goal_xs[init_idx]
            goals = _build_goals(start_pose, lateral_offset)

            print(
                f"\n[stage6] === Init {init_idx}/{args.num_initializations - 1} "
                f"start_x={start_pose[0]:+.3f} goal_x={goal_x:+.3f} "
                f"lateral_offset={lateral_offset:+.3f} "
                f"start_pose={np.array(start_pose).round(4).tolist()} ===",
                flush=True,
            )

            # Shared kwargs for all rollouts from this init.
            # Each rollout does its own full reset internally.
            _rollout_kw = dict(
                env=env, policy=policy, device=device, horizon=args.horizon,
                start_pose=start_pose, goals=goals,
                args=args, init_idx=init_idx,
            )

            # --- Clean reference rollout ---
            print(f"[stage6] init={init_idx} running clean_ref...", flush=True)
            clean_ref = _run_rollout(
                **_rollout_kw,
                noisy=False, noise_seed=None,
                rollout_type="clean_ref", rollout_idx=0,
                save_frames=not args.skip_videos,
            )

            # --- Clean control rollout (determinism check) ---
            print(f"[stage6] init={init_idx} running clean_control...", flush=True)
            clean_ctrl = _run_rollout(
                **_rollout_kw,
                noisy=False, noise_seed=None,
                rollout_type="clean_control", rollout_idx=1,
                save_frames=False,
            )

            # --- Reset fidelity check ---
            T = min(clean_ref.ee_positions.shape[0], clean_ctrl.ee_positions.shape[0])
            if T > 0:
                ee_diff = np.abs(clean_ref.ee_positions[:T] - clean_ctrl.ee_positions[:T])
                fidelity_max_err = float(ee_diff.max())
            else:
                fidelity_max_err = float("inf")

            fidelity_passed = fidelity_max_err <= RESET_FIDELITY_TOLERANCE
            if not fidelity_passed:
                print(
                    f"\n[stage6] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    f"[stage6] RESET FIDELITY FAILED for init={init_idx}!\n"
                    f"[stage6] Max EE diff between clean_ref and clean_control: "
                    f"{fidelity_max_err:.6f} m (tolerance={RESET_FIDELITY_TOLERANCE})\n"
                    f"[stage6] Noisy rollout spread for this init cannot be attributed\n"
                    f"[stage6] to noise alone. Flagging as reset_fidelity_failed.\n"
                    f"[stage6] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                    flush=True,
                )
            else:
                print(
                    f"[stage6] init={init_idx} reset_fidelity OK "
                    f"(max_ee_diff={fidelity_max_err:.2e})",
                    flush=True,
                )

            # --- Noisy rollouts ---
            noisy_rollouts: List[RolloutRecord] = []
            for rollout_idx in range(args.rollouts_per_init):
                ns = _noise_seed(args.seed, init_idx, rollout_idx)
                print(
                    f"[stage6] init={init_idx} noisy rollout {rollout_idx}/{args.rollouts_per_init - 1} "
                    f"noise_seed={ns}...",
                    flush=True,
                )
                r = _run_rollout(
                    **_rollout_kw,
                    noisy=True, noise_seed=ns,
                    rollout_type="noisy", rollout_idx=rollout_idx,
                    save_frames=not args.skip_videos,
                )
                noisy_rollouts.append(r)
                print(
                    f"[stage6] init={init_idx} noisy={rollout_idx} "
                    f"steps={r.steps} success={r.success}",
                    flush=True,
                )

            # --- Save videos ---
            if not args.skip_videos:
                _save_init_videos(
                    output_dir=output_dir,
                    init_idx=init_idx,
                    clean_ref=clean_ref,
                    noisy_rollouts=noisy_rollouts,
                    args=args,
                )

            # --- Compute metrics ---
            noisy_ee = [r.ee_positions for r in noisy_rollouts]
            ee_dev_mean, ee_dev_std = _metric1_ee_dev_from_ref(clean_ref.ee_positions, noisy_ee)
            ee_spread_mean, ee_spread_max = _metric2_ee_pairwise_spread(noisy_ee)
            final_obj_std = _metric3_final_obj_pos_spread(noisy_rollouts)
            action_consistency = _metric4_action_seq_consistency(noisy_rollouts)
            pickup_timing_std, pickup_timing_values = _metric5_pickup_timing_spread(noisy_rollouts)
            noisy_successes = sum(1 for r in noisy_rollouts if r.success)
            noisy_success_rate = noisy_successes / max(len(noisy_rollouts), 1)
            verdict = _band_verdict(
                ee_dev_mean,
                tube_threshold=args.tube_threshold,
                fan_threshold=args.fan_threshold,
            )

            print(
                f"[stage6] init={init_idx} band_verdict={verdict} "
                f"noisy_success_rate={noisy_success_rate:.1%} "
                f"action_seq_consistency={action_consistency:.4f} "
                f"pickup_timing_std={pickup_timing_std:.1f}",
                flush=True,
            )

            # --- Save per-init plots ---
            _save_plots(
                output_dir=output_dir,
                init_idx=init_idx,
                ee_dev_mean=ee_dev_mean,
                ee_dev_std=ee_dev_std,
                ee_spread_mean=ee_spread_mean,
                ee_spread_max=ee_spread_max,
                noisy_rollouts=noisy_rollouts,
            )

            im = InitMetrics(
                init_idx=init_idx,
                start_pose=start_pose,
                reset_fidelity_passed=fidelity_passed,
                reset_fidelity_max_err=fidelity_max_err,
                ee_dev_mean=ee_dev_mean.tolist(),
                ee_dev_std=ee_dev_std.tolist(),
                ee_spread_mean=ee_spread_mean.tolist(),
                ee_spread_max=ee_spread_max.tolist(),
                final_obj_pos_std=final_obj_std.tolist(),
                action_seq_mean_pairwise_l2=action_consistency,
                pickup_timing_std=pickup_timing_std,
                pickup_timing_values=pickup_timing_values,
                noisy_success_rate=noisy_success_rate,
                noisy_successes=noisy_successes,
                rollouts_per_init=args.rollouts_per_init,
                band_verdict=verdict,
            )
            all_init_metrics.append(im)

    finally:
        _destroy_env(env)

    # --- Aggregate metrics and plots ---
    _save_aggregate_plots(output_dir, all_init_metrics)
    _save_summary_json(output_dir, all_init_metrics, args, time.time() - t_start)

    elapsed = time.time() - t_start
    print(f"\n[stage6] DONE elapsed={elapsed/60:.1f}min", flush=True)
    print(f"[stage6] output_dir={output_dir}", flush=True)

    verdicts = [m.band_verdict for m in all_init_metrics]
    from collections import Counter
    verdict_counts = Counter(verdicts)
    print(f"[stage6] band_verdicts={dict(verdict_counts)}", flush=True)

    fidelity_fails = [m.init_idx for m in all_init_metrics if not m.reset_fidelity_passed]
    if fidelity_fails:
        print(f"[stage6] WARNING: reset_fidelity_failed for inits={fidelity_fails}", flush=True)


def _save_init_videos(
    *,
    output_dir: Path,
    init_idx: int,
    clean_ref: RolloutRecord,
    noisy_rollouts: List[RolloutRecord],
    args: argparse.Namespace,
) -> None:
    video_dir = output_dir / "videos" / f"init{init_idx:02d}"
    video_dir.mkdir(parents=True, exist_ok=True)

    if clean_ref.frames is not None and clean_ref.frames.shape[0] > 0:
        annotated = _annotate_frames(
            clean_ref.frames,
            init_idx=init_idx,
            rollout_type="clean_ref",
            rollout_idx=None,
            noise_scale=args.noise_scale,
            success=clean_ref.success,
        )
        _write_video(video_dir / "clean_ref.mp4", annotated, args.video_fps)

    for r in noisy_rollouts:
        if r.frames is not None and r.frames.shape[0] > 0:
            annotated = _annotate_frames(
                r.frames,
                init_idx=init_idx,
                rollout_type="noisy",
                rollout_idx=r.rollout_idx,
                noise_scale=args.noise_scale,
                success=r.success,
            )
            _write_video(
                video_dir / f"noisy_{r.rollout_idx:02d}.mp4",
                annotated,
                args.video_fps,
            )


def _summary_dict(all_init_metrics: List[InitMetrics], args: argparse.Namespace, elapsed_s: float) -> dict:
    from collections import Counter

    def _agg(values: List[float]) -> dict:
        a = np.array(values, dtype=np.float64)
        return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}

    verdicts = [m.band_verdict for m in all_init_metrics]
    verdict_counts = dict(Counter(verdicts))

    # Aggregate band verdict: majority wins; if split, "intermediate".
    if verdict_counts:
        majority = max(verdict_counts, key=verdict_counts.get)
        if verdict_counts[majority] > len(verdicts) / 2:
            agg_verdict = majority
        else:
            agg_verdict = "intermediate"
    else:
        agg_verdict = "intermediate"

    per_init = []
    for m in all_init_metrics:
        per_init.append({
            "init_idx": m.init_idx,
            "start_pose": m.start_pose,
            "reset_fidelity": {
                "passed": m.reset_fidelity_passed,
                "max_err_m": m.reset_fidelity_max_err,
            },
            "band_verdict": m.band_verdict,
            "metrics": {
                "ee_dev_from_ref": {
                    "mean_at_steps": m.ee_dev_mean[:10],   # first 10 steps summary
                    "mean_overall": float(np.mean(m.ee_dev_mean)) if m.ee_dev_mean else 0.0,
                    "max": float(np.max(m.ee_dev_mean)) if m.ee_dev_mean else 0.0,
                },
                "ee_pairwise_spread": {
                    "mean_overall": float(np.mean(m.ee_spread_mean)) if m.ee_spread_mean else 0.0,
                    "max": float(np.max(m.ee_spread_max)) if m.ee_spread_max else 0.0,
                },
                "final_obj_pos_std_xyz": m.final_obj_pos_std,
                "action_seq_mean_pairwise_l2": m.action_seq_mean_pairwise_l2,
                "pickup_timing_std": m.pickup_timing_std,
                "pickup_timing_values": m.pickup_timing_values,
                "noisy_success_rate": m.noisy_success_rate,
                "noisy_successes": m.noisy_successes,
                "rollouts_per_init": m.rollouts_per_init,
            },
        })

    return {
        "band_verdict": agg_verdict,
        "band_verdict_per_init": verdict_counts,
        "noise_config": {
            "noise_scale": args.noise_scale,
            "arm_base_noise": args.arm_base_noise,
            "arm_wrist_noise": args.arm_wrist_noise,
            "thumb_noise": args.thumb_noise,
            "index_noise": args.index_noise,
            "middle_noise": args.middle_noise,
            "ring_noise": args.ring_noise,
            "pinky_noise": args.pinky_noise,
            "ou_theta": args.ou_theta,
            "ou_mu": args.ou_mu,
            "ou_dt": args.ou_dt,
        },
        "run_config": {
            "object_category": args.object_category,
            "object_name": args.object_name,
            "task_name": args.task_name,
            "num_initializations": args.num_initializations,
            "rollouts_per_init": args.rollouts_per_init,
            "horizon": args.horizon,
            "seed": args.seed,
            "tube_threshold_m": args.tube_threshold,
            "fan_threshold_m": args.fan_threshold,
        },
        "aggregate": {
            "action_seq_consistency": _agg(
                [m.action_seq_mean_pairwise_l2 for m in all_init_metrics]
            ),
            "noisy_success_rate": _agg(
                [m.noisy_success_rate for m in all_init_metrics]
            ),
            "max_ee_dev_from_ref": _agg(
                [max(m.ee_dev_mean) if m.ee_dev_mean else 0.0 for m in all_init_metrics]
            ),
            "reset_fidelity_pass_rate": sum(
                1 for m in all_init_metrics if m.reset_fidelity_passed
            ) / max(len(all_init_metrics), 1),
        },
        "elapsed_s": elapsed_s,
        "per_init": per_init,
    }


def _save_summary_json(
    output_dir: Path,
    all_init_metrics: List[InitMetrics],
    args: argparse.Namespace,
    elapsed_s: float,
) -> None:
    summary = _summary_dict(all_init_metrics, args, elapsed_s)
    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stage6] Saved summary to {json_path}", flush=True)
    print(f"[stage6] band_verdict={summary['band_verdict']}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--object-category", type=str, default=DEFAULT_OBJECT_CATEGORY)
    parser.add_argument("--object-name", type=str, default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK_NAME)
    parser.add_argument(
        "--num-initializations", type=int, default=5,
        help="Number of distinct fixed start poses to test.",
    )
    parser.add_argument(
        "--rollouts-per-init", type=int, default=10,
        help="Number of noisy rollouts per initialization.",
    )
    parser.add_argument(
        "--noise-scale", type=float, default=10.0,
        help="Multiplier applied to Stage-5 groupwise base noise defaults.",
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
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/stage6_probe/default"),
        help="Directory for videos, plots, and metrics JSON.",
    )
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument(
        "--num-envs", type=int, default=1,
        help="Must be 1. Multi-env parallelism is not supported in v1.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--skip-videos", action="store_true",
        help="Skip video writing for fast metric-only runs.",
    )
    parser.add_argument(
        "--tube-threshold", type=float, default=DEFAULT_TUBE_THRESHOLD_M,
        help="If mean EE dev stays below this (m) for whole episode → verdict 'tube'.",
    )
    parser.add_argument(
        "--fan-threshold", type=float, default=DEFAULT_FAN_THRESHOLD_M,
        help="If mean EE dev exceeds this (m) by mid-episode → verdict 'fan'.",
    )

    args = parser.parse_args()

    if args.num_envs != 1:
        parser.error(
            f"--num-envs must be 1 (got {args.num_envs}). "
            "Per-init fixed-state resets are stateful and must not be parallelized in v1."
        )

    return args


def main() -> None:
    args = parse_args()
    _run_probe(args)


if __name__ == "__main__":
    main()
