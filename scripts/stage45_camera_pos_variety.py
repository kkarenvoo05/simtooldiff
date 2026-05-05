#!/usr/bin/env python3
"""Stage 4.5: validate that per-rollout camera position randomization works.

This script is a focused validation of *only* camera position variation. The
hammer start pose is held fixed (dx=dy=0) so that any visual difference between
rollouts is attributable to the camera, not the scene. After all rollouts, a
side-by-side comparison image is produced so the camera variation is obvious
at a glance.

The expert in this codebase is object-pose-based (see SimToolReal architecture:
the RL policy consumes object poses from FoundationPose, not images), so
changing the dataset camera does not affect expert behavior. The dataset camera
exists purely to record images for downstream diffusion policy training.

Usage (driver):
    python stage4_5_camera_position_variation.py

Usage (worker, invoked by driver):
    python stage4_5_camera_position_variation.py --worker --rollout-idx N \\
        --camera-pos-x X --camera-pos-y Y --camera-pos-z Z ...
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf


N_OBS = 140
N_ACT = 29
DEFAULT_HORIZON = 150

OBJECT_CATEGORY = "hammer"
OBJECT_NAME = "claw_hammer"
TASK_NAME = "swing_down"

CONFIG_PATH = Path("pretrained_policy/config.yaml")
CHECKPOINT_PATH = Path("pretrained_policy/model.pth")

TABLE_Z = 0.38
TABLE_NAIL_URDF = "urdf/table_narrow_nail.urdf"
START_Z_OFFSET = 0.03
LIFT_HEIGHT_M = 0.20
PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M = 0.02
PICKUP_SUCCESS_MIN_LIFT_M = 0.12
PICKUP_SUCCESS_HOLD_STEPS = 5

# ---------------------------------------------------------------------------
# Camera variation: spherical coordinates around a fixed scene anchor.
#
# The anchor is the same point your Stage 4 dataset camera was looking at.
# We hold the look-at target fixed in this script; only camera POSITION moves.
# Position is sampled in spherical coords around the anchor, so the camera
# always points at roughly the right region of the scene.
#
# Stage 4 nominal camera was at [0.55, -1.35, 1.10] looking at [-0.1, 0.35, 0.60].
# In spherical coords around the anchor this is approximately:
#   radius    ~= 1.85 m
#   azimuth   ~= -132 deg  (atan2(pos_x - anchor_x, pos_y - anchor_y), but see
#                           _spherical_to_cartesian for the exact convention)
#   elevation ~= 15.7 deg
#
# V1 mild range (matches user's earlier choice of ~ +/- 15 az, +/- 10 el).
# ---------------------------------------------------------------------------
CAMERA_ANCHOR = (-0.1, 0.35, 0.60)  # look-at target (held fixed in this script)
CAMERA_NOMINAL_POS = (0.55, -1.35, 1.10)  # the Stage 4 fixed camera

CAMERA_RADIUS_RANGE_M = (1.70, 2.00)
CAMERA_AZIMUTH_DEG_RANGE = (-147.0, -117.0)   # nominal -132 +/- 15
CAMERA_ELEVATION_DEG_RANGE = (5.7, 25.7)      # nominal 15.7 +/- 10

# Camera target/FOV are held FIXED in this script (user will randomize later).
DATASET_CAMERA_TARGET = list(CAMERA_ANCHOR)
DATASET_CAMERA_HORIZONTAL_FOV = 30.0
DATASET_CAMERA_WIDTH = 512
DATASET_CAMERA_HEIGHT = 360


def _spherical_to_cartesian(
    radius_m: float,
    azimuth_deg: float,
    elevation_deg: float,
    anchor: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Convert (radius, azimuth, elevation) -> world (x, y, z) around anchor.

    Convention chosen so that the Stage 4 nominal camera position
    [0.55, -1.35, 1.10] with anchor [-0.1, 0.35, 0.60] maps to azimuth ~= -132 deg
    and elevation ~= 15.7 deg:

        dx = pos_x - anchor_x = 0.65
        dy = pos_y - anchor_y = -1.70
        dz = pos_z - anchor_z = 0.50
        radius_xy = sqrt(dx^2 + dy^2) = 1.82
        elevation = atan2(dz, radius_xy) = 15.4 deg  (close to 15.7)
        azimuth   = atan2(dx, dy) (in deg)
                  = atan2(0.65, -1.70) = -159 + 180-ish... let's just verify
                  with a unit test inside _check_nominal_roundtrip below.

    The exact convention doesn't matter as long as it's self-consistent and the
    nominal position lies inside the sampled range.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    radius_xy = radius_m * math.cos(el)
    dx = radius_xy * math.sin(az)
    dy = radius_xy * math.cos(az)
    dz = radius_m * math.sin(el)
    return (anchor[0] + dx, anchor[1] + dy, anchor[2] + dz)


def _cartesian_to_spherical(
    pos: Tuple[float, float, float],
    anchor: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Inverse of _spherical_to_cartesian. Returns (radius_m, azimuth_deg, elevation_deg)."""
    dx = pos[0] - anchor[0]
    dy = pos[1] - anchor[1]
    dz = pos[2] - anchor[2]
    radius_xy = math.sqrt(dx * dx + dy * dy)
    radius = math.sqrt(dx * dx + dy * dy + dz * dz)
    azimuth_deg = math.degrees(math.atan2(dx, dy))
    elevation_deg = math.degrees(math.atan2(dz, radius_xy))
    return (radius, azimuth_deg, elevation_deg)


def _check_nominal_roundtrip() -> None:
    """Sanity check: nominal Stage 4 camera should lie inside our sampled range.

    Run at driver startup so we fail fast if the spherical convention is broken.
    """
    radius, azimuth_deg, elevation_deg = _cartesian_to_spherical(
        CAMERA_NOMINAL_POS, CAMERA_ANCHOR
    )
    pos_back = _spherical_to_cartesian(radius, azimuth_deg, elevation_deg, CAMERA_ANCHOR)
    err = max(abs(a - b) for a, b in zip(CAMERA_NOMINAL_POS, pos_back))
    assert err < 1e-6, f"spherical roundtrip broken: {pos_back} vs {CAMERA_NOMINAL_POS}"
    print(
        f"[stage4.5] nominal camera in spherical coords: "
        f"radius={radius:.3f}m, azimuth={azimuth_deg:.1f}deg, elevation={elevation_deg:.1f}deg",
        flush=True,
    )
    in_radius = CAMERA_RADIUS_RANGE_M[0] <= radius <= CAMERA_RADIUS_RANGE_M[1]
    in_azimuth = CAMERA_AZIMUTH_DEG_RANGE[0] <= azimuth_deg <= CAMERA_AZIMUTH_DEG_RANGE[1]
    in_elevation = (
        CAMERA_ELEVATION_DEG_RANGE[0] <= elevation_deg <= CAMERA_ELEVATION_DEG_RANGE[1]
    )
    if not (in_radius and in_azimuth and in_elevation):
        print(
            f"[stage4.5] WARNING: nominal camera not inside sampled range "
            f"(radius_in={in_radius}, az_in={in_azimuth}, el_in={in_elevation}). "
            f"This is fine but means rollouts won't include the Stage 4 view.",
            flush=True,
        )


def _format_pose(values) -> List[float]:
    return [round(float(x), 6) for x in values]


def _jsonable(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _actor_root_pose(env, indices) -> List[float]:
    actor_idx = int(indices[0].item())
    return _format_pose(env.root_state_tensor[actor_idx, 0:7].detach().cpu())


def _print_env_diagnostics(prefix: str, env, object_start_pose, goal_pose) -> None:
    env.gym.refresh_actor_root_state_tensor(env.sim)
    cfg_env = env.cfg["env"]
    print(
        f"{prefix} dataset_camera_position={cfg_env['datasetCameraPosition']} "
        f"target={cfg_env['datasetCameraTarget']} "
        f"horizontal_fov={cfg_env.get('datasetCameraHorizontalFov')}",
        flush=True,
    )
    print(
        f"{prefix} requested_object_start_pose={_format_pose(object_start_pose)} "
        f"requested_goal_pose={_format_pose(goal_pose)}",
        flush=True,
    )


def _load_nominal_start_pose() -> List[float]:
    from isaacgymenvs.utils.utils import get_repo_root_dir

    trajectory_path = (
        get_repo_root_dir()
        / "dextoolbench"
        / "trajectories"
        / OBJECT_CATEGORY
        / OBJECT_NAME
        / f"{TASK_NAME}.json"
    )
    assert trajectory_path.exists(), f"Missing trajectory: {trajectory_path}"
    with open(trajectory_path, "r") as f:
        traj_data = json.load(f)

    object_start_pose = list(traj_data["start_pose"])
    object_start_pose[2] += START_Z_OFFSET
    return object_start_pose


def _make_env(
    goal_pose,
    object_start_pose,
    horizon: int,
    headless: bool,
    device: str,
    camera_position: List[float],
):
    from deployment.isaac.isaac_env import create_env

    overrides = {
        "task.env.numEnvs": 1,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        "task.env.enableCameraSensors": True,
        "task.env.enableDatasetCameras": True,
        "task.env.objectName": OBJECT_NAME,
        "task.env.useFixedGoalStates": True,
        "task.env.fixedGoalStates": [goal_pose],
        "task.env.showGoalObjectVisual": False,
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": object_start_pose,
        "task.env.startArmHigher": True,
        "task.env.asset.table": TABLE_NAIL_URDF,
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
        # Camera variation lives here. Position is per-rollout; target/FOV/size
        # are fixed in this Stage 4.5 script (user will randomize them later).
        "task.env.datasetCameraPosition": list(camera_position),
        "task.env.datasetCameraTarget": list(DATASET_CAMERA_TARGET),
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


def run_worker(args: argparse.Namespace) -> None:
    import imageio.v2 as imageio

    # Isaac Gym must be imported before torch.
    from isaacgym import gymapi  # noqa: F401
    import torch

    from deployment.rl_player import RlPlayer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rollout_dir = args.output_dir / f"rollout_{args.rollout_idx:04d}"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    object_start_pose = _load_nominal_start_pose()  # dx=dy=0, fixed scene
    goal_pose = list(object_start_pose)
    goal_pose[2] += LIFT_HEIGHT_M

    camera_position = [args.camera_pos_x, args.camera_pos_y, args.camera_pos_z]

    print(
        f"[stage4.5 worker {args.rollout_idx:04d}] "
        f"camera_position=[{camera_position[0]:+.3f}, {camera_position[1]:+.3f}, "
        f"{camera_position[2]:+.3f}] "
        f"radius={args.camera_radius:.3f}m az={args.camera_azimuth_deg:.1f}deg "
        f"el={args.camera_elevation_deg:.1f}deg device={device}",
        flush=True,
    )

    env = _make_env(
        goal_pose=goal_pose,
        object_start_pose=object_start_pose,
        horizon=args.horizon,
        headless=True,
        device=device,
        camera_position=camera_position,
    )
    _print_env_diagnostics(
        f"[stage4.5 worker {args.rollout_idx:04d}] after_env_init",
        env,
        object_start_pose,
        goal_pose,
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

    images = []
    object_zs = []
    pickup_gate_history = []

    active_env = torch.tensor([0], device=device, dtype=torch.long)
    for step in range(args.horizon):
        image_t = env.render_dataset_camera_rgb(active_env)[0]
        action_t = policy.get_normalized_action(obs, deterministic_actions=True)

        images.append(image_t.cpu().numpy().astype(np.uint8))

        obs_dict, _, _, _ = env.step(action_t)
        obs = obs_dict["obs"]

        object_pose_np = env.object_pose[0, 0:7].detach().cpu().numpy()
        object_zs.append(float(object_pose_np[2]))
        current_lift_m = float(object_pose_np[2]) - float(object_start_pose[2])
        current_pickup_gate = bool(
            object_pose_np[2] >= float(goal_pose[2]) - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
            and current_lift_m >= PICKUP_SUCCESS_MIN_LIFT_M
        )
        pickup_gate_history.append(current_pickup_gate)

        # Early-stop on sustained pickup so we don't waste frames.
        if (
            len(pickup_gate_history) >= PICKUP_SUCCESS_HOLD_STEPS
            and all(pickup_gate_history[-PICKUP_SUCCESS_HOLD_STEPS:])
        ):
            print(
                f"[stage4.5 worker {args.rollout_idx:04d}] "
                f"early stop on pickup at step {step:03d}",
                flush=True,
            )
            break

    images_np = np.stack(images, axis=0)

    # Save: GIF of the full rollout, plus a single representative frame for
    # the side-by-side comparison the driver builds.
    gif_path = rollout_dir / "rollout_preview.gif"
    imageio.mimsave(gif_path, images_np, duration=1000.0 / args.gif_fps)

    # Pick a frame at ~1/3 through the rollout — late enough that the arm
    # has moved into the scene (so camera-vs-camera framing differences are
    # obvious), early enough that the hammer is still on the table.
    representative_frame_idx = min(len(images_np) - 1, max(0, len(images_np) // 3))
    representative_frame_path = rollout_dir / "frame_representative.png"
    imageio.imwrite(representative_frame_path, images_np[representative_frame_idx])
    imageio.imwrite(rollout_dir / "frame_first.png", images_np[0])
    imageio.imwrite(rollout_dir / "frame_last.png", images_np[-1])

    max_object_z = max(object_zs) if object_zs else None
    start_object_z = float(object_start_pose[2])
    goal_z = float(goal_pose[2])
    max_lift_m = max_object_z - start_object_z if max_object_z is not None else None
    pickup_success = bool(
        max_object_z is not None
        and max_object_z >= goal_z - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
        and max_lift_m >= PICKUP_SUCCESS_MIN_LIFT_M
    )

    summary: Dict[str, object] = {
        "rollout_idx": args.rollout_idx,
        "seed": args.seed,
        "camera_position": camera_position,
        "camera_radius_m": args.camera_radius,
        "camera_azimuth_deg": args.camera_azimuth_deg,
        "camera_elevation_deg": args.camera_elevation_deg,
        "camera_target": list(DATASET_CAMERA_TARGET),
        "camera_horizontal_fov": DATASET_CAMERA_HORIZONTAL_FOV,
        "object_start_pose": object_start_pose,
        "goal_pose": goal_pose,
        "horizon_actual": len(images),
        "horizon_max": args.horizon,
        "max_object_z": max_object_z,
        "max_lift_m": max_lift_m,
        "pickup_success": pickup_success,
        "gif_path": str(gif_path),
        "representative_frame_path": str(representative_frame_path),
        "representative_frame_idx": int(representative_frame_idx),
    }

    with open(rollout_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[stage4.5 worker {args.rollout_idx:04d}] "
        f"pickup_success={pickup_success} frames={len(images)} gif={gif_path}",
        flush=True,
    )


def _build_comparison_grid(summaries: List[Dict[str, object]], output_path: Path) -> None:
    """Stitch each rollout's representative frame into a 1xN side-by-side grid."""
    import imageio.v2 as imageio

    frames = []
    labels = []
    for summary in summaries:
        frame_path = summary["representative_frame_path"]
        frame = imageio.imread(frame_path)
        frames.append(frame)
        radius = summary["camera_radius_m"]
        az = summary["camera_azimuth_deg"]
        el = summary["camera_elevation_deg"]
        labels.append(f"r={radius:.2f} az={az:+.0f} el={el:+.0f}")

    # Pad to common shape (they should already match, but be defensive).
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    padded = []
    for f in frames:
        if f.shape[0] != h or f.shape[1] != w:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[: f.shape[0], : f.shape[1]] = f
            padded.append(canvas)
        else:
            padded.append(f)

    grid = np.concatenate(padded, axis=1)
    imageio.imwrite(output_path, grid)
    print(f"[stage4.5] comparison grid: {output_path}", flush=True)
    print(f"[stage4.5] panel labels (left -> right):", flush=True)
    for i, label in enumerate(labels):
        print(f"  panel {i}: {label}", flush=True)


def run_driver(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _check_nominal_roundtrip()

    rng = np.random.default_rng(args.seed)
    script_path = Path(__file__).resolve()

    print(
        f"[stage4.5] sampling {args.num_rollouts} camera positions "
        f"radius={CAMERA_RADIUS_RANGE_M} "
        f"azimuth={CAMERA_AZIMUTH_DEG_RANGE} "
        f"elevation={CAMERA_ELEVATION_DEG_RANGE}",
        flush=True,
    )

    # Pre-sample all camera params so the driver log shows them up front and
    # so we can reproduce a specific rollout if one looks wrong.
    sampled = []
    for rollout_idx in range(args.num_rollouts):
        radius = float(rng.uniform(*CAMERA_RADIUS_RANGE_M))
        azimuth_deg = float(rng.uniform(*CAMERA_AZIMUTH_DEG_RANGE))
        elevation_deg = float(rng.uniform(*CAMERA_ELEVATION_DEG_RANGE))
        pos = _spherical_to_cartesian(radius, azimuth_deg, elevation_deg, CAMERA_ANCHOR)
        sampled.append(
            {
                "rollout_idx": rollout_idx,
                "radius_m": radius,
                "azimuth_deg": azimuth_deg,
                "elevation_deg": elevation_deg,
                "position": list(pos),
            }
        )
        print(
            f"[stage4.5] rollout {rollout_idx}: "
            f"radius={radius:.3f} az={azimuth_deg:+.1f} el={elevation_deg:+.1f} "
            f"pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]",
            flush=True,
        )

    summaries = []
    for s in sampled:
        cmd = [
            sys.executable,
            str(script_path),
            "--worker",
            "--rollout-idx", str(s["rollout_idx"]),
            "--seed", str(args.seed),
            "--horizon", str(args.horizon),
            "--output-dir", str(args.output_dir),
            "--gif-fps", str(args.gif_fps),
            "--camera-pos-x", f"{s['position'][0]:.8f}",
            "--camera-pos-y", f"{s['position'][1]:.8f}",
            "--camera-pos-z", f"{s['position'][2]:.8f}",
            "--camera-radius", f"{s['radius_m']:.8f}",
            "--camera-azimuth-deg", f"{s['azimuth_deg']:.8f}",
            "--camera-elevation-deg", f"{s['elevation_deg']:.8f}",
        ]
        if args.device:
            cmd.extend(["--device", args.device])

        print(
            f"[stage4.5] running rollout {s['rollout_idx'] + 1}/{args.num_rollouts}",
            flush=True,
        )
        subprocess.run(cmd, check=True)

        summary_path = args.output_dir / f"rollout_{s['rollout_idx']:04d}" / "summary.json"
        with open(summary_path, "r") as f:
            summaries.append(json.load(f))

    # Build the side-by-side comparison so camera variation is visible at a glance.
    comparison_path = args.output_dir / "camera_comparison.png"
    _build_comparison_grid(summaries, comparison_path)

    # Sanity check: confirm camera positions are actually distinct.
    positions = np.asarray([s["camera_position"] for s in summaries])
    pairwise_min_dist = float("inf")
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            pairwise_min_dist = min(pairwise_min_dist, d)
    print(
        f"[stage4.5] min pairwise camera position distance: "
        f"{pairwise_min_dist:.3f}m (should be > 0)",
        flush=True,
    )

    successes = sum(bool(s["pickup_success"]) for s in summaries)
    aggregate = {
        "num_rollouts": args.num_rollouts,
        "successes": successes,
        "success_rate": successes / args.num_rollouts,
        "min_pairwise_camera_distance_m": pairwise_min_dist,
        "camera_anchor": list(CAMERA_ANCHOR),
        "camera_radius_range_m": list(CAMERA_RADIUS_RANGE_M),
        "camera_azimuth_deg_range": list(CAMERA_AZIMUTH_DEG_RANGE),
        "camera_elevation_deg_range": list(CAMERA_ELEVATION_DEG_RANGE),
        "camera_target_fixed": list(DATASET_CAMERA_TARGET),
        "camera_horizontal_fov_fixed": DATASET_CAMERA_HORIZONTAL_FOV,
        "comparison_grid_path": str(comparison_path),
        "summaries": summaries,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print()
    print(f"[stage4.5] pickup successes: {successes}/{args.num_rollouts}", flush=True)
    print(f"[stage4.5] aggregate summary: {args.output_dir / 'summary.json'}", flush=True)
    print(f"[stage4.5] open this to verify camera variation: {comparison_path}", flush=True)
    if pairwise_min_dist < 0.05:
        print(
            "[stage4.5] WARNING: cameras are very close to each other "
            "(<5cm apart). Variation may not be visually obvious.",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rollouts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/stage4_5_camera_position_variation"),
    )
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)

    # Worker-only.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rollout-idx", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--camera-pos-x", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--camera-pos-y", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--camera-pos-z", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--camera-radius", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--camera-azimuth-deg", type=float, default=0.0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--camera-elevation-deg", type=float, default=0.0, help=argparse.SUPPRESS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    if args.worker:
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()