#!/usr/bin/env python3
"""Stage 4: test lift_0.20m pickup from randomized hammer x/y starts.

The default driver launches one fresh Python subprocess per rollout. This avoids
IsaacGym viewer/sim lifecycle issues while still testing env recreation with a
new start pose and synchronized lifted goal for each rollout.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

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

# Fixed dataset-camera extrinsics for all rollouts. These are intentionally
# wider than the default task camera so the robot base at y ~= 0.8 and the
# randomized tool region around y ~= 0 both stay in frame.
DATASET_CAMERA_POSITION = [0.55, -1.35, 1.10]
DATASET_CAMERA_TARGET = [-0.1, 0.35, 0.60]
DATASET_CAMERA_HORIZONTAL_FOV = 30.0
DATASET_CAMERA_WIDTH = 512
DATASET_CAMERA_HEIGHT = 360


def _format_pose(values) -> List[float]:
    return [round(float(x), 6) for x in values]


def _jsonable(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _quat_angle_error_rad(q1, q2) -> float:
    q1_np = np.asarray(q1, dtype=np.float64)
    q2_np = np.asarray(q2, dtype=np.float64)
    q1_np = q1_np / np.linalg.norm(q1_np)
    q2_np = q2_np / np.linalg.norm(q2_np)
    dot = float(abs(np.dot(q1_np, q2_np)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


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
        f"{prefix} dataset_camera_attachment=fixed_world "
        f"(created with gym.set_camera_location, not attach_camera_to_body)",
        flush=True,
    )
    print(
        f"{prefix} requested_object_start_pose={_format_pose(object_start_pose)} "
        f"requested_goal_pose={_format_pose(goal_pose)}",
        flush=True,
    )
    print(
        f"{prefix} actual_robot_root_pose={_actor_root_pose(env, env.robot_indices)} "
        f"actual_object_root_pose={_actor_root_pose(env, env.object_indices)} "
        f"actual_table_root_pose={_actor_root_pose(env, env.table_indices)}",
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
    wide_dataset_camera: bool,
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
    }
    if wide_dataset_camera:
        overrides.update(
            {
                "task.env.datasetCameraPosition": DATASET_CAMERA_POSITION,
                "task.env.datasetCameraTarget": DATASET_CAMERA_TARGET,
                "task.env.datasetCameraHorizontalFov": DATASET_CAMERA_HORIZONTAL_FOV,
                "task.env.datasetCameraWidth": DATASET_CAMERA_WIDTH,
                "task.env.datasetCameraHeight": DATASET_CAMERA_HEIGHT,
            }
        )

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

    object_start_pose = _load_nominal_start_pose()
    object_start_pose[0] += args.dx
    object_start_pose[1] += args.dy

    goal_pose = list(object_start_pose)
    goal_pose[2] += LIFT_HEIGHT_M

    print(
        f"[stage4 worker {args.rollout_idx:04d}] "
        f"dx={args.dx:+.3f}, dy={args.dy:+.3f}, device={device}",
        flush=True,
    )

    env = _make_env(
        goal_pose=goal_pose,
        object_start_pose=object_start_pose,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
        wide_dataset_camera=args.wide_dataset_camera,
    )
    _print_env_diagnostics(
        f"[stage4 worker {args.rollout_idx:04d}] after_env_init",
        env,
        object_start_pose,
        goal_pose,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    env.set_env_state(checkpoint[0]["env_state"])
    _print_env_diagnostics(
        f"[stage4 worker {args.rollout_idx:04d}] after_checkpoint_restore",
        env,
        object_start_pose,
        goal_pose,
    )

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
    _print_env_diagnostics(
        f"[stage4 worker {args.rollout_idx:04d}] after_zero_action_step",
        env,
        object_start_pose,
        goal_pose,
    )

    images = []
    states = []
    actions = []
    object_zs = []
    pickup_gate_history = []
    pos_errors = []
    quat_errors_rad = []
    keypoint_errors = []
    keypoint_errors_fixed_size = []
    max_successes_seen = 0

    active_env = torch.tensor([0], device=device, dtype=torch.long)
    for step in range(args.horizon):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            print(f"[stage4 worker {args.rollout_idx:04d}] viewer closed", flush=True)
            break

        image_t = env.render_dataset_camera_rgb(active_env)[0]
        action_t = policy.get_normalized_action(obs, deterministic_actions=True)

        images.append(image_t.cpu().numpy().astype(np.uint8))
        if args.save_tuples:
            states.append(obs[0].detach().cpu().numpy().astype(np.float32))
            actions.append(action_t[0].detach().cpu().numpy().astype(np.float32))

        obs_dict, _, done, _ = env.step(action_t)
        obs = obs_dict["obs"]

        successes = int(env.successes[0].item())
        max_successes_seen = max(max_successes_seen, successes)
        object_pose_np = env.object_pose[0, 0:7].detach().cpu().numpy()
        goal_pose_np = env.goal_pose[0, 0:7].detach().cpu().numpy()
        object_zs.append(float(object_pose_np[2]))
        current_lift_m = float(object_pose_np[2]) - float(object_start_pose[2])
        current_pickup_gate = bool(
            object_pose_np[2] >= float(goal_pose[2]) - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
            and current_lift_m >= PICKUP_SUCCESS_MIN_LIFT_M
        )
        pickup_gate_history.append(current_pickup_gate)
        pos_errors.append(float(np.linalg.norm(object_pose_np[0:3] - goal_pose_np[0:3])))
        quat_errors_rad.append(
            _quat_angle_error_rad(object_pose_np[3:7], goal_pose_np[3:7])
        )
        keypoint_errors.append(float(env.keypoints_max_dist[0].item()))
        keypoint_errors_fixed_size.append(
            float(env.keypoints_max_dist_fixed_size[0].item())
        )

        if step % 30 == 0:
            print(
                f"[stage4 worker {args.rollout_idx:04d}] step={step:03d} "
                f"done={bool(done[0].item())} "
                f"successes={successes}/{env.max_consecutive_successes}",
                flush=True,
            )

        if (
            args.stop_on_pickup_success
            and len(pickup_gate_history) >= args.pickup_success_hold_steps
            and all(pickup_gate_history[-args.pickup_success_hold_steps :])
        ):
            print(
                f"[stage4 worker {args.rollout_idx:04d}] "
                f"early stop: pickup gate held for "
                f"{args.pickup_success_hold_steps} steps at step={step:03d}",
                flush=True,
            )
            break

    images_np = np.stack(images, axis=0)
    frame_absdiff = (
        np.abs(images_np[1:].astype(np.int16) - images_np[:-1].astype(np.int16))
        .sum(axis=(1, 2, 3))
        if len(images_np) > 1
        else np.asarray([], dtype=np.int64)
    )
    nonzero_frame_diffs = int(np.count_nonzero(frame_absdiff))
    max_frame_absdiff = int(frame_absdiff.max()) if frame_absdiff.size else 0
    print(
        f"[stage4 worker {args.rollout_idx:04d}] image_sequence "
        f"frames={len(images_np)} nonzero_frame_diffs={nonzero_frame_diffs} "
        f"max_frame_absdiff={max_frame_absdiff}",
        flush=True,
    )

    gif_path = rollout_dir / "rollout_preview.gif"
    imageio.mimsave(gif_path, images_np, duration=1000.0 / args.gif_fps)
    imageio.imwrite(rollout_dir / "frame_0000.png", images_np[0])
    imageio.imwrite(rollout_dir / "frame_mid.png", images_np[len(images_np) // 2])
    imageio.imwrite(rollout_dir / "frame_last.png", images_np[-1])

    tuple_path = None
    if args.save_tuples:
        tuple_path = rollout_dir / "rollout_tuples.npz"
        np.savez_compressed(
            tuple_path,
            img=images_np,
            obs=np.stack(states, axis=0),
            action=np.stack(actions, axis=0),
            object_start_pose=np.asarray(object_start_pose, dtype=np.float32),
            goal_pose=np.asarray(goal_pose, dtype=np.float32),
        )

    max_consecutive_successes = int(env.max_consecutive_successes)
    env_pose_success = max_successes_seen >= max_consecutive_successes
    final_object_pose = _format_pose(env.object_pose[0, 0:7].detach().cpu())
    final_goal_pose = _format_pose(env.goal_pose[0, 0:7].detach().cpu())
    max_object_z = max(object_zs) if object_zs else None
    start_object_z = float(object_start_pose[2])
    goal_z = float(goal_pose[2])
    max_lift_m = max_object_z - start_object_z if max_object_z is not None else None
    pickup_success = bool(
        max_object_z is not None
        and max_object_z >= goal_z - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
        and max_lift_m >= PICKUP_SUCCESS_MIN_LIFT_M
    )
    max_consecutive_pickup_gate_steps = 0
    current_consecutive_pickup_gate_steps = 0
    for gate_value in pickup_gate_history:
        current_consecutive_pickup_gate_steps = (
            current_consecutive_pickup_gate_steps + 1 if gate_value else 0
        )
        max_consecutive_pickup_gate_steps = max(
            max_consecutive_pickup_gate_steps,
            current_consecutive_pickup_gate_steps,
        )
    min_keypoint_error = min(keypoint_errors) if keypoint_errors else None
    min_keypoint_error_fixed_size = (
        min(keypoint_errors_fixed_size) if keypoint_errors_fixed_size else None
    )
    min_pos_error = min(pos_errors) if pos_errors else None
    min_quat_error_rad = min(quat_errors_rad) if quat_errors_rad else None
    success_tolerance = float(env.success_tolerance)
    keypoint_success_tolerance = success_tolerance * float(env.keypoint_scale)
    summary: Dict[str, object] = {
        "rollout_idx": args.rollout_idx,
        "seed": args.seed,
        "dx": args.dx,
        "dy": args.dy,
        "object_start_pose": object_start_pose,
        "goal_pose": goal_pose,
        "horizon": len(images),
        "max_successes_seen": max_successes_seen,
        "max_consecutive_successes": max_consecutive_successes,
        "env_pose_success": env_pose_success,
        "pickup_success": pickup_success,
        "success": pickup_success,
        "stopped_on_pickup_success": bool(
            args.stop_on_pickup_success
            and max_consecutive_pickup_gate_steps >= args.pickup_success_hold_steps
        ),
        "pickup_success_hold_steps": args.pickup_success_hold_steps,
        "max_consecutive_pickup_gate_steps": max_consecutive_pickup_gate_steps,
        "max_object_z": max_object_z,
        "start_object_z": start_object_z,
        "goal_z": goal_z,
        "max_lift_m": max_lift_m,
        "pickup_success_goal_z_tolerance_m": PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M,
        "pickup_success_min_lift_m": PICKUP_SUCCESS_MIN_LIFT_M,
        "final_object_pose": final_object_pose,
        "final_goal_pose": final_goal_pose,
        "final_pos_error": pos_errors[-1] if pos_errors else None,
        "min_pos_error": min_pos_error,
        "final_quat_error_rad": quat_errors_rad[-1] if quat_errors_rad else None,
        "final_quat_error_deg": (
            math.degrees(quat_errors_rad[-1]) if quat_errors_rad else None
        ),
        "min_quat_error_rad": min_quat_error_rad,
        "min_quat_error_deg": (
            math.degrees(min_quat_error_rad)
            if min_quat_error_rad is not None
            else None
        ),
        "final_keypoint_error": keypoint_errors[-1] if keypoint_errors else None,
        "min_keypoint_error": min_keypoint_error,
        "final_keypoint_error_fixed_size": (
            keypoint_errors_fixed_size[-1] if keypoint_errors_fixed_size else None
        ),
        "min_keypoint_error_fixed_size": min_keypoint_error_fixed_size,
        "success_tolerance": success_tolerance,
        "keypoint_success_tolerance": keypoint_success_tolerance,
        "fixed_size_keypoint_reward": bool(env.cfg["env"]["fixedSizeKeypointReward"]),
        "wide_dataset_camera": args.wide_dataset_camera,
        "dataset_camera_position": _jsonable(env.cfg["env"]["datasetCameraPosition"]),
        "dataset_camera_target": _jsonable(env.cfg["env"]["datasetCameraTarget"]),
        "dataset_camera_horizontal_fov": _jsonable(
            env.cfg["env"].get("datasetCameraHorizontalFov")
        ),
        "dataset_camera_width": _jsonable(env.cfg["env"].get("datasetCameraWidth")),
        "dataset_camera_height": _jsonable(env.cfg["env"].get("datasetCameraHeight")),
        "nonzero_frame_diffs": nonzero_frame_diffs,
        "max_frame_absdiff": max_frame_absdiff,
        "gif_path": str(gif_path),
        "tuple_path": str(tuple_path) if tuple_path else None,
    }

    summary_path = rollout_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[stage4 worker {args.rollout_idx:04d}] "
        f"pickup_success={pickup_success} "
        f"env_pose_success={env_pose_success} "
        f"max_successes={max_successes_seen}/{max_consecutive_successes} "
        f"gif={gif_path}",
        flush=True,
    )


def run_driver(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    script_path = Path(__file__).resolve()

    print(
        f"[stage4] running {args.num_rollouts} randomized rollouts "
        f"with xy_range=+/-{args.xy_range:.3f}m",
        flush=True,
    )

    summaries = []
    for rollout_idx in range(args.num_rollouts):
        dx, dy = rng.uniform(-args.xy_range, args.xy_range, size=2)
        cmd = [
            sys.executable,
            str(script_path),
            "--worker",
            "--rollout-idx",
            str(rollout_idx),
            "--seed",
            str(args.seed),
            "--dx",
            f"{dx:.8f}",
            "--dy",
            f"{dy:.8f}",
            "--horizon",
            str(args.horizon),
            "--output-dir",
            str(args.output_dir),
            "--gif-fps",
            str(args.gif_fps),
        ]
        if args.viewer:
            cmd.append("--viewer")
        if args.save_tuples:
            cmd.append("--save-tuples")
        if args.device:
            cmd.extend(["--device", args.device])
        if args.wide_dataset_camera:
            cmd.append("--wide-dataset-camera")
        if args.stop_on_pickup_success:
            cmd.append("--stop-on-pickup-success")
        cmd.extend(["--pickup-success-hold-steps", str(args.pickup_success_hold_steps)])

        print(
            f"[stage4] rollout {rollout_idx + 1}/{args.num_rollouts}: "
            f"dx={dx:+.3f}, dy={dy:+.3f}",
            flush=True,
        )
        subprocess.run(cmd, check=True)

        summary_path = args.output_dir / f"rollout_{rollout_idx:04d}" / "summary.json"
        with open(summary_path, "r") as f:
            summaries.append(json.load(f))

    successes = sum(bool(summary["pickup_success"]) for summary in summaries)
    env_pose_successes = sum(bool(summary["env_pose_success"]) for summary in summaries)
    success_rate = successes / args.num_rollouts
    aggregate = {
        "num_rollouts": args.num_rollouts,
        "successes": successes,
        "success_rate": success_rate,
        "env_pose_successes": env_pose_successes,
        "env_pose_success_rate": env_pose_successes / args.num_rollouts,
        "success_metric": "pickup_success",
        "xy_range": args.xy_range,
        "seed": args.seed,
        "pass_gate": successes >= args.min_successes,
        "min_successes": args.min_successes,
        "summaries": summaries,
    }
    aggregate_path = args.output_dir / "summary.json"
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print()
    print(
        f"[stage4] success rate: {successes}/{args.num_rollouts} "
        f"({success_rate:.0%})",
        flush=True,
    )
    print(
        f"[stage4] strict env pose success rate: "
        f"{env_pose_successes}/{args.num_rollouts} "
        f"({env_pose_successes / args.num_rollouts:.0%})",
        flush=True,
    )
    print(f"[stage4] aggregate summary: {aggregate_path}", flush=True)
    if aggregate["pass_gate"]:
        print("[stage4] PASS: success rate meets the >=7/10 gate", flush=True)
    else:
        print("[stage4] FAIL: debug before scaling to Stage 5", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rollouts", type=int, default=10)
    parser.add_argument("--min-successes", type=int, default=7)
    parser.add_argument("--xy-range", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/stage4_random_starts_lift_020"),
    )
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the IsaacGym viewer in each worker rollout.",
    )
    parser.add_argument(
        "--save-tuples",
        action="store_true",
        help="Also save full image/obs/action tuple .npz files per rollout.",
    )
    parser.add_argument(
        "--wide-dataset-camera",
        action="store_true",
        help="Use fixed wide dataset camera extrinsics that frame robot base and tool.",
    )
    parser.add_argument(
        "--stop-on-pickup-success",
        action="store_true",
        help="End each worker rollout once the pickup gate is held for several frames.",
    )
    parser.add_argument(
        "--pickup-success-hold-steps",
        type=int,
        default=PICKUP_SUCCESS_HOLD_STEPS,
        help="Number of consecutive frames required before early-stopping on pickup.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rollout-idx", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--dx", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--dy", type=float, default=0.0, help=argparse.SUPPRESS)
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
