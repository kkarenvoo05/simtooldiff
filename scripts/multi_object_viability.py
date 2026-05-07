#!/usr/bin/env python3
"""Per-object viability check: 1 lift-by-20cm rollout for each of the 12 DexToolBench objects.

Driver mode spawns one subprocess per object (mirrors stage 4's pattern to avoid
IsaacGym lifecycle issues across env tear-downs). Each worker loads the object's
trajectory json, runs a single rollout with horizon=250, captures a GIF, and
writes summary.json with pickup_success.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from omegaconf import OmegaConf


N_OBS = 140
N_ACT = 29
DEFAULT_HORIZON = 250
CONFIG_PATH = Path("pretrained_policy/config.yaml")
CHECKPOINT_PATH = Path("pretrained_policy/model.pth")

TABLE_Z = 0.38
TABLE_NAIL_URDF = "urdf/table_narrow_nail.urdf"
START_Z_OFFSET = 0.03
LIFT_HEIGHT_M = 0.20
PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M = 0.02
PICKUP_SUCCESS_MIN_LIFT_M = 0.12
PICKUP_SUCCESS_HOLD_STEPS = 5

DATASET_CAMERA_POSITION = [0.55, -1.35, 1.10]
DATASET_CAMERA_TARGET = [-0.1, 0.35, 0.60]
DATASET_CAMERA_HORIZONTAL_FOV = 30.0
DATASET_CAMERA_WIDTH = 512
DATASET_CAMERA_HEIGHT = 360


# (category, instance, task_name_for_start_pose) — task choice doesn't matter
# for our purposes, we only read start_pose; pick alphabetically first per object.
OBJECTS: List[Tuple[str, str, str]] = [
    ("brush",       "blue_brush",        "sweep_forward"),
    ("brush",       "red_brush",         "sweep_forward"),
    ("eraser",      "flat_eraser",       "wipe_c"),
    ("eraser",      "handle_eraser",     "wipe_c"),
    ("hammer",      "claw_hammer",       "swing_down"),
    ("hammer",      "mallet_hammer",     "swing_down"),
    ("marker",      "sharpie_marker",    "draw_smile"),
    ("marker",      "staples_marker",    "draw_smile"),
    ("screwdriver", "long_screwdriver",  "spin_horizontal"),
    ("screwdriver", "short_screwdriver", "spin_horizontal"),
    ("spatula",     "flat_spatula",      "flip_over"),
    ("spatula",     "spoon_spatula",     "flip_over"),
]


def _format_pose(values) -> List[float]:
    return [round(float(x), 6) for x in values]


def _load_start_pose(category: str, name: str, task: str) -> List[float]:
    from isaacgymenvs.utils.utils import get_repo_root_dir
    p = get_repo_root_dir() / "dextoolbench" / "trajectories" / category / name / f"{task}.json"
    assert p.exists(), p
    with open(p) as f:
        traj = json.load(f)
    pose = list(traj["start_pose"])
    pose[2] += START_Z_OFFSET
    return pose


def _make_env(*, object_name: str, goal_pose, object_start_pose, horizon, headless, device):
    from deployment.isaac.isaac_env import create_env
    overrides = {
        "task.env.numEnvs": 1,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        "task.env.enableCameraSensors": True,
        "task.env.enableDatasetCameras": True,
        "task.env.objectName": object_name,
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


def run_worker(args: argparse.Namespace) -> None:
    import imageio.v2 as imageio
    from isaacgym import gymapi  # noqa: F401
    import torch
    from deployment.rl_player import RlPlayer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rollout_dir = args.output_dir / f"{args.object_category}__{args.object_name}"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    object_start_pose = _load_start_pose(args.object_category, args.object_name, args.task_name)
    goal_pose = list(object_start_pose)
    goal_pose[2] += LIFT_HEIGHT_M

    print(
        f"[viability {args.object_category}/{args.object_name}] start_pose={_format_pose(object_start_pose)} "
        f"goal={_format_pose(goal_pose)} device={device}",
        flush=True,
    )

    env = _make_env(
        object_name=args.object_name,
        goal_pose=goal_pose,
        object_start_pose=object_start_pose,
        horizon=args.horizon,
        headless=True,
        device=device,
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
    pickup_gate_history: List[bool] = []
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
        pickup_gate = bool(
            object_pose_np[2] >= float(goal_pose[2]) - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
            and current_lift_m >= PICKUP_SUCCESS_MIN_LIFT_M
        )
        pickup_gate_history.append(pickup_gate)
        if (
            len(pickup_gate_history) >= PICKUP_SUCCESS_HOLD_STEPS
            and all(pickup_gate_history[-PICKUP_SUCCESS_HOLD_STEPS:])
        ):
            print(
                f"[viability {args.object_category}/{args.object_name}] "
                f"early stop: pickup gate held at step={step}",
                flush=True,
            )
            break

    images_np = np.stack(images, axis=0)
    gif_path = rollout_dir / "rollout.gif"
    imageio.mimsave(gif_path, images_np, duration=1000.0 / args.gif_fps)
    imageio.imwrite(rollout_dir / "frame_first.png", images_np[0])
    imageio.imwrite(rollout_dir / "frame_last.png", images_np[-1])

    max_z = max(object_zs) if object_zs else None
    max_lift = (max_z - float(object_start_pose[2])) if max_z is not None else None
    pickup_success = bool(
        max_z is not None
        and max_z >= float(goal_pose[2]) - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
        and max_lift >= PICKUP_SUCCESS_MIN_LIFT_M
    )
    summary = {
        "object_category": args.object_category,
        "object_name": args.object_name,
        "task_name": args.task_name,
        "object_start_pose": object_start_pose,
        "goal_pose": goal_pose,
        "horizon_used": len(images),
        "max_object_z": max_z,
        "max_lift_m": max_lift,
        "goal_z": float(goal_pose[2]),
        "pickup_success": pickup_success,
        "gif_path": str(gif_path),
    }
    with open(rollout_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[viability {args.object_category}/{args.object_name}] "
        f"pickup_success={pickup_success} max_lift={max_lift:.3f}m horizon_used={len(images)}",
        flush=True,
    )


def run_driver(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    summaries = []
    for category, name, task in OBJECTS:
        cmd = [
            sys.executable, str(script_path),
            "--worker",
            "--object-category", category,
            "--object-name", name,
            "--task-name", task,
            "--horizon", str(args.horizon),
            "--output-dir", str(args.output_dir),
            "--gif-fps", str(args.gif_fps),
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        print(f"\n[viability driver] >>> {category}/{name}", flush=True)
        rc = subprocess.run(cmd).returncode
        summary_path = args.output_dir / f"{category}__{name}" / "summary.json"
        if rc != 0 or not summary_path.exists():
            print(f"[viability driver] !!! {category}/{name} FAILED (rc={rc})", flush=True)
            summaries.append({
                "object_category": category, "object_name": name, "task_name": task,
                "pickup_success": False, "max_lift_m": float("nan"),
                "horizon_used": 0, "error": f"subprocess rc={rc}",
                "object_start_pose": None, "goal_pose": None,
                "max_object_z": None, "goal_z": None, "gif_path": None,
            })
            continue
        with open(summary_path) as f:
            summaries.append(json.load(f))

    with open(args.output_dir / "aggregate.json", "w") as f:
        json.dump({"objects": summaries}, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f"{'category':<12} {'instance':<22} {'success':<8} {'max_lift':<10} {'steps':<6}", flush=True)
    print("-" * 80, flush=True)
    n_pass = 0
    for s in summaries:
        ok = "PASS" if s["pickup_success"] else "FAIL"
        if s["pickup_success"]:
            n_pass += 1
        ml = s.get("max_lift_m")
        ml_str = f"{ml:>7.3f}m" if isinstance(ml, (int, float)) and ml == ml else "    n/a"
        print(
            f"{s['object_category']:<12} {s['object_name']:<22} {ok:<8} "
            f"{ml_str} {s['horizon_used']:>5d}",
            flush=True,
        )
    print("-" * 80, flush=True)
    print(f"{n_pass}/{len(summaries)} objects passed pickup-success gate", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--output-dir", type=Path, default=Path("data/multi_object_viability"))
    p.add_argument("--gif-fps", type=int, default=15)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--object-category", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--object-name", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--task-name", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


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
