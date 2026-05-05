#!/usr/bin/env python3
"""Stage 3: capture one camera-synchronized lift_0.20m expert rollout."""

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

# Isaac Gym must be imported before torch.
from isaacgym import gymapi  # noqa: F401
import torch

from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer
from isaacgymenvs.utils.utils import get_repo_root_dir


N_OBS = 140
N_ACT = 29
DEFAULT_HORIZON = 150
CONTROL_HZ = 60.0

OBJECT_CATEGORY = "hammer"
OBJECT_NAME = "claw_hammer"
TASK_NAME = "swing_down"

CONFIG_PATH = Path("pretrained_policy/config.yaml")
CHECKPOINT_PATH = Path("pretrained_policy/model.pth")

TABLE_Z = 0.38
TABLE_NAIL_URDF = "urdf/table_narrow_nail.urdf"
START_Z_OFFSET = 0.03
LIFT_HEIGHT_M = 0.20

# Match Stage 4: fixed wide camera extrinsics for dataset collection.
DATASET_CAMERA_POSITION = [0.55, -1.35, 1.10]
DATASET_CAMERA_TARGET = [0.05, 0.35, 0.60]
DATASET_CAMERA_HORIZONTAL_FOV = 85.0
DATASET_CAMERA_WIDTH = 384
DATASET_CAMERA_HEIGHT = 384


def _format_pose(values):
    return [round(float(x), 6) for x in values]


def _actor_root_pose(env, indices):
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


def make_env(
    goal_pose,
    object_start_pose,
    horizon: int,
    headless: bool,
    device: str,
    wide_dataset_camera: bool,
):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/stage3_camera_capture_lift_020"),
    )
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Also open the IsaacGym viewer while capturing.",
    )
    parser.add_argument(
        "--wide-dataset-camera",
        action="store_true",
        help="Use fixed wide dataset camera extrinsics that frame robot base and tool.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trajectory_path = (
        repo_root
        / "dextoolbench"
        / "trajectories"
        / OBJECT_CATEGORY
        / OBJECT_NAME
        / f"{TASK_NAME}.json"
    )

    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"
    assert trajectory_path.exists(), f"Missing trajectory: {trajectory_path}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tuple_path = args.output_dir / "rollout_tuples.npz"
    gif_path = args.output_dir / "rollout_preview.gif"

    with open(trajectory_path, "r") as f:
        traj_data = json.load(f)

    object_start_pose = list(traj_data["start_pose"])
    object_start_pose[2] += START_Z_OFFSET

    goal_pose = list(object_start_pose)
    goal_pose[2] += LIFT_HEIGHT_M

    print(f"[stage3] device={device}")
    print(f"[stage3] trajectory={trajectory_path}")
    print(f"[stage3] object_start_pose={object_start_pose}")
    print(f"[stage3] goal_pose={goal_pose}")
    print(f"[stage3] output_dir={args.output_dir}")
    print("[stage3] creating camera env...")

    env = make_env(
        goal_pose=goal_pose,
        object_start_pose=object_start_pose,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
        wide_dataset_camera=args.wide_dataset_camera,
    )
    _print_env_diagnostics(
        "[stage3] after_env_init",
        env,
        object_start_pose,
        goal_pose,
    )

    print("[stage3] loading checkpoint env_state and policy...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    env.set_env_state(checkpoint[0]["env_state"])
    _print_env_diagnostics(
        "[stage3] after_checkpoint_restore",
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
        "[stage3] after_zero_action_step",
        env,
        object_start_pose,
        goal_pose,
    )

    images = []
    states = []
    actions = []

    active_env = torch.tensor([0], device=device, dtype=torch.long)
    print(f"[stage3] capturing {args.horizon} synchronized tuples...")
    for step in range(args.horizon):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            print("[stage3] viewer closed; stopping early")
            break

        # Required tuple order: render image -> query policy -> save tuple -> step env.
        image_t = env.render_dataset_camera_rgb(active_env)[0]
        action_t = policy.get_normalized_action(obs, deterministic_actions=True)

        images.append(image_t.cpu().numpy().astype(np.uint8))
        states.append(obs[0].detach().cpu().numpy().astype(np.float32))
        actions.append(action_t[0].detach().cpu().numpy().astype(np.float32))

        obs_dict, _, done, _ = env.step(action_t)
        obs = obs_dict["obs"]

        if step % 30 == 0:
            successes = int(env.successes[0].item())
            print(
                f"[stage3] step={step:03d} "
                f"done={bool(done[0].item())} "
                f"successes={successes}/{env.max_consecutive_successes}"
            )

        if args.viewer:
            time.sleep(1.0 / CONTROL_HZ)

    images_np = np.stack(images, axis=0)
    states_np = np.stack(states, axis=0)
    actions_np = np.stack(actions, axis=0)

    np.savez_compressed(
        tuple_path,
        img=images_np,
        obs=states_np,
        action=actions_np,
        object_start_pose=np.asarray(object_start_pose, dtype=np.float32),
        goal_pose=np.asarray(goal_pose, dtype=np.float32),
    )
    imageio.mimsave(gif_path, images_np, fps=args.gif_fps)

    print(f"[stage3] saved tuples: {tuple_path}")
    print(
        f"[stage3] tuple shapes: img={images_np.shape}, "
        f"obs={states_np.shape}, action={actions_np.shape}"
    )
    print(f"[stage3] saved GIF: {gif_path}")
    print("[stage3] inspect the GIF before moving to Stage 4")


if __name__ == "__main__":
    main()
