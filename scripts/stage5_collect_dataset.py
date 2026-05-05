#!/usr/bin/env python3
"""Stage 5: collect clean successful DexToolBench pickup rollouts into Diffusion Policy Zarr.

Architecture:
- ONE persistent env, created once. No teardown/recreation loop.
- Per-env start-pose randomization via task.env.resetPositionNoiseX/Y, so each
  auto-reset draws a fresh hammer XY start within +/- xy_range meters.
- Per-env rolling buffers. When an env finishes an episode (done == True),
  we gate on pickup success and either append to Zarr or discard.
- Strict ordering preserved from Stage 4: render image -> query policy ->
  save (img, obs, action) -> step env.

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
import time
from pathlib import Path
from typing import List

import numpy as np
from omegaconf import OmegaConf

# Isaac Gym import order matters: import isaacgym before torch.
from isaacgym import gymapi  # noqa: F401
import torch

import zarr
from numcodecs import Blosc


N_OBS = 140
N_ACT = 29
DEFAULT_HORIZON = 150
MIN_EPISODE_LEN = 30  # discard implausibly short episodes

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

DATASET_CAMERA_POSITION = [0.55, -1.35, 1.10]
DATASET_CAMERA_TARGET = [-0.1, 0.35, 0.60]
DATASET_CAMERA_HORIZONTAL_FOV = 30.0
DATASET_CAMERA_WIDTH = 512
DATASET_CAMERA_HEIGHT = 360


def _jsonable(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


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
    *,
    num_envs: int,
    nominal_start_pose: List[float],
    nominal_goal_pose: List[float],
    xy_range: float,
    horizon: int,
    headless: bool,
    device: str,
):
    """Create env with in-env XY start randomization.

    Note: useFixedInitObjectPose=True still applies (so the nominal pose anchors
    the reset distribution), and resetPositionNoiseX/Y add per-env uniform noise
    on each reset.
    """
    from deployment.isaac.isaac_env import create_env

    overrides = {
        "task.env.numEnvs": num_envs,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        # Cameras MUST be on; we read RGB from the dataset camera every step.
        "task.env.enableCameraSensors": True,
        "task.env.enableDatasetCameras": True,
        "task.env.objectName": OBJECT_NAME,
        "task.env.useFixedGoalStates": True,
        "task.env.fixedGoalStates": [nominal_goal_pose],
        "task.env.showGoalObjectVisual": False,
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": nominal_start_pose,
        "task.env.startArmHigher": True,
        "task.env.asset.table": TABLE_NAIL_URDF,
        "task.env.tableResetZ": TABLE_Z,
        # >>> Per-reset XY randomization. This is the change vs. previous Stage 5.
        "task.env.resetPositionNoiseX": float(xy_range),
        "task.env.resetPositionNoiseY": float(xy_range),
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

    root.attrs["schema"] = "diffusion_policy_image_dataset_v1"
    root.attrs["object_name"] = OBJECT_NAME
    root.attrs["task_name"] = TASK_NAME
    root.attrs["state_dim"] = N_OBS
    root.attrs["action_dim"] = N_ACT
    root.attrs["img_height"] = img_h
    root.attrs["img_width"] = img_w
    return root


def _append_episode(root, img: np.ndarray, state: np.ndarray, action: np.ndarray):
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

    new_n = int(data["img"].shape[0]) + img.shape[0]
    data["img"].append(img, axis=0)
    data["state"].append(state, axis=0)
    data["action"].append(action, axis=0)
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
) -> bool:
    if not object_zs:
        return False
    max_object_z = max(object_zs)
    max_lift_m = max_object_z - object_start_z
    return bool(
        max_object_z >= goal_z - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
        and max_lift_m >= PICKUP_SUCCESS_MIN_LIFT_M
    )


def collect(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    root = _open_or_create_zarr(
        args.output_zarr,
        img_h=DATASET_CAMERA_HEIGHT,
        img_w=DATASET_CAMERA_WIDTH,
        resume=args.resume,
    )
    n0, e0 = _current_counts(root)
    print(f"[stage5] output_zarr={args.output_zarr}", flush=True)
    print(f"[stage5] starting counts: transitions={n0}, episodes={e0}", flush=True)
    print(
        f"[stage5] num_envs={args.num_envs}, target_transitions={args.target_transitions}, "
        f"xy_range=+/-{args.xy_range:.3f}m",
        flush=True,
    )

    nominal_start_pose = _load_nominal_start_pose()
    nominal_goal_pose = list(nominal_start_pose)
    nominal_goal_pose[2] += LIFT_HEIGHT_M
    nominal_start_z = float(nominal_start_pose[2])
    goal_z = float(nominal_goal_pose[2])

    print("[debug] creating env (one-time)...", flush=True)
    env = _make_env(
        num_envs=args.num_envs,
        nominal_start_pose=nominal_start_pose,
        nominal_goal_pose=nominal_goal_pose,
        xy_range=args.xy_range,
        horizon=args.horizon,
        headless=not args.viewer,
        device=device,
    )
    print("[debug] env created", flush=True)

    env.gym.refresh_actor_root_state_tensor(env.sim)
    print(
        f"[stage5] camera pos={_jsonable(env.cfg['env']['datasetCameraPosition'])} "
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

    # Prime obs with a single zero-action step (matches Stage 4).
    zero_action = torch.zeros((env.num_envs, N_ACT), device=device)
    print("[debug] priming with zero action...", flush=True)
    obs_dict, _, _, _ = env.step(zero_action)
    obs = obs_dict["obs"]
    print("[debug] primed", flush=True)

    active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)

    # Per-env rolling buffers. We reset an env's buffer when that env finishes.
    per_env_imgs:    List[List[np.ndarray]] = [[] for _ in range(env.num_envs)]
    per_env_states:  List[List[np.ndarray]] = [[] for _ in range(env.num_envs)]
    per_env_actions: List[List[np.ndarray]] = [[] for _ in range(env.num_envs)]
    per_env_object_zs: List[List[float]] = [[] for _ in range(env.num_envs)]
    # Track the actual start z each env resets to. Currently constant since
    # resetPositionNoiseZ=0, but cheap to track and future-proof.
    per_env_start_z: List[float] = [nominal_start_z] * env.num_envs

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
            print(f"[stage5] hit --max-steps={args.max_steps}, stopping", flush=True)
            break
        if (
            args.max_attempted_episodes is not None
            and attempted_episodes >= args.max_attempted_episodes
        ):
            print(
                f"[stage5] hit --max-attempted-episodes={args.max_attempted_episodes}, stopping",
                flush=True,
            )
            break
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            print("[stage5] viewer closed; stopping", flush=True)
            break

        # render -> query policy -> save tuple -> step env
        image_t = env.render_dataset_camera_rgb(active_envs)
        action_t = policy.get_normalized_action(obs, deterministic_actions=True)

        image_np = image_t.detach().cpu().numpy().astype(np.uint8)
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        action_np = action_t.detach().cpu().numpy().astype(np.float32)

        for env_i in range(env.num_envs):
            per_env_imgs[env_i].append(image_np[env_i])
            per_env_states[env_i].append(obs_np[env_i])
            per_env_actions[env_i].append(action_np[env_i])

        obs_dict, _, done, _ = env.step(action_t)
        obs = obs_dict["obs"]

        object_pose_np = env.object_pose[:, 0:7].detach().cpu().numpy()
        for env_i in range(env.num_envs):
            per_env_object_zs[env_i].append(float(object_pose_np[env_i, 2]))

        # Harvest any envs that finished this step.
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

                    appended = _append_episode(root, img_ep, state_ep, action_ep)
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

            # Clear this env's buffer regardless of outcome.
            per_env_imgs[env_i].clear()
            per_env_states[env_i].clear()
            per_env_actions[env_i].clear()
            per_env_object_zs[env_i].clear()
            # Update start_z from the post-reset object pose. After env.step()
            # with done[i]=True, the env auto-resets and object_pose reflects
            # the new randomized start.
            per_env_start_z[env_i] = float(object_pose_np[env_i, 2])

        step_idx += 1

        if step_idx % 30 == 0:
            n_transitions, n_episodes = _current_counts(root)
            elapsed = max(time.time() - t_start, 1e-6)
            rate = (n_transitions - n0) / elapsed
            remaining = max(args.target_transitions - n_transitions, 0)
            eta_min = remaining / max(rate, 1e-6) / 60.0
            success_rate = written_episodes / max(attempted_episodes, 1)
            print(
                f"[stage5] step={step_idx} "
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

    n_transitions, n_episodes = _current_counts(root)
    elapsed = time.time() - t_start
    print("\n[stage5] DONE", flush=True)
    print(f"[stage5] transitions={n_transitions}", flush=True)
    print(f"[stage5] episodes={n_episodes}", flush=True)
    print(f"[stage5] attempted={attempted_episodes} written={written_episodes}", flush=True)
    print(
        f"[stage5] discarded short={discarded_short} "
        f"unsuccessful={discarded_unsuccessful} invalid={discarded_invalid}",
        flush=True,
    )
    print(f"[stage5] elapsed={elapsed/60:.1f}min", flush=True)
    print(f"[stage5] zarr={args.output_zarr}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--target-transitions", type=int, default=50000)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Hard cap on total env.step() iterations (safety bound).",
    )
    parser.add_argument("--max-attempted-episodes", type=int, default=None)
    parser.add_argument("--xy-range", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--output-zarr",
        type=Path,
        default=Path("data/stage5_claw_hammer_v1.zarr"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument(
        "--save-preview-every",
        type=int,
        default=20,
        help="Save a GIF every N written episodes. Use 0 to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())