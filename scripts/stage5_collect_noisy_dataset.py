#!/usr/bin/env python3
"""Stage 5/6: collect PDP-style noisy DexToolBench pickup rollouts into Diffusion Policy Zarr.

This script is intentionally conservative:
- Uses the same core env/policy/camera/pickup-success logic as Stage 4.
- Runs num_envs in parallel.
- Samples ONE randomized hammer start per env recreation batch.
- Executes OU-correlated noisy actions in the environment.
- Supports noisy_clean: save noisy rollout states/images with clean expert action labels.
- Supports noisy_noisy: save noisy rollout states/images with noisy executed action labels.
- Filters: only successful pickup episodes are appended to dataset.zarr.
- Stores full 140-dim obs as state, 29-dim action, uint8 RGB images.

Recommended first smoke test:
    python stage5_collect_noisy_dataset.py \
      --num-envs 4 \
      --target-transitions 1000 \
      --max-batches 5 \
      --output-zarr data/stage5_claw_hammer_v1_smoke.zarr

Full-ish run:
    python stage5_collect_noisy_dataset.py \
      --num-envs 16 \
      --target-transitions 50000 \
      --output-zarr data/stage5_claw_hammer_v1.zarr
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

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

OBJECT_CATEGORY = "hammer"
OBJECT_NAME = "claw_hammer"
TASK_NAME = "swing_down"

CONFIG_PATH = Path("pretrained_policy/config.yaml")
CHECKPOINT_PATH = Path("pretrained_policy/model.pth")

TABLE_Z = 0.38
TABLE_NAIL_URDF = "urdf/table_narrow_nail.urdf"
START_Z_OFFSET = 0.03
LIFT_HEIGHT_M = 0.20

# Same pickup gate as Stage 4.
PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M = 0.02
PICKUP_SUCCESS_MIN_LIFT_M = 0.12
PICKUP_SUCCESS_HOLD_STEPS = 5

# Same wide dataset camera as Stage 4.
DATASET_CAMERA_POSITION = [0.55, -1.35, 1.10]
DATASET_CAMERA_TARGET = [-0.1, 0.35, 0.60]
DATASET_CAMERA_HORIZONTAL_FOV = 30.0
DATASET_CAMERA_WIDTH = 512
DATASET_CAMERA_HEIGHT = 360

# Optional per-action-index noise overrides.
# Fill these with DexToolBench action indices if you want joint-group-specific noise.
# If left empty, the script applies uniform OU noise to all 29 action dimensions.
HIP_IDXS: List[int] = []
KNEE_IDXS: List[int] = []
ANKLE_IDXS: List[int] = []



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
    goal_pose: List[float],
    object_start_pose: List[float],
    horizon: int,
    headless: bool,
    device: str,
):
    from deployment.isaac.isaac_env import create_env

    overrides = {
        "task.env.numEnvs": num_envs,
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

    data = root["data"]
    meta = root["meta"]

    old_n = data["img"].shape[0]
    new_n = old_n + img.shape[0]

    data["img"].append(img, axis=0)
    data["state"].append(state, axis=0)
    data["action"].append(action, axis=0)
    meta["episode_ends"].append(np.asarray([new_n], dtype=np.int64), axis=0)


def _current_counts(root):
    n_transitions = int(root["data"]["img"].shape[0])
    n_episodes = int(root["meta"]["episode_ends"].shape[0])
    return n_transitions, n_episodes


def _episode_pickup_success(
    object_zs: List[float],
    pickup_gate_history: List[bool],
    object_start_z: float,
    goal_z: float,
    hold_steps: int,
) -> bool:
    if not object_zs:
        return False
    max_object_z = max(object_zs)
    max_lift_m = max_object_z - object_start_z

    z_gate = (
        max_object_z >= goal_z - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
        and max_lift_m >= PICKUP_SUCCESS_MIN_LIFT_M
    )

    # max_consecutive = 0
    # cur = 0
    # for v in pickup_gate_history:
    #     cur = cur + 1 if v else 0
    #     max_consecutive = max(max_consecutive, cur)

    return bool(z_gate)


def _destroy_env(env):
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


def collect(args: argparse.Namespace) -> None:
    from deployment.rl_player import RlPlayer

    assert CONFIG_PATH.exists(), f"Missing policy config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing policy checkpoint: {CHECKPOINT_PATH}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    root = _open_or_create_zarr(
        args.output_zarr,
        img_h=DATASET_CAMERA_HEIGHT,
        img_w=DATASET_CAMERA_WIDTH,
        resume=args.resume,
    )
    n0, e0 = _current_counts(root)
    print(f"[stage5-noisy] output_zarr={args.output_zarr}")
    print(f"[stage5-noisy] variant={args.variant} noise_level={args.noise_level} theta={args.ou_theta} dt={args.ou_dt}")
    print(f"[stage5-noisy] starting counts: transitions={n0}, episodes={e0}")
    print(f"[stage5-noisy] num_envs={args.num_envs}, target_transitions={args.target_transitions}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    attempted_episodes = 0
    written_episodes = 0
    batch_idx = 0
    t_start = time.time()

    while True:
        n_transitions, n_episodes = _current_counts(root)
        if n_transitions >= args.target_transitions:
            break
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        if args.max_attempted_episodes is not None and attempted_episodes >= args.max_attempted_episodes:
            break

        nominal = _load_nominal_start_pose()
        dx, dy = rng.uniform(-args.xy_range, args.xy_range, size=2)
        object_start_pose = list(nominal)
        object_start_pose[0] += float(dx)
        object_start_pose[1] += float(dy)

        goal_pose = list(object_start_pose)
        goal_pose[2] += LIFT_HEIGHT_M

        print(
            f"\n[stage5-noisy] batch={batch_idx:04d} "
            f"dx={dx:+.3f}, dy={dy:+.3f}, "
            f"current={n_transitions}/{args.target_transitions}",
            flush=True,
        )

        env = _make_env(
            num_envs=args.num_envs,
            goal_pose=goal_pose,
            object_start_pose=object_start_pose,
            horizon=args.horizon,
            headless=not args.viewer,
            device=device,
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

        # OU noise state persists across timesteps within this rollout batch.
        noise_state = torch.zeros((env.num_envs, N_ACT), device=device)
        sqrt_dt = math.sqrt(args.ou_dt)

        per_env_imgs = [[] for _ in range(env.num_envs)]
        per_env_states = [[] for _ in range(env.num_envs)]
        per_env_actions = [[] for _ in range(env.num_envs)]
        per_env_object_zs = [[] for _ in range(env.num_envs)]
        per_env_pickup_gate = [[] for _ in range(env.num_envs)]

        active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)

        for step in range(args.horizon):
            if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
                print("[stage5-noisy] viewer closed; ending collection loop", flush=True)
                break

            # Required order from Stage 3: render image -> query policy -> save tuple -> step env.
            # For PDP-style variants, the rollout state/image is produced by stepping with noisy actions.
            #   noisy_clean: save clean expert action as label.
            #   noisy_noisy: save noisy executed action as label.
            image_t = env.render_dataset_camera_rgb(active_envs)
            clean_action_t = policy.get_normalized_action(obs, deterministic_actions=True)

            if args.variant in ("noisy_clean", "noisy_noisy"):
                sigma_noise = torch.normal(
                    mean=0.0,
                    std=args.noise_level,
                    size=clean_action_t.shape,
                    device=device,
                )
                if HIP_IDXS:
                    sigma_noise[:, HIP_IDXS] = torch.normal(
                        mean=0.0,
                        std=args.hip_noise,
                        size=clean_action_t[:, HIP_IDXS].shape,
                        device=device,
                    )
                if KNEE_IDXS:
                    sigma_noise[:, KNEE_IDXS] = torch.normal(
                        mean=0.0,
                        std=args.knee_noise,
                        size=clean_action_t[:, KNEE_IDXS].shape,
                        device=device,
                    )
                if ANKLE_IDXS:
                    sigma_noise[:, ANKLE_IDXS] = torch.normal(
                        mean=0.0,
                        std=args.ankle_noise,
                        size=clean_action_t[:, ANKLE_IDXS].shape,
                        device=device,
                    )

                noise_state = (
                    noise_state
                    + args.ou_theta * (args.ou_mu - noise_state) * args.ou_dt
                    + sigma_noise * sqrt_dt
                )
                executed_action_t = clean_action_t + noise_state
            else:
                executed_action_t = clean_action_t

            if args.variant == "noisy_noisy":
                action_to_save_t = executed_action_t
            else:
                action_to_save_t = clean_action_t

            image_np = image_t.detach().cpu().numpy().astype(np.uint8)
            obs_np = obs.detach().cpu().numpy().astype(np.float32)
            action_np = action_to_save_t.detach().cpu().numpy().astype(np.float32)

            for env_i in range(env.num_envs):
                per_env_imgs[env_i].append(image_np[env_i])
                per_env_states[env_i].append(obs_np[env_i])
                per_env_actions[env_i].append(action_np[env_i])

            obs_dict, _, done, _ = env.step(executed_action_t)
            obs = obs_dict["obs"]

            object_pose_np = env.object_pose[:, 0:7].detach().cpu().numpy()
            goal_pose_np = env.goal_pose[:, 0:7].detach().cpu().numpy()
            for env_i in range(env.num_envs):
                obj_z = float(object_pose_np[env_i, 2])
                start_z = float(object_start_pose[2])
                goal_z = float(goal_pose[2])
                current_lift_m = obj_z - start_z
                gate = bool(
                    obj_z >= goal_z - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
                    and current_lift_m >= PICKUP_SUCCESS_MIN_LIFT_M
                )
                per_env_object_zs[env_i].append(obj_z)
                per_env_pickup_gate[env_i].append(gate)

            if step % 30 == 0:
                successes_now = int((env.successes >= env.max_consecutive_successes).sum().item())
                print(
                    f"[stage5-noisy] batch={batch_idx:04d} step={step:03d} "
                    f"strict_env_pose_successes_now={successes_now}/{env.num_envs}",
                    flush=True,
                )

        batch_successes = 0
        for env_i in range(env.num_envs):
            attempted_episodes += 1
            success = _episode_pickup_success(
                object_zs=per_env_object_zs[env_i],
                pickup_gate_history=per_env_pickup_gate[env_i],
                object_start_z=float(object_start_pose[2]),
                goal_z=float(goal_pose[2]),
                hold_steps=args.pickup_success_hold_steps,
            )

            if not success:
                continue

            img_ep = np.stack(per_env_imgs[env_i], axis=0).astype(np.uint8)
            state_ep = np.stack(per_env_states[env_i], axis=0).astype(np.float32)
            action_ep = np.stack(per_env_actions[env_i], axis=0).astype(np.float32)

            _append_episode(root, img_ep, state_ep, action_ep)
            written_episodes += 1
            batch_successes += 1

            if args.save_preview_every and written_episodes % args.save_preview_every == 0:
                import imageio.v2 as imageio
                preview_dir = args.output_zarr.parent / f"{args.output_zarr.stem}_previews"
                preview_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(
                    preview_dir / f"episode_{written_episodes:05d}.gif",
                    img_ep,
                    duration=1000.0 / args.gif_fps,
                )

        n_transitions, n_episodes = _current_counts(root)
        elapsed = max(time.time() - t_start, 1e-6)
        rate = (n_transitions - n0) / elapsed
        remaining = max(args.target_transitions - n_transitions, 0)
        eta_min = remaining / max(rate, 1e-6) / 60.0

        print(
            f"[stage5-noisy] batch={batch_idx:04d} successes_written={batch_successes}/{env.num_envs} "
            f"total_transitions={n_transitions}/{args.target_transitions} "
            f"total_episodes={n_episodes} attempted={attempted_episodes} "
            f"write_success_rate={written_episodes / max(attempted_episodes, 1):.1%} "
            f"rate={rate:.1f} transitions/sec eta={eta_min:.1f} min",
            flush=True,
        )

        root.attrs["last_batch_idx"] = batch_idx
        root.attrs["attempted_episodes"] = attempted_episodes
        root.attrs["written_episodes"] = written_episodes
        root.attrs["xy_range"] = args.xy_range
        root.attrs["horizon"] = args.horizon
        root.attrs["lift_height_m"] = LIFT_HEIGHT_M
        root.attrs["pickup_success_hold_steps"] = args.pickup_success_hold_steps
        root.attrs["variant"] = args.variant
        root.attrs["noise_level"] = args.noise_level
        root.attrs["hip_noise"] = args.hip_noise
        root.attrs["knee_noise"] = args.knee_noise
        root.attrs["ankle_noise"] = args.ankle_noise
        root.attrs["ou_theta"] = args.ou_theta
        root.attrs["ou_mu"] = args.ou_mu
        root.attrs["ou_dt"] = args.ou_dt

        _destroy_env(env)
        batch_idx += 1

    n_transitions, n_episodes = _current_counts(root)
    print("\n[stage5-noisy] DONE")
    print(f"[stage5-noisy] transitions={n_transitions}")
    print(f"[stage5-noisy] episodes={n_episodes}")
    print(f"[stage5-noisy] zarr={args.output_zarr}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--target-transitions", type=int, default=50000)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-attempted-episodes", type=int, default=None)
    parser.add_argument("--xy-range", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--output-zarr",
        type=Path,
        default=Path("data/stage5_claw_hammer_noisy_clean.zarr"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument(
        "--save-preview-every",
        type=int,
        default=20,
        help="Save a GIF every N written successful episodes. Use 0 to disable.",
    )
    parser.add_argument(
        "--pickup-success-hold-steps",
        type=int,
        default=PICKUP_SUCCESS_HOLD_STEPS,
    )
    parser.add_argument(
        "--variant",
        choices=["clean", "noisy_clean", "noisy_noisy"],
        default="noisy_clean",
        help="clean: execute/save clean actions; noisy_clean: execute noisy but save clean labels; noisy_noisy: execute/save noisy labels.",
    )
    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--hip-noise", type=float, default=0.10)
    parser.add_argument("--knee-noise", type=float, default=0.10)
    parser.add_argument("--ankle-noise", type=float, default=0.10)
    parser.add_argument("--ou-theta", type=float, default=0.15)
    parser.add_argument("--ou-mu", type=float, default=0.0)
    parser.add_argument("--ou-dt", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
