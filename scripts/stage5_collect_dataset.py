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
# Horizon needs headroom over the policy's worst-case rollout length on
# randomized starts. At xy_range=0.10 we observed pickups completing in 124-167
# steps (stage 4 horizon=250 sweep); 150 truncates ~half of them.
DEFAULT_HORIZON = 250
MIN_EPISODE_LEN = 30  # discard implausibly short episodes

DEFAULT_OBJECT_CATEGORY = "hammer"
DEFAULT_OBJECT_NAME = "claw_hammer"
DEFAULT_TASK_NAME = "swing_down"

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
# Height divisible by 16 (and 32) so MP4 encoders don't pad and image encoders
# with strided convs don't hit weird off-by-one shapes.
DATASET_CAMERA_HEIGHT = 384


def _jsonable(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _load_nominal_start_pose(category: str, name: str, task: str) -> List[float]:
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
    seed: int,
    object_name: str,
):
    """Create env with the nominal start pose anchored.

    XY randomization is NOT done via task.env.resetPositionNoiseX/Y: the env
    short-circuits those when useFixedInitObjectPose=True (env.py:3491-3493
    sets the per-reset random offsets to zero). We instead mutate
    env.object_init_state[env_ids, 0:2] from the script before each reset; see
    randomize_object_init_xy() in collect().
    """
    from deployment.isaac.isaac_env import create_env

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
        "task.env.fixedGoalStates": [nominal_goal_pose],
        "task.env.showGoalObjectVisual": False,
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": nominal_start_pose,
        "task.env.startArmHigher": True,
        "task.env.asset.table": TABLE_NAIL_URDF,
        "task.env.tableResetZ": TABLE_Z,
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        f"[stage5] object={args.object_category}/{args.object_name} "
        f"task={args.task_name} object_id={args.object_id} category_id={args.category_id}",
        flush=True,
    )
    print(
        f"[stage5] num_envs={args.num_envs}, target_transitions={args.target_transitions}, "
        f"xy_range=+/-{args.xy_range:.3f}m",
        flush=True,
    )

    # Persist the id->name mappings as we encounter new objects/categories.
    object_registry = dict(root.attrs.get("object_id_to_name", {}))
    category_registry = dict(root.attrs.get("category_id_to_name", {}))
    object_registry[str(args.object_id)] = f"{args.object_category}/{args.object_name}"
    category_registry[str(args.category_id)] = args.object_category
    root.attrs["object_id_to_name"] = object_registry
    root.attrs["category_id_to_name"] = category_registry

    nominal_start_pose = _load_nominal_start_pose(
        args.object_category, args.object_name, args.task_name
    )
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
        seed=args.seed,
        object_name=args.object_name,
    )
    print("[debug] env created", flush=True)

    # Anchor the per-env nominal init XY so randomization is always relative to
    # the canonical start, not to the previous (already-randomized) reset.
    nominal_init_xy = env.object_init_state[:, 0:2].detach().clone()

    def randomize_object_init_xy(env_id_list: List[int]) -> None:
        """Mutate env.object_init_state[env_ids, 0:2] in place, in [-xy_range, +xy_range]
        relative to the per-env nominal anchor. Reads on next env reset."""
        if not env_id_list:
            return
        env_ids = torch.tensor(env_id_list, device=env.object_init_state.device, dtype=torch.long)
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

    # Randomize for the very first reset, which fires inside the priming step
    # below (vec_task initializes reset_buf to all-ones, so the first env.step
    # auto-resets every env).
    randomize_object_init_xy(list(range(env.num_envs)))

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
    # Start Z per env. The auto-reset for env_i happens inside the env.step()
    # AFTER done[i]=True is observed (in pre_physics_step), so we can only
    # capture the post-reset start Z one iteration later — see just_reset below.
    per_env_start_z: List[float] = [nominal_start_z] * env.num_envs
    # When True for env_i, the upcoming iteration's pre-step image/obs/action
    # still reflect the just-finished episode's terminal (lifted) state — the
    # reset hasn't physically happened yet. We skip those samples and instead
    # capture per_env_start_z from object_pose AFTER env.step() applies the reset.
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
            if just_reset[env_i]:
                # Pre-step samples this iteration belong to the prior (already
                # harvested) episode's terminal state, not the new episode.
                continue
            per_env_imgs[env_i].append(image_np[env_i])
            per_env_states[env_i].append(obs_np[env_i])
            per_env_actions[env_i].append(action_np[env_i])

        obs_dict, _, done, _ = env.step(action_t)
        obs = obs_dict["obs"]

        object_pose_np = env.object_pose[:, 0:7].detach().cpu().numpy()
        for env_i in range(env.num_envs):
            if just_reset[env_i]:
                # env.step() above ran the auto-reset for env_i; object_pose
                # now reflects the new randomized start. Lock it in and resume
                # data collection on subsequent iterations.
                per_env_start_z[env_i] = float(object_pose_np[env_i, 2])
                just_reset[env_i] = False
                continue
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

                    appended = _append_episode(
                        root, img_ep, state_ep, action_ep,
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

            # Clear this env's buffer regardless of outcome.
            per_env_imgs[env_i].clear()
            per_env_states[env_i].clear()
            per_env_actions[env_i].clear()
            per_env_object_zs[env_i].clear()
            # Defer start_z capture: the env hasn't been reset yet (object_pose
            # here is still the lifted terminal pose). The next iteration's
            # env.step() applies the reset; we capture start_z from its
            # post-step object_pose.
            just_reset[env_i] = True

        # Randomize XY for every env that's about to reset on the next step.
        # Done in one batched call so the RNG draw order is deterministic in
        # env_id order, matching the seeded torch global RNG.
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