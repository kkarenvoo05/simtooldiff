"""Collect SimToolReal expert rollouts in Diffusion Policy Zarr format."""

from dataclasses import dataclass
import faulthandler
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import tyro

try:
    import zarr
except ImportError as exc:
    raise ImportError(
        "zarr is required to collect Diffusion Policy datasets. Install this repo "
        "after the updated pyproject.toml, or run "
        f"`{sys.executable} -m pip install 'zarr<3'`."
    ) from exc

faulthandler.enable()

# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
import torch
# isort: on

import math

from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer


N_OBS = 140
N_ACT = 29


def log(message: str) -> None:
    print(f"[collect_dataset] {message}", file=sys.stderr, flush=True)


@dataclass
class CollectDatasetArgs:
    config_path: Path = Path("pretrained_policy/config.yaml")
    """Path to the pretrained SimToolReal policy config."""

    checkpoint_path: Path = Path("pretrained_policy/model.pth")
    """Path to the pretrained SimToolReal policy checkpoint."""

    output_path: Path = Path("data/simtoolreal_pickup.zarr")
    """Output Zarr directory."""

    num_episodes: int = 1024
    """Number of episodes to collect."""

    horizon: int = 75
    """Pickup rollout length per episode."""

    num_envs: int = 256
    """Parallel Isaac Gym env count. Lower this if rendering runs out of memory."""

    object_name: str = "handle_head_primitives"
    """Object distribution/name to load. handle_head_primitives samples training tools."""

    headless: bool = True
    """Run without an interactive viewer."""

    device: Optional[str] = None
    """Torch/Isaac device. Defaults to cuda when available."""

    deterministic_actions: bool = True
    """Use deterministic expert policy actions."""

    chunk_size: int = 1024
    """Zarr chunk length along the timestep axis."""

    log_every_steps: int = 10
    """Print rollout progress every N timesteps."""


def _create_output_arrays(args: CollectDatasetArgs, total_steps: int):
    log(f"Creating output Zarr at {args.output_path} with {total_steps} transitions")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(args.output_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    chunks = min(args.chunk_size, total_steps)
    images = data.create_dataset(
        "img",
        shape=(total_steps, 224, 224, 3),
        chunks=(chunks, 224, 224, 3),
        dtype="uint8",
        overwrite=True,
    )
    states = data.create_dataset(
        "state",
        shape=(total_steps, N_OBS),
        chunks=(chunks, N_OBS),
        dtype="float32",
        overwrite=True,
    )
    actions = data.create_dataset(
        "action",
        shape=(total_steps, N_ACT),
        chunks=(chunks, N_ACT),
        dtype="float32",
        overwrite=True,
    )
    episode_ends = meta.create_dataset(
        "episode_ends",
        shape=(args.num_episodes,),
        chunks=(min(args.num_episodes, 1024),),
        dtype="int64",
        overwrite=True,
    )
    return images, states, actions, episode_ends


def _load_env(args: CollectDatasetArgs, device: str) -> SimToolReal:
    log(
        f"Creating IsaacGym env: num_envs={args.num_envs}, "
        f"object_name={args.object_name}, device={device}, headless={args.headless}"
    )
    env = create_env(
        config_path=str(args.config_path),
        headless=args.headless,
        device=device,
        enable_viewer_sync_at_start=False,
        overrides={
            "task.env.numEnvs": args.num_envs,
            "task.env.objectName": args.object_name,
            "task.env.enableCameraSensors": True,
            "task.env.enableDatasetCameras": True,
            "task.env.datasetCameraWidth": 224,
            "task.env.datasetCameraHeight": 224,
            "task.env.capture_video": False,
            "task.env.episodeLength": args.horizon + 2,
            "task.env.forceScale": 0.0,
            "task.env.torqueScale": 0.0,
            "task.env.linVelImpulseScale": 0.0,
            "task.env.angVelImpulseScale": 0.0,
        },
    )

    log(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    log("Checkpoint loaded; applying checkpoint env state")
    env.set_env_state(checkpoint[0]["env_state"])
    log("Environment is ready")
    return env


def _initial_observation(env: SimToolReal, device: str) -> torch.Tensor:
    log("Stepping once with zero actions to get initial observation")
    zero_action = torch.zeros((env.num_envs, N_ACT), device=device)
    obs_dict, _, _, _ = env.step(zero_action)
    log(f"Initial observation shape: {tuple(obs_dict['obs'].shape)}")
    return obs_dict["obs"]


def main() -> None:
    args = tyro.cli(CollectDatasetArgs)
    assert args.config_path.exists(), f"Config not found: {args.config_path}"
    assert args.checkpoint_path.exists(), f"Checkpoint not found: {args.checkpoint_path}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    total_steps = args.num_episodes * args.horizon
    log(
        f"Starting collection: episodes={args.num_episodes}, horizon={args.horizon}, "
        f"num_envs={args.num_envs}, total_steps={total_steps}"
    )
    images_zarr, states_zarr, actions_zarr, episode_ends_zarr = _create_output_arrays(
        args, total_steps
    )

    env = _load_env(args, device)
    log("Creating RL policy player")
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=str(args.config_path),
        checkpoint_path=str(args.checkpoint_path),
        device=device,
        num_envs=env.num_envs,
    )
    log("Policy is ready")

    obs = _initial_observation(env, device)
    num_batches = math.ceil(args.num_episodes / env.num_envs)
    write_idx = 0
    episode_idx = 0

    for batch_idx in range(num_batches):
        batch_episodes = min(env.num_envs, args.num_episodes - episode_idx)
        active = torch.arange(batch_episodes, device=device)
        batch_start_idx = write_idx
        log(
            f"Batch {batch_idx + 1}/{num_batches}: collecting {batch_episodes} "
            f"episodes starting at transition {batch_start_idx}"
        )

        for timestep in range(args.horizon):
            if timestep == 0 or (timestep + 1) % args.log_every_steps == 0:
                log(
                    f"Batch {batch_idx + 1}/{num_batches}: "
                    f"timestep {timestep + 1}/{args.horizon}"
                )
            imgs = env.render_dataset_camera_rgb(active)
            action = policy.get_normalized_action(
                obs, deterministic_actions=args.deterministic_actions
            )

            step_indices = batch_start_idx + (
                np.arange(batch_episodes, dtype=np.int64) * args.horizon + timestep
            )
            images_zarr[step_indices, :, :, :] = imgs.cpu().numpy()
            states_zarr[step_indices, :] = (
                obs[:batch_episodes].detach().cpu().numpy().astype(np.float32)
            )
            actions_zarr[step_indices, :] = (
                action[:batch_episodes].detach().cpu().numpy().astype(np.float32)
            )

            obs_dict, _, _, _ = env.step(action)
            obs = obs_dict["obs"]

        for local_episode_idx in range(batch_episodes):
            episode_ends_zarr[episode_idx] = (
                batch_start_idx + (local_episode_idx + 1) * args.horizon
            )
            episode_idx += 1
        write_idx += batch_episodes * args.horizon

        env.reset_buf[:] = 1
        obs = _initial_observation(env, device)
        policy.reset()
        log(
            f"Collected batch {batch_idx + 1}/{num_batches}: "
            f"{episode_idx}/{args.num_episodes} episodes, {write_idx} transitions"
        )

    assert write_idx == total_steps, f"Expected {total_steps} steps, wrote {write_idx}"
    log(f"Saved Diffusion Policy dataset to {args.output_path}")


if __name__ == "__main__":
    main()
