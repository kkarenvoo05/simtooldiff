#!/usr/bin/env python3
"""Stage 1: single-env viewer rollout for claw_hammer/swing_down."""

import json
import time
from pathlib import Path

# Isaac Gym must be imported before torch.
from isaacgym import gymapi  # noqa: F401
import torch

from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer
from isaacgymenvs.utils.utils import get_repo_root_dir


N_OBS = 140
N_ACT = 29
HORIZON = 150
CONTROL_HZ = 60.0

OBJECT_CATEGORY = "hammer"
OBJECT_NAME = "claw_hammer"
TASK_NAME = "swing_down"

CONFIG_PATH = Path("pretrained_policy/config.yaml")
CHECKPOINT_PATH = Path("pretrained_policy/model.pth")

TABLE_Z = 0.38
TABLE_NAIL_URDF = "urdf/table_narrow_nail.urdf"
START_Z_OFFSET = 0.03


def main() -> None:
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

    with open(trajectory_path, "r") as f:
        traj_data = json.load(f)

    # Match the repo's canonical eval behavior: keep goals unchanged, lift only init pose.
    object_start_pose = list(traj_data["start_pose"])
    object_start_pose[2] += START_Z_OFFSET

    print(f"[stage1] device={device}")
    print(f"[stage1] trajectory={trajectory_path}")
    print(f"[stage1] num_goals={len(traj_data['goals'])}")
    print("[stage1] creating viewer env...")

    env = create_env(
        config_path=str(CONFIG_PATH),
        headless=False,
        device=device,
        episode_length=HORIZON + 2,
        overrides={
            "task.env.numEnvs": 1,
            "task.env.envSpacing": 0.4,
            "task.env.capture_video": False,
            "task.env.enableCameraSensors": False,
            "task.env.objectName": OBJECT_NAME,
            "task.env.useFixedGoalStates": True,
            "task.env.fixedGoalStates": traj_data["goals"],
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
        },
    )

    print("[stage1] loading checkpoint env_state and policy...")
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

    print(f"[stage1] running {HORIZON} viewer steps...")
    for step in range(HORIZON):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            print("[stage1] viewer closed; stopping early")
            break

        action = policy.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, _, done, _ = env.step(action)
        obs = obs_dict["obs"]

        if step % 30 == 0:
            successes = int(env.successes[0].item())
            print(
                f"[stage1] step={step:03d} "
                f"done={bool(done[0].item())} "
                f"successes={successes}/{env.max_consecutive_successes}"
            )

        time.sleep(1.0 / CONTROL_HZ)

    print("[stage1] rollout finished. Leaving viewer open for 3 seconds...")
    time.sleep(3.0)


if __name__ == "__main__":
    main()
