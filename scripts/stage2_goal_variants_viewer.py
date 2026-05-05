#!/usr/bin/env python3
"""Stage 2: viewer rollout for one lifted-goal variant on claw_hammer/swing_down."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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


@dataclass(frozen=True)
class GoalVariant:
    name: str
    description: str
    goals: List[List[float]]


def make_lift_goal(start_pose: List[float], lift_m: float) -> List[List[float]]:
    goal = list(start_pose)
    goal[2] += lift_m
    return [goal]


def make_env(
    goals: List[List[float]],
    object_start_pose: List[float],
    horizon: int,
    device: str,
):
    return create_env(
        config_path=str(CONFIG_PATH),
        headless=False,
        device=device,
        episode_length=horizon + 2,
        overrides={
            "task.env.numEnvs": 1,
            "task.env.envSpacing": 0.4,
            "task.env.capture_video": False,
            "task.env.enableCameraSensors": False,
            "task.env.objectName": OBJECT_NAME,
            "task.env.useFixedGoalStates": True,
            "task.env.fixedGoalStates": goals,
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


def run_variant(
    variant: GoalVariant,
    object_start_pose: List[float],
    horizon: int,
    hold_seconds: float,
    device: str,
) -> None:
    print()
    print("=" * 80)
    print(f"[stage2] variant: {variant.name}")
    print(f"[stage2] {variant.description}")
    print(f"[stage2] goals={variant.goals}")
    print("=" * 80)
    print("[stage2] creating viewer env...")

    env = make_env(
        goals=variant.goals,
        object_start_pose=object_start_pose,
        horizon=horizon,
        device=device,
    )

    print("[stage2] loading checkpoint env_state and policy...")
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

    print(f"[stage2] running {horizon} viewer steps...")
    last_step = 0
    for step in range(horizon):
        last_step = step
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            print("[stage2] viewer closed; stopping this variant early")
            break

        action = policy.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, _, done, _ = env.step(action)
        obs = obs_dict["obs"]

        if step % 30 == 0:
            successes = int(env.successes[0].item())
            print(
                f"[stage2] {variant.name} step={step:03d} "
                f"done={bool(done[0].item())} "
                f"successes={successes}/{env.max_consecutive_successes}"
            )

        time.sleep(1.0 / CONTROL_HZ)

    successes = int(env.successes[0].item())
    goal_pct = 100.0 * successes / env.max_consecutive_successes
    print(
        f"[stage2] {variant.name} finished at step={last_step}; "
        f"goal_pct={goal_pct:.0f}%"
    )
    print(f"[stage2] holding viewer for {hold_seconds:.1f}s...")
    time.sleep(hold_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("lift_0.15m", "lift_0.20m", "first_3_swing_down_goals"),
        required=True,
        help="Goal variant to run. Run this script once per variant.",
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
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

    with open(trajectory_path, "r") as f:
        traj_data = json.load(f)

    object_start_pose = list(traj_data["start_pose"])
    object_start_pose[2] += START_Z_OFFSET

    variants: Dict[str, GoalVariant] = {
        "lift_0.15m": GoalVariant(
            name="lift_0.15m",
            description="Single custom goal: actual start pose lifted by 15cm.",
            goals=make_lift_goal(object_start_pose, 0.15),
        ),
        "lift_0.20m": GoalVariant(
            name="lift_0.20m",
            description="Single custom goal: actual start pose lifted by 20cm.",
            goals=make_lift_goal(object_start_pose, 0.20),
        ),
        "first_3_swing_down_goals": GoalVariant(
            name="first_3_swing_down_goals",
            description="First 3 canonical swing_down goals, unchanged.",
            goals=traj_data["goals"][:3],
        ),
    }

    print(f"[stage2] device={device}")
    print(f"[stage2] trajectory={trajectory_path}")
    print(f"[stage2] object_start_pose={object_start_pose}")

    run_variant(
        variant=variants[args.variant],
        object_start_pose=object_start_pose,
        horizon=args.horizon,
        hold_seconds=args.hold_seconds,
        device=device,
    )

    print()
    print(f"[stage2] {args.variant} complete")


if __name__ == "__main__":
    main()
