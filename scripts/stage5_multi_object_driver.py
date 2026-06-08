#!/usr/bin/env python3
"""Driver: run stage5_collect_dataset.py per-object for multi-object collection.

Builds two zarrs:
  - data/stage5_train.zarr: 9 training objects
  - data/stage5_ood.zarr:   3 held-out objects (eval only)

Subprocess-per-object because IsaacGym dislikes recreating a sim with a different
object asset in the same process.

Each per-object run uses --resume so it appends to the shared zarr. Because the
stage 5 main loop checks total_transitions >= target_transitions, the driver
passes an absolute cumulative target after each object completes.

Collection types:
  - pickup: preserved earlier pickup-only Stage 5 collector
  - pick_place: current pick-and-place collector
  - pick_place_release: experimental pick-place-release collector

Action variants:
  - clean: persistent-env clean expert rollouts
  - noisy_clean: noisy execution with clean action labels
  - noisy_noisy: noisy execution with noisy action labels
"""

import argparse
import json
import subprocess
import sys
import zarr
from dataclasses import dataclass
from pathlib import Path
from typing import List

from stage5_noise_config import resolve_noise_config


# (category_id, object_id, object_category, object_name, task_name)
# IDs are stable across train/ood splits so the same object always has the
# same numerical id no matter which zarr it appears in.
@dataclass(frozen=True)
class ObjectSpec:
    object_id: int
    category_id: int
    object_category: str
    object_name: str
    task_name: str


# Global registry. Category ids alphabetical; object ids = first appearance.
_REGISTRY: List[ObjectSpec] = [
    # category_id 0: brush
    ObjectSpec(0, 0, "brush",       "blue_brush",        "sweep_forward"),
    ObjectSpec(1, 0, "brush",       "red_brush",         "sweep_forward"),
    # category_id 1: eraser
    ObjectSpec(2, 1, "eraser",      "flat_eraser",       "wipe_c"),
    ObjectSpec(3, 1, "eraser",      "handle_eraser",     "wipe_c"),
    # category_id 2: hammer (claw_hammer trains, mallet_hammer is OOD-instance)
    ObjectSpec(4, 2, "hammer",      "claw_hammer",       "swing_down"),
    ObjectSpec(5, 2, "hammer",      "mallet_hammer",     "swing_down"),
    # category_id 3: marker
    ObjectSpec(6, 3, "marker",      "sharpie_marker",    "draw_smile"),
    ObjectSpec(7, 3, "marker",      "staples_marker",    "draw_smile"),
    # category_id 4: screwdriver
    ObjectSpec(8, 4, "screwdriver", "long_screwdriver",  "spin_horizontal"),
    ObjectSpec(9, 4, "screwdriver", "short_screwdriver", "spin_horizontal"),
    # category_id 5: spatula (fully held-out OOD-category)
    ObjectSpec(10, 5, "spatula",     "flat_spatula",      "flip_over"),
    ObjectSpec(11, 5, "spatula",     "spoon_spatula",     "flip_over"),
]


TRAIN_OBJECT_IDS = {0, 1, 2, 3, 4, 6, 7, 8, 9}
OOD_INSTANCE_IDS = {5}                # mallet_hammer
OOD_CATEGORY_IDS = {10, 11}           # both spatulas


def _split(name: str) -> List[ObjectSpec]:
    if name == "train":
        ids = TRAIN_OBJECT_IDS
    elif name == "ood":
        ids = OOD_INSTANCE_IDS | OOD_CATEGORY_IDS
    else:
        raise ValueError(name)
    return [s for s in _REGISTRY if s.object_id in ids]


def _current_transition_count(zarr_path: Path) -> int:
    if not zarr_path.exists():
        return 0
    try:
        root = zarr.open(str(zarr_path), mode="r")
    except zarr.errors.PathNotFoundError:
        print(
            f"[multi-object driver] output store exists but is not a readable zarr root yet: "
            f"{zarr_path}; treating as empty",
            flush=True,
        )
        return 0
    if "data" not in root or "img" not in root["data"]:
        return 0
    return int(root["data"]["img"].shape[0])


def _run_one(spec: ObjectSpec, args: argparse.Namespace) -> None:
    current = _current_transition_count(args.output_zarr)
    cumulative_target = current + args.per_object_transitions
    collector_script = (
        "scripts/stage5_collect_dataset.py"
        if args.collection_type == "pickup"
        else "scripts/collect_dataset_pick_place_release.py"
    )
    cmd = [
        sys.executable, collector_script,
        "--collection-type", args.collection_type,
        "--object-category", spec.object_category,
        "--object-name", spec.object_name,
        "--task-name", spec.task_name,
        "--object-id", str(spec.object_id),
        "--category-id", str(spec.category_id),
        "--num-envs", str(args.num_envs),
        "--target-transitions", str(cumulative_target),
        "--horizon", str(args.horizon),
        "--xy-range", str(args.xy_range),
        "--start-z-offset", str(args.start_z_offset),
        "--seed", str(args.seed + spec.object_id),
        "--output-zarr", str(args.output_zarr),
        "--resume",
        "--save-preview-every", str(args.save_preview_every),
    ]
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if args.max_attempted_episodes is not None:
        cmd.extend(["--max-attempted-episodes", str(args.max_attempted_episodes)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.viewer:
        cmd.append("--viewer")
    if args.log_every_step:
        cmd.append("--log-every-step")

    if args.collection_type != "pickup":
        cmd.extend(
            [
                "--lift-height", str(args.lift_height),
                "--lateral-offset-range", str(args.lateral_offset_range),
                "--place-height", str(args.place_height),
                "--place-hold-goals", str(args.place_hold_goals),
                "--table-x-half-extent", str(args.table_x_half_extent),
                "--table-x-inset-margin", str(args.table_x_inset_margin),
                "--min-effective-transport", str(args.min_effective_transport),
            ]
        )

    cmd.extend(["--variant", args.variant])

    # Forward phase-gated-noise + blunt-fallback flags to the collector. The
    # collector applies the multipliers (its wrapped _resolve_noise_args), so we
    # forward the raw multipliers here rather than baking them into
    # resolve_noise_config below (which would double-apply).
    cmd.extend([
        "--noise-phase-gating", str(args.noise_phase_gating),
        "--closure-proximity-threshold", str(args.closure_proximity_threshold),
        "--closure-palm-z-threshold", str(args.closure_palm_z_threshold),
        "--closure-noise-scale", str(args.closure_noise_scale),
        "--closure-groups", str(args.closure_groups),
        "--closure-window-padding", str(args.closure_window_padding),
        "--finger-noise-multiplier", str(args.finger_noise_multiplier),
        "--wrist-noise-multiplier", str(args.wrist_noise_multiplier),
    ])

    if args.collection_type != "pick_place_release" and args.max_batches is not None:
        cmd.extend(["--max-batches", str(args.max_batches)])

    if args.variant != "clean":
        noise_config = resolve_noise_config(
            noise_scale=args.noise_scale,
            arm_base_noise=args.arm_base_noise,
            arm_wrist_noise=args.arm_wrist_noise,
            thumb_noise=args.thumb_noise,
            index_noise=args.index_noise,
            middle_noise=args.middle_noise,
            ring_noise=args.ring_noise,
            pinky_noise=args.pinky_noise,
        )
        cmd.extend(
            [
                "--noise-strategy", args.noise_strategy,
                "--noise-scale", str(args.noise_scale),
                "--arm-base-noise", str(noise_config["arm_base"]),
                "--arm-wrist-noise", str(noise_config["arm_wrist"]),
                "--thumb-noise", str(noise_config["thumb"]),
                "--index-noise", str(noise_config["index"]),
                "--middle-noise", str(noise_config["middle"]),
                "--ring-noise", str(noise_config["ring"]),
                "--pinky-noise", str(noise_config["pinky"]),
                "--ou-theta", str(args.ou_theta),
                "--ou-mu", str(args.ou_mu),
                "--ou-dt", str(args.ou_dt),
            ]
        )
        if args.noise_strategy == "anchored_recovery":
            cmd.extend(
                [
                    "--anchored-branches-per-rollout", str(args.anchored_branches_per_rollout),
                    "--anchored-branch-min-step", str(args.anchored_branch_min_step),
                    "--anchored-branch-max-step", str(args.anchored_branch_max_step),
                    "--anchored-perturb-steps", str(args.anchored_perturb_steps),
                    "--anchored-recovery-steps", str(args.anchored_recovery_steps),
                ]
            )

    if args.collection_type == "pickup":
        if args.variant != "clean":
            cmd.extend(["--pickup-success-hold-steps", str(args.pickup_success_hold_steps)])
        cmd.extend(["--rollouts-per-init", str(args.rollouts_per_init)])
    elif args.collection_type == "pick_place":
        if args.variant != "clean":
            cmd.extend(["--pickup-success-hold-steps", str(args.pickup_success_hold_steps)])
    elif args.collection_type == "pick_place_release":
        cmd.extend(
            [
                "--release-steps", str(args.release_steps),
                "--release-arm-mode", args.release_arm_mode,
                "--release-hand-blend", str(args.release_hand_blend),
                "--release-xy-tolerance", str(args.release_xy_tolerance),
                "--release-z-tolerance", str(args.release_z_tolerance),
                "--release-speed-tolerance", str(args.release_speed_tolerance),
                "--table-y-half-extent", str(args.table_y_half_extent),
                "--table-y-inset-margin", str(args.table_y_inset_margin),
                "--place-goal-x-margin", str(args.place_goal_x_margin),
                "--place-goal-y-margin", str(args.place_goal_y_margin),
                "--rollouts-per-init", str(args.rollouts_per_init),
            ]
        )
    elif args.collection_type != "pickup":
        raise ValueError(f"Unsupported --collection-type={args.collection_type}")

    print(
        f"\n[multi-object driver] >>> object_id={spec.object_id} "
        f"{spec.object_category}/{spec.object_name} "
        f"target={cumulative_target} (delta={args.per_object_transitions}) "
        f"collection_type={args.collection_type} "
        f"variant={args.variant}",
        flush=True,
    )
    subprocess.run(cmd, check=True)


def _validate_mode_args(args: argparse.Namespace) -> None:
    if args.collection_type == "pick_place_release" and args.max_batches is not None:
        raise ValueError("--max-batches is not supported for --collection-type pick_place_release.")
    if args.noise_strategy == "anchored_recovery":
        if args.variant != "noisy_clean":
            raise ValueError("--noise-strategy anchored_recovery requires --variant noisy_clean")
        if args.num_envs != 1:
            raise ValueError("--noise-strategy anchored_recovery requires --num-envs 1")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("train", "ood"), required=True)
    p.add_argument("--output-zarr", type=Path, required=True)
    p.add_argument("--per-object-transitions", type=int, required=True,
                   help="How many transitions to collect per object on top of the existing zarr.")
    p.add_argument(
        "--collection-type",
        choices=("pickup", "pick_place", "pick_place_release"),
        default="pick_place",
    )
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--xy-range", type=float, default=0.10)
    p.add_argument("--start-z-offset", type=float, default=0.0)
    p.add_argument("--lift-height", type=float, default=0.20)
    p.add_argument("--lateral-offset-range", type=float, default=0.15)
    p.add_argument("--place-height", type=float, default=0.02)
    p.add_argument("--place-hold-goals", type=int, default=10)
    p.add_argument("--table-x-half-extent", type=float, default=0.475 / 2.0)
    p.add_argument("--table-x-inset-margin", type=float, default=0.06)
    p.add_argument("--table-y-half-extent", type=float, default=0.4 / 2.0)
    p.add_argument("--table-y-inset-margin", type=float, default=0.04)
    p.add_argument("--place-goal-x-margin", type=float, default=0.10)
    p.add_argument("--place-goal-y-margin", type=float, default=0.06)
    p.add_argument("--min-effective-transport", type=float, default=0.05)
    p.add_argument("--release-steps", type=int, default=45)
    p.add_argument("--release-arm-mode", choices=("hold", "policy"), default="hold")
    p.add_argument("--release-hand-blend", type=float, default=1.0)
    p.add_argument("--release-xy-tolerance", type=float, default=0.05)
    p.add_argument("--release-z-tolerance", type=float, default=0.04)
    p.add_argument("--release-speed-tolerance", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-attempted-episodes", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--log-every-step", action="store_true")
    p.add_argument("--save-preview-every", type=int, default=0,
                   help="Save a GIF every N written episodes per object (0 disables).")
    p.add_argument("--variant", choices=("clean", "noisy_clean", "noisy_noisy"), default="clean")
    p.add_argument("--pickup-success-hold-steps", type=int, default=5)
    p.add_argument(
        "--noise-strategy",
        choices=("continuous_ou", "anchored_recovery"),
        default="continuous_ou",
    )
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--arm-base-noise", type=float, default=None)
    p.add_argument("--arm-wrist-noise", type=float, default=None)
    p.add_argument("--thumb-noise", type=float, default=None)
    p.add_argument("--index-noise", type=float, default=None)
    p.add_argument("--middle-noise", type=float, default=None)
    p.add_argument("--ring-noise", type=float, default=None)
    p.add_argument("--pinky-noise", type=float, default=None)
    p.add_argument("--ou-theta", type=float, default=0.15)
    p.add_argument("--ou-mu", type=float, default=0.0)
    p.add_argument("--ou-dt", type=float, default=1.0)
    # --- Phase-gated noise for grasp closure (forwarded to the collector) ---
    p.add_argument("--noise-phase-gating", choices=["off", "proximity", "palm_z"], default="off")
    p.add_argument("--closure-proximity-threshold", type=float, default=0.08)
    p.add_argument("--closure-palm-z-threshold", type=float, default=0.66)
    p.add_argument("--closure-noise-scale", type=float, default=0.0)
    p.add_argument("--closure-groups", type=str, default="wrist,thumb,index,middle,ring,pinky")
    p.add_argument("--closure-window-padding", type=int, default=5)
    p.add_argument("--finger-noise-multiplier", type=float, default=1.0)
    p.add_argument("--wrist-noise-multiplier", type=float, default=1.0)
    p.add_argument("--anchored-branches-per-rollout", type=int, default=3)
    p.add_argument("--anchored-branch-min-step", type=int, default=10)
    p.add_argument("--anchored-branch-max-step", type=str, default="auto")
    p.add_argument("--anchored-perturb-steps", type=int, default=3)
    p.add_argument("--anchored-recovery-steps", type=int, default=15)
    p.add_argument("--rollouts-per-init", type=int, default=1,
                   help="Number of noisy rollouts per sampled start_pose (pick_place_release only).")
    args = p.parse_args()
    _validate_mode_args(args)

    specs = _split(args.split)
    args.output_zarr.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[multi-object driver] split={args.split} "
        f"collection_type={args.collection_type} variant={args.variant} "
        f"objects={[s.object_name for s in specs]}",
        flush=True,
    )

    for spec in specs:
        _run_one(spec, args)

    final = _current_transition_count(args.output_zarr)
    print(f"\n[multi-object driver] DONE: {args.output_zarr} has {final} transitions across {len(specs)} objects", flush=True)


if __name__ == "__main__":
    main()
