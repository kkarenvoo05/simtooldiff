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
"""

import argparse
import json
import subprocess
import sys
import zarr
from dataclasses import dataclass
from pathlib import Path
from typing import List


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
    root = zarr.open(str(zarr_path), mode="r")
    if "data" not in root or "img" not in root["data"]:
        return 0
    return int(root["data"]["img"].shape[0])


def _run_one(spec: ObjectSpec, args: argparse.Namespace) -> None:
    current = _current_transition_count(args.output_zarr)
    cumulative_target = current + args.per_object_transitions
    cmd = [
        sys.executable, "scripts/stage5_collect_dataset.py",
        "--object-category", spec.object_category,
        "--object-name", spec.object_name,
        "--task-name", spec.task_name,
        "--object-id", str(spec.object_id),
        "--category-id", str(spec.category_id),
        "--num-envs", str(args.num_envs),
        "--target-transitions", str(cumulative_target),
        "--horizon", str(args.horizon),
        "--xy-range", str(args.xy_range),
        "--seed", str(args.seed + spec.object_id),  # different seed per object
        "--output-zarr", str(args.output_zarr),
        "--resume",
        "--save-preview-every", str(args.save_preview_every),
    ]
    if args.max_attempted_episodes is not None:
        cmd.extend(["--max-attempted-episodes", str(args.max_attempted_episodes)])
    print(
        f"\n[multi-object driver] >>> object_id={spec.object_id} "
        f"{spec.object_category}/{spec.object_name} "
        f"target={cumulative_target} (delta={args.per_object_transitions})",
        flush=True,
    )
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("train", "ood"), required=True)
    p.add_argument("--output-zarr", type=Path, required=True)
    p.add_argument("--per-object-transitions", type=int, required=True,
                   help="How many transitions to collect per object on top of the existing zarr.")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=250)
    p.add_argument("--xy-range", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-attempted-episodes", type=int, default=None)
    p.add_argument("--save-preview-every", type=int, default=0,
                   help="Save a GIF every N written episodes per object (0 disables).")
    args = p.parse_args()

    specs = _split(args.split)
    args.output_zarr.parent.mkdir(parents=True, exist_ok=True)
    print(f"[multi-object driver] split={args.split} objects={[s.object_name for s in specs]}", flush=True)

    for spec in specs:
        _run_one(spec, args)

    final = _current_transition_count(args.output_zarr)
    print(f"\n[multi-object driver] DONE: {args.output_zarr} has {final} transitions across {len(specs)} objects", flush=True)


if __name__ == "__main__":
    main()
