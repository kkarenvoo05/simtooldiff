#!/usr/bin/env python3
"""Run a short long-table pick/place debug rollout and save camera artifacts.

This is a thin wrapper around `collect_dataset_pick_place_release.py` intended
for fast qualitative checks:

1. dataset camera framing
2. end-to-end transport behavior
3. release-stage trajectory behavior

It runs the collector in `--dry-run` mode, converts the saved rollout MP4s to
GIF previews, and extracts the first video frame as a PNG for quick camera
inspection.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _build_cmd(args: argparse.Namespace) -> List[str]:
    script_path = Path(__file__).resolve().with_name("collect_dataset_pick_place_release.py")
    output_zarr = args.output_dir / f"{args.collection_type}_debug.zarr"
    video_dir = args.output_dir / f"{args.collection_type}_videos"

    cmd = [
        sys.executable,
        str(script_path),
        "--collection-type",
        args.collection_type,
        "--object-category",
        args.object_category,
        "--object-name",
        args.object_name,
        "--task-name",
        args.task_name,
        "--output-zarr",
        str(output_zarr),
        "--dry-run",
        "--dry-run-rollouts",
        str(args.rollouts),
        "--dry-run-video-dir",
        str(video_dir),
        "--save-preview-every",
        "0",
        "--seed",
        str(args.seed),
        "--variant",
        args.variant,
        "--xy-range",
        str(args.xy_range),
        "--start-z-offset",
        str(args.start_z_offset),
        "--lift-height",
        str(args.lift_height),
        "--lateral-offset-range",
        str(args.lateral_offset_range),
        "--place-height",
        str(args.place_height),
        "--place-hold-goals",
        str(args.place_hold_goals),
        "--table-x-half-extent",
        str(args.table_x_half_extent),
        "--table-x-inset-margin",
        str(args.table_x_inset_margin),
        "--table-y-half-extent",
        str(args.table_y_half_extent),
        "--table-y-inset-margin",
        str(args.table_y_inset_margin),
        "--place-goal-x-margin",
        str(args.place_goal_x_margin),
        "--place-goal-y-margin",
        str(args.place_goal_y_margin),
        "--min-effective-transport",
        str(args.min_effective_transport),
        "--horizon",
        str(args.horizon),
        "--log-every-step",
    ]

    if args.variant != "clean":
        cmd.extend(
            [
                "--noise-scale",
                str(args.noise_scale),
                "--ou-theta",
                str(args.ou_theta),
                "--ou-mu",
                str(args.ou_mu),
                "--ou-dt",
                str(args.ou_dt),
            ]
        )

    if args.collection_type == "pick_place_release":
        cmd.extend(
            [
                "--release-steps",
                str(args.release_steps),
                "--release-arm-mode",
                args.release_arm_mode,
                "--release-hand-blend",
                str(args.release_hand_blend),
                "--release-xy-tolerance",
                str(args.release_xy_tolerance),
                "--release-z-tolerance",
                str(args.release_z_tolerance),
                "--release-speed-tolerance",
                str(args.release_speed_tolerance),
            ]
        )

    if args.device:
        cmd.extend(["--device", args.device])
    if args.viewer:
        cmd.append("--viewer")

    return cmd


def _extract_first_frame(video_dir: Path, output_path: Path) -> Path:
    import imageio.v2 as imageio

    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No debug videos found in {video_dir}")

    reader = imageio.get_reader(videos[0])
    try:
        frame0 = reader.get_data(0)
    finally:
        reader.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, frame0)
    return videos[0]


def _write_gif_previews(video_dir: Path, fps: int) -> List[Path]:
    import imageio.v2 as imageio

    gif_paths: List[Path] = []
    for video_path in sorted(video_dir.glob("*.mp4")):
        reader = imageio.get_reader(video_path)
        try:
            frames = [frame for frame in reader]
        finally:
            reader.close()
        if not frames:
            continue
        gif_path = video_path.with_suffix(".gif")
        imageio.mimsave(
            gif_path,
            frames,
            duration=1000.0 / max(fps, 1),
        )
        gif_paths.append(gif_path)
    return gif_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-type",
        choices=("pick_place", "pick_place_release"),
        default="pick_place_release",
    )
    parser.add_argument("--object-category", default="hammer")
    parser.add_argument("--object-name", default="claw_hammer")
    parser.add_argument("--task-name", default="swing_down")
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        choices=("clean", "noisy_clean", "noisy_noisy"),
        default="clean",
    )
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--ou-theta", type=float, default=0.15)
    parser.add_argument("--ou-mu", type=float, default=0.0)
    parser.add_argument("--ou-dt", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=325)
    parser.add_argument("--xy-range", type=float, default=0.10)
    parser.add_argument("--start-z-offset", type=float, default=0.0)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--lateral-offset-range", type=float, default=0.15)
    parser.add_argument("--place-height", type=float, default=0.02)
    parser.add_argument("--place-hold-goals", type=int, default=10)
    parser.add_argument("--table-x-half-extent", type=float, default=0.30)
    parser.add_argument("--table-x-inset-margin", type=float, default=0.06)
    parser.add_argument("--table-y-half-extent", type=float, default=0.20)
    parser.add_argument("--table-y-inset-margin", type=float, default=0.04)
    parser.add_argument("--place-goal-x-margin", type=float, default=0.10)
    parser.add_argument("--place-goal-y-margin", type=float, default=0.06)
    parser.add_argument("--min-effective-transport", type=float, default=0.05)
    parser.add_argument("--release-steps", type=int, default=0)
    parser.add_argument("--release-arm-mode", choices=("hold", "policy"), default="hold")
    parser.add_argument("--release-hand-blend", type=float, default=1.0)
    parser.add_argument("--release-xy-tolerance", type=float, default=0.05)
    parser.add_argument("--release-z-tolerance", type=float, default=0.04)
    parser.add_argument("--release-speed-tolerance", type=float, default=0.25)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("data/debug_pick_place_release"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(args)
    print("[debug] running:")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    video_dir = args.output_dir / f"{args.collection_type}_videos"
    frame_path = args.output_dir / f"{args.collection_type}_camera_frame0.png"
    gif_paths = _write_gif_previews(video_dir, args.gif_fps)
    video_path = _extract_first_frame(video_dir, frame_path)

    print(f"[debug] video={video_path}")
    for gif_path in gif_paths:
        print(f"[debug] gif={gif_path}")
    print(f"[debug] camera_frame={frame_path}")


if __name__ == "__main__":
    main()
