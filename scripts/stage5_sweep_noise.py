#!/usr/bin/env python3
"""Sweep Stage 5 noisy rollout scales and summarize write-rate/noise-band stats.

Example:
    python scripts/stage5_sweep_noise.py \
      --split train \
      --max-objects 3 \
      --attempted-episodes-per-object 256 \
      --scales 2,6,12,20
"""

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import zarr

from stage5_noise_config import BASE_NOISE_DEFAULTS, resolve_noise_config
from stage5_multi_object_driver import ObjectSpec, _REGISTRY, _split


DEFAULT_SCALES = [2.0, 6.0, 12.0, 20.0]


def _collector_script(collection_type: str) -> str:
    if collection_type == "pickup":
        return "scripts/stage5_collect_dataset.py"
    return "scripts/collect_dataset_pick_place_release.py"


def _parse_scales(raw: str) -> List[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_object_names(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _select_specs(args: argparse.Namespace) -> List[ObjectSpec]:
    if args.object_names:
        wanted = set(_parse_object_names(args.object_names))
        specs = [spec for spec in _REGISTRY if spec.object_name in wanted]
        missing = sorted(wanted - {spec.object_name for spec in specs})
        if missing:
            raise ValueError(f"Unknown object names: {missing}")
    else:
        specs = _split(args.split)

    if args.max_objects is not None:
        specs = specs[: args.max_objects]
    if not specs:
        raise ValueError("No objects selected for sweep.")
    return specs


def _scaled_noises(scale: float) -> Dict[str, float]:
    return resolve_noise_config(
        noise_scale=scale,
        arm_base_noise=None,
        arm_wrist_noise=None,
        thumb_noise=None,
        index_noise=None,
        middle_noise=None,
        ring_noise=None,
        pinky_noise=None,
    )


def _run_one(
    args: argparse.Namespace,
    spec: ObjectSpec,
    scale: float,
    scale_dir: Path,
) -> Dict[str, float]:
    output_zarr = scale_dir / f"{spec.object_id:02d}_{spec.object_name}.zarr"
    noises = _scaled_noises(scale)
    cmd = [
        sys.executable,
        _collector_script(args.collection_type),
        "--collection-type",
        args.collection_type,
        "--object-category",
        spec.object_category,
        "--object-name",
        spec.object_name,
        "--task-name",
        spec.task_name,
        "--object-id",
        str(spec.object_id),
        "--category-id",
        str(spec.category_id),
        "--num-envs",
        str(args.num_envs),
        "--target-transitions",
        str(args.target_transitions_per_object),
        "--max-attempted-episodes",
        str(args.attempted_episodes_per_object),
        "--horizon",
        str(args.horizon),
        "--xy-range",
        str(args.xy_range),
        "--seed",
        str(args.seed + spec.object_id),
        "--output-zarr",
        str(output_zarr),
        "--variant",
        args.variant,
        "--save-preview-every",
        "0",
        "--pickup-success-hold-steps",
        str(args.pickup_success_hold_steps),
        "--arm-base-noise",
        str(noises["arm_base"]),
        "--arm-wrist-noise",
        str(noises["arm_wrist"]),
        "--thumb-noise",
        str(noises["thumb"]),
        "--index-noise",
        str(noises["index"]),
        "--middle-noise",
        str(noises["middle"]),
        "--ring-noise",
        str(noises["ring"]),
        "--pinky-noise",
        str(noises["pinky"]),
        "--ou-theta",
        str(args.ou_theta),
        "--ou-mu",
        str(args.ou_mu),
        "--ou-dt",
        str(args.ou_dt),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.viewer:
        cmd.append("--viewer")

    print(
        f"\n[noise-sweep] scale={scale:g} object={spec.object_category}/{spec.object_name}",
        flush=True,
    )
    subprocess.run(cmd, check=True)

    root = zarr.open(str(output_zarr), mode="r")
    transitions = int(root["data"]["img"].shape[0])
    episodes = int(root["meta"]["episode_ends"].shape[0])
    attempted = int(root.attrs.get("attempted_episodes", root.attrs.get("attempted_rollouts", 0)))
    written = int(root.attrs.get("written_episodes", episodes))
    delta_count = int(root.attrs.get("noise_action_delta_count", 0))
    pickup_attempted = int(root.attrs.get("pickup_timing_attempted_episodes", attempted))
    pickup_gate_count = int(root.attrs.get("pickup_gate_first_step_count", 0))
    pickup_hold_count = int(root.attrs.get("pickup_hold_first_step_count", 0))

    return {
        "scale": scale,
        "object_id": spec.object_id,
        "category_id": spec.category_id,
        "object_category": spec.object_category,
        "object_name": spec.object_name,
        "task_name": spec.task_name,
        "output_zarr": str(output_zarr),
        "attempted_episodes": attempted,
        "written_episodes": written,
        "write_success_rate": written / max(attempted, 1),
        "transitions": transitions,
        "episodes": episodes,
        "mean_episode_len": transitions / max(episodes, 1),
        "noise_action_delta_count": delta_count,
        "noise_action_delta_l2_sum": float(
            root.attrs.get("noise_action_delta_l2_sum", 0.0)
        ),
        "noise_action_delta_l2_sq_sum": float(
            root.attrs.get("noise_action_delta_l2_sq_sum", 0.0)
        ),
        "noise_action_delta_l2_mean": float(
            root.attrs.get("noise_action_delta_l2_mean", 0.0)
        ),
        "noise_action_delta_l2_std": float(
            root.attrs.get("noise_action_delta_l2_std", 0.0)
        ),
        "noise_action_delta_linf_mean": float(
            root.attrs.get("noise_action_delta_linf_mean", 0.0)
        ),
        "pickup_timing_attempted_episodes": pickup_attempted,
        "pickup_gate_first_step_count": pickup_gate_count,
        "pickup_gate_first_step_sum": float(
            root.attrs.get("pickup_gate_first_step_sum", 0.0)
        ),
        "pickup_gate_first_step_mean": float(
            root.attrs.get("pickup_gate_first_step_mean", -1.0)
        ),
        "pickup_gate_trigger_rate": pickup_gate_count / max(pickup_attempted, 1),
        "pickup_hold_first_step_count": pickup_hold_count,
        "pickup_hold_first_step_sum": float(
            root.attrs.get("pickup_hold_first_step_sum", 0.0)
        ),
        "pickup_hold_first_step_mean": float(
            root.attrs.get("pickup_hold_first_step_mean", -1.0)
        ),
        "pickup_hold_trigger_rate": pickup_hold_count / max(pickup_attempted, 1),
        "arm_base_noise": noises["arm_base"],
        "arm_wrist_noise": noises["arm_wrist"],
        "thumb_noise": noises["thumb"],
        "index_noise": noises["index"],
        "middle_noise": noises["middle"],
        "ring_noise": noises["ring"],
        "pinky_noise": noises["pinky"],
    }


def _aggregate_scale(scale: float, rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    attempted = sum(int(row["attempted_episodes"]) for row in rows)
    written = sum(int(row["written_episodes"]) for row in rows)
    transitions = sum(int(row["transitions"]) for row in rows)
    episodes = sum(int(row["episodes"]) for row in rows)
    delta_count = sum(int(row["noise_action_delta_count"]) for row in rows)
    l2_sum = sum(float(row["noise_action_delta_l2_sum"]) for row in rows)
    l2_sq_sum = sum(float(row["noise_action_delta_l2_sq_sum"]) for row in rows)
    linf_weighted_sum = sum(
        float(row["noise_action_delta_linf_mean"]) * int(row["noise_action_delta_count"])
        for row in rows
    )
    pickup_attempted = sum(int(row["pickup_timing_attempted_episodes"]) for row in rows)
    pickup_gate_count = sum(int(row["pickup_gate_first_step_count"]) for row in rows)
    pickup_gate_sum = sum(float(row["pickup_gate_first_step_sum"]) for row in rows)
    pickup_hold_count = sum(int(row["pickup_hold_first_step_count"]) for row in rows)
    pickup_hold_sum = sum(float(row["pickup_hold_first_step_sum"]) for row in rows)

    mean_l2 = l2_sum / max(delta_count, 1)
    mean_l2_sq = l2_sq_sum / max(delta_count, 1)
    std_l2 = max(mean_l2_sq - mean_l2 * mean_l2, 0.0) ** 0.5
    mean_linf = linf_weighted_sum / max(delta_count, 1)

    exemplar = rows[0]
    return {
        "scale": scale,
        "objects": len(rows),
        "attempted_episodes": attempted,
        "written_episodes": written,
        "write_success_rate": written / max(attempted, 1),
        "transitions": transitions,
        "episodes": episodes,
        "mean_episode_len": transitions / max(episodes, 1),
        "noise_action_delta_count": delta_count,
        "noise_action_delta_l2_mean": mean_l2,
        "noise_action_delta_l2_std": std_l2,
        "noise_action_delta_linf_mean": mean_linf,
        "pickup_timing_attempted_episodes": pickup_attempted,
        "pickup_gate_first_step_count": pickup_gate_count,
        "pickup_gate_first_step_mean": (
            pickup_gate_sum / pickup_gate_count if pickup_gate_count > 0 else -1.0
        ),
        "pickup_gate_trigger_rate": pickup_gate_count / max(pickup_attempted, 1),
        "pickup_hold_first_step_count": pickup_hold_count,
        "pickup_hold_first_step_mean": (
            pickup_hold_sum / pickup_hold_count if pickup_hold_count > 0 else -1.0
        ),
        "pickup_hold_trigger_rate": pickup_hold_count / max(pickup_attempted, 1),
        "arm_base_noise": exemplar["arm_base_noise"],
        "arm_wrist_noise": exemplar["arm_wrist_noise"],
        "thumb_noise": exemplar["thumb_noise"],
        "index_noise": exemplar["index_noise"],
        "middle_noise": exemplar["middle_noise"],
        "ring_noise": exemplar["ring_noise"],
        "pinky_noise": exemplar["pinky_noise"],
    }


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-type",
        choices=("pickup", "pick_place", "pick_place_release"),
        default="pick_place",
    )
    parser.add_argument("--split", choices=("train", "ood"), default="train")
    parser.add_argument(
        "--object-names",
        type=str,
        default="",
        help="Optional comma-separated override instead of --split.",
    )
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--variant", choices=("noisy_clean", "noisy_noisy"), default="noisy_clean")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--attempted-episodes-per-object", type=int, default=256)
    parser.add_argument(
        "--target-transitions-per-object",
        type=int,
        default=1_000_000,
        help="Keep this large so the sweep stops on attempted episodes, not transition target.",
    )
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--xy-range", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pickup-success-hold-steps", type=int, default=5)
    parser.add_argument("--ou-theta", type=float, default=0.15)
    parser.add_argument("--ou-mu", type=float, default=0.0)
    parser.add_argument("--ou-dt", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--scales",
        type=str,
        default=",".join(str(scale) for scale in DEFAULT_SCALES),
        help="Comma-separated multipliers applied to the current groupwise noise defaults.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/noise_sweeps"),
    )
    args = parser.parse_args()

    specs = _select_specs(args)
    scales = _parse_scales(args.scales)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "collection_type": args.collection_type,
        "split": args.split,
        "object_names": args.object_names,
        "max_objects": args.max_objects,
        "variant": args.variant,
        "num_envs": args.num_envs,
        "attempted_episodes_per_object": args.attempted_episodes_per_object,
        "target_transitions_per_object": args.target_transitions_per_object,
        "horizon": args.horizon,
        "xy_range": args.xy_range,
        "seed": args.seed,
        "pickup_success_hold_steps": args.pickup_success_hold_steps,
        "ou_theta": args.ou_theta,
        "ou_mu": args.ou_mu,
        "ou_dt": args.ou_dt,
        "scales": scales,
        "default_noises": BASE_NOISE_DEFAULTS,
        "objects": [asdict(spec) for spec in specs],
    }
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    per_object_rows: List[Dict[str, float]] = []
    scale_rows: List[Dict[str, float]] = []

    print(
        f"[noise-sweep] collection_type={args.collection_type} variant={args.variant} scales={scales} "
        f"objects={[spec.object_name for spec in specs]}",
        flush=True,
    )

    for scale in scales:
        scale_dir = args.output_dir / f"scale_{str(scale).replace('.', 'p')}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        scale_run_rows: List[Dict[str, float]] = []
        for spec in specs:
            row = _run_one(args, spec, scale, scale_dir)
            per_object_rows.append(row)
            scale_run_rows.append(row)

        scale_summary = _aggregate_scale(scale, scale_run_rows)
        scale_rows.append(scale_summary)
        print(
            "[noise-sweep] "
            f"scale={scale:g} arm_base={scale_summary['arm_base_noise']:.3f} "
            f"write_success_rate={scale_summary['write_success_rate']:.1%} "
            f"mean_action_delta_l2={scale_summary['noise_action_delta_l2_mean']:.4f} "
            f"mean_pickup_gate_step={scale_summary['pickup_gate_first_step_mean']:.1f} "
            f"mean_pickup_hold_step={scale_summary['pickup_hold_first_step_mean']:.1f} "
            f"mean_episode_len={scale_summary['mean_episode_len']:.1f}",
            flush=True,
        )

    _write_csv(args.output_dir / "per_object_results.csv", per_object_rows)
    _write_csv(args.output_dir / "scale_summary.csv", scale_rows)
    with open(args.output_dir / "scale_summary.json", "w") as f:
        json.dump(scale_rows, f, indent=2)

    print("\n[noise-sweep] summary", flush=True)
    for row in scale_rows:
        print(
            f"[noise-sweep] scale={row['scale']:g} "
            f"arm_base={row['arm_base_noise']:.3f} "
            f"write_success_rate={row['write_success_rate']:.1%} "
            f"mean_action_delta_l2={row['noise_action_delta_l2_mean']:.4f} "
            f"mean_action_delta_linf={row['noise_action_delta_linf_mean']:.4f} "
            f"mean_pickup_gate_step={row['pickup_gate_first_step_mean']:.1f} "
            f"mean_pickup_hold_step={row['pickup_hold_first_step_mean']:.1f} "
            f"mean_episode_len={row['mean_episode_len']:.1f}",
            flush=True,
        )
    print(f"[noise-sweep] wrote outputs under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
