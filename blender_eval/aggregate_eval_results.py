#!/usr/bin/env python3
"""Aggregate per-object eval worker JSONs into eval_blender.py's driver format."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from stage5_multi_object_driver import _split  # noqa: E402


def _first_non_none(per_object: List[dict], key: str):
  for result in per_object:
    value = result.get(key)
    if value is not None:
      return value
  return None


def _load_result_map(result_dir: Path, result_glob: str) -> Dict[str, dict]:
  by_name: Dict[str, dict] = {}
  for path in sorted(result_dir.glob(result_glob)):
    result = json.loads(path.read_text())
    object_name = result.get("object_name")
    if not object_name:
      raise ValueError(f"{path} has no object_name")
    if object_name in by_name:
      raise ValueError(f"Duplicate result for object {object_name!r} in {result_dir}")
    by_name[object_name] = result
  return by_name


def aggregate_results(
  *,
  split: str,
  checkpoint: Path,
  result_dir: Path,
  output_json: Path,
  result_glob: str = "*.json",
  renderer: Optional[str] = None,
  blend_file: Optional[Path] = None,
  lighting_preset: Optional[str] = None,
  engine: Optional[str] = None,
  samples: Optional[int] = None,
  cycles_device: Optional[str] = None,
  episodes_per_object: Optional[int] = None,
  num_envs: Optional[int] = None,
  xy_range: Optional[float] = None,
  horizon: Optional[int] = None,
  render_width: Optional[int] = None,
  render_height: Optional[int] = None,
  video_dir: Optional[Path] = None,
  eval_task: Optional[str] = None,
) -> dict:
  specs = _split(split)
  by_name = _load_result_map(result_dir, result_glob)
  missing = [spec.object_name for spec in specs if spec.object_name not in by_name]
  if missing:
    raise FileNotFoundError(
      f"Missing results for split={split}: {missing}. "
      f"Searched {result_dir / result_glob}"
    )

  per_object = [by_name[spec.object_name] for spec in specs]
  total_attempted = sum(int(r.get("attempted", 0)) for r in per_object)
  total_succeeded = sum(int(r.get("succeeded", 0)) for r in per_object)
  total_stable_succeeded = sum(int(r.get("stable_succeeded", 0)) for r in per_object)
  total_pick_place_succeeded = sum(int(r.get("pick_place_succeeded", 0)) for r in per_object)
  total_release_goal_succeeded = sum(int(r.get("release_goal_succeeded", 0)) for r in per_object)
  total_release_stable_succeeded = sum(
    int(r.get("release_stable_succeeded", 0)) for r in per_object)
  overall = total_succeeded / max(total_attempted, 1)
  stable_overall = total_stable_succeeded / max(total_attempted, 1)
  eval_task = eval_task if eval_task is not None else (_first_non_none(per_object, "eval_task") or "pickup")

  summary = {
    "checkpoint": str(checkpoint),
    "eval_task": eval_task,
    "renderer": renderer if renderer is not None else _first_non_none(per_object, "renderer"),
    "blend_file": (
      str(blend_file) if blend_file is not None else _first_non_none(per_object, "blend_file")
    ),
    "lighting_preset": (
      lighting_preset if lighting_preset is not None
      else _first_non_none(per_object, "lighting_preset")
    ),
    "engine": engine if engine is not None else _first_non_none(per_object, "engine"),
    "samples": samples if samples is not None else _first_non_none(per_object, "samples"),
    "cycles_device": (
      cycles_device if cycles_device is not None
      else _first_non_none(per_object, "cycles_device")
    ),
    "split": split,
    "episodes_per_object": episodes_per_object,
    "num_envs": num_envs,
    "xy_range": xy_range,
    "horizon": horizon,
    "render_width": (
      render_width if render_width is not None
      else _first_non_none(per_object, "render_width")
    ),
    "render_height": (
      render_height if render_height is not None
      else _first_non_none(per_object, "render_height")
    ),
    "video_dir": str(video_dir) if video_dir is not None else None,
    "overall_success_rate": overall,
    "stable_success_rate": stable_overall,
    "total_attempted": total_attempted,
    "total_succeeded": total_succeeded,
    "total_stable_succeeded": total_stable_succeeded,
    "total_pick_place_succeeded": total_pick_place_succeeded,
    "total_release_goal_succeeded": total_release_goal_succeeded,
    "total_release_stable_succeeded": total_release_stable_succeeded,
    "pick_place_success_rate": (
      total_pick_place_succeeded / max(total_attempted, 1)
      if eval_task == "pick_place_release" else None
    ),
    "release_goal_success_rate": (
      total_release_goal_succeeded / max(total_attempted, 1)
      if eval_task == "pick_place_release" else None
    ),
    "pickup_success_hold_steps": _first_non_none(per_object, "pickup_success_hold_steps"),
    "per_object": per_object,
  }
  output_json.parent.mkdir(parents=True, exist_ok=True)
  output_json.write_text(json.dumps(summary, indent=2))
  return summary


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--split", choices=("train", "ood"), required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--result-dir", type=Path, required=True)
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--result-glob", default="*.json")
  parser.add_argument("--renderer", default="blender")
  parser.add_argument("--blend-file", type=Path, default=None)
  parser.add_argument("--lighting-preset", default=None)
  parser.add_argument("--engine", default="cycles")
  parser.add_argument("--samples", type=int, default=None)
  parser.add_argument("--cycles-device", default=None)
  parser.add_argument("--episodes-per-object", type=int, default=None)
  parser.add_argument("--num-envs", type=int, default=None)
  parser.add_argument("--xy-range", type=float, default=None)
  parser.add_argument("--horizon", type=int, default=None)
  parser.add_argument("--render-width", type=int, default=None)
  parser.add_argument("--render-height", type=int, default=None)
  parser.add_argument("--video-dir", type=Path, default=None)
  parser.add_argument("--eval-task", choices=("pickup", "pick_place_release"), default=None)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  summary = aggregate_results(
    split=args.split,
    checkpoint=args.checkpoint,
    result_dir=args.result_dir,
    output_json=args.output_json,
    result_glob=args.result_glob,
    renderer=args.renderer,
    blend_file=args.blend_file,
    lighting_preset=args.lighting_preset,
    engine=args.engine,
    samples=args.samples,
    cycles_device=args.cycles_device,
    episodes_per_object=args.episodes_per_object,
    num_envs=args.num_envs,
    xy_range=args.xy_range,
    horizon=args.horizon,
    render_width=args.render_width,
    render_height=args.render_height,
    video_dir=args.video_dir,
    eval_task=args.eval_task,
  )
  print(
    f"[aggregate-eval] {summary['split']} "
    f"{summary['total_succeeded']}/{summary['total_attempted']} = "
    f"{summary['overall_success_rate']:.1%} "
    f"stable={summary['total_stable_succeeded']}/{summary['total_attempted']} = "
    f"{summary['stable_success_rate']:.1%}"
  )
  print(f"[aggregate-eval] saved {args.output_json}")


if __name__ == "__main__":
  main()
