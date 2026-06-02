import json

import pytest

from blender_eval.aggregate_eval_results import aggregate_results


def _write_result(path, spec, attempted, succeeded):
  path.write_text(json.dumps({
    "object_id": spec.object_id,
    "category_id": spec.category_id,
    "object_name": spec.object_name,
    "object_category": spec.object_category,
    "task_name": spec.task_name,
    "renderer": "blender",
    "blend_file": "/repo/assets/blender/templates/simtool_lab.blend",
    "engine": "cycles",
    "samples": 96,
    "cycles_device": "gpu",
    "render_width": 512,
    "render_height": 384,
    "attempted": attempted,
    "succeeded": succeeded,
    "success_rate": succeeded / max(attempted, 1),
  }))


def test_aggregate_train_results_preserves_registry_order(tmp_path):
  from stage5_multi_object_driver import _split

  result_dir = tmp_path / "object_results"
  result_dir.mkdir()
  specs = _split("train")
  for i, spec in enumerate(reversed(specs)):
    _write_result(result_dir / f"{spec.object_name}.json", spec, attempted=2, succeeded=i % 2)

  output_json = tmp_path / "photoreal_train.json"
  summary = aggregate_results(
    split="train",
    checkpoint=tmp_path / "checkpoint.ckpt",
    result_dir=result_dir,
    output_json=output_json,
    episodes_per_object=2,
    num_envs=1,
    xy_range=0.1,
    horizon=250,
    video_dir=tmp_path / "photoreal_train_videos",
  )

  assert output_json.exists()
  assert summary["split"] == "train"
  assert summary["renderer"] == "blender"
  assert summary["engine"] == "cycles"
  assert summary["samples"] == 96
  assert summary["cycles_device"] == "gpu"
  assert summary["total_attempted"] == 2 * len(specs)
  assert [r["object_name"] for r in summary["per_object"]] == [
    spec.object_name for spec in specs
  ]


def test_aggregate_results_errors_on_missing_object(tmp_path):
  result_dir = tmp_path / "object_results"
  result_dir.mkdir()

  with pytest.raises(FileNotFoundError, match="Missing results"):
    aggregate_results(
      split="ood",
      checkpoint=tmp_path / "checkpoint.ckpt",
      result_dir=result_dir,
      output_json=tmp_path / "photoreal_ood.json",
    )


def test_aggregate_pick_place_release_totals(tmp_path, monkeypatch):
  from stage5_multi_object_driver import ObjectSpec

  spec = ObjectSpec(4, 2, "hammer", "claw_hammer", "swing_down")
  result_dir = tmp_path / "object_results"
  result_dir.mkdir()
  # Pose-based (diffusion-fair) per-object worker schema: 4 attempted, 1 a
  # degenerate spawn (invalid_start) -> 3 valid; 2 release successes over the
  # 3 valid spawns; funnel + failure breakdown carried through.
  (result_dir / "claw_hammer.json").write_text(json.dumps({
    "object_id": spec.object_id,
    "category_id": spec.category_id,
    "object_name": spec.object_name,
    "object_category": spec.object_category,
    "task_name": spec.task_name,
    "eval_task": "pick_place_release",
    "renderer": "blender",
    "attempted": 4,
    "valid_attempted": 3,
    "invalid_start": 1,
    "release_succeeded": 2,
    "lifted": 3,
    "transported": 3,
    "placed_near_goal": 2,
    "release_stable": 2,
    "release_success_rate": 2 / 3,
    "failure_breakdown": {"invalid_start": 1, "no_place": 1},
    "release_xy_tolerance": 0.06,
    "min_lift_height": 0.10,
  }))
  monkeypatch.setattr(
    "blender_eval.aggregate_eval_results._split",
    lambda split: [spec],
  )

  summary = aggregate_results(
    split="train",
    checkpoint=tmp_path / "checkpoint.ckpt",
    result_dir=result_dir,
    output_json=tmp_path / "photoreal_ppr.json",
  )

  assert summary["eval_task"] == "pick_place_release"
  assert summary["total_attempted"] == 4
  assert summary["total_valid_attempted"] == 3
  assert summary["total_invalid_start"] == 1
  assert summary["total_release_succeeded"] == 2
  assert summary["total_lifted"] == 3
  assert summary["total_transported"] == 3
  assert summary["total_placed_near_goal"] == 2
  assert summary["total_release_stable"] == 2
  assert summary["overall_release_success_rate"] == 2 / 3
  assert summary["failure_breakdown"] == {"invalid_start": 1, "no_place": 1}
  # Headline is over valid spawns, not all attempts.
  assert "total_succeeded" not in summary
  assert summary["release_xy_tolerance"] == 0.06
