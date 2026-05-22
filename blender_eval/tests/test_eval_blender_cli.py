import json
import os
from pathlib import Path
from types import SimpleNamespace

from blender_eval.eval_blender import (
  DEFAULT_BLEND_FILE,
  LIGHTING_PRESET_ENV,
  _apply_driver_defaults,
  _apply_photoreal_defaults,
  run_driver,
)
from stage5_multi_object_driver import ObjectSpec


def test_blender_eval_defaults_to_master_template():
  args = SimpleNamespace(renderer="blender", blend_file=None, lighting_preset=None)

  _apply_photoreal_defaults(args)

  assert args.blend_file == DEFAULT_BLEND_FILE
  assert args.blend_file.exists()


def test_blender_eval_lighting_preset_sets_child_env(monkeypatch):
  monkeypatch.delenv(LIGHTING_PRESET_ENV, raising=False)
  custom_template = Path("/tmp/custom_scene.blend")
  args = SimpleNamespace(
    renderer="blender",
    blend_file=custom_template,
    lighting_preset="softbox_grid",
  )

  _apply_photoreal_defaults(args)

  assert args.blend_file == custom_template
  assert os.environ[LIGHTING_PRESET_ENV] == "softbox_grid"


def test_driver_defaults_match_eval_video_dir_layout(tmp_path):
  args = SimpleNamespace(
    output_json=tmp_path / "photoreal_eval.json",
    video_dir=None,
    max_success_previews=2,
    max_failure_previews=2,
  )

  _apply_driver_defaults(args)

  assert args.video_dir == tmp_path / "photoreal_eval_videos"


def test_driver_summary_uses_worker_render_size_when_unset(tmp_path, monkeypatch):
  spec = ObjectSpec(4, 2, "hammer", "claw_hammer", "swing_down")

  def fake_run_one(worker_spec, args, result_path):
    assert worker_spec == spec
    return {
      "object_id": spec.object_id,
      "category_id": spec.category_id,
      "object_name": spec.object_name,
      "object_category": spec.object_category,
      "task_name": spec.task_name,
      "renderer": "blender",
      "blend_file": str(DEFAULT_BLEND_FILE),
      "engine": "cycles",
      "samples": 96,
      "cycles_device": "gpu",
      "render_width": 512,
      "render_height": 384,
      "attempted": 1,
      "succeeded": 0,
      "success_rate": 0.0,
    }

  monkeypatch.setattr("blender_eval.eval_blender._split", lambda split: [spec])
  monkeypatch.setattr("blender_eval.eval_blender._run_one", fake_run_one)

  args = SimpleNamespace(
    checkpoint=tmp_path / "checkpoint.ckpt",
    renderer="blender",
    blend_file=DEFAULT_BLEND_FILE,
    lighting_preset=None,
    engine="cycles",
    samples=96,
    cycles_device="gpu",
    split="train",
    episodes_per_object=1,
    num_envs=1,
    xy_range=0.1,
    horizon=250,
    render_width=None,
    render_height=None,
    output_json=tmp_path / "photoreal_eval.json",
    video_dir=None,
    max_success_previews=0,
    max_failure_previews=0,
    gif_fps=10,
    device="cuda:0",
  )

  run_driver(args)

  summary = json.loads(args.output_json.read_text())
  assert summary["render_width"] == 512
  assert summary["render_height"] == 384
  assert summary["total_attempted"] == 1
