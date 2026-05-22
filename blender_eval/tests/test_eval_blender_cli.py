import os
from pathlib import Path
from types import SimpleNamespace

from blender_eval.eval_blender import (
  DEFAULT_BLEND_FILE,
  LIGHTING_PRESET_ENV,
  _apply_driver_defaults,
  _apply_photoreal_defaults,
)


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
