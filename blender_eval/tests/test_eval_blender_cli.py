import json
import os
from pathlib import Path
from types import SimpleNamespace

from blender_eval.eval_blender import (
  DEFAULT_BLEND_FILE,
  DEFAULT_PICK_PLACE_RELEASE_HORIZON,
  DEFAULT_PICKUP_HORIZON,
  LIGHTING_PRESET_ENV,
  _apply_driver_defaults,
  _apply_photoreal_defaults,
  _record_release_failure,
  _run_one,
  run_driver,
  run_worker,
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


def test_driver_defaults_set_task_specific_horizon(tmp_path):
  pickup_args = SimpleNamespace(
    eval_task="pickup",
    horizon=None,
    output_json=tmp_path / "pickup.json",
    video_dir=None,
    max_success_previews=0,
    max_failure_previews=0,
  )
  ppr_args = SimpleNamespace(
    eval_task="pick_place_release",
    horizon=None,
    output_json=tmp_path / "ppr.json",
    video_dir=None,
    max_success_previews=0,
    max_failure_previews=0,
  )

  _apply_driver_defaults(pickup_args)
  _apply_driver_defaults(ppr_args)

  assert pickup_args.horizon == DEFAULT_PICKUP_HORIZON
  assert ppr_args.horizon == DEFAULT_PICK_PLACE_RELEASE_HORIZON


def test_run_worker_dispatches_pick_place_release_before_pickup_imports(monkeypatch):
  args = SimpleNamespace(eval_task="pick_place_release")
  calls = []

  def fake_ppr_worker(worker_args):
    calls.append(worker_args)

  monkeypatch.setattr("blender_eval.eval_blender.run_pick_place_release_worker", fake_ppr_worker)

  run_worker(args)

  assert calls == [args]


def test_driver_applies_photoreal_defaults_and_summary_shape(tmp_path, monkeypatch):
  spec = ObjectSpec(4, 2, "hammer", "claw_hammer", "swing_down")

  def fake_run_one(worker_spec, args, result_path):
    assert worker_spec == spec
    assert args.blend_file == DEFAULT_BLEND_FILE
    assert args.video_dir == tmp_path / "photoreal_eval_videos"
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
    blend_file=None,
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
    max_success_previews=2,
    max_failure_previews=2,
    gif_fps=10,
    device="cuda:0",
  )

  run_driver(args)

  summary = json.loads(args.output_json.read_text())
  assert summary["render_width"] == 512
  assert summary["render_height"] == 384
  assert summary["blend_file"] == str(DEFAULT_BLEND_FILE)
  assert summary["video_dir"] == str(tmp_path / "photoreal_eval_videos")
  assert summary["total_attempted"] == 1


def test_run_one_passes_blender_worker_args_and_video_subdir(tmp_path, monkeypatch):
  spec = ObjectSpec(4, 2, "hammer", "claw_hammer", "swing_down")
  result_path = tmp_path / "result.json"
  calls = []

  def fake_run(cmd, check, env):
    calls.append((cmd, check, env))
    result_path.write_text(json.dumps({
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
      "succeeded": 1,
      "success_rate": 1.0,
    }))

  monkeypatch.setattr("blender_eval.eval_blender.subprocess.run", fake_run)

  args = SimpleNamespace(
    checkpoint=tmp_path / "checkpoint.ckpt",
    renderer="blender",
    blend_file=DEFAULT_BLEND_FILE,
    lighting_preset="softbox_grid",
    engine="cycles",
    samples=96,
    cycles_device="gpu",
    blender="/opt/blender/blender",
    num_envs=1,
    episodes_per_object=1,
    horizon=250,
    xy_range=0.1,
    start_z_offset=0.015,
    seed=10,
    device="cuda:0",
    result_json=result_path,
    max_success_previews=2,
    max_failure_previews=2,
    gif_fps=10,
    render_width=None,
    render_height=None,
    video_dir=tmp_path / "videos",
  )

  result = _run_one(spec, args, result_path)

  cmd, check, env = calls[0]
  assert check is True
  assert "--worker" in cmd
  assert cmd[cmd.index("--renderer") + 1] == "blender"
  assert cmd[cmd.index("--engine") + 1] == "cycles"
  assert cmd[cmd.index("--samples") + 1] == "96"
  assert cmd[cmd.index("--cycles-device") + 1] == "gpu"
  assert cmd[cmd.index("--blender") + 1] == "/opt/blender/blender"
  assert cmd[cmd.index("--blend-file") + 1] == str(DEFAULT_BLEND_FILE)
  assert cmd[cmd.index("--lighting-preset") + 1] == "softbox_grid"
  assert cmd[cmd.index("--video-dir") + 1] == str(tmp_path / "videos" / "claw_hammer")
  assert cmd[cmd.index("--start-z-offset") + 1] == "0.015"
  assert cmd[cmd.index("--seed") + 1] == "14"
  assert "LD_LIBRARY_PATH" in env
  assert result["renderer"] == "blender"


def test_run_one_passes_pick_place_release_worker_args(tmp_path, monkeypatch):
  spec = ObjectSpec(4, 2, "hammer", "claw_hammer", "swing_down")
  result_path = tmp_path / "result.json"
  calls = []

  def fake_run(cmd, check, env):
    calls.append((cmd, check, env))
    result_path.write_text(json.dumps({
      "object_id": spec.object_id,
      "category_id": spec.category_id,
      "object_name": spec.object_name,
      "object_category": spec.object_category,
      "task_name": spec.task_name,
      "eval_task": "pick_place_release",
      "renderer": "stub",
      "attempted": 1,
      "succeeded": 1,
      "success_rate": 1.0,
      "pick_place_succeeded": 1,
      "release_goal_succeeded": 1,
      "stable_succeeded": 1,
    }))

  monkeypatch.setattr("blender_eval.eval_blender.subprocess.run", fake_run)

  args = SimpleNamespace(
    checkpoint=tmp_path / "checkpoint.ckpt",
    renderer="stub",
    eval_task="pick_place_release",
    blend_file=None,
    lighting_preset=None,
    engine="cycles",
    samples=96,
    cycles_device="gpu",
    blender="blender",
    num_envs=1,
    episodes_per_object=1,
    horizon=325,
    xy_range=0.1,
    start_z_offset=0.0,
    seed=10,
    device="cuda:0",
    result_json=result_path,
    max_success_previews=0,
    max_failure_previews=0,
    gif_fps=10,
    pickup_success_hold_steps=5,
    render_width=None,
    render_height=None,
    video_dir=None,
    lift_height=0.2,
    place_height=0.02,
    place_hold_goals=10,
    release_steps=45,
    release_arm_mode="hold",
    release_hand_blend=1.0,
    release_xy_tolerance=0.05,
    release_z_tolerance=0.04,
    release_speed_tolerance=0.25,
    table_x_half_extent=0.3,
    table_x_inset_margin=0.06,
    table_y_half_extent=0.2,
    table_y_inset_margin=0.04,
    place_goal_x_margin=0.10,
    place_goal_y_margin=0.06,
    min_effective_transport=0.05,
  )

  result = _run_one(spec, args, result_path)

  cmd, check, _ = calls[0]
  assert check is True
  assert cmd[cmd.index("--eval-task") + 1] == "pick_place_release"
  assert cmd[cmd.index("--horizon") + 1] == "325"
  assert cmd[cmd.index("--release-steps") + 1] == "45"
  assert cmd[cmd.index("--release-arm-mode") + 1] == "hold"
  assert cmd[cmd.index("--release-hand-blend") + 1] == "1.0"
  assert cmd[cmd.index("--start-z-offset") + 1] == "0.0"
  assert cmd[cmd.index("--seed") + 1] == "14"
  assert result["eval_task"] == "pick_place_release"


def test_release_failure_breakdown_counts_drop_reattempt_once():
  failure_breakdown = {"drop_reattempt": 0}

  _record_release_failure(
    failure_breakdown,
    release_success=False,
    failure_stage="drop_reattempt",
  )

  assert failure_breakdown["drop_reattempt"] == 1
