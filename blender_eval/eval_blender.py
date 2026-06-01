#!/usr/bin/env python3
"""Closed-loop evaluation with swappable renderer (stub, IsaacGym, or Blender).

Mirrors scripts/eval_diffusion_policy.py with one change: the image source is
a Renderer instance instead of env.render_dataset_camera_rgb(). Everything else
— action/obs timing, history management, success criterion — is identical to
ensure A/B parity when using --renderer isaacgym.

Usage (driver):
    python blender_eval/eval_blender.py \\
        --checkpoint /path/to/checkpoint.ckpt \\
        --renderer blender \\
        --split train \\
        --episodes-per-object 32 \\
        --output-json data/blender_eval/eval.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

# Re-use the stage5 registry/split logic.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BLEND_FILE = REPO_ROOT / "assets" / "blender" / "templates" / "simtool_lab.blend"
LIGHTING_PRESET_ENV = "SIMTOOLDIFF_LIGHTING_PRESET"
DEFAULT_PICKUP_SUCCESS_HOLD_STEPS = 5
DEFAULT_PICKUP_HORIZON = 250
DEFAULT_PICK_PLACE_RELEASE_HORIZON = 325
DEFAULT_RELEASE_STEPS = 45
DEFAULT_LIFT_HEIGHT_M = 0.20
DEFAULT_PLACE_HEIGHT_M = 0.02
DEFAULT_PLACE_HOLD_GOALS = 10
DEFAULT_RELEASE_XY_TOLERANCE_M = 0.05
DEFAULT_RELEASE_Z_TOLERANCE_M = 0.04
DEFAULT_RELEASE_SPEED_TOLERANCE_MPS = 0.25
DEFAULT_LONG_TABLE_X_HALF_EXTENT_M = 0.60 / 2.0
DEFAULT_LONG_TABLE_X_INSET_MARGIN_M = 0.06
DEFAULT_TABLE_Y_HALF_EXTENT_M = 0.4 / 2.0
DEFAULT_TABLE_Y_INSET_MARGIN_M = 0.04
DEFAULT_PLACE_GOAL_X_MARGIN_M = 0.10
DEFAULT_PLACE_GOAL_Y_MARGIN_M = 0.06
DEFAULT_MIN_EFFECTIVE_TRANSPORT_M = 0.05
DEFAULT_START_Z_OFFSET = 0.0

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from stage5_multi_object_driver import _split, ObjectSpec  # noqa: E402


# ----------------------------- driver -----------------------------------------

def _first_non_none(per_object, key):
  for result in per_object:
    value = result.get(key)
    if value is not None:
      return value
  return None


def _run_one(spec: ObjectSpec, args, result_path: Path) -> dict:
  pickup_success_hold_steps = getattr(
    args, "pickup_success_hold_steps", DEFAULT_PICKUP_SUCCESS_HOLD_STEPS)
  eval_task = getattr(args, "eval_task", "pickup")
  cmd = [
    sys.executable, str(Path(__file__).resolve()),
    "--worker",
    "--checkpoint", str(args.checkpoint),
    "--renderer", args.renderer,
    "--eval-task", eval_task,
    "--object-category", spec.object_category,
    "--object-name", spec.object_name,
    "--task-name", spec.task_name,
    "--object-id", str(spec.object_id),
    "--category-id", str(spec.category_id),
    "--num-envs", str(args.num_envs),
    "--episodes-per-object", str(args.episodes_per_object),
    "--horizon", str(args.horizon),
    "--xy-range", str(args.xy_range),
    "--seed", str(args.seed + spec.object_id),
    "--device", args.device,
    "--result-json", str(result_path),
    "--max-success-previews", str(args.max_success_previews),
    "--max-failure-previews", str(args.max_failure_previews),
    "--gif-fps", str(args.gif_fps),
    "--pickup-success-hold-steps", str(pickup_success_hold_steps),
  ]
  if eval_task == "pick_place_release":
    cmd += [
      "--start-z-offset", str(args.start_z_offset),
      "--lift-height", str(args.lift_height),
      "--place-height", str(args.place_height),
      "--place-hold-goals", str(args.place_hold_goals),
      "--release-steps", str(args.release_steps),
      "--release-arm-mode", args.release_arm_mode,
      "--release-hand-blend", str(args.release_hand_blend),
      "--release-xy-tolerance", str(args.release_xy_tolerance),
      "--release-z-tolerance", str(args.release_z_tolerance),
      "--release-speed-tolerance", str(args.release_speed_tolerance),
      "--table-x-half-extent", str(args.table_x_half_extent),
      "--table-x-inset-margin", str(args.table_x_inset_margin),
      "--table-y-half-extent", str(args.table_y_half_extent),
      "--table-y-inset-margin", str(args.table_y_inset_margin),
      "--place-goal-x-margin", str(args.place_goal_x_margin),
      "--place-goal-y-margin", str(args.place_goal_y_margin),
      "--min-effective-transport", str(args.min_effective_transport),
    ]
  if args.render_width is not None:
    cmd += ["--render-width", str(args.render_width)]
  if args.render_height is not None:
    cmd += ["--render-height", str(args.render_height)]
  if args.renderer == "blender":
    cmd += [
      "--engine", args.engine,
      "--samples", str(args.samples),
      "--cycles-device", args.cycles_device,
      "--blender", args.blender,
    ]
    if args.blend_file is not None:
      cmd += ["--blend-file", str(args.blend_file)]
    if args.lighting_preset is not None:
      cmd += ["--lighting-preset", args.lighting_preset]
  if args.video_dir is not None:
    cmd += ["--video-dir", str(args.video_dir / spec.object_name)]

  print(
    f"\n[eval-driver] >>> object_id={spec.object_id} "
    f"{spec.object_category}/{spec.object_name} task={spec.task_name} "
    f"renderer={args.renderer} eval_task={eval_task}",
    flush=True,
  )
  env = os.environ.copy()
  conda_lib = str(Path(sys.prefix) / "lib")
  existing = env.get("LD_LIBRARY_PATH", "")
  env["LD_LIBRARY_PATH"] = f"{conda_lib}:{existing}" if existing else conda_lib
  subprocess.run(cmd, check=True, env=env)
  return json.loads(result_path.read_text())


def run_driver(args) -> None:
  _apply_photoreal_defaults(args)
  _apply_driver_defaults(args)

  specs = _split(args.split)
  eval_task = getattr(args, "eval_task", "pickup")
  print(
    f"[eval-driver] split={args.split} renderer={args.renderer} "
    f"eval_task={eval_task} objects={[s.object_name for s in specs]}",
    flush=True,
  )
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  if args.video_dir is not None:
    args.video_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval-driver] previews -> {args.video_dir}", flush=True)

  per_object = []
  with tempfile.TemporaryDirectory(prefix="eval_blender_") as tmpdir:
    for spec in specs:
      result_path = Path(tmpdir) / f"_result_{spec.object_name}.json"
      try:
        result = _run_one(spec, args, result_path)
      except subprocess.CalledProcessError as e:
        result = {
          "object_id": spec.object_id,
          "category_id": spec.category_id,
          "object_name": spec.object_name,
          "object_category": spec.object_category,
          "task_name": spec.task_name,
          "eval_task": eval_task,
          "renderer": args.renderer,
          "blend_file": str(args.blend_file) if args.blend_file is not None else None,
          "lighting_preset": args.lighting_preset,
          "engine": args.engine if args.renderer == "blender" else None,
          "samples": args.samples if args.renderer == "blender" else None,
          "cycles_device": args.cycles_device if args.renderer == "blender" else None,
          "render_width": args.render_width,
          "render_height": args.render_height,
          "attempted": 0,
          "succeeded": 0,
          "success_rate": None,
          "stable_succeeded": 0,
          "stable_success_rate": None,
          "error": f"Worker exited {e.returncode}",
        }
      per_object.append(result)

  total_attempted = sum(r.get("attempted", 0) for r in per_object)
  total_succeeded = sum(r.get("succeeded", 0) for r in per_object)
  total_stable_succeeded = sum(r.get("stable_succeeded", 0) for r in per_object)
  total_pick_place_succeeded = sum(r.get("pick_place_succeeded", 0) for r in per_object)
  total_release_goal_succeeded = sum(r.get("release_goal_succeeded", 0) for r in per_object)
  total_release_stable_succeeded = sum(
    r.get("release_stable_succeeded", 0) for r in per_object)
  overall = total_succeeded / max(total_attempted, 1)
  stable_overall = total_stable_succeeded / max(total_attempted, 1)

  summary = {
    "checkpoint": str(args.checkpoint),
    "eval_task": eval_task,
    "renderer": args.renderer,
    "blend_file": str(args.blend_file) if args.blend_file is not None else None,
    "lighting_preset": args.lighting_preset,
    "engine": args.engine if args.renderer == "blender" else None,
    "samples": args.samples if args.renderer == "blender" else None,
    "cycles_device": args.cycles_device if args.renderer == "blender" else None,
    "split": args.split,
    "episodes_per_object": args.episodes_per_object,
    "num_envs": args.num_envs,
    "xy_range": args.xy_range,
    "horizon": args.horizon,
    "render_width": (
      args.render_width if args.render_width is not None
      else _first_non_none(per_object, "render_width")
    ),
    "render_height": (
      args.render_height if args.render_height is not None
      else _first_non_none(per_object, "render_height")
    ),
    "video_dir": str(args.video_dir) if args.video_dir is not None else None,
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
    "pickup_success_hold_steps": getattr(
      args, "pickup_success_hold_steps", DEFAULT_PICKUP_SUCCESS_HOLD_STEPS),
    "per_object": per_object,
  }
  if eval_task == "pick_place_release":
    summary.update({
      "start_z_offset": args.start_z_offset,
      "lift_height": args.lift_height,
      "place_height": args.place_height,
      "place_hold_goals": args.place_hold_goals,
      "release_steps": args.release_steps,
      "release_arm_mode": args.release_arm_mode,
      "release_hand_blend": args.release_hand_blend,
      "release_xy_tolerance": args.release_xy_tolerance,
      "release_z_tolerance": args.release_z_tolerance,
      "release_speed_tolerance": args.release_speed_tolerance,
    })
  args.output_json.write_text(json.dumps(summary, indent=2))

  print(f"\n[eval-driver] renderer={args.renderer}", flush=True)
  print(
    f"[eval-driver] overall: {total_succeeded}/{total_attempted} = {overall:.1%}",
    flush=True,
  )
  print(
    f"[eval-driver] stable: {total_stable_succeeded}/{total_attempted} = "
    f"{stable_overall:.1%}",
    flush=True,
  )
  if eval_task == "pick_place_release":
    print(
      f"[eval-driver] pick_place: {total_pick_place_succeeded}/{total_attempted} = "
      f"{total_pick_place_succeeded / max(total_attempted, 1):.1%}",
      flush=True,
    )
    print(
      f"[eval-driver] release_goal: {total_release_goal_succeeded}/{total_attempted} = "
      f"{total_release_goal_succeeded / max(total_attempted, 1):.1%}",
      flush=True,
    )
  for r in per_object:
    if r.get("success_rate") is None:
      print(f"  - {r['object_name']:<22s} ERROR: {r.get('error')}")
    else:
      print(
        f"  - {r['object_name']:<22s} "
        f"{r['succeeded']:>4d}/{r['attempted']:>4d} = {r['success_rate']:.1%} "
        f"stable={r.get('stable_succeeded', 0):>4d}/{r['attempted']:>4d} "
        f"= {r.get('stable_success_rate', 0.0):.1%}"
      )
  print(f"[eval-driver] saved {args.output_json}", flush=True)


# ----------------------------- worker -----------------------------------------

def run_worker(args) -> None:
  if getattr(args, "eval_task", "pickup") == "pick_place_release":
    run_pick_place_release_worker(args)
    return

  # IsaacGym must be imported before torch.
  from isaacgym import gymapi  # noqa: F401
  import numpy as np
  import torch
  import torch.nn.functional as F
  import dill
  import hydra
  from omegaconf import OmegaConf

  import isaacgymenvs  # noqa: F401
  OmegaConf.register_new_resolver("eval", eval, replace=True)

  from stage5_collect_dataset import (
    _make_env,
    _load_nominal_start_pose,
    N_ACT,
    LIFT_HEIGHT_M,
  )
  from blender_eval.success_criteria import (
    PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M,
    PICKUP_SUCCESS_MIN_LIFT_M,
    compute_pickup_success_metrics,
  )
  from blender_eval.renderer_interface import StubRenderer, IsaacGymRenderer
  from blender_eval.blender_renderer import BlenderRenderer
  from blender_eval.asset_manifest import get_robot_mesh_manifest, get_object_mesh_path
  from blender_eval.state_extraction import extract_render_state

  device = args.device
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # --- Load policy from checkpoint ---
  ckpt_path = Path(args.checkpoint).expanduser().resolve()
  # Keep the full checkpoint payload on CPU. The epoch-50 checkpoint is ~4 GB
  # and includes tensors that are only needed for loading; putting the whole
  # payload on CUDA can exhaust 8 GB GPUs before IsaacGym even initializes.
  payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
  cfg = payload["cfg"]
  ws_cls = hydra.utils.get_class(cfg._target_)
  workspace = ws_cls(cfg, output_dir=tempfile.mkdtemp(prefix="dp_eval_"))
  workspace.load_payload(payload)
  policy = workspace.ema_model if cfg.training.use_ema and workspace.ema_model is not None else workspace.model
  policy.to(device).eval()
  if torch.cuda.is_available() and str(device).startswith("cuda"):
    torch.cuda.empty_cache()

  n_obs_steps = int(cfg.n_obs_steps)
  n_action_steps = int(cfg.n_action_steps)
  state_dim = int(cfg.task.state_dim)
  image_shape = tuple(int(x) for x in cfg.task.image_shape)
  target_h, target_w = image_shape[1], image_shape[2]
  print(
    f"[eval-worker] policy: To={n_obs_steps} k={n_action_steps} "
    f"state_dim={state_dim} image={image_shape} renderer={args.renderer}",
    flush=True,
  )

  # --- Build env ---
  nominal_start_pose = _load_nominal_start_pose(
    args.object_category, args.object_name, args.task_name
  )
  nominal_goal_pose = list(nominal_start_pose)
  nominal_goal_pose[2] += LIFT_HEIGHT_M
  nominal_start_z = float(nominal_start_pose[2])
  goal_z = float(nominal_goal_pose[2])

  print("[eval-worker] creating env...", flush=True)
  env = _make_env(
    num_envs=args.num_envs,
    nominal_start_pose=nominal_start_pose,
    nominal_goal_pose=nominal_goal_pose,
    xy_range=args.xy_range,
    horizon=args.horizon,
    headless=True,
    device=device,
    seed=args.seed,
    object_name=args.object_name,
  )
  nominal_init_xy = env.object_init_state[:, 0:2].detach().clone()

  def randomize_object_init_xy(env_id_list):
    if not env_id_list:
      return
    env_ids = torch.tensor(env_id_list, device=env.object_init_state.device, dtype=torch.long)
    deltas = (
      torch.rand((len(env_id_list), 2), device=env.object_init_state.device,
                  dtype=env.object_init_state.dtype) * 2.0 - 1.0
    ) * float(args.xy_range)
    env.object_init_state[env_ids, 0:2] = nominal_init_xy[env_ids] + deltas

  randomize_object_init_xy(list(range(env.num_envs)))
  env.gym.refresh_actor_root_state_tensor(env.sim)

  active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)

  # --- Set up renderer (all return (B, H, W, 3) uint8 batches) ---
  render_w = args.render_width or 512
  render_h = args.render_height or 384
  mesh_manifest = None
  _blender_tmpdir = None
  if args.renderer == "blender" and args.lighting_preset is not None:
    os.environ[LIGHTING_PRESET_ENV] = args.lighting_preset

  if args.renderer == "stub":
    renderer = StubRenderer(env.num_envs, render_w, render_h)
  elif args.renderer == "isaacgym":
    renderer = IsaacGymRenderer(env, active_envs)
  elif args.renderer == "blender":
    mesh_manifest = get_robot_mesh_manifest()
    tool_mesh_path = str(get_object_mesh_path(args.object_name))
    from blender_eval.camera_params import isaacgym_to_blender_camera
    from stage5_collect_dataset import (
      DATASET_CAMERA_POSITION, DATASET_CAMERA_TARGET,
      DATASET_CAMERA_HORIZONTAL_FOV,
    )

    _blender_tmpdir = tempfile.mkdtemp(prefix="blender_cfg_")
    _manifest_dict = {
      name: {
        "mesh_path": str(info.mesh_path),
        "material_rgba": list(info.material_rgba) if info.material_rgba else None,
      }
      for name, info in mesh_manifest.items()
    }
    _manifest_path = os.path.join(_blender_tmpdir, "manifest.json")
    with open(_manifest_path, "w") as f:
      json.dump(_manifest_dict, f)

    cam_params = isaacgym_to_blender_camera(
      DATASET_CAMERA_HORIZONTAL_FOV, render_w, render_h,
      DATASET_CAMERA_POSITION, DATASET_CAMERA_TARGET,
    )
    _camera_path = os.path.join(_blender_tmpdir, "camera.json")
    with open(_camera_path, "w") as f:
      json.dump({
        "focal_length_mm": cam_params.focal_length_mm,
        "sensor_width_mm": cam_params.sensor_width_mm,
        "sensor_height_mm": cam_params.sensor_height_mm,
        "location": list(cam_params.location),
        "rotation_quaternion_wxyz": list(cam_params.rotation_quaternion_wxyz),
      }, f)

    renderer = BlenderRenderer(
      num_envs=env.num_envs,
      width=render_w,
      height=render_h,
      manifest_path=_manifest_path,
      camera_path=_camera_path,
      tool_mesh_path=tool_mesh_path,
      engine=args.engine,
      samples=args.samples,
      cycles_device=args.cycles_device,
      blend_file=str(args.blend_file) if args.blend_file is not None else None,
      blender_executable=args.blender,
    )
  else:
    raise ValueError(f"Unknown renderer: {args.renderer}")

  # mesh_manifest is only needed for renderers that use pose state (blender).
  # stub and isaacgym renderers ignore it — keep it None to avoid unnecessary
  # state extraction overhead and body-name dependency in stub mode.

  # --- Prime obs ---
  zero_action = torch.zeros((env.num_envs, N_ACT), device=device)
  obs_dict, _, _, _ = env.step(zero_action)
  obs = obs_dict["obs"]

  def render_step():
    """Return (native_uint8_np, normalized_resized_tensor).

    Timing contract: called AFTER env.step(), matching eval_diffusion_policy.py.
    All renderers return (B, H, W, 3) uint8 numpy.
    """
    states = None
    if mesh_manifest is not None:
      states = [
        extract_render_state(env, i, args.object_name, mesh_manifest)
        for i in range(env.num_envs)
      ]
    native_np = renderer.render(states)
    raw_t = torch.from_numpy(native_np).to(device)
    img = raw_t.float() / 255.0
    img = img.permute(0, 3, 1, 2).contiguous()
    if img.shape[-2:] != (target_h, target_w):
      img = F.interpolate(
        img, size=(target_h, target_w),
        mode="bilinear", align_corners=False,
      )
    return native_np, img

  def current_agent_pos():
    return obs[:, :state_dim].float().contiguous()

  # --- Preview bookkeeping ---
  save_previews = (args.video_dir is not None) and (
    args.max_success_previews > 0 or args.max_failure_previews > 0
  )
  if save_previews:
    Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
  preview_caps = {"success": int(args.max_success_previews), "fail": int(args.max_failure_previews)}
  preview_counts = {"success": 0, "fail": 0}
  MIN_PREVIEW_FRAMES = 8

  # --- Seed history buffers (matches eval_diffusion_policy.py exactly) ---
  init_native, init_normalized = render_step()
  image_history = deque(
    [init_normalized.clone() for _ in range(n_obs_steps)],
    maxlen=n_obs_steps,
  )
  state_history = deque(
    [current_agent_pos().clone() for _ in range(n_obs_steps)],
    maxlen=n_obs_steps,
  )
  per_env_preview_imgs = [
    [init_native[i]] if save_previews else [] for i in range(env.num_envs)
  ]

  per_env_object_zs = [[] for _ in range(env.num_envs)]
  per_env_pickup_gate_history = [[] for _ in range(env.num_envs)]
  per_env_start_z = [nominal_start_z] * env.num_envs
  just_reset = [False] * env.num_envs

  attempted = 0
  succeeded = 0
  stable_succeeded = 0
  target = args.episodes_per_object

  # --- Main eval loop (timing matches eval_diffusion_policy.py exactly) ---
  try:
    while attempted < target:
      with torch.no_grad():
        obs_input = {
          "image": torch.stack(list(image_history), dim=1),
          "agent_pos": torch.stack(list(state_history), dim=1),
        }
        action_seq = policy.predict_action(obs_input)["action"]
      action_seq = torch.clamp(action_seq, -1.0, 1.0)

      for k in range(action_seq.shape[1]):
        if attempted >= target:
          break
        action_t = action_seq[:, k]
        obs_dict, _, done, _ = env.step(action_t)
        obs = obs_dict["obs"]

        object_pose_np = env.object_pose[:, 0:7].detach().cpu().numpy()
        for env_i in range(env.num_envs):
          if just_reset[env_i]:
            per_env_start_z[env_i] = float(object_pose_np[env_i, 2])
            just_reset[env_i] = False
            continue
          object_z = float(object_pose_np[env_i, 2])
          per_env_object_zs[env_i].append(object_z)
          per_env_pickup_gate_history[env_i].append(bool(
            object_z >= goal_z - PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
            and object_z - per_env_start_z[env_i] >= PICKUP_SUCCESS_MIN_LIFT_M
          ))

        done_np = done.detach().cpu().numpy().astype(bool)
        done_indices = [i for i, d in enumerate(done_np) if d]
        native_np, normalized_img = render_step()

        for env_i in done_indices:
          if attempted >= target:
            break
          attempted += 1
          metrics = compute_pickup_success_metrics(
            object_zs=per_env_object_zs[env_i],
            object_start_z=per_env_start_z[env_i],
            goal_z=goal_z,
            pickup_gate_history=per_env_pickup_gate_history[env_i],
            hold_steps=args.pickup_success_hold_steps,
          )
          ok = metrics.success
          if ok:
            succeeded += 1
          if metrics.stable_success:
            stable_succeeded += 1
          per_env_object_zs[env_i].clear()
          per_env_pickup_gate_history[env_i].clear()
          just_reset[env_i] = True

          if save_previews:
            outcome = "success" if ok else "fail"
            frames = per_env_preview_imgs[env_i]
            if (preview_counts[outcome] < preview_caps[outcome]
                and len(frames) >= MIN_PREVIEW_FRAMES):
              idx = preview_counts[outcome]
              gif_path = Path(args.video_dir) / f"{outcome}_{idx:02d}_attempt{attempted:03d}.gif"
              imageio.mimsave(gif_path, np.stack(frames, axis=0), duration=1000.0 / max(args.gif_fps, 1))
              preview_counts[outcome] += 1
          if save_previews:
            per_env_preview_imgs[env_i] = []

        randomize_object_init_xy(done_indices)

        if save_previews:
          for env_i in range(env.num_envs):
            per_env_preview_imgs[env_i].append(native_np[env_i])
        image_history.append(normalized_img)
        state_history.append(current_agent_pos())

      sr = succeeded / max(attempted, 1)
      stable_sr = stable_succeeded / max(attempted, 1)
      print(
        f"[eval-worker] {args.object_name}: {succeeded}/{attempted} ({sr:.1%}) "
        f"stable={stable_succeeded}/{attempted} ({stable_sr:.1%})",
        flush=True,
      )

  finally:
    renderer.close()
    if _blender_tmpdir:
      shutil.rmtree(_blender_tmpdir, ignore_errors=True)

  success_rate = succeeded / max(attempted, 1)
  stable_success_rate = stable_succeeded / max(attempted, 1)
  result = {
    "object_id": args.object_id,
    "category_id": args.category_id,
    "object_name": args.object_name,
    "object_category": args.object_category,
    "task_name": args.task_name,
    "renderer": args.renderer,
    "blend_file": str(args.blend_file) if args.blend_file is not None else None,
    "lighting_preset": args.lighting_preset,
    "engine": args.engine if args.renderer == "blender" else None,
    "samples": args.samples if args.renderer == "blender" else None,
    "cycles_device": args.cycles_device if args.renderer == "blender" else None,
    "render_width": render_w,
    "render_height": render_h,
    "attempted": attempted,
    "succeeded": succeeded,
    "success_rate": success_rate,
    "stable_succeeded": stable_succeeded,
    "stable_success_rate": stable_success_rate,
    "pickup_success_hold_steps": args.pickup_success_hold_steps,
    "preview_dir": str(args.video_dir) if save_previews else None,
    "previews_saved": preview_counts if save_previews else {"success": 0, "fail": 0},
  }
  Path(args.result_json).write_text(json.dumps(result, indent=2))
  print(
    f"[eval-worker] DONE {args.object_name}: "
    f"{succeeded}/{attempted} = {success_rate:.1%} "
    f"stable={stable_succeeded}/{attempted} = {stable_success_rate:.1%}",
    flush=True,
  )


def run_pick_place_release_worker(args) -> None:
  # IsaacGym must be imported before torch.
  from isaacgym import gymapi  # noqa: F401
  import numpy as np
  import torch
  import torch.nn.functional as F
  import dill
  import hydra
  from omegaconf import OmegaConf

  import isaacgymenvs  # noqa: F401
  OmegaConf.register_new_resolver("eval", eval, replace=True)

  import collect_dataset_pick_place_release as ppr
  from blender_eval.renderer_interface import StubRenderer, IsaacGymRenderer
  from blender_eval.blender_renderer import BlenderRenderer
  from blender_eval.asset_manifest import get_robot_mesh_manifest, get_object_mesh_path
  from blender_eval.state_extraction import extract_render_state

  device = args.device
  if args.num_envs != 1:
    print(
      f"[eval-worker-ppr] forcing --num-envs=1 for pick_place_release "
      f"(got {args.num_envs})",
      flush=True,
    )
    args.num_envs = 1
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  rng = np.random.default_rng(args.seed)

  ckpt_path = Path(args.checkpoint).expanduser().resolve()
  payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
  cfg = payload["cfg"]
  ws_cls = hydra.utils.get_class(cfg._target_)
  workspace = ws_cls(cfg, output_dir=tempfile.mkdtemp(prefix="dp_eval_ppr_"))
  workspace.load_payload(payload)
  policy = workspace.ema_model if cfg.training.use_ema and workspace.ema_model is not None else workspace.model
  policy.to(device).eval()
  if torch.cuda.is_available() and str(device).startswith("cuda"):
    torch.cuda.empty_cache()

  n_obs_steps = int(cfg.n_obs_steps)
  state_dim = int(cfg.task.state_dim)
  image_shape = tuple(int(x) for x in cfg.task.image_shape)
  target_h, target_w = image_shape[1], image_shape[2]
  print(
    f"[eval-worker-ppr] policy: To={n_obs_steps} "
    f"state_dim={state_dim} image={image_shape} renderer={args.renderer}",
    flush=True,
  )

  nominal_start_pose = ppr._load_nominal_start_pose(
    args.object_category,
    args.object_name,
    args.task_name,
    start_z_offset=args.start_z_offset,
  )
  bootstrap_goal_x = ppr._default_goal_x(args)
  bootstrap_goals, _, _ = ppr._build_pick_place_release_goals(
    nominal_start_pose,
    goal_x=bootstrap_goal_x,
    lift_height=args.lift_height,
    place_height=args.place_height,
    place_hold_goals=args.place_hold_goals,
    release_hold_goals=args.release_steps,
  )

  print("[eval-worker-ppr] creating long-table env...", flush=True)
  env = ppr._make_long_table_env(
    num_envs=1,
    nominal_start_pose=nominal_start_pose,
    goal_poses=bootstrap_goals,
    horizon=args.horizon,
    headless=True,
    device=device,
    seed=args.seed,
    object_name=args.object_name,
  )
  env.gym.refresh_actor_root_state_tensor(env.sim)
  if ppr.CHECKPOINT_PATH.exists():
    checkpoint = torch.load(ppr.CHECKPOINT_PATH, map_location=device)
    env.set_env_state(checkpoint[0]["env_state"])
  else:
    print(
      f"[eval-worker-ppr] warning: env checkpoint not found: {ppr.CHECKPOINT_PATH}",
      flush=True,
    )

  active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)
  render_w = args.render_width or 512
  render_h = args.render_height or 384
  mesh_manifest = None
  _blender_tmpdir = None
  if args.renderer == "blender" and args.lighting_preset is not None:
    os.environ[LIGHTING_PRESET_ENV] = args.lighting_preset

  if args.renderer == "stub":
    renderer = StubRenderer(env.num_envs, render_w, render_h)
  elif args.renderer == "isaacgym":
    renderer = IsaacGymRenderer(env, active_envs)
  elif args.renderer == "blender":
    mesh_manifest = get_robot_mesh_manifest()
    tool_mesh_path = str(get_object_mesh_path(args.object_name))
    from blender_eval.camera_params import isaacgym_to_blender_camera

    _blender_tmpdir = tempfile.mkdtemp(prefix="blender_cfg_ppr_")
    _manifest_dict = {
      name: {
        "mesh_path": str(info.mesh_path),
        "material_rgba": list(info.material_rgba) if info.material_rgba else None,
      }
      for name, info in mesh_manifest.items()
    }
    _manifest_path = os.path.join(_blender_tmpdir, "manifest.json")
    with open(_manifest_path, "w") as f:
      json.dump(_manifest_dict, f)

    cam_params = isaacgym_to_blender_camera(
      ppr.DATASET_CAMERA_HORIZONTAL_FOV, render_w, render_h,
      ppr.DATASET_CAMERA_POSITION, ppr.DATASET_CAMERA_TARGET,
    )
    _camera_path = os.path.join(_blender_tmpdir, "camera.json")
    with open(_camera_path, "w") as f:
      json.dump({
        "focal_length_mm": cam_params.focal_length_mm,
        "sensor_width_mm": cam_params.sensor_width_mm,
        "sensor_height_mm": cam_params.sensor_height_mm,
        "location": list(cam_params.location),
        "rotation_quaternion_wxyz": list(cam_params.rotation_quaternion_wxyz),
      }, f)

    renderer = BlenderRenderer(
      num_envs=env.num_envs,
      width=render_w,
      height=render_h,
      manifest_path=_manifest_path,
      camera_path=_camera_path,
      tool_mesh_path=tool_mesh_path,
      engine=args.engine,
      samples=args.samples,
      cycles_device=args.cycles_device,
      blend_file=str(args.blend_file) if args.blend_file is not None else None,
      blender_executable=args.blender,
    )
  else:
    raise ValueError(f"Unknown renderer: {args.renderer}")

  def render_step():
    states = None
    if mesh_manifest is not None:
      states = [
        extract_render_state(env, i, args.object_name, mesh_manifest)
        for i in range(env.num_envs)
      ]
    native_np = renderer.render(states)
    raw_t = torch.from_numpy(native_np).to(device)
    img = raw_t.float() / 255.0
    img = img.permute(0, 3, 1, 2).contiguous()
    if img.shape[-2:] != (target_h, target_w):
      img = F.interpolate(
        img, size=(target_h, target_w),
        mode="bilinear", align_corners=False,
      )
    return native_np, img

  save_previews = (args.video_dir is not None) and (
    args.max_success_previews > 0 or args.max_failure_previews > 0
  )
  if save_previews:
    Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
  preview_caps = {"success": int(args.max_success_previews), "fail": int(args.max_failure_previews)}
  preview_counts = {"success": 0, "fail": 0}
  min_preview_frames = 8

  attempted = 0
  succeeded = 0
  pick_place_succeeded = 0
  release_goal_succeeded = 0
  release_stable_succeeded = 0
  failure_breakdown = {name: 0 for name in ppr.RELEASE_GOAL_STAGE_NAMES}
  failure_breakdown["drop_reattempt"] = 0
  final_metrics = {}
  open_hand_action = ppr._compute_open_hand_action(env)

  try:
    while attempted < args.episodes_per_object:
      start_pose, goal_x, lateral_offset = ppr._sample_end_to_end_start_and_goal_release(
        env,
        rng,
        nominal_start_pose,
        args,
      )
      goals, release_start_goal_idx, place_goal = ppr._build_pick_place_release_goals(
        start_pose,
        goal_x=goal_x,
        lift_height=args.lift_height,
        place_height=args.place_height,
        place_hold_goals=args.place_hold_goals,
        release_hold_goals=args.release_steps,
      )
      if not ppr._place_goal_in_safe_zone(
        env,
        place_goal,
        table_x_half_extent=args.table_x_half_extent,
        table_x_inset_margin=args.place_goal_x_margin,
        table_y_half_extent=args.table_y_half_extent,
        table_y_inset_margin=args.place_goal_y_margin,
      ):
        print(
          f"[eval-worker-ppr] skipped unsafe place goal "
          f"start={np.array(start_pose).round(4).tolist()} "
          f"goal={np.array(place_goal).round(4).tolist()}",
          flush=True,
        )
        continue

      env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
      env.trajectory_states = torch.tensor(
        goals,
        device=env.device,
        dtype=env.trajectory_states.dtype,
      )
      env.max_consecutive_successes = len(goals)
      env.object_init_state[env_ids, 0:7] = torch.tensor(
        [start_pose],
        device=env.device,
        dtype=env.object_init_state.dtype,
      )
      env.cfg["env"]["tableObjectZOffset"] = float(start_pose[2] - ppr.TABLE_Z)
      env.reset_idx(env_ids, tensor_reset=True)

      zero_action = torch.zeros((env.num_envs, ppr.N_ACT), device=device)
      obs_dict, _, _, _ = env.step(zero_action)
      obs = obs_dict["obs"]

      init_native, init_normalized = render_step()
      image_history = deque(
        [init_normalized.clone() for _ in range(n_obs_steps)],
        maxlen=n_obs_steps,
      )
      state_history = deque(
        [obs[:, :state_dim].float().contiguous().clone() for _ in range(n_obs_steps)],
        maxlen=n_obs_steps,
      )
      preview_frames = [init_native[0]] if save_previews else []

      max_successes_seen = 0
      final_successes = 0
      release_start_step = None
      release_phase_per_step = []
      stage_names = []
      goal_dists = []
      successes_per_step = []
      dropped_after_lift = False
      drop_step = None
      drop_successes_before = None
      reattempted_after_drop = False
      max_object_height_above_init_m = 0.0
      steps_executed = 0

      step = 0
      episode_done = False
      while step < args.horizon and not episode_done:
        with torch.no_grad():
          obs_input = {
            "image": torch.stack(list(image_history), dim=1),
            "agent_pos": torch.stack(list(state_history), dim=1),
          }
          action_seq = policy.predict_action(obs_input)["action"]
        action_seq = torch.clamp(action_seq, -1.0, 1.0)

        for k in range(action_seq.shape[1]):
          if step >= args.horizon:
            break
          current_successes = int(env.successes[0].item())
          in_release_phase = args.release_steps > 0 and current_successes >= release_start_goal_idx
          if in_release_phase and release_start_step is None:
            release_start_step = step

          action_t = action_seq[:, k]
          if in_release_phase:
            action_t = ppr._build_release_action(
              policy_action_t=action_t,
              open_hand_action=open_hand_action,
              arm_mode=args.release_arm_mode,
              hand_blend=args.release_hand_blend,
            )

          obs_dict, _, done, _ = env.step(action_t)
          obs = obs_dict["obs"]
          steps_executed += 1

          final_successes = int(env.successes[0].item())
          max_successes_seen = max(max_successes_seen, final_successes)
          goal_dist = float(env.keypoints_max_dist[0].item())
          object_height_above_init_m = float(
            env.object_pose[0, 2].item() - env.object_init_state[0, 2].item()
          )
          max_object_height_above_init_m = max(
            max_object_height_above_init_m,
            object_height_above_init_m,
          )
          dropped_now = ppr._drop_detected_after_pickup_attempt(
            object_height_above_init_m=object_height_above_init_m,
            max_object_height_above_init_m=max_object_height_above_init_m,
            lifted_object=bool(env.lifted_object[0].item()),
            in_release_phase=in_release_phase,
          )
          if dropped_now and not dropped_after_lift:
            dropped_after_lift = True
            drop_step = step
            drop_successes_before = final_successes
          elif (
            dropped_after_lift
            and not reattempted_after_drop
            and drop_successes_before is not None
            and final_successes > drop_successes_before
          ):
            reattempted_after_drop = True

          stage_name = (
            "release" if in_release_phase
            else ppr._release_goal_stage_name(current_successes, release_start_goal_idx)
          )
          stage_names.append(stage_name)
          goal_dists.append(goal_dist)
          successes_per_step.append(final_successes)
          release_phase_per_step.append(bool(in_release_phase))

          native_np, normalized_img = render_step()
          if save_previews:
            preview_frames.append(native_np[0])
          image_history.append(normalized_img)
          state_history.append(obs[:, :state_dim].float().contiguous())

          step += 1
          if final_successes >= len(goals) or bool(done[0].item()):
            episode_done = True
            break

      attempted += 1
      final_object_pose_np = env.object_pose[0, 0:7].detach().cpu().numpy()
      final_object_linvel_np = env.object_linvel[0].detach().cpu().numpy()
      place_goal_np = np.asarray(place_goal, dtype=np.float32)
      final_place_xy_error_m = float(np.linalg.norm(final_object_pose_np[0:2] - place_goal_np[0:2]))
      final_place_z_error_m = float(abs(final_object_pose_np[2] - place_goal_np[2]))
      final_place_pos_error_m = float(np.linalg.norm(final_object_pose_np[0:3] - place_goal_np[0:3]))
      final_object_speed_mps = float(np.linalg.norm(final_object_linvel_np))
      entered_release_phase = release_start_step is not None
      release_steps_executed = sum(1 for value in release_phase_per_step if value)
      final_object_on_table = ppr._final_object_on_table(
        env,
        final_object_pose_np,
        table_x_half_extent=args.table_x_half_extent,
        table_x_inset_margin=args.table_x_inset_margin,
        table_y_half_extent=args.table_y_half_extent,
        table_y_inset_margin=args.table_y_inset_margin,
      )
      (
        pick_place_success,
        release_goal_success,
        release_stable,
        release_success,
        failure_stage,
      ) = ppr._classify_release_outcome(
        max_successes_seen=max_successes_seen,
        release_start_goal_idx=release_start_goal_idx,
        total_goals=len(goals),
        entered_release_phase=entered_release_phase,
        final_object_on_table=final_object_on_table,
        final_place_xy_error_m=final_place_xy_error_m,
        final_place_z_error_m=final_place_z_error_m,
        final_object_speed_mps=final_object_speed_mps,
        release_xy_tolerance=args.release_xy_tolerance,
        release_z_tolerance=args.release_z_tolerance,
        release_speed_tolerance=args.release_speed_tolerance,
        reattempted_after_drop=reattempted_after_drop,
      )

      if pick_place_success:
        pick_place_succeeded += 1
      if release_goal_success:
        release_goal_succeeded += 1
      if release_stable:
        release_stable_succeeded += 1
      if release_success:
        succeeded += 1
      elif failure_stage is not None:
        failure_breakdown[failure_stage] = failure_breakdown.get(failure_stage, 0) + 1
      if reattempted_after_drop:
        failure_breakdown["drop_reattempt"] += 1

      final_metrics = {
        "steps": steps_executed,
        "release_start_step": release_start_step,
        "release_steps_executed": release_steps_executed,
        "max_successes_seen": max_successes_seen,
        "final_successes": final_successes,
        "failure_stage": failure_stage,
        "final_object_on_table": final_object_on_table,
        "final_place_pos_error_m": final_place_pos_error_m,
        "final_place_xy_error_m": final_place_xy_error_m,
        "final_place_z_error_m": final_place_z_error_m,
        "final_object_speed_mps": final_object_speed_mps,
        "dropped_after_lift": dropped_after_lift,
        "drop_step": drop_step,
        "drop_successes_before": drop_successes_before,
        "reattempted_after_drop": reattempted_after_drop,
        "goal_x": goal_x,
        "lateral_offset": lateral_offset,
      }

      if save_previews:
        outcome = "success" if release_success else "fail"
        if (
          preview_counts[outcome] < preview_caps[outcome]
          and len(preview_frames) >= min_preview_frames
        ):
          idx = preview_counts[outcome]
          gif_path = Path(args.video_dir) / f"{outcome}_{idx:02d}_attempt{attempted:03d}.gif"
          imageio.mimsave(
            gif_path,
            np.stack(preview_frames, axis=0),
            duration=1000.0 / max(args.gif_fps, 1),
          )
          preview_counts[outcome] += 1

      print(
        f"[eval-worker-ppr] {args.object_name}: release={succeeded}/{attempted} "
        f"({succeeded / max(attempted, 1):.1%}) "
        f"pick_place={pick_place_succeeded}/{attempted} "
        f"stable={release_stable_succeeded}/{attempted}",
        flush=True,
      )
  finally:
    renderer.close()
    if _blender_tmpdir:
      shutil.rmtree(_blender_tmpdir, ignore_errors=True)

  success_rate = succeeded / max(attempted, 1)
  stable_success_rate = release_stable_succeeded / max(attempted, 1)
  result = {
    "object_id": args.object_id,
    "category_id": args.category_id,
    "object_name": args.object_name,
    "object_category": args.object_category,
    "task_name": args.task_name,
    "eval_task": "pick_place_release",
    "renderer": args.renderer,
    "blend_file": str(args.blend_file) if args.blend_file is not None else None,
    "lighting_preset": args.lighting_preset,
    "engine": args.engine if args.renderer == "blender" else None,
    "samples": args.samples if args.renderer == "blender" else None,
    "cycles_device": args.cycles_device if args.renderer == "blender" else None,
    "render_width": render_w,
    "render_height": render_h,
    "attempted": attempted,
    "succeeded": succeeded,
    "success_rate": success_rate,
    "stable_succeeded": release_stable_succeeded,
    "stable_success_rate": stable_success_rate,
    "pick_place_succeeded": pick_place_succeeded,
    "pick_place_success_rate": pick_place_succeeded / max(attempted, 1),
    "release_goal_succeeded": release_goal_succeeded,
    "release_goal_success_rate": release_goal_succeeded / max(attempted, 1),
    "release_stable_succeeded": release_stable_succeeded,
    "release_stable_success_rate": stable_success_rate,
    "failure_breakdown": failure_breakdown,
    "release_steps": args.release_steps,
    "release_arm_mode": args.release_arm_mode,
    "release_hand_blend": args.release_hand_blend,
    "release_xy_tolerance": args.release_xy_tolerance,
    "release_z_tolerance": args.release_z_tolerance,
    "release_speed_tolerance": args.release_speed_tolerance,
    "start_z_offset": args.start_z_offset,
    "lift_height": args.lift_height,
    "place_height": args.place_height,
    "place_hold_goals": args.place_hold_goals,
    "last_episode": final_metrics,
    "preview_dir": str(args.video_dir) if save_previews else None,
    "previews_saved": preview_counts if save_previews else {"success": 0, "fail": 0},
  }
  Path(args.result_json).write_text(json.dumps(result, indent=2))
  print(
    f"[eval-worker-ppr] DONE {args.object_name}: "
    f"release={succeeded}/{attempted} = {success_rate:.1%} "
    f"pick_place={pick_place_succeeded}/{attempted} = "
    f"{pick_place_succeeded / max(attempted, 1):.1%}",
    flush=True,
  )


# ----------------------------- argparse / main --------------------------------

def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--renderer", choices=("stub", "isaacgym", "blender"), default="stub")
  p.add_argument("--eval-task", choices=("pickup", "pick_place_release"), default="pickup",
                 help="Task success/eval logic to run.")
  p.add_argument("--engine", choices=("cycles", "eevee"), default="cycles",
                 help="Blender render engine (only used with --renderer blender)")
  p.add_argument("--samples", type=int, default=96,
                 help="Cycles render samples (only used with --renderer blender)")
  p.add_argument("--cycles-device", choices=("auto", "gpu", "cpu"), default="auto",
                 help="Cycles compute device for Blender: auto, gpu, or cpu")
  p.add_argument("--blender", default="blender",
                 help="Path to Blender executable (only used with --renderer blender)")
  p.add_argument("--blend-file", type=Path, default=None,
                 help="Optional master .blend template for static Blender scene visuals")
  p.add_argument("--lighting-preset", default=None,
                 choices=(
                   "clean_key_fill", "reference_area", "softbox_grid",
                   "softbox_overhead", "softbox_wrap", "softbox_rim",
                   "spot_accent",
                 ),
                 help=(
                   "Optional scripted lighting preset override for Blender. "
                   "If omitted, the master .blend template lighting is used."
                 ))
  p.add_argument("--split", choices=("train", "ood"), default="train")
  p.add_argument("--episodes-per-object", type=int, default=32)
  p.add_argument("--num-envs", type=int, default=8)
  p.add_argument("--horizon", type=int, default=None)
  p.add_argument("--xy-range", type=float, default=0.10)
  p.add_argument("--start-z-offset", type=float, default=DEFAULT_START_Z_OFFSET)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--render-width", type=int, default=None,
                 help="Native render width (default: env camera config)")
  p.add_argument("--render-height", type=int, default=None,
                 help="Native render height (default: env camera config)")
  p.add_argument("--output-json", type=Path, default=Path("data/blender_eval/eval.json"))
  p.add_argument("--max-success-previews", type=int, default=2)
  p.add_argument("--max-failure-previews", type=int, default=2)
  p.add_argument("--gif-fps", type=int, default=10)
  p.add_argument("--pickup-success-hold-steps", type=int, default=5,
                 help="Consecutive pickup-gate steps required for stable_success_rate.")
  p.add_argument("--lift-height", type=float, default=DEFAULT_LIFT_HEIGHT_M)
  p.add_argument("--place-height", type=float, default=DEFAULT_PLACE_HEIGHT_M)
  p.add_argument("--place-hold-goals", type=int, default=DEFAULT_PLACE_HOLD_GOALS)
  p.add_argument("--release-steps", type=int, default=DEFAULT_RELEASE_STEPS)
  p.add_argument("--release-arm-mode", choices=("hold", "policy"), default="hold")
  p.add_argument("--release-hand-blend", type=float, default=1.0)
  p.add_argument("--release-xy-tolerance", type=float, default=DEFAULT_RELEASE_XY_TOLERANCE_M)
  p.add_argument("--release-z-tolerance", type=float, default=DEFAULT_RELEASE_Z_TOLERANCE_M)
  p.add_argument("--release-speed-tolerance", type=float, default=DEFAULT_RELEASE_SPEED_TOLERANCE_MPS)
  p.add_argument("--table-x-half-extent", type=float, default=DEFAULT_LONG_TABLE_X_HALF_EXTENT_M)
  p.add_argument("--table-x-inset-margin", type=float, default=DEFAULT_LONG_TABLE_X_INSET_MARGIN_M)
  p.add_argument("--table-y-half-extent", type=float, default=DEFAULT_TABLE_Y_HALF_EXTENT_M)
  p.add_argument("--table-y-inset-margin", type=float, default=DEFAULT_TABLE_Y_INSET_MARGIN_M)
  p.add_argument("--place-goal-x-margin", type=float, default=DEFAULT_PLACE_GOAL_X_MARGIN_M)
  p.add_argument("--place-goal-y-margin", type=float, default=DEFAULT_PLACE_GOAL_Y_MARGIN_M)
  p.add_argument("--min-effective-transport", type=float, default=DEFAULT_MIN_EFFECTIVE_TRANSPORT_M)

  # Worker-only args (passed by driver)
  p.add_argument("--object-category", default=None)
  p.add_argument("--object-name", default=None)
  p.add_argument("--task-name", default=None)
  p.add_argument("--object-id", type=int, default=0)
  p.add_argument("--category-id", type=int, default=0)
  p.add_argument("--result-json", type=Path, default=None)
  p.add_argument("--video-dir", type=Path, default=None)

  return p.parse_args()


def _apply_photoreal_defaults(args) -> None:
  """Make Blender eval use the committed photoreal template unless overridden."""
  if args.renderer != "blender":
    return
  if args.blend_file is None:
    if not DEFAULT_BLEND_FILE.exists():
      raise FileNotFoundError(
        f"Default photoreal blend template not found: {DEFAULT_BLEND_FILE}"
      )
    args.blend_file = DEFAULT_BLEND_FILE
  if args.lighting_preset is not None:
    os.environ[LIGHTING_PRESET_ENV] = args.lighting_preset


def _apply_driver_defaults(args) -> None:
  """Match eval_diffusion_policy.py preview-directory behavior."""
  if not hasattr(args, "eval_task"):
    args.eval_task = "pickup"
  if not hasattr(args, "pickup_success_hold_steps"):
    args.pickup_success_hold_steps = DEFAULT_PICKUP_SUCCESS_HOLD_STEPS
  if getattr(args, "horizon", None) is None:
    args.horizon = (
      DEFAULT_PICK_PLACE_RELEASE_HORIZON
      if args.eval_task == "pick_place_release"
      else DEFAULT_PICKUP_HORIZON
    )
  if args.video_dir is None and (
    args.max_success_previews > 0 or args.max_failure_previews > 0
  ):
    args.video_dir = args.output_json.parent / f"{args.output_json.stem}_videos"


def main():
  args = parse_args()
  _apply_photoreal_defaults(args)
  _apply_driver_defaults(args)
  if args.worker:
    run_worker(args)
  else:
    run_driver(args)


if __name__ == "__main__":
  main()
