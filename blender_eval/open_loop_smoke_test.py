#!/usr/bin/env python3
"""Open-loop Blender render smoke test — validates the renderer half of the
bridge WITHOUT IsaacGym.

Builds the mesh manifest from the real URDF, computes a zero-pose forward
kinematics robot configuration (all joint angles = 0, so each link's world
transform is just the product of joint-origin transforms from the root), and
drives the real BlenderRenderer IPC to render one frame. Exercises STL/OBJ
importers, camera setup, the FIFO protocol, and the Cycles render call together.

This is the open-loop check from BLENDER_EVAL_PROGRESS.md "first validation
milestone", minus a real collected rollout (which requires the cluster). It
confirms the bpy script runs and that the coordinate-frame + camera math
produce an upright, correctly-framed robot.

Usage:
    python blender_eval/open_loop_smoke_test.py \\
        --blender /path/to/blender --object claw_hammer --out /tmp/render.png

Verified on Blender 4.2.9 LTS (headless, Cycles CPU).
"""
import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from blender_eval.asset_manifest import (  # noqa: E402
  get_robot_mesh_manifest, get_object_mesh_path, ROBOT_URDF,
)
from blender_eval.pose_conversion import urdf_origin_to_matrix, matrix_to_pose  # noqa: E402
from blender_eval.camera_params import isaacgym_to_blender_camera  # noqa: E402
from blender_eval.state_extraction import RenderState  # noqa: E402
from blender_eval.blender_renderer import BlenderRenderer  # noqa: E402


def zero_pose_link_world_matrices(urdf_path):
  """World transform of every link at zero joint angles = product of joint
  origins along the chain from root (a joint's rotation at angle 0 is identity)."""
  root = ET.parse(str(urdf_path)).getroot()
  child_to = {}
  for j in root.findall("joint"):
    parent = j.find("parent").get("link")
    child = j.find("child").get("link")
    o = j.find("origin")
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
    child_to[child] = (parent, urdf_origin_to_matrix(xyz, rpy))

  all_links = {l.get("name") for l in root.findall("link")}
  cache = {}

  def world(link):
    if link in cache:
      return cache[link]
    if link not in child_to:
      cache[link] = np.eye(4)
      return cache[link]
    parent, origin = child_to[link]
    cache[link] = world(parent) @ origin
    return cache[link]

  return {l: world(l) for l in all_links}


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--blender", default="blender", help="Blender executable (default: PATH)")
  ap.add_argument("--object", default="claw_hammer", help="Tool object name")
  ap.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
  ap.add_argument("--samples", type=int, default=8)
  ap.add_argument("--width", type=int, default=256)
  ap.add_argument("--height", type=int, default=192)
  ap.add_argument("--hfov-deg", type=float, default=58.0,
                  help="Camera horizontal FOV; eval reads the real value from stage5")
  ap.add_argument("--out", default="/tmp/smoke_render.png")
  args = ap.parse_args()

  manifest = get_robot_mesh_manifest()
  link_world = zero_pose_link_world_matrices(ROBOT_URDF)

  mesh_poses = {}
  for link_name, info in manifest.items():
    mesh_mat = link_world.get(link_name, np.eye(4)) @ info.visual_origin
    pos, quat_xyzw = matrix_to_pose(mesh_mat)
    mesh_poses[link_name] = (pos.astype(np.float32), quat_xyzw.astype(np.float32))

  state = RenderState(
    mesh_poses=mesh_poses,
    object_pos=np.array([0.0, 0.0, 0.50], dtype=np.float32),
    object_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    object_name=args.object,
  )

  cam = isaacgym_to_blender_camera(
    horizontal_fov_deg=args.hfov_deg, width=args.width, height=args.height,
    camera_position=[0.0, -1.0, 1.03], camera_target=[0.0, 0.0, 0.53],
  )

  tmp = tempfile.mkdtemp(prefix="smoke_")
  manifest_path = os.path.join(tmp, "manifest.json")
  with open(manifest_path, "w") as f:
    json.dump({k: {"mesh_path": str(v.mesh_path)} for k, v in manifest.items()}, f)
  camera_path = os.path.join(tmp, "camera.json")
  with open(camera_path, "w") as f:
    json.dump({
      "focal_length_mm": cam.focal_length_mm,
      "sensor_width_mm": cam.sensor_width_mm,
      "sensor_height_mm": cam.sensor_height_mm,
      "location": list(cam.location),
      "rotation_quaternion_wxyz": list(cam.rotation_quaternion_wxyz),
    }, f)

  print(f"[smoke] launching Blender ({len(manifest)} robot meshes + 1 tool)...", flush=True)
  r = BlenderRenderer(
    num_envs=1, width=args.width, height=args.height,
    manifest_path=manifest_path, camera_path=camera_path,
    tool_mesh_path=str(get_object_mesh_path(args.object)),
    engine=args.engine, samples=args.samples,
    blender_executable=args.blender,
  )
  try:
    imgs = r.render([state])
  finally:
    r.close()

  img = imgs[0]
  Image.fromarray(img).save(args.out)
  uniq = len(np.unique(img.reshape(-1, 3), axis=0))
  print(f"[smoke] shape={imgs.shape} unique_colors={uniq} "
        f"mean={img.mean():.1f} min={img.min()} max={img.max()}", flush=True)
  print(f"[smoke] saved {args.out}", flush=True)
  if uniq <= 10:
    print("[smoke] WARN: near-uniform image — check framing/lighting", flush=True)
    return 1
  print("[smoke] PASS: non-trivial geometry rendered", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
