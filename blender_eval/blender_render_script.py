#!/usr/bin/env python3
# UNVERIFIED — This script runs inside Blender's Python interpreter, not the
# simtooldiff venv. It cannot be tested on a headless box without Blender.
#
# Usage:
#   blender --background --python blender_eval/blender_render_script.py -- \
#     --manifest /path/to/manifest.json \
#     --camera /path/to/camera.json \
#     --engine cycles \
#     --width 512 --height 384
#
# IPC protocol:
#   Commands:  JSON lines on stdin (one per frame with mesh_poses + object state).
#   Responses: READY and image paths written to a named pipe (FIFO) passed as
#              --response-fifo. This avoids Blender's own stdout pollution
#              (version banners, render progress) from corrupting the protocol.
#   Both sides must flush after every write to avoid pipe deadlocks.

"""Blender headless render script for SimToolDiff photorealistic evaluation.

Expects to be run inside Blender's Python:
    blender --background --python this_script.py -- [args]

Loads STL/OBJ meshes once at startup, then enters a render loop reading
pose updates from stdin and writing rendered image paths to stdout.
"""

import json
import os
import sys
import tempfile

# ---- Blender imports (only available inside blender --python) ----
try:
  import bpy
  import mathutils
except ImportError:
  print("ERROR: This script must be run inside Blender's Python interpreter.")
  print("Usage: blender --background --python blender_render_script.py -- [args]")
  sys.exit(1)


def parse_args():
  """Parse args after the '--' separator in blender's command line."""
  argv = sys.argv
  if "--" in argv:
    argv = argv[argv.index("--") + 1:]
  else:
    argv = []

  import argparse
  p = argparse.ArgumentParser()
  p.add_argument("--manifest", type=str, required=True,
                 help="Path to mesh manifest JSON (from asset_manifest)")
  p.add_argument("--camera", type=str, required=True,
                 help="Path to camera params JSON (from camera_params)")
  p.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
  p.add_argument("--width", type=int, default=512)
  p.add_argument("--height", type=int, default=384)
  p.add_argument("--samples", type=int, default=64,
                 help="Cycles render samples (lower = faster, noisier)")
  p.add_argument("--blend-file", type=str, default=None,
                 help="Optional .blend scene template to load (lighting, materials)")
  p.add_argument("--response-fifo", type=str, required=True,
                 help="Path to named pipe (FIFO) for protocol responses")
  return p.parse_args(argv)


def setup_scene(args):
  """Configure render engine, resolution, and basic scene."""
  scene = bpy.context.scene

  # Load scene template if provided
  if args.blend_file and os.path.exists(args.blend_file):
    bpy.ops.wm.open_mainfile(filepath=args.blend_file)
    scene = bpy.context.scene
  else:
    # Clear default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Basic lighting (will be replaced when a .blend template is authored)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.data.angle = 0.1  # Soft shadows

    # World background
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
      bg.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)

  # Render engine
  if args.engine == "cycles":
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    # Use GPU if available
    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
      prefs.preferences.compute_device_type = 'CUDA'
      for device in prefs.preferences.devices:
        device.use = True
  else:
    scene.render.engine = 'BLENDER_EEVEE_NEXT'

  # Resolution
  scene.render.resolution_x = args.width
  scene.render.resolution_y = args.height
  scene.render.resolution_percentage = 100
  scene.render.image_settings.file_format = 'PNG'

  # Table: simple plane at z=0.38 (TABLE_Z from stage5)
  if not bpy.data.objects.get("table"):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0.4, 0.38))
    table = bpy.context.active_object
    table.name = "table"
    # Gray material
    mat = bpy.data.materials.new("TableMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
      bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
    table.data.materials.append(mat)

  return scene


def import_meshes(manifest_path: str):
  """Import all meshes from the manifest, keyed by link name.

  Returns dict mapping link_name -> bpy.types.Object.
  """
  with open(manifest_path) as f:
    manifest = json.load(f)

  objects = {}
  for link_name, info in manifest.items():
    mesh_path = info["mesh_path"]
    if not os.path.exists(mesh_path):
      print(f"WARNING: mesh not found: {mesh_path}", file=sys.stderr)
      continue

    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".stl":
      bpy.ops.import_mesh.stl(filepath=mesh_path)
    elif ext == ".obj":
      # Blender's OBJ importer auto-resolves sibling .mtl files
      bpy.ops.wm.obj_import(filepath=mesh_path)
    else:
      print(f"WARNING: unsupported mesh format: {ext} for {link_name}", file=sys.stderr)
      continue

    obj = bpy.context.selected_objects[-1]
    obj.name = link_name
    obj.rotation_mode = 'QUATERNION'
    objects[link_name] = obj

  return objects


def import_tool_mesh(mesh_path: str, object_name: str):
  """Import the tool object mesh."""
  ext = os.path.splitext(mesh_path)[1].lower()
  if ext == ".stl":
    bpy.ops.import_mesh.stl(filepath=mesh_path)
  elif ext == ".obj":
    bpy.ops.wm.obj_import(filepath=mesh_path)
  else:
    raise ValueError(f"Unsupported format: {ext}")

  obj = bpy.context.selected_objects[-1]
  obj.name = f"tool_{object_name}"
  obj.rotation_mode = 'QUATERNION'
  return obj


def setup_camera(camera_path: str, scene):
  """Create and position the camera to match IsaacGym's viewpoint."""
  with open(camera_path) as f:
    params = json.load(f)

  cam_data = bpy.data.cameras.new("EvalCamera")
  cam_data.lens = params["focal_length_mm"]
  cam_data.sensor_width = params["sensor_width_mm"]
  cam_data.sensor_height = params["sensor_height_mm"]
  cam_data.sensor_fit = 'HORIZONTAL'  # REQUIRED for HFOV match

  cam_obj = bpy.data.objects.new("EvalCamera", cam_data)
  bpy.context.collection.objects.link(cam_obj)

  cam_obj.location = params["location"]
  cam_obj.rotation_mode = 'QUATERNION'
  cam_obj.rotation_quaternion = params["rotation_quaternion_wxyz"]

  scene.camera = cam_obj
  return cam_obj


def update_poses(robot_objects, tool_object, state):
  """Set location and rotation for each mesh object from render state.

  Quaternions arrive in wxyz order (Blender convention), converted from
  IsaacGym xyzw by serialize_render_state() in state_extraction.py.
  """
  mesh_poses = state.get("mesh_poses", {})
  for link_name, (pos, quat_wxyz) in mesh_poses.items():
    obj = robot_objects.get(link_name)
    if obj is None:
      continue
    obj.location = pos
    obj.rotation_quaternion = quat_wxyz

  if tool_object and "object_pos" in state:
    tool_object.location = state["object_pos"]
    tool_object.rotation_quaternion = state["object_quat_wxyz"]


def render_frame(scene, output_dir):
  """Render current scene to a PNG file. Returns the file path."""
  fd, path = tempfile.mkstemp(suffix=".png", dir=output_dir)
  os.close(fd)
  scene.render.filepath = path
  bpy.ops.render.render(write_still=True)
  return path


def main():
  args = parse_args()
  scene = setup_scene(args)

  # Load manifest and import meshes
  robot_objects = import_meshes(args.manifest)
  print(f"Imported {len(robot_objects)} robot meshes", file=sys.stderr, flush=True)

  # Camera
  setup_camera(args.camera, scene)

  # Temp dir for rendered frames
  render_dir = tempfile.mkdtemp(prefix="blender_eval_")

  # Open the response FIFO for writing. The parent process created the FIFO
  # and is blocking on open() for reading, so this unblocks both sides.
  response_fifo = open(args.response_fifo, "w")

  # Tool object (imported on first frame when we know the object name)
  tool_object = None
  current_tool_name = None

  # Signal ready via FIFO (NOT stdout — Blender pollutes stdout)
  response_fifo.write("READY\n")
  response_fifo.flush()

  # Main render loop: read JSON commands from stdin
  for line in sys.stdin:
    line = line.strip()
    if not line:
      continue
    if line == "QUIT":
      break

    try:
      state = json.loads(line)
    except json.JSONDecodeError as e:
      print(f"ERROR: malformed JSON: {e}", file=sys.stderr, flush=True)
      # MUST still write a response so the parent doesn't deadlock.
      response_fifo.write(f"ERROR: malformed JSON: {e}\n")
      response_fifo.flush()
      continue

    try:
      # Import tool mesh on first frame or if object changes
      obj_name = state.get("object_name")
      tool_mesh_path = state.get("tool_mesh_path")
      if obj_name != current_tool_name and tool_mesh_path:
        if tool_object:
          bpy.data.objects.remove(tool_object, do_unlink=True)
        tool_object = import_tool_mesh(tool_mesh_path, obj_name)
        current_tool_name = obj_name

      # Poses arrive with quaternions in wxyz (Blender convention),
      # converted from IsaacGym xyzw by serialize_render_state().
      update_poses(robot_objects, tool_object, state)

      # Render
      img_path = render_frame(scene, render_dir)

      # Write image path to FIFO (NOT stdout)
      response_fifo.write(img_path + "\n")
      response_fifo.flush()
    except Exception as e:
      print(f"ERROR: render failed: {e}", file=sys.stderr, flush=True)
      response_fifo.write(f"ERROR: render failed: {e}\n")
      response_fifo.flush()

  response_fifo.close()


if __name__ == "__main__":
  main()
