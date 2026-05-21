#!/usr/bin/env python3
# This script runs inside Blender's Python interpreter, not the simtooldiff venv.
# VERIFIED to run end-to-end on Blender 4.2.9 LTS (headless, Cycles CPU) via
# blender_eval/open_loop_smoke_test.py: imports all 36 robot STL meshes + the
# tool OBJ, configures the camera, and renders a frame over the FIFO protocol.
# Still UNVERIFIED against a real collected rollout (needs the cluster) and the
# authored .blend scene template (HDRI/PBR lighting) does not exist yet.
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


def _import_mesh_file(mesh_path: str):
  """Import one STL/OBJ mesh, returning the newly-created bpy object.

  Robust across Blender versions:
    - STL: prefer the 4.0+ C++ importer (bpy.ops.wm.stl_import); fall back to
      the legacy add-on operator (bpy.ops.import_mesh.stl), removed in 4.2.
    - OBJ: bpy.ops.wm.obj_import (4.x). The .mtl sibling is auto-resolved.
  Imported-object capture is done by diffing the object set before/after,
  which is reliable regardless of which operator selected what.
  """
  ext = os.path.splitext(mesh_path)[1].lower()
  before = set(bpy.data.objects)

  if ext == ".stl":
    if hasattr(bpy.ops.wm, "stl_import"):
      bpy.ops.wm.stl_import(filepath=mesh_path)
    elif hasattr(bpy.ops.import_mesh, "stl"):
      bpy.ops.import_mesh.stl(filepath=mesh_path)
    else:
      raise RuntimeError("No STL importer available in this Blender build")
  elif ext == ".obj":
    if hasattr(bpy.ops.wm, "obj_import"):
      bpy.ops.wm.obj_import(filepath=mesh_path)
    elif hasattr(bpy.ops, "import_scene") and hasattr(bpy.ops.import_scene, "obj"):
      bpy.ops.import_scene.obj(filepath=mesh_path)
    else:
      raise RuntimeError("No OBJ importer available in this Blender build")
  else:
    raise ValueError(f"Unsupported mesh format: {ext}")

  new_objs = [o for o in bpy.data.objects if o not in before]
  if not new_objs:
    raise RuntimeError(f"Import produced no new object: {mesh_path}")
  # An OBJ may import as several objects; join them into one for a clean handle.
  if len(new_objs) > 1:
    bpy.ops.object.select_all(action='DESELECT')
    for o in new_objs:
      o.select_set(True)
    bpy.context.view_layer.objects.active = new_objs[0]
    bpy.ops.object.join()
    new_objs = [bpy.context.view_layer.objects.active]
  return new_objs[0]


def _eevee_engine_name():
  """Return the EEVEE engine identifier valid for this Blender version.

  Blender 4.2+ renamed the engine to BLENDER_EEVEE_NEXT; 4.0-4.1 use
  BLENDER_EEVEE.
  """
  try:
    items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    names = {item.identifier for item in items}
  except (KeyError, AttributeError):
    names = set()
  if "BLENDER_EEVEE_NEXT" in names:
    return "BLENDER_EEVEE_NEXT"
  return "BLENDER_EEVEE"


def _set_principled_input(bsdf, names, value):
  for name in names:
    if name in bsdf.inputs:
      bsdf.inputs[name].default_value = value
      return


def _make_material(
  name,
  rgba,
  roughness=0.55,
  metallic=0.0,
  specular=0.5,
  coat=0.0,
):
  """Create or update a simple Principled material."""
  mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
  mat.use_nodes = True
  bsdf = mat.node_tree.nodes.get("Principled BSDF")
  if bsdf:
    _set_principled_input(bsdf, ("Base Color",), tuple(rgba))
    _set_principled_input(bsdf, ("Roughness",), roughness)
    _set_principled_input(bsdf, ("Metallic",), metallic)
    _set_principled_input(bsdf, ("Specular IOR Level", "Specular"), specular)
    _set_principled_input(bsdf, ("Coat Weight", "Clearcoat"), coat)
  return mat


def _make_wood_material(name):
  """Create a subtle procedural wood-like table material."""
  mat = _make_material(
    name,
    (0.70, 0.48, 0.30, 1.0),
    roughness=0.74,
    metallic=0.0,
    specular=0.25,
    coat=0.03,
  )
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  bsdf = nodes.get("Principled BSDF")
  if bsdf:
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 9.0
    noise.inputs["Roughness"].default_value = 0.58
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.24
    ramp.color_ramp.elements[0].color = (0.42, 0.28, 0.18, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.68, 0.50, 0.34, 1.0)
    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.018
    bump.inputs["Distance"].default_value = 0.025
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    if "Normal" in bsdf.inputs:
      links.new(noise.outputs["Fac"], bump.inputs["Height"])
      links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
  return mat


def _robot_material(link_name, rgba):
  """Choose PBR-ish params for URDF robot colors."""
  if link_name.startswith("iiwa14_link_") and rgba[0] > 0.9 and rgba[1] > 0.25:
    return _make_material(
      f"urdf_{link_name}_material",
      rgba,
      roughness=0.48,
      specular=0.42,
      coat=0.06,
    )
  if link_name.startswith("iiwa14_link_") and max(rgba[:3]) < 0.45:
    return _make_material(
      f"urdf_{link_name}_material",
      rgba,
      roughness=0.45,
      metallic=0.35,
      specular=0.38,
    )
  return _make_material(
    f"urdf_{link_name}_material", rgba, roughness=0.48, specular=0.45
  )


def _assign_material(obj, mat):
  if not hasattr(obj.data, "materials"):
    return
  obj.data.materials.clear()
  obj.data.materials.append(mat)


def _shade_smooth(obj):
  if not hasattr(obj.data, "polygons"):
    return
  for poly in obj.data.polygons:
    poly.use_smooth = True


def _add_bevel(obj, amount, segments=2):
  bevel = obj.modifiers.new(name="softened_edges", type="BEVEL")
  bevel.width = amount
  bevel.segments = segments
  bevel.affect = "EDGES"
  obj.modifiers.new(name="weighted_normals", type="WEIGHTED_NORMAL")


def _look_at(obj, target):
  direction = mathutils.Vector(target) - obj.location
  obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_area_light(name, location, target, energy, size):
  bpy.ops.object.light_add(type="AREA", location=location)
  light = bpy.context.active_object
  light.name = name
  light.data.energy = energy
  light.data.size = size
  _look_at(light, target)
  return light


def _add_spot_light(name, location, target, energy, size, blend=0.65):
  bpy.ops.object.light_add(type="SPOT", location=location)
  light = bpy.context.active_object
  light.name = name
  light.data.energy = energy
  light.data.spot_size = size
  light.data.spot_blend = blend
  _look_at(light, target)
  return light


def _set_color_management(scene):
  """Prefer a product-render transform, falling back across Blender versions."""
  available = {
    item.identifier
    for item in bpy.types.ColorManagedViewSettings.bl_rna.properties[
      "view_transform"
    ].enum_items
  }
  if "AgX" in available:
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
  elif "Filmic" in available:
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
  else:
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
  scene.view_settings.exposure = -1.0
  scene.view_settings.gamma = 1.0


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

    # Product-style deterministic lighting: broad softboxes for shape, plus
    # focused spots on the table/tool area. This is still reproducible and does
    # not require a hand-authored .blend file.
    scene_center = (0.0, 0.05, 0.72)
    _add_area_light(
      "key_softbox",
      location=(-0.85, -1.05, 1.75),
      target=scene_center,
      energy=44.0,
      size=2.15,
    )
    _add_area_light(
      "fill_softbox",
      location=(1.10, -0.55, 1.25),
      target=scene_center,
      energy=18.0,
      size=3.00,
    )
    _add_area_light(
      "rim_softbox",
      location=(0.15, 1.20, 1.50),
      target=(0.0, 0.10, 0.80),
      energy=16.0,
      size=1.50,
    )
    _add_spot_light(
      "tool_spot",
      location=(-0.45, -0.70, 1.35),
      target=(-0.02, 0.04, 0.60),
      energy=5.0,
      size=1.15,
      blend=0.85,
    )
    _add_spot_light(
      "hand_spot",
      location=(0.65, -0.95, 1.75),
      target=(0.02, 0.02, 0.92),
      energy=4.0,
      size=1.05,
      blend=0.85,
    )

    # World background
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
      bg.inputs["Color"].default_value = (0.025, 0.026, 0.028, 1.0)
      bg.inputs["Strength"].default_value = 0.035

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
    scene.render.engine = _eevee_engine_name()

  # Resolution
  scene.render.resolution_x = args.width
  scene.render.resolution_y = args.height
  scene.render.resolution_percentage = 100
  scene.render.image_settings.file_format = 'PNG'
  _set_color_management(scene)

  # IsaacGym scene furniture from assets/urdf/table_narrow_nail.urdf.
  # The table actor is centered at (0, 0, TABLE_Z); the URDF box is centered
  # on that actor with size 0.475 x 0.4 x 0.3, plus a small gray nail.
  table_mat = _make_wood_material("table_wood")
  nail_mat = _make_material(
    "table_nail_grey", (0.5, 0.5, 0.5, 1.0), roughness=0.3, metallic=0.65
  )
  if not bpy.data.objects.get("table"):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.38))
    table = bpy.context.active_object
    table.name = "table"
    table.dimensions = (0.475, 0.4, 0.3)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    _add_bevel(table, amount=0.006, segments=3)
    _assign_material(table, table_mat)

  if not bpy.data.objects.get("table_nail"):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-0.16, 0.06, 0.38 + 0.175))
    nail = bpy.context.active_object
    nail.name = "table_nail"
    nail.dimensions = (0.03, 0.03, 0.06)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    _add_bevel(nail, amount=0.002, segments=2)
    _assign_material(nail, nail_mat)

  # Neutral studio floor/backdrop. No checkerboard is required for evaluation;
  # this keeps the rendered observation focused on the robot, tool, and table.
  if not bpy.data.objects.get("floor"):
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.active_object
    floor.name = "floor"
    floor_mat = _make_material(
      "studio_floor_matte", (0.48, 0.48, 0.46, 1.0), roughness=0.82, specular=0.25
    )
    floor.data.materials.append(floor_mat)

  if not bpy.data.objects.get("studio_backdrop"):
    bpy.ops.mesh.primitive_plane_add(
      size=4.0, location=(0.0, 1.45, 1.20), rotation=(1.57079632679, 0.0, 0.0)
    )
    backdrop = bpy.context.active_object
    backdrop.name = "studio_backdrop"
    backdrop.data.materials.append(
      _make_material(
        "studio_backdrop_matte",
        (0.30, 0.30, 0.30, 1.0),
        roughness=0.9,
        specular=0.18,
      )
    )

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

    try:
      obj = _import_mesh_file(mesh_path)
    except (ValueError, RuntimeError) as e:
      print(f"WARNING: could not import mesh for {link_name}: {e}", file=sys.stderr)
      continue

    obj.name = link_name
    obj.rotation_mode = 'QUATERNION'
    _shade_smooth(obj)
    rgba = info.get("material_rgba")
    if rgba is not None:
      _assign_material(obj, _robot_material(link_name, rgba))
    objects[link_name] = obj

  return objects


def import_tool_mesh(mesh_path: str, object_name: str):
  """Import the tool object mesh."""
  obj = _import_mesh_file(mesh_path)
  obj.name = f"tool_{object_name}"
  obj.rotation_mode = 'QUATERNION'
  _shade_smooth(obj)
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
