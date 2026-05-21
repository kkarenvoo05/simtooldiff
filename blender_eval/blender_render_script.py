#!/usr/bin/env python3
# This script runs inside Blender's Python interpreter, not the simtooldiff venv.
# VERIFIED to run end-to-end on Blender 4.2.9 LTS (headless, Cycles CPU) via
# blender_eval/open_loop_smoke_test.py: imports all 36 robot STL meshes + the
# tool OBJ, configures the camera, and renders a frame over the FIFO protocol.
# Still UNVERIFIED against a real collected rollout (needs the cluster). The
# default scene is scripted for parity testing; an optional .blend template can
# still override lighting/materials if needed.
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

USE_MOTION_BLUR = False
USE_DOF = False
CYCLES_SEED = 0
CYCLES_ADAPTIVE_THRESHOLD = 0.008
CYCLES_MIN_SAMPLES = 32

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


def _add_roughness_variation(mat, bsdf, roughness, variation, scale=42.0):
  if variation <= 0 or "Roughness" not in bsdf.inputs:
    return
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  noise = nodes.new(type="ShaderNodeTexNoise")
  noise.inputs["Scale"].default_value = scale
  noise.inputs["Detail"].default_value = 9.0
  noise.inputs["Roughness"].default_value = 0.62
  ramp = nodes.new(type="ShaderNodeValToRGB")
  low = max(0.02, roughness - variation)
  high = min(0.98, roughness + variation)
  ramp.color_ramp.elements[0].position = 0.18
  ramp.color_ramp.elements[0].color = (low, low, low, 1.0)
  ramp.color_ramp.elements[1].position = 1.0
  ramp.color_ramp.elements[1].color = (high, high, high, 1.0)
  links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
  links.new(ramp.outputs["Color"], bsdf.inputs["Roughness"])


def _add_shader_bevel(mat, bsdf, radius):
  if radius <= 0 or "Normal" not in bsdf.inputs:
    return
  bevel = mat.node_tree.nodes.new(type="ShaderNodeBevel")
  bevel.inputs["Radius"].default_value = radius
  mat.node_tree.links.new(bevel.outputs["Normal"], bsdf.inputs["Normal"])


def _make_material(
  name,
  rgba,
  roughness=0.55,
  metallic=0.0,
  specular=0.5,
  coat=0.0,
  roughness_variation=0.0,
  bevel_radius=0.0,
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
    _add_roughness_variation(mat, bsdf, roughness, roughness_variation)
    _add_shader_bevel(mat, bsdf, bevel_radius)
  return mat


def _make_wood_material(name):
  """Create a procedural light-oak material with grain and normal variation."""
  mat = _make_material(
    name,
    (0.66, 0.49, 0.30, 1.0),
    roughness=0.58,
    metallic=0.0,
    specular=0.36,
    coat=0.08,
    roughness_variation=0.12,
  )
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  bsdf = nodes.get("Principled BSDF")
  if bsdf:
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 22.0
    noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Roughness"].default_value = 0.58
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.12
    ramp.color_ramp.elements[0].color = (0.52, 0.36, 0.20, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.72, 0.54, 0.34, 1.0)
    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.018
    bump.inputs["Distance"].default_value = 0.006
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    if "Normal" in bsdf.inputs:
      links.new(noise.outputs["Fac"], bump.inputs["Height"])
      links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
  return mat


def _make_concrete_material(name, base=(0.42, 0.43, 0.41, 1.0), roughness=0.86):
  mat = _make_material(
    name,
    base,
    roughness=roughness,
    specular=0.18,
    roughness_variation=0.08,
  )
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  bsdf = nodes.get("Principled BSDF")
  if bsdf:
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 26.0
    noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Roughness"].default_value = 0.58
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.12
    ramp.color_ramp.elements[0].color = (0.28, 0.29, 0.28, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = base
    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.030
    bump.inputs["Distance"].default_value = 0.030
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    if "Normal" in bsdf.inputs:
      links.new(noise.outputs["Fac"], bump.inputs["Height"])
      links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
  return mat


def _make_marble_material(name):
  mat = _make_material(
    name,
    (0.78, 0.74, 0.66, 1.0),
    roughness=0.26,
    specular=0.46,
    coat=0.10,
    roughness_variation=0.06,
  )
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  bsdf = nodes.get("Principled BSDF")
  if bsdf:
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 15.0
    noise.inputs["Roughness"].default_value = 0.64
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[0].color = (0.50, 0.47, 0.41, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.88, 0.84, 0.76, 1.0)
    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.018
    bump.inputs["Distance"].default_value = 0.010
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    if "Normal" in bsdf.inputs:
      links.new(noise.outputs["Fac"], bump.inputs["Height"])
      links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
  return mat


def _make_emission_material(name, color, strength):
  mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
  mat.use_nodes = True
  nodes = mat.node_tree.nodes
  nodes.clear()
  emission = nodes.new(type="ShaderNodeEmission")
  emission.inputs["Color"].default_value = color
  emission.inputs["Strength"].default_value = strength
  output = nodes.new(type="ShaderNodeOutputMaterial")
  mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
  return mat


def _robot_material(link_name, rgba):
  """Choose PBR-ish params for URDF robot colors."""
  if link_name.startswith("iiwa14_link_") and rgba[0] > 0.9 and rgba[1] > 0.25:
    return _make_material(
      f"urdf_{link_name}_material",
      rgba,
      roughness=0.36,
      specular=0.56,
      coat=0.28,
      roughness_variation=0.08,
      bevel_radius=0.0012,
    )
  if link_name.startswith("iiwa14_link_") and max(rgba[:3]) < 0.45:
    return _make_material(
      f"urdf_{link_name}_material",
      rgba,
      roughness=0.32,
      metallic=0.70,
      specular=0.54,
      roughness_variation=0.10,
      bevel_radius=0.0008,
    )
  return _make_material(
    f"urdf_{link_name}_material",
    rgba,
    roughness=0.50,
    specular=0.42,
    roughness_variation=0.06,
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
  bevel = obj.modifiers.get("softened_edges") or obj.modifiers.new(
    name="softened_edges", type="BEVEL"
  )
  bevel.width = amount
  bevel.segments = segments
  bevel.affect = "EDGES"
  if not obj.modifiers.get("weighted_normals"):
    obj.modifiers.new(name="weighted_normals", type="WEIGHTED_NORMAL")


def _look_at(obj, target):
  direction = mathutils.Vector(target) - obj.location
  obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_cube(name, location, dimensions, mat=None, bevel=0.0, segments=2):
  obj = bpy.data.objects.get(name)
  if obj is None:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
  else:
    obj.location = location
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
  obj.dimensions = dimensions
  bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
  if bevel > 0.0:
    _add_bevel(obj, amount=bevel, segments=segments)
  if mat:
    _assign_material(obj, mat)
  return obj


def _add_area_light(name, location, target, energy, size, size_y=None):
  bpy.ops.object.light_add(type="AREA", location=location)
  light = bpy.context.active_object
  light.name = name
  light.data.energy = energy
  if size_y is not None and hasattr(light.data, "shape"):
    light.data.shape = "RECTANGLE"
    light.data.size = size
    if hasattr(light.data, "size_y"):
      light.data.size_y = size_y
  else:
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


def _add_lab_strip_light(target):
  """Visible overhead fixture plus an actual rectangular emitter."""
  housing_mat = _make_material(
    "overhead_fixture_black", (0.015, 0.014, 0.012, 1.0), roughness=0.62, specular=0.25
  )
  diffuser_mat = _make_emission_material(
    "overhead_diffuser_emission", (1.0, 0.95, 0.84, 1.0), strength=2.0
  )
  cable_mat = _make_material(
    "fixture_cable_dark", (0.035, 0.035, 0.033, 1.0), roughness=0.72, specular=0.2
  )

  _add_cube(
    "overhead_strip_housing",
    location=(0.08, -0.24, 1.54),
    dimensions=(1.25, 0.09, 0.045),
    mat=housing_mat,
    bevel=0.004,
  )
  _add_cube(
    "overhead_strip_diffuser",
    location=(0.08, -0.255, 1.515),
    dimensions=(1.14, 0.018, 0.012),
    mat=diffuser_mat,
    bevel=0.003,
  )
  for x in (-0.42, 0.58):
    _add_cube(
      f"overhead_strip_cable_{x:.1f}",
      location=(x, -0.24, 1.78),
      dimensions=(0.012, 0.012, 0.48),
      mat=cable_mat,
      bevel=0.002,
    )
  _add_area_light(
    "overhead_rect_key",
    location=(0.08, -0.255, 1.49),
    target=target,
    energy=95.0,
    size=1.20,
    size_y=0.28,
  )


def _set_color_management(scene):
  """Prefer a product-render transform, falling back across Blender versions."""
  transforms = {
    item.identifier
    for item in bpy.types.ColorManagedViewSettings.bl_rna.properties[
      "view_transform"
    ].enum_items
  }
  if "AgX" in transforms:
    scene.view_settings.view_transform = "AgX"
  elif "Filmic" in transforms:
    scene.view_settings.view_transform = "Filmic"
  else:
    scene.view_settings.view_transform = "Standard"

  looks = {
    item.identifier
    for item in bpy.types.ColorManagedViewSettings.bl_rna.properties["look"].enum_items
  }
  if "Medium High Contrast" in looks:
    scene.view_settings.look = "Medium High Contrast"
  elif "AgX - Medium High Contrast" in looks:
    scene.view_settings.look = "AgX - Medium High Contrast"
  elif "None" in looks:
    scene.view_settings.look = "None"

  scene.view_settings.exposure = -1.15
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

    # Lab/product lighting: deterministic, sharp policy-observation frames
    # with no per-frame randomization, no DoF, and no motion blur.
    scene_center = (0.0, 0.05, 0.72)
    _add_lab_strip_light(scene_center)
    _add_area_light(
      "key_softbox",
      location=(-1.15, -1.10, 1.55),
      target=scene_center,
      energy=42.0,
      size=1.60,
    )
    _add_area_light(
      "fill_softbox",
      location=(1.35, -0.70, 1.05),
      target=scene_center,
      energy=14.0,
      size=2.70,
    )
    _add_area_light(
      "rim_softbox",
      location=(0.35, 1.20, 1.35),
      target=(0.0, 0.10, 0.80),
      energy=14.0,
      size=1.25,
    )
    _add_spot_light(
      "tool_spot",
      location=(-0.55, -0.95, 1.30),
      target=(-0.02, 0.04, 0.60),
      energy=1.4,
      size=1.25,
      blend=0.85,
    )

    # World background
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
      bg.inputs["Color"].default_value = (0.46, 0.47, 0.45, 1.0)
      bg.inputs["Strength"].default_value = 0.07

  # Render engine
  if args.engine == "cycles":
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = CYCLES_SEED
    scene.cycles.max_bounces = 10
    scene.cycles.diffuse_bounces = 5
    scene.cycles.glossy_bounces = 6
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.sample_clamp_indirect = 10.0
    scene.cycles.blur_glossy = 0.35
    if hasattr(scene.cycles, "use_adaptive_sampling"):
      scene.cycles.use_adaptive_sampling = True
    if hasattr(scene.cycles, "adaptive_threshold"):
      scene.cycles.adaptive_threshold = CYCLES_ADAPTIVE_THRESHOLD
    if hasattr(scene.cycles, "adaptive_min_samples"):
      scene.cycles.adaptive_min_samples = max(1, min(args.samples, CYCLES_MIN_SAMPLES))
    if hasattr(scene.cycles, "caustics_reflective"):
      scene.cycles.caustics_reflective = True
    if hasattr(scene.cycles, "caustics_refractive"):
      scene.cycles.caustics_refractive = True

    scene.render.use_persistent_data = True
    scene.render.use_motion_blur = USE_MOTION_BLUR

    # Prefer OptiX/CUDA on NVIDIA, but keep the script runnable on CPU-only hosts.
    scene.cycles.device = 'CPU'
    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
      for compute_type in ("OPTIX", "CUDA"):
        try:
          prefs.preferences.compute_device_type = compute_type
          prefs.preferences.get_devices()
        except Exception:
          continue
        enabled = False
        for device in prefs.preferences.devices:
          device.use = device.type in {"OPTIX", "CUDA"}
          enabled = enabled or device.use
        if enabled:
          scene.cycles.device = 'GPU'
          try:
            scene.cycles.denoiser = 'OPTIX'
          except Exception:
            pass
          break
  else:
    scene.render.engine = _eevee_engine_name()
    scene.render.use_motion_blur = USE_MOTION_BLUR

  # Resolution
  scene.render.resolution_x = args.width
  scene.render.resolution_y = args.height
  scene.render.resolution_percentage = 100
  scene.render.image_settings.file_format = 'PNG'
  _set_color_management(scene)

  # IsaacGym scene furniture from assets/urdf/table_narrow_nail.urdf.
  # The table actor is centered at (0, 0, TABLE_Z); the URDF box is centered
  # on that actor with size 0.475 x 0.4 x 0.3, plus a small gray nail.
  table_mat = _make_wood_material("table_light_oak")
  nail_mat = _make_marble_material("table_nail_marble")
  floor_mat = _make_concrete_material(
    "lab_floor_matte", base=(0.49, 0.50, 0.47, 1.0), roughness=0.82
  )
  wall_mat = _make_concrete_material(
    "lab_wall_concrete", base=(0.39, 0.40, 0.38, 1.0), roughness=0.88
  )
  beam_mat = _make_material(
    "lab_wall_beam_dark", (0.17, 0.18, 0.17, 1.0), roughness=0.74, specular=0.20
  )
  bench_mat = _make_material(
    "rear_bench_laminate", (0.62, 0.61, 0.56, 1.0), roughness=0.64, specular=0.28
  )
  bench_leg_mat = _make_material(
    "rear_bench_frame", (0.12, 0.13, 0.13, 1.0), roughness=0.55, metallic=0.55
  )

  _add_cube(
    "table",
    location=(0.0, 0.0, 0.38),
    dimensions=(0.475, 0.4, 0.3),
    mat=table_mat,
    bevel=0.010,
    segments=4,
  )
  _add_cube(
    "table_nail",
    location=(-0.16, 0.06, 0.38 + 0.175),
    dimensions=(0.03, 0.03, 0.06),
    mat=nail_mat,
    bevel=0.002,
    segments=2,
  )

  # A neutral lab scene replaces the old checkerboard/gradient placeholder.
  if not bpy.data.objects.get("floor"):
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.active_object
    floor.name = "floor"
    floor.data.materials.append(floor_mat)

  _add_cube(
    "concrete_back_wall",
    location=(0.0, 1.58, 0.88),
    dimensions=(3.2, 0.08, 1.76),
    mat=wall_mat,
  )
  for x in (-1.08, 1.08):
    _add_cube(
      f"wall_vertical_beam_{x:.1f}",
      location=(x, 1.52, 0.88),
      dimensions=(0.055, 0.10, 1.78),
      mat=beam_mat,
      bevel=0.002,
    )
  _add_cube(
    "wall_mid_seam",
    location=(0.0, 1.515, 0.88),
    dimensions=(3.2, 0.10, 0.022),
    mat=beam_mat,
  )
  _add_cube(
    "rear_workbench_top",
    location=(0.25, 0.92, 0.44),
    dimensions=(1.65, 0.42, 0.055),
    mat=bench_mat,
    bevel=0.006,
  )
  for x in (-0.45, 0.95):
    for y in (0.76, 1.08):
      _add_cube(
        f"rear_workbench_leg_{x:.1f}_{y:.1f}",
        location=(x, y, 0.22),
        dimensions=(0.035, 0.035, 0.42),
        mat=bench_leg_mat,
        bevel=0.002,
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
  cam_data.dof.use_dof = USE_DOF

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
