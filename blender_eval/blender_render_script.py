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
import math
import os
import sys
import tempfile

SIMTOOLDIFF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HDRI_PATH = os.path.join(
  SIMTOOLDIFF_ROOT, "assets", "blender", "hdri", "workshop_1k.hdr"
)
USE_MOTION_BLUR = False
USE_DOF = False
CYCLES_SEED = 0
CYCLES_ADAPTIVE_THRESHOLD = 0.005
CYCLES_MIN_SAMPLES = 48
LIGHTING_PRESET_ENV = "SIMTOOLDIFF_LIGHTING_PRESET"
DEFAULT_LIGHTING_PRESET = "softbox_grid"
_DIAGNOSTICS_PRINTED = False

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


def _get_principled_bsdf(mat):
  if not mat or not mat.use_nodes:
    return None
  for node in mat.node_tree.nodes:
    if node.type == "BSDF_PRINCIPLED":
      return node
  return mat.node_tree.nodes.get("Principled BSDF")


def _asset_path(*parts):
  return os.path.join(SIMTOOLDIFF_ROOT, "assets", "blender", *parts)


def _load_image(path, *, colorspace=None):
  image = bpy.data.images.get(os.path.basename(path))
  if image is None:
    image = bpy.data.images.load(path)
  if colorspace:
    try:
      image.colorspace_settings.name = colorspace
    except TypeError:
      pass
  return image


def _resolve_hdri_path():
  override = os.environ.get("SIMTOOLDIFF_HDRI_PATH")
  if override:
    if os.path.exists(override):
      return override
    print(f"WARNING: SIMTOOLDIFF_HDRI_PATH does not exist: {override}", file=sys.stderr)
  if os.path.exists(DEFAULT_HDRI_PATH):
    return DEFAULT_HDRI_PATH
  print(f"WARNING: default HDRI not found: {DEFAULT_HDRI_PATH}", file=sys.stderr)
  return None


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


def _add_noise_bump(mat, bsdf, *, scale=80.0, detail=10.0, strength=0.01, distance=0.004):
  if "Normal" not in bsdf.inputs:
    return
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  normal_input = bsdf.inputs["Normal"]
  previous_normal = normal_input.links[0].from_socket if normal_input.links else None
  for link in list(normal_input.links):
    links.remove(link)

  noise = nodes.new(type="ShaderNodeTexNoise")
  noise.inputs["Scale"].default_value = scale
  noise.inputs["Detail"].default_value = detail
  noise.inputs["Roughness"].default_value = 0.58
  bump = nodes.new(type="ShaderNodeBump")
  bump.inputs["Strength"].default_value = strength
  bump.inputs["Distance"].default_value = distance
  if previous_normal is not None and "Normal" in bump.inputs:
    links.new(previous_normal, bump.inputs["Normal"])
  links.new(noise.outputs["Fac"], bump.inputs["Height"])
  links.new(bump.outputs["Normal"], normal_input)


def _add_base_color_variation(mat, bsdf, color_low, color_high, scale=24.0, detail=8.0):
  if "Base Color" not in bsdf.inputs:
    return
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  noise = nodes.new(type="ShaderNodeTexNoise")
  noise.inputs["Scale"].default_value = scale
  noise.inputs["Detail"].default_value = detail
  noise.inputs["Roughness"].default_value = 0.58
  ramp = nodes.new(type="ShaderNodeValToRGB")
  ramp.color_ramp.elements[0].position = 0.18
  ramp.color_ramp.elements[0].color = color_low
  ramp.color_ramp.elements[1].position = 1.0
  ramp.color_ramp.elements[1].color = color_high
  links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
  links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])


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


def _enhance_existing_material(
  mat,
  name,
  *,
  roughness=0.55,
  metallic=0.0,
  specular=0.35,
  coat=0.0,
  roughness_variation=0.04,
  bump_strength=0.0,
  bump_distance=0.003,
  bump_scale=80.0,
):
  """Keep imported texture nodes but replace the weak OBJ material response."""
  if mat is None:
    return _make_material(
      name,
      (0.55, 0.55, 0.55, 1.0),
      roughness=roughness,
      metallic=metallic,
      specular=specular,
      coat=coat,
      roughness_variation=roughness_variation,
    )
  mat.name = name
  mat.use_nodes = True
  bsdf = mat.node_tree.nodes.get("Principled BSDF")
  if bsdf:
    _set_principled_input(bsdf, ("Roughness",), roughness)
    _set_principled_input(bsdf, ("Metallic",), metallic)
    _set_principled_input(bsdf, ("Specular IOR Level", "Specular"), specular)
    _set_principled_input(bsdf, ("Coat Weight", "Clearcoat"), coat)
    _add_roughness_variation(mat, bsdf, roughness, roughness_variation)
    if bump_strength > 0.0:
      _add_noise_bump(
        mat,
        bsdf,
        scale=bump_scale,
        detail=10.0,
        strength=bump_strength,
        distance=bump_distance,
      )
  return mat


def _make_textured_pbr_material(
  name,
  diffuse_path,
  roughness_path,
  normal_path,
  *,
  roughness=0.5,
  specular=0.35,
  coat=0.0,
  normal_strength=0.10,
  texture_scale=(1.0, 1.0, 1.0),
):
  """Create a PBR material from image maps, falling back to scalar inputs."""
  mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
  mat.use_nodes = True
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  nodes.clear()

  output = nodes.new(type="ShaderNodeOutputMaterial")
  bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
  links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
  _set_principled_input(bsdf, ("Metallic",), 0.0)
  _set_principled_input(bsdf, ("Roughness",), roughness)
  _set_principled_input(bsdf, ("Specular IOR Level", "Specular"), specular)
  _set_principled_input(bsdf, ("Coat Weight", "Clearcoat"), coat)

  texcoord = nodes.new(type="ShaderNodeTexCoord")
  mapping = nodes.new(type="ShaderNodeMapping")
  if "Scale" in mapping.inputs:
    mapping.inputs["Scale"].default_value = texture_scale
  links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])

  if diffuse_path and os.path.exists(diffuse_path):
    diffuse = nodes.new(type="ShaderNodeTexImage")
    diffuse.image = _load_image(diffuse_path, colorspace="sRGB")
    diffuse.extension = "REPEAT"
    links.new(mapping.outputs["Vector"], diffuse.inputs["Vector"])
    links.new(diffuse.outputs["Color"], bsdf.inputs["Base Color"])

  if roughness_path and os.path.exists(roughness_path) and "Roughness" in bsdf.inputs:
    rough = nodes.new(type="ShaderNodeTexImage")
    rough.image = _load_image(roughness_path, colorspace="Non-Color")
    rough.extension = "REPEAT"
    links.new(mapping.outputs["Vector"], rough.inputs["Vector"])
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

  if normal_path and os.path.exists(normal_path) and "Normal" in bsdf.inputs:
    normal_tex = nodes.new(type="ShaderNodeTexImage")
    normal_tex.image = _load_image(normal_path, colorspace="Non-Color")
    normal_tex.extension = "REPEAT"
    normal_map = nodes.new(type="ShaderNodeNormalMap")
    normal_map.inputs["Strength"].default_value = normal_strength
    links.new(mapping.outputs["Vector"], normal_tex.inputs["Vector"])
    links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

  return mat


def _make_wood_material(
  name,
  *,
  roughness=0.50,
  specular=0.34,
  coat=0.035,
  normal_strength=0.23,
  texture_scale=(2.35, 2.35, 1.0),
):
  """Create a procedural light-oak material with grain and normal variation."""
  diffuse_path = _asset_path("textures", "oak_veneer_01", "oak_veneer_01_diff_1k.jpg")
  roughness_path = _asset_path("textures", "oak_veneer_01", "oak_veneer_01_rough_1k.jpg")
  normal_path = _asset_path("textures", "oak_veneer_01", "oak_veneer_01_nor_gl_1k.jpg")
  if os.path.exists(diffuse_path) and os.path.exists(roughness_path) and os.path.exists(normal_path):
    return _make_textured_pbr_material(
      name,
      diffuse_path,
      roughness_path,
      normal_path,
      roughness=roughness,
      specular=specular,
      coat=coat,
      normal_strength=normal_strength,
      texture_scale=texture_scale,
    )

  mat = _make_material(
    name,
    (0.60, 0.44, 0.27, 1.0),
    roughness=max(roughness, 0.58),
    metallic=0.0,
    specular=specular,
    coat=coat,
    roughness_variation=0.08,
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
    ramp.color_ramp.elements[0].color = (0.46, 0.31, 0.17, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.69, 0.52, 0.33, 1.0)
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
  diffuse_path = _asset_path("textures", "marble_01", "marble_01_diff_1k.jpg")
  roughness_path = _asset_path("textures", "marble_01", "marble_01_rough_1k.jpg")
  normal_path = _asset_path("textures", "marble_01", "marble_01_nor_gl_1k.jpg")
  if os.path.exists(diffuse_path) and os.path.exists(roughness_path) and os.path.exists(normal_path):
    return _make_textured_pbr_material(
      name,
      diffuse_path,
      roughness_path,
      normal_path,
      roughness=0.22,
      specular=0.48,
      coat=0.12,
      normal_strength=0.09,
      texture_scale=(1.0, 1.0, 1.0),
    )

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


def _setup_world_hdri(hdri_path, strength=1.0, visible_to_camera=True):
  world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
  bpy.context.scene.world = world
  world.use_nodes = True
  nodes = world.node_tree.nodes
  links = world.node_tree.links
  nodes.clear()

  output = nodes.new(type="ShaderNodeOutputWorld")
  background = nodes.new(type="ShaderNodeBackground")
  background.inputs["Strength"].default_value = strength

  if hdri_path and os.path.exists(hdri_path):
    env = nodes.new(type="ShaderNodeTexEnvironment")
    env.image = _load_image(hdri_path, colorspace="Linear Rec.709")
    links.new(env.outputs["Color"], background.inputs["Color"])
  else:
    background.inputs["Color"].default_value = (0.46, 0.47, 0.45, 1.0)

  links.new(background.outputs["Background"], output.inputs["Surface"])
  if hasattr(world, "cycles_visibility"):
    world.cycles_visibility.camera = visible_to_camera
  return world


def _render_diagnostics_enabled():
  return os.environ.get("SIMTOOLDIFF_RENDER_DIAGNOSTICS", "").lower() in {
    "1", "true", "yes", "on",
  }


def _input_value(bsdf, names):
  if not bsdf:
    return None
  for name in names:
    if name in bsdf.inputs:
      value = bsdf.inputs[name].default_value
      try:
        return tuple(round(float(x), 4) for x in value)
      except TypeError:
        return round(float(value), 4)
  return None


def _input_link_count(bsdf, names):
  if not bsdf:
    return 0
  for name in names:
    if name in bsdf.inputs:
      return len(bsdf.inputs[name].links)
  return 0


def _print_material_diagnostics(label, obj):
  if obj is None:
    print(f"DIAG_MATERIAL {label} missing_object", file=sys.stderr, flush=True)
    return
  if not hasattr(obj.data, "materials"):
    print(f"DIAG_MATERIAL {label} no_material_data", file=sys.stderr, flush=True)
    return
  materials = [m for m in obj.data.materials if m is not None]
  print(
    f"DIAG_MATERIAL {label} object={obj.name} slots={[m.name for m in materials]}",
    file=sys.stderr,
    flush=True,
  )
  for mat in materials:
    bsdf = _get_principled_bsdf(mat)
    print(
      "DIAG_MATERIAL_NODE "
      f"{label}/{mat.name} "
      f"base={_input_value(bsdf, ('Base Color',))} "
      f"metallic={_input_value(bsdf, ('Metallic',))} "
      f"roughness={_input_value(bsdf, ('Roughness',))} "
      f"coat={_input_value(bsdf, ('Coat Weight', 'Clearcoat'))} "
      f"base_links={_input_link_count(bsdf, ('Base Color',))} "
      f"roughness_links={_input_link_count(bsdf, ('Roughness',))} "
      f"normal_links={_input_link_count(bsdf, ('Normal',))}",
      file=sys.stderr,
      flush=True,
    )


def _print_render_diagnostics(scene, robot_objects, tool_object):
  world = scene.world
  print(f"DIAG_RENDER_ENGINE_SETUP {scene.render.engine}", file=sys.stderr, flush=True)
  if scene.render.engine == "CYCLES":
    print(
      "DIAG_CYCLES "
      f"device={scene.cycles.device} samples={scene.cycles.samples} "
      f"seed={scene.cycles.seed} denoise={scene.cycles.use_denoising}",
      file=sys.stderr,
      flush=True,
    )

  print(
    f"DIAG_RESOLUTION {scene.render.resolution_x} {scene.render.resolution_y} "
    f"{scene.render.resolution_percentage}",
    file=sys.stderr,
    flush=True,
  )
  cam = scene.camera
  if cam:
    hfov = 2.0 * math.atan((cam.data.sensor_width / 2.0) / cam.data.lens)
    print(
      "DIAG_CAMERA "
      f"name={cam.name} loc={tuple(round(float(v), 4) for v in cam.location)} "
      f"lens={cam.data.lens:.4f} sensor=({cam.data.sensor_width:.4f},"
      f"{cam.data.sensor_height:.4f}) hfov={math.degrees(hfov):.4f} "
      f"dof={cam.data.dof.use_dof}",
      file=sys.stderr,
      flush=True,
    )

  if world:
    print(
      f"DIAG_WORLD exists=True use_nodes={world.use_nodes} name={world.name}",
      file=sys.stderr,
      flush=True,
    )
    if hasattr(world, "cycles_visibility"):
      print(
        f"DIAG_WORLD_CAMERA_VISIBLE {world.cycles_visibility.camera}",
        file=sys.stderr,
        flush=True,
      )
    if world.use_nodes and world.node_tree:
      nodes = [(n.name, n.type) for n in world.node_tree.nodes]
      links = [
        f"{l.from_node.name}:{l.from_socket.name}->{l.to_node.name}:{l.to_socket.name}"
        for l in world.node_tree.links
      ]
      env_images = [
        n.image.filepath if getattr(n, "image", None) else None
        for n in world.node_tree.nodes
        if n.type == "TEX_ENVIRONMENT"
      ]
      print(f"DIAG_WORLD_NODES {nodes}", file=sys.stderr, flush=True)
      print(f"DIAG_WORLD_LINKS {links}", file=sys.stderr, flush=True)
      print(f"DIAG_HDRI_IMAGES {env_images}", file=sys.stderr, flush=True)

  lights = [
    (
      obj.name,
      obj.data.type,
      round(float(obj.data.energy), 4),
      tuple(round(float(v), 4) for v in obj.location),
    )
    for obj in bpy.data.objects
    if obj.type == "LIGHT"
  ]
  print(
    f"DIAG_LIGHTING_PRESET {scene.get('simtooldiff_lighting_preset', 'template')}",
    file=sys.stderr,
    flush=True,
  )
  print(f"DIAG_LIGHTS {lights}", file=sys.stderr, flush=True)
  print(
    f"DIAG_OBJECT_NAMES {sorted(obj.name for obj in bpy.data.objects)}",
    file=sys.stderr,
    flush=True,
  )

  _print_material_diagnostics("arm_orange_link_1", bpy.data.objects.get("iiwa14_link_1"))
  _print_material_diagnostics("arm_orange_link_2", bpy.data.objects.get("iiwa14_link_2"))
  _print_material_diagnostics("arm_gray_link_7", bpy.data.objects.get("iiwa14_link_7"))
  _print_material_diagnostics(
    "hammer", tool_object or next((o for o in bpy.data.objects if o.name.startswith("tool_")), None)
  )
  _print_material_diagnostics("box_table", bpy.data.objects.get("table"))
  _print_material_diagnostics("small_block", bpy.data.objects.get("table_nail"))


def _robot_material(link_name, rgba):
  """Choose PBR-ish params for URDF robot colors."""
  if link_name.startswith("iiwa14_link_") and rgba[0] > 0.9 and rgba[1] > 0.25:
    mat = _make_material(
      f"urdf_{link_name}_material",
      (0.72, 0.27, 0.030, rgba[3]),
      roughness=0.46,
      specular=0.46,
      coat=0.18,
      roughness_variation=0.14,
      bevel_radius=0.0016,
    )
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
      _add_base_color_variation(
        mat,
        bsdf,
        (0.54, 0.18, 0.018, rgba[3]),
        (0.88, 0.35, 0.050, rgba[3]),
        scale=13.0,
        detail=8.0,
      )
      _add_noise_bump(
        mat,
        bsdf,
        scale=95.0,
        detail=12.0,
        strength=0.007,
        distance=0.0035,
      )
    return mat
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


def _make_steel_material(name, base=(0.55, 0.54, 0.51, 1.0), roughness=0.28):
  steel = _make_material(
    name,
    base,
    roughness=roughness,
    metallic=1.0,
    specular=0.50,
    roughness_variation=0.10,
    bevel_radius=0.0008,
  )
  steel_bsdf = steel.node_tree.nodes.get("Principled BSDF")
  if steel_bsdf:
    _add_base_color_variation(
      steel,
      steel_bsdf,
      (0.38, 0.37, 0.35, 1.0),
      (0.72, 0.70, 0.66, 1.0),
      scale=75.0,
      detail=10.0,
    )
    _add_noise_bump(steel, steel_bsdf, scale=120.0, strength=0.004, distance=0.002)
  return steel


def _make_rubber_material(name, base=(0.018, 0.017, 0.015, 1.0), roughness=0.64):
  rubber = _make_material(
    name,
    base,
    roughness=roughness,
    metallic=0.0,
    specular=0.24,
    roughness_variation=0.11,
    bevel_radius=0.0005,
  )
  bsdf = rubber.node_tree.nodes.get("Principled BSDF")
  if bsdf:
    _add_base_color_variation(
      rubber,
      bsdf,
      tuple(max(0.0, c * 0.45) for c in base[:3]) + (base[3],),
      tuple(min(1.0, c * 1.65 + 0.01) for c in base[:3]) + (base[3],),
      scale=48.0,
      detail=8.0,
    )
    _add_noise_bump(rubber, bsdf, scale=90.0, strength=0.010, distance=0.004)
  return rubber


def _make_plastic_material(name, base, roughness=0.48, coat=0.08):
  mat = _make_material(
    name,
    base,
    roughness=roughness,
    metallic=0.0,
    specular=0.34,
    coat=coat,
    roughness_variation=0.08,
    bevel_radius=0.0004,
  )
  bsdf = mat.node_tree.nodes.get("Principled BSDF")
  if bsdf:
    _add_noise_bump(mat, bsdf, scale=70.0, strength=0.004, distance=0.002)
  return mat


def _make_bristle_material(name, base=(0.72, 0.62, 0.36, 1.0)):
  mat = _make_material(
    name,
    base,
    roughness=0.88,
    metallic=0.0,
    specular=0.10,
    roughness_variation=0.16,
    bevel_radius=0.0002,
  )
  bsdf = mat.node_tree.nodes.get("Principled BSDF")
  if bsdf:
    _add_base_color_variation(
      mat,
      bsdf,
      tuple(max(0.0, c * 0.70) for c in base[:3]) + (base[3],),
      tuple(min(1.0, c * 1.25) for c in base[:3]) + (base[3],),
      scale=140.0,
      detail=12.0,
    )
    _add_noise_bump(mat, bsdf, scale=180.0, strength=0.020, distance=0.006)
  return mat


def _mesh_bounds(obj):
  verts = obj.data.vertices
  mins = [min(v.co[i] for v in verts) for i in range(3)]
  maxs = [max(v.co[i] for v in verts) for i in range(3)]
  spans = [max(maxs[i] - mins[i], 1e-6) for i in range(3)]
  return mins, maxs, spans


def _poly_center(obj, poly):
  verts = obj.data.vertices
  return sum(
    (verts[i].co for i in poly.vertices), mathutils.Vector((0.0, 0.0, 0.0))
  ) / len(poly.vertices)


def _x_norm(center, bounds):
  mins, _, spans = bounds
  return (center.x - mins[0]) / spans[0]


def _z_norm(center, bounds):
  mins, _, spans = bounds
  return (center.z - mins[2]) / spans[2]


def _replace_material_slots(obj, materials):
  obj.data.materials.clear()
  for mat in materials:
    obj.data.materials.append(mat)


def _apply_claw_hammer_materials(obj):
  """Assign head/handle PBR materials to the single imported hammer mesh."""
  rubber = _make_rubber_material("claw_hammer_handle_rubber")
  steel = _make_steel_material("claw_hammer_head_steel")
  _replace_material_slots(obj, [rubber, steel])
  bounds = _mesh_bounds(obj)
  for poly in obj.data.polygons:
    center = _poly_center(obj, poly)
    # The claw hammer OBJ is a single mesh. The head/claw geometry occupies the
    # wide end and the handle stays comparatively narrow through the middle.
    x = _x_norm(center, bounds)
    is_head = x > 0.80 or abs(center.y) > 0.022 or x < 0.10
    poly.material_index = 1 if is_head else 0


def _apply_mallet_materials(obj, imported):
  head = _make_rubber_material("mallet_head_rubber", (0.025, 0.030, 0.035, 1.0), 0.70)
  handle = _enhance_existing_material(
    imported,
    "mallet_handle_textured_plastic",
    roughness=0.50,
    specular=0.30,
    coat=0.04,
    roughness_variation=0.08,
    bump_strength=0.005,
  )
  _replace_material_slots(obj, [handle, head])
  bounds = _mesh_bounds(obj)
  for poly in obj.data.polygons:
    center = _poly_center(obj, poly)
    x = _x_norm(center, bounds)
    poly.material_index = 1 if x < 0.30 or (x < 0.42 and abs(center.y) > 0.026) else 0


def _apply_screwdriver_materials(obj, object_name, imported):
  handle = _enhance_existing_material(
    imported,
    f"{object_name}_handle_textured_plastic",
    roughness=0.50,
    specular=0.32,
    coat=0.08,
    roughness_variation=0.08,
    bump_strength=0.004,
  )
  shaft = _make_steel_material(f"{object_name}_brushed_steel", roughness=0.24)
  tip = _make_steel_material(f"{object_name}_dark_tip", (0.20, 0.20, 0.19, 1.0), 0.30)
  _replace_material_slots(obj, [handle, shaft, tip])
  bounds = _mesh_bounds(obj)
  for poly in obj.data.polygons:
    x = _x_norm(_poly_center(obj, poly), bounds)
    poly.material_index = 2 if x > 0.92 else 1 if x > 0.38 else 0


def _apply_marker_materials(obj, object_name, imported):
  # Marker labels and cap colors live in the source texture, so preserve them.
  mat = _enhance_existing_material(
    imported,
    f"{object_name}_printed_plastic",
    roughness=0.43,
    specular=0.36,
    coat=0.14,
    roughness_variation=0.05,
    bump_strength=0.003,
    bump_scale=95.0,
  )
  _replace_material_slots(obj, [mat])


def _apply_eraser_materials(obj, object_name, imported):
  body = _enhance_existing_material(
    imported,
    f"{object_name}_matte_rubber_texture",
    roughness=0.76,
    specular=0.16,
    coat=0.0,
    roughness_variation=0.10,
    bump_strength=0.010,
    bump_scale=110.0,
  )
  if object_name == "handle_eraser":
    bristles = _make_bristle_material("handle_eraser_bristles", (0.72, 0.66, 0.34, 1.0))
    _replace_material_slots(obj, [body, bristles])
    bounds = _mesh_bounds(obj)
    for poly in obj.data.polygons:
      center = _poly_center(obj, poly)
      poly.material_index = 1 if _z_norm(center, bounds) < 0.24 else 0
  else:
    _replace_material_slots(obj, [body])


def _apply_brush_materials(obj, object_name, imported):
  plastic = _enhance_existing_material(
    imported,
    f"{object_name}_textured_plastic",
    roughness=0.54,
    specular=0.30,
    coat=0.06,
    roughness_variation=0.08,
    bump_strength=0.004,
  )
  bristle_color = (
    (0.72, 0.63, 0.38, 1.0)
    if object_name == "blue_brush"
    else (0.025, 0.023, 0.021, 1.0)
  )
  bristles = _make_bristle_material(f"{object_name}_bristles", bristle_color)
  _replace_material_slots(obj, [plastic, bristles])
  bounds = _mesh_bounds(obj)
  for poly in obj.data.polygons:
    center = _poly_center(obj, poly)
    x = _x_norm(center, bounds)
    poly.material_index = 1 if x < 0.33 or (x < 0.50 and abs(center.y) > 0.035) else 0


def _apply_spatula_materials(obj, object_name, imported):
  handle = _enhance_existing_material(
    imported,
    f"{object_name}_handle_texture",
    roughness=0.58,
    specular=0.26,
    coat=0.04,
    roughness_variation=0.08,
    bump_strength=0.004,
  )
  if object_name == "flat_spatula":
    blade = _make_steel_material("flat_spatula_satin_blade", (0.46, 0.45, 0.42, 1.0), 0.34)
  else:
    blade = _make_rubber_material("spoon_spatula_silicone_bowl", (0.025, 0.024, 0.022, 1.0), 0.72)
  _replace_material_slots(obj, [handle, blade])
  bounds = _mesh_bounds(obj)
  for poly in obj.data.polygons:
    center = _poly_center(obj, poly)
    x = _x_norm(center, bounds)
    is_blade = x < 0.34 or (x < 0.46 and abs(center.y) > 0.025)
    poly.material_index = 1 if is_blade else 0


def _apply_tool_materials(obj, object_name):
  imported = obj.data.materials[0] if obj.data.materials else None
  if object_name == "claw_hammer":
    _apply_claw_hammer_materials(obj)
  elif object_name == "mallet_hammer":
    _apply_mallet_materials(obj, imported)
  elif object_name in {"long_screwdriver", "short_screwdriver"}:
    _apply_screwdriver_materials(obj, object_name, imported)
  elif object_name in {"sharpie_marker", "staples_marker"}:
    _apply_marker_materials(obj, object_name, imported)
  elif object_name in {"flat_eraser", "handle_eraser"}:
    _apply_eraser_materials(obj, object_name, imported)
  elif object_name in {"blue_brush", "red_brush"}:
    _apply_brush_materials(obj, object_name, imported)
  elif object_name in {"flat_spatula", "spoon_spatula"}:
    _apply_spatula_materials(obj, object_name, imported)
  else:
    mat = _enhance_existing_material(
      imported,
      f"{object_name}_textured_pbr",
      roughness=0.56,
      specular=0.30,
      coat=0.04,
      roughness_variation=0.08,
      bump_strength=0.004,
    )
    _replace_material_slots(obj, [mat])


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


def _assign_table_box_materials(obj, top_mat, side_mat):
  """Give the table separate top/side responses while preserving its geometry."""
  obj.data.materials.clear()
  obj.data.materials.append(top_mat)
  obj.data.materials.append(side_mat)
  for poly in obj.data.polygons:
    poly.material_index = 0 if poly.normal.z > 0.5 else 1


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
    "overhead_diffuser_emission", (1.0, 0.95, 0.84, 1.0), strength=0.9
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
    energy=26.0,
    size=1.20,
    size_y=0.28,
  )


def _setup_clean_eval_lighting(target=(0.0, 0.0, 0.45)):
  """One dominant key plus weak fill for clearer, stable Cycles shadows."""
  target = mathutils.Vector(target)
  _add_area_light(
    "Key",
    location=(0.80, -0.90, 1.60),
    target=target,
    energy=145.0,
    size=0.52,
    size_y=0.34,
  )
  _add_area_light(
    "Fill",
    location=(-1.20, -0.50, 1.00),
    target=target,
    energy=22.0,
    size=2.00,
  )


def _clear_scene_lights():
  for obj in list(bpy.data.objects):
    if obj.type == "LIGHT":
      bpy.data.objects.remove(obj, do_unlink=True)


def _setup_reference_area_lighting():
  """Area-light-dominant lab setup, with HDRI only as weak reflection fill."""
  target = (0.0, 0.03, 0.62)
  _add_area_light(
    "OverheadStrip",
    location=(0.10, -0.32, 1.48),
    target=target,
    energy=68.0,
    size=1.45,
    size_y=0.18,
  )
  _add_area_light(
    "CameraSoftbox",
    location=(-0.85, -1.22, 1.20),
    target=(0.0, 0.02, 0.60),
    energy=38.0,
    size=1.10,
    size_y=0.75,
  )
  _add_area_light(
    "RearRim",
    location=(0.82, 0.76, 1.20),
    target=(0.02, 0.06, 0.66),
    energy=18.0,
    size=0.72,
  )
  _add_area_light(
    "ShadowFill",
    location=(1.18, -0.92, 0.92),
    target=(0.0, 0.04, 0.54),
    energy=4.0,
    size=2.40,
  )


def _setup_softbox_grid_lighting():
  """Softer product-lighting variant: broad key plus strip and rim."""
  _add_area_light(
    "LargeLeftSoftbox",
    location=(-0.84, -1.22, 1.38),
    target=(0.0, 0.02, 0.62),
    energy=58.0,
    size=1.36,
    size_y=0.86,
  )
  _add_area_light(
    "TopStrip",
    location=(0.08, -0.34, 1.54),
    target=(0.0, 0.02, 0.62),
    energy=50.0,
    size=1.48,
    size_y=0.18,
  )
  _add_area_light(
    "RightFill",
    location=(1.25, -0.80, 0.95),
    target=(0.0, 0.02, 0.54),
    energy=4.5,
    size=2.60,
  )
  _add_area_light(
    "BackEdge",
    location=(0.20, 0.92, 1.22),
    target=(0.02, 0.04, 0.66),
    energy=12.0,
    size=0.74,
  )


def _setup_softbox_overhead_lighting():
  """Softbox variant with the strip light doing more of the shadow work."""
  _add_area_light(
    "LargeLeftSoftbox",
    location=(-0.72, -1.30, 1.28),
    target=(0.0, 0.02, 0.60),
    energy=42.0,
    size=1.42,
    size_y=0.92,
  )
  _add_area_light(
    "TopStrip",
    location=(0.12, -0.40, 1.58),
    target=(0.0, 0.03, 0.62),
    energy=70.0,
    size=1.36,
    size_y=0.16,
  )
  _add_area_light(
    "RightFill",
    location=(1.20, -0.84, 0.95),
    target=(0.0, 0.02, 0.54),
    energy=3.5,
    size=2.70,
  )
  _add_area_light(
    "BackEdge",
    location=(0.18, 0.90, 1.24),
    target=(0.02, 0.04, 0.66),
    energy=12.0,
    size=0.74,
  )


def _setup_softbox_wrap_lighting():
  """Softbox variant with broader camera-side wrap and lower contrast."""
  _add_area_light(
    "LargeLeftSoftbox",
    location=(-1.18, -1.06, 1.22),
    target=(0.0, 0.02, 0.60),
    energy=76.0,
    size=1.80,
    size_y=1.05,
  )
  _add_area_light(
    "TopStrip",
    location=(0.02, -0.30, 1.50),
    target=(0.0, 0.02, 0.62),
    energy=34.0,
    size=1.55,
    size_y=0.20,
  )
  _add_area_light(
    "RightFill",
    location=(1.30, -0.86, 0.98),
    target=(0.0, 0.02, 0.54),
    energy=7.0,
    size=2.80,
  )
  _add_area_light(
    "BackEdge",
    location=(-0.12, 0.98, 1.20),
    target=(0.02, 0.04, 0.66),
    energy=10.0,
    size=0.92,
  )


def _setup_softbox_rim_lighting():
  """Softbox variant with more separation on the hammer and orange arm."""
  _add_area_light(
    "LargeLeftSoftbox",
    location=(-0.84, -1.22, 1.38),
    target=(0.0, 0.02, 0.62),
    energy=54.0,
    size=1.36,
    size_y=0.86,
  )
  _add_area_light(
    "TopStrip",
    location=(0.08, -0.34, 1.54),
    target=(0.0, 0.02, 0.62),
    energy=46.0,
    size=1.48,
    size_y=0.18,
  )
  _add_area_light(
    "RightFill",
    location=(1.25, -0.80, 0.95),
    target=(0.0, 0.02, 0.54),
    energy=3.8,
    size=2.60,
  )
  _add_area_light(
    "BackEdge",
    location=(0.34, 0.84, 1.28),
    target=(0.02, 0.04, 0.66),
    energy=24.0,
    size=0.64,
  )


def _setup_spot_accent_lighting():
  """Sharper variant with an area key and controlled spot accents."""
  _add_area_light(
    "TopStrip",
    location=(0.08, -0.28, 1.50),
    target=(0.0, 0.04, 0.62),
    energy=38.0,
    size=1.35,
    size_y=0.18,
  )
  _add_area_light(
    "FocusedKey",
    location=(0.82, -1.02, 1.42),
    target=(0.0, 0.04, 0.62),
    energy=64.0,
    size=0.52,
    size_y=0.34,
  )
  _add_area_light(
    "BroadFill",
    location=(-1.22, -0.74, 1.05),
    target=(0.0, 0.04, 0.55),
    energy=6.0,
    size=2.20,
  )
  _add_area_light(
    "RearEdge",
    location=(0.15, 0.92, 1.15),
    target=(0.02, 0.05, 0.65),
    energy=16.0,
    size=0.70,
  )
  _add_spot_light(
    "ToolSpot",
    location=(-0.48, -1.05, 1.18),
    target=(0.02, 0.05, 0.60),
    energy=34.0,
    size=0.78,
    blend=0.84,
  )


def _setup_lighting_preset(preset):
  """Install a deterministic lighting preset into the current scene."""
  preset = (preset or DEFAULT_LIGHTING_PRESET).strip().lower()
  _clear_scene_lights()

  if preset == "clean_key_fill":
    _setup_world_hdri(_resolve_hdri_path(), strength=0.28, visible_to_camera=False)
    _setup_clean_eval_lighting()
  elif preset == "reference_area":
    _setup_world_hdri(_resolve_hdri_path(), strength=0.02, visible_to_camera=False)
    _setup_reference_area_lighting()
  elif preset == "softbox_grid":
    _setup_world_hdri(_resolve_hdri_path(), strength=0.02, visible_to_camera=False)
    _setup_softbox_grid_lighting()
  elif preset == "softbox_overhead":
    _setup_world_hdri(_resolve_hdri_path(), strength=0.02, visible_to_camera=False)
    _setup_softbox_overhead_lighting()
  elif preset == "softbox_wrap":
    _setup_world_hdri(_resolve_hdri_path(), strength=0.02, visible_to_camera=False)
    _setup_softbox_wrap_lighting()
  elif preset == "softbox_rim":
    _setup_world_hdri(_resolve_hdri_path(), strength=0.02, visible_to_camera=False)
    _setup_softbox_rim_lighting()
  elif preset == "spot_accent":
    _setup_world_hdri(_resolve_hdri_path(), strength=0.015, visible_to_camera=False)
    _setup_spot_accent_lighting()
  else:
    raise RuntimeError(
      f"Unknown lighting preset {preset!r}. Expected one of: "
      "clean_key_fill, reference_area, softbox_grid, softbox_overhead, "
      "softbox_wrap, softbox_rim, spot_accent."
    )
  bpy.context.scene["simtooldiff_lighting_preset"] = preset
  return preset


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

  scene.view_settings.exposure = -1.20
  scene.view_settings.gamma = 1.0


def _add_default_static_scene():
  """Create the fallback scripted static scene used when no .blend is supplied."""
  # IsaacGym scene furniture from assets/urdf/table_narrow_nail.urdf.
  # The table actor is centered at (0, 0, TABLE_Z); the URDF box is centered
  # on that actor with size 0.475 x 0.4 x 0.3, plus a small gray nail.
  table_top_mat = _make_wood_material(
    "table_light_oak_top",
    roughness=0.48,
    specular=0.34,
    coat=0.03,
    normal_strength=0.26,
    texture_scale=(2.6, 2.6, 1.0),
  )
  table_side_mat = _make_wood_material(
    "table_light_oak_side",
    roughness=0.56,
    specular=0.28,
    coat=0.02,
    normal_strength=0.18,
    texture_scale=(1.2, 3.8, 1.0),
  )
  nail_mat = _make_marble_material("table_nail_marble")
  floor_mat = _make_concrete_material(
    "lab_floor_matte", base=(0.49, 0.50, 0.47, 1.0), roughness=0.82
  )
  wall_mat = _make_concrete_material(
    "clean_back_wall_matte", base=(0.47, 0.48, 0.46, 1.0), roughness=0.88
  )
  bench_mat = _make_material(
    "rear_bench_laminate", (0.62, 0.61, 0.56, 1.0), roughness=0.64, specular=0.28
  )
  bench_leg_mat = _make_material(
    "rear_bench_frame", (0.12, 0.13, 0.13, 1.0), roughness=0.55, metallic=0.55
  )

  table = _add_cube(
    "table",
    location=(0.0, 0.0, 0.38),
    dimensions=(0.475, 0.4, 0.3),
    mat=None,
    bevel=0.010,
    segments=4,
  )
  _assign_table_box_materials(table, table_top_mat, table_side_mat)
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
    "clean_back_wall",
    location=(0.0, 1.42, 0.82),
    dimensions=(4.50, 0.055, 1.75),
    mat=wall_mat,
    bevel=0.0,
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


def _as_tuple(values):
  return tuple(round(float(v), 4) for v in values)


def _max_abs_delta(a, b):
  return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _validate_template_scene(blend_file):
  """Validate the static-scene contract for an externally authored template."""
  required = {
    "table": {
      "location": (0.0, 0.0, 0.38),
      "dimensions": (0.475, 0.4, 0.3),
    },
    "table_nail": {
      "location": (-0.16, 0.06, 0.555),
      "dimensions": (0.03, 0.03, 0.06),
    },
    "floor": {},
  }
  missing = [name for name in required if bpy.data.objects.get(name) is None]
  if missing:
    raise RuntimeError(
      f"Blend template {blend_file} is missing required static objects: {missing}. "
      "The template must include at least table, table_nail, and floor."
    )

  for name, expected in required.items():
    obj = bpy.data.objects[name]
    if "location" in expected:
      delta = _max_abs_delta(obj.location, expected["location"])
      if delta > 0.003:
        print(
          "WARNING: template object location differs from IsaacGym visual contract: "
          f"{name} actual={_as_tuple(obj.location)} expected={expected['location']}",
          file=sys.stderr,
          flush=True,
        )
    if "dimensions" in expected:
      delta = _max_abs_delta(obj.dimensions, expected["dimensions"])
      if delta > 0.010:
        print(
          "WARNING: template object dimensions differ from IsaacGym visual contract: "
          f"{name} actual={_as_tuple(obj.dimensions)} expected={expected['dimensions']}",
          file=sys.stderr,
          flush=True,
        )

  has_light = any(obj.type == "LIGHT" for obj in bpy.data.objects)
  world = bpy.context.scene.world
  has_hdri = bool(
    world and world.use_nodes and world.node_tree and any(
      node.type == "TEX_ENVIRONMENT" for node in world.node_tree.nodes
    )
  )
  if not has_light:
    print(
      f"WARNING: blend template {blend_file} contains no lights.",
      file=sys.stderr,
      flush=True,
    )
  if not has_hdri:
    print(
      f"WARNING: blend template {blend_file} has no HDRI Environment Texture.",
      file=sys.stderr,
      flush=True,
    )


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
  p.add_argument("--cycles-device", choices=("auto", "gpu", "cpu"), default="auto",
                 help="Cycles compute device. Use cpu when policy+IsaacGym need GPU VRAM.")
  p.add_argument("--blend-file", type=str, default=None,
                 help="Optional .blend scene template to load (lighting, materials)")
  p.add_argument("--response-fifo", type=str, required=True,
                 help="Path to named pipe (FIFO) for protocol responses")
  return p.parse_args(argv)


def setup_scene(args):
  """Configure render engine, resolution, and basic scene."""
  scene = bpy.context.scene
  using_template = False

  # Load scene template if provided
  if args.blend_file:
    if not os.path.exists(args.blend_file):
      raise FileNotFoundError(f"Blend template not found: {args.blend_file}")
    bpy.ops.wm.open_mainfile(filepath=args.blend_file)
    scene = bpy.context.scene
    using_template = True
    if os.environ.get(LIGHTING_PRESET_ENV):
      _setup_lighting_preset(os.environ[LIGHTING_PRESET_ENV])
  else:
    # Clear default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # The HDRI provides weak reflection fill only. Area lights do the visible
    # shadow shaping so the render does not inherit the HDRI's flat ambient look.
    _setup_lighting_preset(
      os.environ.get(LIGHTING_PRESET_ENV, DEFAULT_LIGHTING_PRESET)
    )

  # Render engine
  if args.engine == "cycles":
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = CYCLES_SEED
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 6
    scene.cycles.glossy_bounces = 8
    scene.cycles.transmission_bounces = 12
    scene.cycles.transparent_max_bounces = 12
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
    scene.render.film_transparent = False

    # Prefer OptiX/CUDA on NVIDIA unless CPU is explicitly requested. CPU is
    # useful on 8 GB GPUs when policy + IsaacGym already consume most VRAM.
    scene.cycles.device = 'CPU'
    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs and args.cycles_device in {"auto", "gpu"}:
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
    if args.cycles_device == "gpu" and scene.cycles.device != 'GPU':
      raise RuntimeError("Cycles GPU requested but no OPTIX/CUDA device was enabled")
  else:
    scene.render.engine = _eevee_engine_name()
    scene.render.use_motion_blur = USE_MOTION_BLUR

  # Resolution
  scene.render.resolution_x = args.width
  scene.render.resolution_y = args.height
  scene.render.resolution_percentage = 100
  scene.render.image_settings.file_format = 'PNG'
  _set_color_management(scene)

  if using_template:
    _validate_template_scene(args.blend_file)
  else:
    _add_default_static_scene()

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
  _apply_tool_materials(obj, object_name)
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
  if _render_diagnostics_enabled():
    print(
      f"DIAG_RENDER_ENGINE_BEFORE_RENDER {scene.render.engine}",
      file=sys.stderr,
      flush=True,
    )
  bpy.ops.render.render(write_still=True)
  return path


def main():
  global _DIAGNOSTICS_PRINTED
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

      if _render_diagnostics_enabled() and not _DIAGNOSTICS_PRINTED:
        _print_render_diagnostics(scene, robot_objects, tool_object)
        _DIAGNOSTICS_PRINTED = True

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
