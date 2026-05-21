#!/usr/bin/env python3
"""Build the master SimToolDiff Blender scene template.

Run from the repository root with Blender:

  /tmp/blender-4.2.9-linux-x64/blender --background \
    --python assets/blender/templates/build_simtool_lab_template.py

The generated .blend owns only static visual scene content. Closed-loop eval
still imports the robot/tool meshes and updates their poses from IsaacGym.
"""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

try:
  import bpy
except ImportError:
  print("ERROR: run this script with Blender, not the simtooldiff venv Python.")
  raise


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "assets" / "blender" / "templates" / "simtool_lab.blend"


def _parse_args():
  argv = sys.argv
  if "--" in argv:
    argv = argv[argv.index("--") + 1:]
  else:
    argv = []

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--samples", type=int, default=64)
  return parser.parse_args(argv)


def main():
  args = _parse_args()
  sys.path.insert(0, str(ROOT))

  from blender_eval import blender_render_script as render_script

  scene_args = SimpleNamespace(
    blend_file=None,
    engine="cycles",
    width=512,
    height=384,
    samples=args.samples,
    cycles_device="auto",
  )
  scene = render_script.setup_scene(scene_args)
  scene["simtooldiff_template_contract"] = "static_scene_v1"
  scene["simtooldiff_runtime_owns"] = (
    "EvalCamera, render settings, robot link meshes, tool mesh, all moving poses"
  )
  scene["simtooldiff_template_owns"] = (
    "world/HDRI, static lights, static lab furniture, static materials"
  )

  args.output.parent.mkdir(parents=True, exist_ok=True)
  bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
  try:
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
  except Exception as exc:
    print(f"WARNING: could not make image paths relative: {exc}", file=sys.stderr)
  print(f"saved {args.output}")


if __name__ == "__main__":
  main()
