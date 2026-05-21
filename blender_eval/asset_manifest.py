"""URDF parsing to build a mesh manifest with collapsed-link offsets.

Parses the integrated KUKA+Sharpa URDF to map each visual-mesh-bearing link
to its mesh path, visual origin, and (if collapsed) the fixed-joint transform
chain to its surviving parent body.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from blender_eval.pose_conversion import urdf_origin_to_matrix

SIMTOOLDIFF_ROOT = Path(__file__).resolve().parent.parent
URDF_DIR = SIMTOOLDIFF_ROOT / "assets" / "urdf"
ROBOT_URDF = (
  URDF_DIR / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf"
)
TOOL_DIR = URDF_DIR / "dextoolbench"


@dataclass
class LinkMeshInfo:
  """Visual mesh information for one link."""
  mesh_path: Path
  visual_origin: np.ndarray = field(default_factory=lambda: np.eye(4))
  is_collapsed: bool = False
  surviving_parent: Optional[str] = None
  joint_chain_offset: Optional[np.ndarray] = None

  @property
  def offset_from_parent(self) -> Optional[np.ndarray]:
    """Full 4x4 offset = joint_chain_offset @ visual_origin (if collapsed)."""
    if self.joint_chain_offset is None:
      return None
    return self.joint_chain_offset @ self.visual_origin


def _parse_origin(elem: Optional[ET.Element]) -> np.ndarray:
  """Parse an <origin xyz="..." rpy="..."> element into a 4x4 transform."""
  if elem is None:
    return np.eye(4)
  xyz_str = elem.get("xyz", "0 0 0")
  rpy_str = elem.get("rpy", "0 0 0")
  xyz = [float(x) for x in xyz_str.split()]
  rpy = [float(x) for x in rpy_str.split()]
  return urdf_origin_to_matrix(xyz, rpy)


def parse_urdf_visual_meshes(urdf_path: Path) -> Dict[str, LinkMeshInfo]:
  """Parse URDF and return visual mesh info for each link that has one.

  Handles collapse_fixed_joints=True by:
  1. Building the joint tree (parent→child, joint type, joint origin).
  2. Identifying which links are connected via fixed joints and would collapse.
  3. For collapsed links with visual meshes, computing the accumulated
     fixed-joint transform from the surviving parent.
  """
  tree = ET.parse(str(urdf_path))
  root = tree.getroot()
  urdf_dir = urdf_path.parent

  # 1. Parse all links: name → (has_visual, mesh_path, visual_origin)
  link_visuals: Dict[str, Tuple[Optional[Path], np.ndarray]] = {}
  for link_elem in root.findall("link"):
    name = link_elem.get("name")
    visual = link_elem.find("visual")
    if visual is not None:
      geom = visual.find("geometry")
      mesh = geom.find("mesh") if geom is not None else None
      if mesh is not None:
        mesh_filename = mesh.get("filename")
        mesh_path = (urdf_dir / mesh_filename).resolve()
        visual_origin = _parse_origin(visual.find("origin"))
        link_visuals[name] = (mesh_path, visual_origin)
      else:
        link_visuals[name] = (None, np.eye(4))
    else:
      link_visuals[name] = (None, np.eye(4))

  # 2. Parse all joints: child → (parent, joint_type, joint_origin)
  child_to_parent: Dict[str, Tuple[str, str, np.ndarray]] = {}
  for joint_elem in root.findall("joint"):
    joint_type = joint_elem.get("type")
    parent_name = joint_elem.find("parent").get("link")
    child_name = joint_elem.find("child").get("link")
    origin = _parse_origin(joint_elem.find("origin"))
    child_to_parent[child_name] = (parent_name, joint_type, origin)

  # 3. For each link, determine if it's collapsed and find the surviving parent.
  def _is_link_collapsed(link_name: str) -> bool:
    """A link is collapsed if its parent joint is fixed."""
    if link_name not in child_to_parent:
      return False
    _, joint_type, _ = child_to_parent[link_name]
    return joint_type == "fixed"

  def resolve_collapse(link_name: str) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
    """Determine collapse status for a link.

    Returns:
      (is_collapsed, surviving_parent_name, joint_chain_offset_4x4)
      If not collapsed, surviving_parent and offset are None.
    """
    if not _is_link_collapsed(link_name):
      return False, None, None

    # Walk up through fixed joints. Only accumulate origins of FIXED joints.
    accumulated = np.eye(4)
    current = link_name
    while current in child_to_parent:
      parent, joint_type, joint_origin = child_to_parent[current]
      if joint_type != "fixed":
        return True, current, accumulated
      accumulated = joint_origin @ accumulated
      current = parent

    # Reached root through only fixed joints — root is the surviving body.
    return True, current, accumulated

  # 4. Build the manifest
  manifest: Dict[str, LinkMeshInfo] = {}
  for link_name, (mesh_path, visual_origin) in link_visuals.items():
    if mesh_path is None:
      continue  # No visual mesh for this link

    collapsed, surviving_parent, chain_offset = resolve_collapse(link_name)
    manifest[link_name] = LinkMeshInfo(
      mesh_path=mesh_path,
      visual_origin=visual_origin,
      is_collapsed=collapsed,
      surviving_parent=surviving_parent,
      joint_chain_offset=chain_offset,
    )

  return manifest


def get_robot_mesh_manifest() -> Dict[str, LinkMeshInfo]:
  """Get mesh manifest for the KUKA iiwa14 + Sharpa hand."""
  return parse_urdf_visual_meshes(ROBOT_URDF)


def get_object_mesh_path(object_name: str) -> Path:
  """Get the visual mesh path for a DexToolBench tool object.

  Returns the .obj file path. Blender's OBJ importer auto-resolves sibling
  .mtl files from the same directory, so no separate material path is needed.
  """
  # Find the object's URDF by searching the dextoolbench directory
  for category_dir in TOOL_DIR.iterdir():
    if not category_dir.is_dir():
      continue
    obj_dir = category_dir / object_name
    if obj_dir.is_dir():
      urdf_path = obj_dir / f"{object_name}.urdf"
      if urdf_path.exists():
        # Parse the URDF to find the mesh
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        for link_elem in root.findall("link"):
          visual = link_elem.find("visual")
          if visual is not None:
            geom = visual.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is not None:
              mesh_filename = mesh.get("filename")
              return (urdf_path.parent / mesh_filename).resolve()
  raise FileNotFoundError(f"No mesh found for object '{object_name}' in {TOOL_DIR}")
