"""Extract per-link world-frame poses from a live IsaacGym environment.

Reads env.rigid_body_states for surviving links and composes fixed-joint
offsets for collapsed links, producing the full set of mesh poses needed
for Blender rendering.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from blender_eval.asset_manifest import LinkMeshInfo
from blender_eval.pose_conversion import (
  matrix_to_pose,
  pose_to_matrix,
  quat_xyzw_to_wxyz,
)


@dataclass
class RenderState:
  """All state needed to render one frame in Blender.

  mesh_poses: world-frame (pos, quat_xyzw) for every visual mesh, including
    composed visual_origin and collapsed-link offsets.
  object_pos / object_quat_xyzw: tool object world pose.
  object_name: which tool (for selecting the correct mesh).
  """
  mesh_poses: Dict[str, Tuple[np.ndarray, np.ndarray]]
  object_pos: np.ndarray
  object_quat_xyzw: np.ndarray
  object_name: str


def get_surviving_body_names(env) -> List[str]:
  """Return robot link names that survive collapse_fixed_joints.

  Reads env.rigid_body_name_to_idx and filters for "robot/" prefix.
  """
  names = []
  for key in env.rigid_body_name_to_idx:
    if key.startswith("robot/"):
      names.append(key[len("robot/"):])
  return sorted(names)


def extract_render_state(
  env,
  env_idx: int,
  object_name: str,
  mesh_manifest: Dict[str, LinkMeshInfo],
) -> RenderState:
  """Extract rendering state for one environment.

  For each link in the mesh manifest:
    - If not collapsed: reads world pose from rigid_body_states, composes
      with visual_origin.
    - If collapsed: reads surviving parent's world pose, composes with
      joint_chain_offset @ visual_origin.

  Args:
    env: The IsaacGym SimToolReal environment.
    env_idx: Which environment index to extract from.
    object_name: Tool name (e.g. "claw_hammer").
    mesh_manifest: From asset_manifest.get_robot_mesh_manifest().

  Returns:
    RenderState with all mesh poses and object pose.
  """
  rb_states = env.rigid_body_states  # (num_envs, num_bodies, 13)
  name_to_idx = env.rigid_body_name_to_idx

  # Cache: surviving body name → world pose (4×4)
  _pose_cache: Dict[str, np.ndarray] = {}

  def _get_body_world_matrix(body_name: str) -> np.ndarray:
    if body_name in _pose_cache:
      return _pose_cache[body_name]
    key = f"robot/{body_name}"
    if key not in name_to_idx:
      raise KeyError(
        f"Body '{body_name}' not found in rigid_body_name_to_idx. "
        f"Available robot bodies: {[k for k in name_to_idx if k.startswith('robot/')]}"
      )
    body_idx = name_to_idx[key]
    state = rb_states[env_idx, body_idx, :7].detach().cpu().numpy()
    pos = state[:3].astype(np.float64)
    quat_xyzw = state[3:7].astype(np.float64)
    mat = pose_to_matrix(pos, quat_xyzw)
    _pose_cache[body_name] = mat
    return mat

  mesh_poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

  for link_name, info in mesh_manifest.items():
    if not info.is_collapsed:
      # Surviving link: read directly, compose with visual_origin
      world_mat = _get_body_world_matrix(link_name)
      mesh_mat = world_mat @ info.visual_origin
    else:
      # Collapsed link: read parent, compose chain offset + visual_origin
      parent_mat = _get_body_world_matrix(info.surviving_parent)
      mesh_mat = parent_mat @ info.offset_from_parent

    pos, quat_xyzw = matrix_to_pose(mesh_mat)
    mesh_poses[link_name] = (pos.astype(np.float32), quat_xyzw.astype(np.float32))

  # Object pose
  obj_pose = env.object_pose[env_idx, :7].detach().cpu().numpy()
  object_pos = obj_pose[:3].astype(np.float32)
  object_quat_xyzw = obj_pose[3:7].astype(np.float32)

  return RenderState(
    mesh_poses=mesh_poses,
    object_pos=object_pos,
    object_quat_xyzw=object_quat_xyzw,
    object_name=object_name,
  )


def serialize_render_state(state: RenderState, tool_mesh_path: str = "") -> dict:
  """Serialize RenderState to a JSON-compatible dict for the Blender subprocess.

  Converts all quaternions from IsaacGym xyzw to Blender wxyz at this boundary.
  This is the ONLY place the xyzw→wxyz conversion happens for the IPC path.
  """
  mesh_poses = {}
  for link_name, (pos, quat_xyzw) in state.mesh_poses.items():
    q_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
    mesh_poses[link_name] = [pos.tolist(), q_wxyz.tolist()]

  obj_q_wxyz = quat_xyzw_to_wxyz(state.object_quat_xyzw)

  return {
    "mesh_poses": mesh_poses,
    "object_pos": state.object_pos.tolist(),
    "object_quat_wxyz": obj_q_wxyz.tolist(),
    "object_name": state.object_name,
    "tool_mesh_path": tool_mesh_path,
  }
