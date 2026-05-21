"""Convert IsaacGym camera parameters to Blender camera parameters.

IsaacGym specifies cameras via position, look-at target, and horizontal FOV.
Blender cameras use focal length (mm), sensor size (mm), and an object transform.
Blender cameras look along local -Z with local +Y as up.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from blender_eval.pose_conversion import (
  quat_to_rotation_matrix,
  quat_wxyz_to_xyzw,
  quat_xyzw_to_wxyz,
  rotation_matrix_to_quat_xyzw,
)


@dataclass
class BlenderCameraParams:
  """Parameters sufficient to configure a Blender camera to match IsaacGym.

  In Blender, set:
    camera.data.lens = focal_length_mm
    camera.data.sensor_width = sensor_width_mm
    camera.data.sensor_height = sensor_height_mm
    camera.data.sensor_fit = 'HORIZONTAL'  # REQUIRED for HFOV match
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    camera.location = location
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = rotation_quaternion_wxyz
  """
  focal_length_mm: float
  sensor_width_mm: float
  sensor_height_mm: float
  resolution_x: int
  resolution_y: int
  location: Tuple[float, float, float]
  rotation_quaternion_wxyz: Tuple[float, float, float, float]


def _look_at_rotation_matrix(
  camera_pos: np.ndarray,
  target_pos: np.ndarray,
  world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
  """Build a 3x3 rotation matrix for a Blender camera looking at a target.

  Blender camera convention: local -Z is forward, local +Y is up.
  So the rotation matrix columns (in world frame) are:
    col 0 (local X) = right
    col 1 (local Y) = up
    col 2 (local Z) = -forward  (since camera looks along -Z)
  """
  forward = target_pos - camera_pos
  forward = forward / np.linalg.norm(forward)

  right = np.cross(forward, world_up)
  right_norm = np.linalg.norm(right)
  if right_norm < 1e-10:
    # Camera is looking straight up or down; pick an arbitrary right
    right = np.array([1.0, 0.0, 0.0])
  else:
    right = right / right_norm

  up = np.cross(right, forward)
  up = up / np.linalg.norm(up)

  # Build rotation: columns are local axes in world frame.
  # Blender camera: local X = right, local Y = up, local Z = -forward
  R = np.column_stack([right, up, -forward])
  return R


def isaacgym_to_blender_camera(
  horizontal_fov_deg: float,
  width: int,
  height: int,
  camera_position: List[float],
  camera_target: List[float],
  sensor_width_mm: float = 36.0,
) -> BlenderCameraParams:
  """Convert IsaacGym camera parameters to Blender camera parameters.

  Args:
    horizontal_fov_deg: Horizontal field of view in degrees.
    width: Image width in pixels.
    height: Image height in pixels.
    camera_position: [x, y, z] camera position in world frame.
    camera_target: [x, y, z] look-at target in world frame.
    sensor_width_mm: Blender sensor width in mm (default 36mm, full-frame).

  Returns:
    BlenderCameraParams with all values needed to configure the Blender camera.
  """
  hfov_rad = math.radians(horizontal_fov_deg)
  focal_length_mm = (sensor_width_mm / 2.0) / math.tan(hfov_rad / 2.0)
  sensor_height_mm = sensor_width_mm * height / width

  cam_pos = np.array(camera_position, dtype=np.float64)
  cam_target = np.array(camera_target, dtype=np.float64)

  R = _look_at_rotation_matrix(cam_pos, cam_target)
  q_xyzw = rotation_matrix_to_quat_xyzw(R)
  q_wxyz = quat_xyzw_to_wxyz(q_xyzw)

  return BlenderCameraParams(
    focal_length_mm=focal_length_mm,
    sensor_width_mm=sensor_width_mm,
    sensor_height_mm=sensor_height_mm,
    resolution_x=width,
    resolution_y=height,
    location=tuple(cam_pos.tolist()),
    rotation_quaternion_wxyz=tuple(q_wxyz.tolist()),
  )


def blender_camera_hfov_deg(focal_length_mm: float, sensor_width_mm: float) -> float:
  """Back-compute horizontal FOV in degrees from focal length."""
  return 2.0 * math.degrees(math.atan2(sensor_width_mm / 2.0, focal_length_mm))


def camera_forward_vector(q_wxyz: Tuple[float, float, float, float]) -> np.ndarray:
  """Extract the world-frame forward direction from a Blender camera quaternion.

  Blender cameras look along local -Z, so forward = R @ [0, 0, -1].
  """
  q_xyzw = quat_wxyz_to_xyzw(np.array(q_wxyz))
  R = quat_to_rotation_matrix(q_xyzw)
  return R @ np.array([0.0, 0.0, -1.0])


def camera_up_vector(q_wxyz: Tuple[float, float, float, float]) -> np.ndarray:
  """Extract the world-frame up direction from a Blender camera quaternion.

  Blender cameras have local +Y as up, so up = R @ [0, 1, 0].
  """
  q_xyzw = quat_wxyz_to_xyzw(np.array(q_wxyz))
  R = quat_to_rotation_matrix(q_xyzw)
  return R @ np.array([0.0, 1.0, 0.0])


def project_point_to_image(
  world_point: np.ndarray,
  camera_pos: np.ndarray,
  q_wxyz: Tuple[float, float, float, float],
  focal_length_mm: float,
  sensor_width_mm: float,
  sensor_height_mm: float,
  resolution_x: int,
  resolution_y: int,
) -> Tuple[float, float]:
  """Project a world point onto the image plane using a pinhole camera model.

  Returns (u, v) pixel coordinates where (0,0) is top-left.
  """
  q_xyzw = quat_wxyz_to_xyzw(np.array(q_wxyz))
  R = quat_to_rotation_matrix(q_xyzw)

  # Transform world point to camera-local frame
  p_local = R.T @ (world_point - camera_pos)

  # Blender camera: forward is -Z, right is +X, up is +Y
  # p_local = (right, up, -depth)
  x_local, y_local, z_local = p_local
  depth = -z_local  # depth is positive when point is in front of camera

  # Focal length in pixels
  fx = focal_length_mm * resolution_x / sensor_width_mm
  fy = focal_length_mm * resolution_y / sensor_height_mm

  # Project
  u = fx * x_local / depth + resolution_x / 2.0
  v = -fy * y_local / depth + resolution_y / 2.0  # -y because image Y is downward

  return u, v
