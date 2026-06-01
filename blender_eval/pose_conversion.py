"""Coordinate-frame and quaternion conversion between IsaacGym and Blender.

Both systems are Z-up right-handed, so no axis swap is needed.
The only reordering is quaternion components: IsaacGym (x,y,z,w) vs Blender (w,x,y,z).
"""

from typing import Sequence, Tuple

import numpy as np


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
  """Reorder quaternion from IsaacGym (x,y,z,w) to Blender (w,x,y,z)."""
  q = np.asarray(q, dtype=np.float64)
  return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
  """Reorder quaternion from Blender (w,x,y,z) to IsaacGym (x,y,z,w)."""
  q = np.asarray(q, dtype=np.float64)
  return q[..., [1, 2, 3, 0]]


def quat_to_rotation_matrix(q_xyzw: np.ndarray) -> np.ndarray:
  """Convert quaternion (x,y,z,w) to 3x3 rotation matrix."""
  q = np.asarray(q_xyzw, dtype=np.float64)
  x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

  x2, y2, z2 = x * x, y * y, z * z
  xy, xz, yz = x * y, x * z, y * z
  wx, wy, wz = w * x, w * y, w * z

  R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
  R[..., 0, 0] = 1.0 - 2.0 * (y2 + z2)
  R[..., 0, 1] = 2.0 * (xy - wz)
  R[..., 0, 2] = 2.0 * (xz + wy)
  R[..., 1, 0] = 2.0 * (xy + wz)
  R[..., 1, 1] = 1.0 - 2.0 * (x2 + z2)
  R[..., 1, 2] = 2.0 * (yz - wx)
  R[..., 2, 0] = 2.0 * (xz - wy)
  R[..., 2, 1] = 2.0 * (yz + wx)
  R[..., 2, 2] = 1.0 - 2.0 * (x2 + y2)
  return R


def rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
  """Convert 3x3 rotation matrix to quaternion (x,y,z,w).

  Uses Shepperd's method for numerical stability.
  Always returns the quaternion with w >= 0 (canonical form).
  """
  R = np.asarray(R, dtype=np.float64)
  trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

  # Shepperd: pick the largest of w, x, y, z to avoid division by small number
  choices = np.stack([
    trace,
    R[..., 0, 0],
    R[..., 1, 1],
    R[..., 2, 2],
  ], axis=-1)
  best = np.argmax(choices, axis=-1)

  q = np.zeros(R.shape[:-2] + (4,), dtype=np.float64)

  # Case 0: w is largest
  s = np.sqrt(np.maximum(trace + 1.0, 0.0)) * 2.0  # 4w
  s = np.where(s == 0, 1.0, s)  # avoid div by zero
  w = s / 4.0
  x = (R[..., 2, 1] - R[..., 1, 2]) / s
  y = (R[..., 0, 2] - R[..., 2, 0]) / s
  z = (R[..., 1, 0] - R[..., 0, 1]) / s

  mask0 = best == 0
  q[mask0] = np.stack([x, y, z, w], axis=-1)[mask0]

  # Case 1: x is largest
  s = np.sqrt(np.maximum(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2], 0.0)) * 2.0
  s = np.where(s == 0, 1.0, s)
  x = s / 4.0
  w = (R[..., 2, 1] - R[..., 1, 2]) / s
  y = (R[..., 0, 1] + R[..., 1, 0]) / s
  z = (R[..., 0, 2] + R[..., 2, 0]) / s

  mask1 = best == 1
  q[mask1] = np.stack([x, y, z, w], axis=-1)[mask1]

  # Case 2: y is largest
  s = np.sqrt(np.maximum(1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2], 0.0)) * 2.0
  s = np.where(s == 0, 1.0, s)
  y = s / 4.0
  w = (R[..., 0, 2] - R[..., 2, 0]) / s
  x = (R[..., 0, 1] + R[..., 1, 0]) / s
  z = (R[..., 1, 2] + R[..., 2, 1]) / s

  mask2 = best == 2
  q[mask2] = np.stack([x, y, z, w], axis=-1)[mask2]

  # Case 3: z is largest
  s = np.sqrt(np.maximum(1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1], 0.0)) * 2.0
  s = np.where(s == 0, 1.0, s)
  z = s / 4.0
  w = (R[..., 1, 0] - R[..., 0, 1]) / s
  x = (R[..., 0, 2] + R[..., 2, 0]) / s
  y = (R[..., 1, 2] + R[..., 2, 1]) / s

  mask3 = best == 3
  q[mask3] = np.stack([x, y, z, w], axis=-1)[mask3]

  # Canonicalize: ensure w >= 0
  sign = np.where(q[..., 3:4] < 0, -1.0, 1.0)
  q = q * sign

  # Normalize
  q = q / np.linalg.norm(q, axis=-1, keepdims=True)
  return q


def pose_to_matrix(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
  """Build 4x4 homogeneous transform from position + IsaacGym quaternion."""
  pos = np.asarray(pos, dtype=np.float64)
  R = quat_to_rotation_matrix(quat_xyzw)
  mat = np.eye(4, dtype=np.float64)
  mat[:3, :3] = R
  mat[:3, 3] = pos
  return mat


def matrix_to_pose(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Extract position and IsaacGym quaternion (x,y,z,w) from 4x4 matrix."""
  mat = np.asarray(mat, dtype=np.float64)
  pos = mat[:3, 3].copy()
  R = mat[:3, :3]
  quat_xyzw = rotation_matrix_to_quat_xyzw(R)
  return pos, quat_xyzw


def urdf_origin_to_matrix(
  xyz: Sequence[float] = (0.0, 0.0, 0.0),
  rpy: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
  """Convert a URDF <origin xyz="..." rpy="..."> to a 4x4 transform.

  RPY is applied as extrinsic XYZ (equivalently intrinsic ZYX):
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
  This matches the URDF specification.
  """
  roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])

  cr, sr = np.cos(roll), np.sin(roll)
  cp, sp = np.cos(pitch), np.sin(pitch)
  cy, sy = np.cos(yaw), np.sin(yaw)

  # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
  R = np.array([
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr],
  ], dtype=np.float64)

  mat = np.eye(4, dtype=np.float64)
  mat[:3, :3] = R
  mat[:3, 3] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
  return mat
