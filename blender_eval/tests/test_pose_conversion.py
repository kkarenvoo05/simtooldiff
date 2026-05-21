import numpy as np
import numpy.testing as npt
import pytest

from blender_eval.pose_conversion import (
  matrix_to_pose,
  pose_to_matrix,
  quat_to_rotation_matrix,
  quat_xyzw_to_wxyz,
  quat_wxyz_to_xyzw,
  rotation_matrix_to_quat_xyzw,
  urdf_origin_to_matrix,
)

SIN45 = np.sin(np.pi / 4)
COS45 = np.cos(np.pi / 4)


class TestQuatReorder:
  def test_xyzw_to_wxyz(self):
    q = np.array([1.0, 2.0, 3.0, 4.0])
    npt.assert_array_equal(quat_xyzw_to_wxyz(q), [4.0, 1.0, 2.0, 3.0])

  def test_wxyz_to_xyzw(self):
    q = np.array([4.0, 1.0, 2.0, 3.0])
    npt.assert_array_equal(quat_wxyz_to_xyzw(q), [1.0, 2.0, 3.0, 4.0])

  def test_identity_xyzw_to_wxyz(self):
    # IsaacGym identity: (0,0,0,1). Blender identity: (1,0,0,0).
    npt.assert_array_equal(
      quat_xyzw_to_wxyz([0, 0, 0, 1]), [1, 0, 0, 0]
    )

  def test_roundtrip(self):
    q = np.array([0.1, 0.2, 0.3, 0.9])
    npt.assert_allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q)), q)

  def test_batch(self):
    q = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
    result = quat_xyzw_to_wxyz(q)
    assert result.shape == (2, 4)
    npt.assert_array_equal(result[0], [4, 1, 2, 3])
    npt.assert_array_equal(result[1], [8, 5, 6, 7])


class TestQuatToRotationMatrix:
  def test_identity(self):
    R = quat_to_rotation_matrix(np.array([0, 0, 0, 1.0]))
    npt.assert_allclose(R, np.eye(3), atol=1e-12)

  def test_90deg_about_z(self):
    # quat: (0, 0, sin(45), cos(45)) rotates [1,0,0] to [0,1,0]
    q = np.array([0, 0, SIN45, COS45])
    R = quat_to_rotation_matrix(q)
    v = R @ np.array([1, 0, 0])
    npt.assert_allclose(v, [0, 1, 0], atol=1e-12)

  def test_90deg_about_x(self):
    # quat: (sin(45), 0, 0, cos(45)) rotates [0,1,0] to [0,0,1]
    q = np.array([SIN45, 0, 0, COS45])
    R = quat_to_rotation_matrix(q)
    v = R @ np.array([0, 1, 0])
    npt.assert_allclose(v, [0, 0, 1], atol=1e-12)

  def test_90deg_about_y(self):
    # quat: (0, sin(45), 0, cos(45)) rotates [0,0,1] to [1,0,0]
    q = np.array([0, SIN45, 0, COS45])
    R = quat_to_rotation_matrix(q)
    v = R @ np.array([0, 0, 1])
    npt.assert_allclose(v, [1, 0, 0], atol=1e-12)

  def test_orthogonal(self):
    q = np.array([0.1, 0.2, 0.3, 0.9])
    q = q / np.linalg.norm(q)
    R = quat_to_rotation_matrix(q)
    npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


class TestRotationMatrixToQuat:
  def test_identity(self):
    q = rotation_matrix_to_quat_xyzw(np.eye(3))
    npt.assert_allclose(q, [0, 0, 0, 1], atol=1e-12)

  def test_90deg_about_z_roundtrip(self):
    q_orig = np.array([0, 0, SIN45, COS45])
    R = quat_to_rotation_matrix(q_orig)
    q_back = rotation_matrix_to_quat_xyzw(R)
    # q and -q represent the same rotation; canonical form has w >= 0
    npt.assert_allclose(q_back, q_orig, atol=1e-12)

  def test_180deg_about_z(self):
    # quat: (0, 0, 1, 0) — w=0, canonical sign is ambiguous but we flip to w>=0
    q = np.array([0, 0, 1.0, 0])
    R = quat_to_rotation_matrix(q)
    q_back = rotation_matrix_to_quat_xyzw(R)
    # Either (0,0,1,0) or (0,0,-1,0) is fine as long as it reconstructs R
    R_check = quat_to_rotation_matrix(q_back)
    npt.assert_allclose(R_check, R, atol=1e-12)


class TestPoseToMatrix:
  def test_identity(self):
    mat = pose_to_matrix(np.array([0, 0, 0.0]), np.array([0, 0, 0, 1.0]))
    npt.assert_allclose(mat, np.eye(4), atol=1e-12)

  def test_pure_translation(self):
    mat = pose_to_matrix(np.array([1, 2, 3.0]), np.array([0, 0, 0, 1.0]))
    expected = np.eye(4)
    expected[:3, 3] = [1, 2, 3]
    npt.assert_allclose(mat, expected, atol=1e-12)

  def test_rotation_and_translation(self):
    q = np.array([0, 0, SIN45, COS45])  # 90° about Z
    mat = pose_to_matrix(np.array([5, 6, 7.0]), q)
    assert mat.shape == (4, 4)
    # Translation
    npt.assert_allclose(mat[:3, 3], [5, 6, 7], atol=1e-12)
    # Rotation: [1,0,0] → [0,1,0]
    npt.assert_allclose(mat[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    # Bottom row
    npt.assert_allclose(mat[3, :], [0, 0, 0, 1], atol=1e-12)


class TestMatrixToPose:
  def test_identity(self):
    pos, q = matrix_to_pose(np.eye(4))
    npt.assert_allclose(pos, [0, 0, 0], atol=1e-12)
    npt.assert_allclose(q, [0, 0, 0, 1], atol=1e-12)

  def test_roundtrip(self):
    pos_orig = np.array([1.5, -2.3, 0.7])
    q_orig = np.array([0.1, 0.2, 0.3, 0.9])
    q_orig = q_orig / np.linalg.norm(q_orig)

    mat = pose_to_matrix(pos_orig, q_orig)
    pos_back, q_back = matrix_to_pose(mat)

    npt.assert_allclose(pos_back, pos_orig, atol=1e-12)
    # q and -q are the same rotation; compare canonical (w >= 0)
    if q_orig[3] < 0:
      q_orig = -q_orig
    npt.assert_allclose(q_back, q_orig, atol=1e-10)


class TestUrdfOriginToMatrix:
  def test_identity_origin(self):
    mat = urdf_origin_to_matrix(xyz=(0, 0, 0), rpy=(0, 0, 0))
    npt.assert_allclose(mat, np.eye(4), atol=1e-12)

  def test_pure_translation(self):
    mat = urdf_origin_to_matrix(xyz=(0, 0, 0.045), rpy=(0, 0, 0))
    expected = np.eye(4)
    expected[2, 3] = 0.045
    npt.assert_allclose(mat, expected, atol=1e-12)

  def test_pure_yaw_90(self):
    mat = urdf_origin_to_matrix(xyz=(0, 0, 0), rpy=(0, 0, np.pi / 2))
    # Rz(90°): [1,0,0] → [0,1,0]
    npt.assert_allclose(mat[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    npt.assert_allclose(mat[:3, 3], [0, 0, 0], atol=1e-12)

  def test_elastomer_offset(self):
    """Verify the elastomer fixed-joint offset rpy=(pi/2, -pi/2, 0)."""
    mat = urdf_origin_to_matrix(xyz=(0, 0, 0), rpy=(np.pi / 2, -np.pi / 2, 0))
    R = mat[:3, :3]
    # Verify orthogonality
    npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)
    # Zero translation
    npt.assert_allclose(mat[:3, 3], [0, 0, 0], atol=1e-12)

  def test_iiwa_ee_joint(self):
    """iiwa14_joint_ee: xyz=(0, 0, 0.045), rpy=(0, 0, 0)."""
    mat = urdf_origin_to_matrix(xyz=(0, 0, 0.045), rpy=(0, 0, 0))
    npt.assert_allclose(mat[:3, :3], np.eye(3), atol=1e-12)
    npt.assert_allclose(mat[:3, 3], [0, 0, 0.045], atol=1e-12)

  def test_sharpa_mount_joint(self):
    """sharpa_mount_joint: xyz=(0, 0, 0.05), rpy=(0, 0, -pi/2)."""
    mat = urdf_origin_to_matrix(xyz=(0, 0, 0.05), rpy=(0, 0, -np.pi / 2))
    # Rz(-90°): [1,0,0] → [0,-1,0]
    npt.assert_allclose(mat[:3, :3] @ [1, 0, 0], [0, -1, 0], atol=1e-12)
    npt.assert_allclose(mat[:3, 3], [0, 0, 0.05], atol=1e-12)

  def test_compose_two_transforms(self):
    """Compose iiwa14_joint_ee and sharpa_mount_joint offsets."""
    ee = urdf_origin_to_matrix(xyz=(0, 0, 0.045), rpy=(0, 0, 0))
    mount = urdf_origin_to_matrix(xyz=(0, 0, 0.05), rpy=(0, 0, -np.pi / 2))
    composed = ee @ mount
    # Total Z translation: 0.045 + 0.05 = 0.095
    npt.assert_allclose(composed[:3, 3], [0, 0, 0.095], atol=1e-12)
    # Rotation is just Rz(-90°)
    npt.assert_allclose(composed[:3, :3] @ [1, 0, 0], [0, -1, 0], atol=1e-12)
