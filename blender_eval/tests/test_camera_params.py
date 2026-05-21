import math

import numpy as np
import numpy.testing as npt
import pytest

from blender_eval.camera_params import (
  BlenderCameraParams,
  blender_camera_hfov_deg,
  camera_forward_vector,
  camera_up_vector,
  isaacgym_to_blender_camera,
  project_point_to_image,
)

# The actual training camera configuration
CAMERA_POS = [0.55, -1.35, 1.10]
CAMERA_TARGET = [-0.1, 0.35, 0.60]
HFOV = 30.0
WIDTH = 512
HEIGHT = 384


class TestFovRoundtrip:
  def test_30deg(self):
    params = isaacgym_to_blender_camera(30.0, 512, 384, [0, 0, 1], [0, 0, 0])
    hfov_back = blender_camera_hfov_deg(params.focal_length_mm, params.sensor_width_mm)
    assert abs(hfov_back - 30.0) < 1e-10

  def test_90deg(self):
    params = isaacgym_to_blender_camera(90.0, 512, 384, [0, 0, 1], [0, 0, 0])
    hfov_back = blender_camera_hfov_deg(params.focal_length_mm, params.sensor_width_mm)
    assert abs(hfov_back - 90.0) < 1e-10

  def test_90deg_focal_is_half_sensor(self):
    # At 90° HFOV: focal = sensor_width / (2 * tan(45°)) = sensor_width / 2
    params = isaacgym_to_blender_camera(90.0, 512, 384, [0, 0, 1], [0, 0, 0])
    assert abs(params.focal_length_mm - params.sensor_width_mm / 2.0) < 1e-10

  def test_known_geometry_18mm(self):
    # HFOV=90°, sensor_width=36mm → focal = 36/(2*tan(45°)) = 18mm
    params = isaacgym_to_blender_camera(90.0, 512, 384, [0, 0, 1], [0, 0, 0],
                                        sensor_width_mm=36.0)
    assert abs(params.focal_length_mm - 18.0) < 1e-10


class TestSensorHeight:
  def test_aspect_ratio(self):
    params = isaacgym_to_blender_camera(30.0, 512, 384, [0, 0, 1], [0, 0, 0])
    expected = params.sensor_width_mm * 384 / 512
    assert abs(params.sensor_height_mm - expected) < 1e-10

  def test_512x360(self):
    params = isaacgym_to_blender_camera(30.0, 512, 360, [0, 0, 1], [0, 0, 0])
    expected = 36.0 * 360 / 512
    assert abs(params.sensor_height_mm - expected) < 1e-10


class TestForwardAxisOrientation:
  def test_forward_points_to_target(self):
    """Camera forward should point from position toward target."""
    params = isaacgym_to_blender_camera(HFOV, WIDTH, HEIGHT, CAMERA_POS, CAMERA_TARGET)
    fwd = camera_forward_vector(params.rotation_quaternion_wxyz)

    expected_dir = np.array(CAMERA_TARGET) - np.array(CAMERA_POS)
    expected_dir = expected_dir / np.linalg.norm(expected_dir)

    dot = np.dot(fwd, expected_dir)
    assert dot > 0.999, f"Forward dot product {dot} should be ~1.0"

  def test_up_has_positive_z(self):
    """Camera up vector should have positive Z (not upside down)."""
    params = isaacgym_to_blender_camera(HFOV, WIDTH, HEIGHT, CAMERA_POS, CAMERA_TARGET)
    up = camera_up_vector(params.rotation_quaternion_wxyz)
    assert up[2] > 0, f"Camera up Z = {up[2]}, should be positive"

  def test_looking_straight_down(self):
    """Camera at (0,0,5) looking at (0,0,0) — forward should be (0,0,-1)."""
    params = isaacgym_to_blender_camera(30.0, 512, 384, [0, 0, 5], [0, 0, 0])
    fwd = camera_forward_vector(params.rotation_quaternion_wxyz)
    npt.assert_allclose(fwd, [0, 0, -1], atol=1e-10)

  def test_looking_along_positive_y(self):
    """Camera at (0,-5,0) looking at (0,0,0) — forward should be (0,1,0)."""
    params = isaacgym_to_blender_camera(30.0, 512, 384, [0, -5, 0], [0, 0, 0])
    fwd = camera_forward_vector(params.rotation_quaternion_wxyz)
    npt.assert_allclose(fwd, [0, 1, 0], atol=1e-10)


class TestKnownPointProjection:
  def test_target_projects_to_center(self):
    """The camera target should project to roughly the image center."""
    params = isaacgym_to_blender_camera(HFOV, WIDTH, HEIGHT, CAMERA_POS, CAMERA_TARGET)
    u, v = project_point_to_image(
      np.array(CAMERA_TARGET),
      np.array(CAMERA_POS),
      params.rotation_quaternion_wxyz,
      params.focal_length_mm,
      params.sensor_width_mm,
      params.sensor_height_mm,
      params.resolution_x,
      params.resolution_y,
    )
    # Should be close to center (256, 192)
    assert abs(u - WIDTH / 2.0) < 2.0, f"u={u}, expected ~{WIDTH/2}"
    assert abs(v - HEIGHT / 2.0) < 2.0, f"v={v}, expected ~{HEIGHT/2}"

  def test_point_right_of_center(self):
    """A point offset to camera's right should have u > width/2."""
    params = isaacgym_to_blender_camera(90.0, 512, 384, [0, -5, 0], [0, 0, 0])
    # Camera at (0,-5,0) looking at (0,0,0). Camera right is +X.
    # Point at (1, 0, 0) should be to the right of center.
    u, v = project_point_to_image(
      np.array([1.0, 0.0, 0.0]),
      np.array([0.0, -5.0, 0.0]),
      params.rotation_quaternion_wxyz,
      params.focal_length_mm,
      params.sensor_width_mm,
      params.sensor_height_mm,
      params.resolution_x,
      params.resolution_y,
    )
    assert u > 256, f"u={u}, expected > 256 (right of center)"


class TestResolution:
  def test_resolution_passed_through(self):
    params = isaacgym_to_blender_camera(30.0, 512, 384, [0, 0, 1], [0, 0, 0])
    assert params.resolution_x == 512
    assert params.resolution_y == 384

  def test_location_passed_through(self):
    params = isaacgym_to_blender_camera(30.0, 512, 384, [1, 2, 3], [0, 0, 0])
    npt.assert_allclose(params.location, (1.0, 2.0, 3.0))
