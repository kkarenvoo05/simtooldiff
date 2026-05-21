import numpy as np
import numpy.testing as npt
import pytest

from blender_eval.asset_manifest import (
  ROBOT_URDF,
  TOOL_DIR,
  get_object_mesh_path,
  get_robot_mesh_manifest,
  parse_urdf_visual_meshes,
)
from blender_eval.pose_conversion import urdf_origin_to_matrix


class TestRobotManifest:
  @pytest.fixture(scope="class")
  def manifest(self):
    return get_robot_mesh_manifest()

  def test_urdf_exists(self):
    assert ROBOT_URDF.exists(), f"URDF not found: {ROBOT_URDF}"

  def test_manifest_not_empty(self, manifest):
    assert len(manifest) > 0

  def test_all_mesh_paths_exist(self, manifest):
    for link_name, info in manifest.items():
      assert info.mesh_path.exists(), (
        f"Mesh not found for {link_name}: {info.mesh_path}"
      )

  def test_kuka_links_present(self, manifest):
    for i in range(8):
      assert f"iiwa14_link_{i}" in manifest, f"Missing iiwa14_link_{i}"

  def test_kuka_links_not_collapsed(self, manifest):
    for i in range(8):
      info = manifest[f"iiwa14_link_{i}"]
      assert not info.is_collapsed, f"iiwa14_link_{i} should not be collapsed"

  def test_elastomer_links_collapsed(self, manifest):
    elastomers = [
      "left_thumb_elastomer",
      "left_index_elastomer",
      "left_middle_elastomer",
      "left_ring_elastomer",
      "left_pinky_elastomer",
    ]
    for name in elastomers:
      assert name in manifest, f"Missing {name}"
      info = manifest[name]
      assert info.is_collapsed, f"{name} should be collapsed"
      assert info.surviving_parent is not None
      assert info.joint_chain_offset is not None

  def test_elastomer_surviving_parents(self, manifest):
    expected_parents = {
      "left_thumb_elastomer": "left_thumb_DP",
      "left_index_elastomer": "left_index_DP",
      "left_middle_elastomer": "left_middle_DP",
      "left_ring_elastomer": "left_ring_DP",
      "left_pinky_elastomer": "left_pinky_DP",
    }
    for name, expected_parent in expected_parents.items():
      info = manifest[name]
      assert info.surviving_parent == expected_parent, (
        f"{name}: expected parent {expected_parent}, got {info.surviving_parent}"
      )

  def test_index_elastomer_offset(self, manifest):
    """The elastomer joint origin is xyz=(0,0,0) rpy=(pi/2, -pi/2, 0)."""
    info = manifest["left_index_elastomer"]
    expected = urdf_origin_to_matrix(
      xyz=(0, 0, 0), rpy=(np.pi / 2, -np.pi / 2, 0)
    )
    npt.assert_allclose(info.joint_chain_offset, expected, atol=1e-10)

  def test_palm_collapsed_into_iiwa_link7(self, manifest):
    """left_hand_C_MC should be collapsed (connected via fixed joints through
    iiwa14_link_ee and sharpa_mount to iiwa14_link_7)."""
    if "left_hand_C_MC" in manifest:
      info = manifest["left_hand_C_MC"]
      assert info.is_collapsed, "left_hand_C_MC should be collapsed"
      assert info.surviving_parent == "iiwa14_link_7", (
        f"Expected surviving parent iiwa14_link_7, got {info.surviving_parent}"
      )

  def test_visual_origins_are_identity(self, manifest):
    """Most visual origins in this URDF are identity."""
    identity = np.eye(4)
    identity_count = 0
    for info in manifest.values():
      if np.allclose(info.visual_origin, identity, atol=1e-10):
        identity_count += 1
    # At least 90% should be identity
    assert identity_count / len(manifest) > 0.8, (
      f"Only {identity_count}/{len(manifest)} visual origins are identity"
    )

  def test_offset_from_parent_property(self, manifest):
    """offset_from_parent should be joint_chain_offset @ visual_origin."""
    info = manifest["left_index_elastomer"]
    expected = info.joint_chain_offset @ info.visual_origin
    npt.assert_allclose(info.offset_from_parent, expected, atol=1e-12)

  def test_non_collapsed_offset_is_none(self, manifest):
    info = manifest["iiwa14_link_0"]
    assert info.offset_from_parent is None


class TestObjectMeshPath:
  def test_claw_hammer(self):
    path = get_object_mesh_path("claw_hammer")
    assert path.exists()
    assert path.suffix in (".obj", ".OBJ")

  def test_flat_spatula(self):
    path = get_object_mesh_path("flat_spatula")
    assert path.exists()

  def test_nonexistent_raises(self):
    with pytest.raises(FileNotFoundError):
      get_object_mesh_path("nonexistent_tool_xyz")

  def test_all_known_tools(self):
    known_tools = [
      "claw_hammer", "mallet_hammer",
      "long_screwdriver", "short_screwdriver",
      "blue_brush", "red_brush",
      "flat_spatula", "spoon_spatula",
      "sharpie_marker", "staples_marker",
      "flat_eraser", "handle_eraser",
    ]
    for tool in known_tools:
      path = get_object_mesh_path(tool)
      assert path.exists(), f"Mesh not found for {tool}: {path}"
