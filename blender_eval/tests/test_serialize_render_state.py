import numpy as np
import numpy.testing as npt

from blender_eval.state_extraction import RenderState, serialize_render_state
from blender_eval.pose_conversion import quat_xyzw_to_wxyz


class TestSerializeRenderState:
  def _make_state(self):
    """Create a RenderState with known xyzw quaternions."""
    return RenderState(
      mesh_poses={
        "link_a": (
          np.array([1.0, 2.0, 3.0], dtype=np.float32),
          np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32),  # xyzw
        ),
        "link_b": (
          np.array([4.0, 5.0, 6.0], dtype=np.float32),
          np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # identity xyzw
        ),
      },
      object_pos=np.array([7.0, 8.0, 9.0], dtype=np.float32),
      object_quat_xyzw=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),  # xyzw
      object_name="test_tool",
    )

  def test_mesh_poses_are_wxyz(self):
    state = self._make_state()
    d = serialize_render_state(state)

    # link_a: xyzw (0.1, 0.2, 0.3, 0.9) → wxyz (0.9, 0.1, 0.2, 0.3)
    pos_a, quat_a = d["mesh_poses"]["link_a"]
    npt.assert_allclose(pos_a, [1.0, 2.0, 3.0], atol=1e-6)
    npt.assert_allclose(quat_a, [0.9, 0.1, 0.2, 0.3], atol=1e-6)

  def test_identity_quat_converted(self):
    state = self._make_state()
    d = serialize_render_state(state)

    # link_b: xyzw (0,0,0,1) → wxyz (1,0,0,0)
    _, quat_b = d["mesh_poses"]["link_b"]
    npt.assert_allclose(quat_b, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

  def test_object_quat_is_wxyz(self):
    state = self._make_state()
    d = serialize_render_state(state)

    # xyzw (0.5, 0.5, 0.5, 0.5) → wxyz (0.5, 0.5, 0.5, 0.5)
    npt.assert_allclose(d["object_quat_wxyz"], [0.5, 0.5, 0.5, 0.5], atol=1e-6)

  def test_object_name_passed_through(self):
    state = self._make_state()
    d = serialize_render_state(state)
    assert d["object_name"] == "test_tool"

  def test_tool_mesh_path_included(self):
    state = self._make_state()
    d = serialize_render_state(state, tool_mesh_path="/path/to/mesh.obj")
    assert d["tool_mesh_path"] == "/path/to/mesh.obj"

  def test_json_serializable(self):
    """The output must be JSON-serializable (no numpy arrays)."""
    import json
    state = self._make_state()
    d = serialize_render_state(state)
    # This will raise TypeError if any numpy arrays remain
    json.dumps(d)

  def test_conversion_matches_quat_xyzw_to_wxyz(self):
    """Verify the conversion matches the standalone function."""
    state = self._make_state()
    d = serialize_render_state(state)

    expected = quat_xyzw_to_wxyz(state.object_quat_xyzw).tolist()
    npt.assert_allclose(d["object_quat_wxyz"], expected, atol=1e-6)
