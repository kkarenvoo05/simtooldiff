"""GPU tests for state_extraction. Requires a live IsaacGym environment.

Run with: python -m pytest blender_eval/tests/test_state_extraction.py -m gpu -v

These tests are skipped if CUDA is not available or if PyTorch CUDA doesn't
support the current GPU (e.g. Blackwell on Python 3.8).
"""

import numpy as np
import pytest

try:
  import torch
  _cuda_ok = torch.cuda.is_available()
  if _cuda_ok:
    try:
      torch.randn(1, device="cuda")
    except RuntimeError:
      _cuda_ok = False
except ImportError:
  _cuda_ok = False

gpu = pytest.mark.skipif(not _cuda_ok, reason="CUDA not available or incompatible GPU")


@gpu
class TestStateExtraction:
  """Tests that require a live IsaacGym env."""

  @pytest.fixture(scope="class")
  def env_and_manifest(self):
    """Create env once for all tests in this class."""
    import sys
    from pathlib import Path

    # IsaacGym must be imported before torch
    from isaacgym import gymapi  # noqa: F401
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    from stage5_collect_dataset import _make_env, _load_nominal_start_pose, LIFT_HEIGHT_M

    from blender_eval.asset_manifest import get_robot_mesh_manifest

    nominal_start = _load_nominal_start_pose("hammer", "claw_hammer", "swing_down")
    nominal_goal = list(nominal_start)
    nominal_goal[2] += LIFT_HEIGHT_M

    env = _make_env(
      num_envs=1,
      nominal_start_pose=nominal_start,
      nominal_goal_pose=nominal_goal,
      xy_range=0.0,
      horizon=50,
      headless=True,
      device="cuda:0",
      seed=0,
      object_name="claw_hammer",
    )

    # Step once to populate buffers
    zero_action = torch.zeros((1, 29), device="cuda:0")
    env.step(zero_action)

    manifest = get_robot_mesh_manifest()
    yield env, manifest

  def test_extract_state(self, env_and_manifest):
    env, manifest = env_and_manifest
    from blender_eval.state_extraction import extract_render_state

    state = extract_render_state(env, 0, "claw_hammer", manifest)
    assert len(state.mesh_poses) == len(manifest)

  def test_positions_finite(self, env_and_manifest):
    env, manifest = env_and_manifest
    from blender_eval.state_extraction import extract_render_state

    state = extract_render_state(env, 0, "claw_hammer", manifest)
    for name, (pos, quat) in state.mesh_poses.items():
      assert np.all(np.isfinite(pos)), f"{name} pos not finite: {pos}"
      assert np.all(np.isfinite(quat)), f"{name} quat not finite: {quat}"

  def test_quaternions_unit_norm(self, env_and_manifest):
    env, manifest = env_and_manifest
    from blender_eval.state_extraction import extract_render_state

    state = extract_render_state(env, 0, "claw_hammer", manifest)
    for name, (pos, quat) in state.mesh_poses.items():
      norm = np.linalg.norm(quat)
      assert abs(norm - 1.0) < 1e-3, f"{name} quat norm={norm}"

  def test_object_pos_matches_env(self, env_and_manifest):
    env, manifest = env_and_manifest
    from blender_eval.state_extraction import extract_render_state

    state = extract_render_state(env, 0, "claw_hammer", manifest)
    env_pos = env.object_pos[0].detach().cpu().numpy()
    np.testing.assert_allclose(state.object_pos, env_pos, atol=1e-5)

  def test_surviving_bodies_cross_reference(self, env_and_manifest):
    env, manifest = env_and_manifest
    from blender_eval.state_extraction import get_surviving_body_names

    surviving = set(get_surviving_body_names(env))
    for name, info in manifest.items():
      if not info.is_collapsed:
        assert name in surviving, (
          f"Non-collapsed link '{name}' not in rigid_body_name_to_idx"
        )
      else:
        assert name not in surviving, (
          f"Collapsed link '{name}' unexpectedly in rigid_body_name_to_idx"
        )
