from blender_eval.success_criteria import (
  PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M,
  PICKUP_SUCCESS_MIN_LIFT_M,
  compute_pickup_success_metrics,
  pickup_success,
  stable_pickup_success,
)


class TestPickupSuccess:
  def test_empty_zs_fails(self):
    assert not pickup_success([], 0.4, 0.6)

  def test_below_goal_fails(self):
    assert not pickup_success([0.4, 0.45, 0.5], 0.4, 0.6)

  def test_above_goal_succeeds(self):
    assert pickup_success([0.4, 0.5, 0.59], 0.4, 0.6)

  def test_exact_goal_succeeds(self):
    assert pickup_success([0.6], 0.4, 0.6)

  def test_within_tolerance(self):
    # goal_z=0.6, tolerance=0.02 → need max_z >= 0.58
    assert pickup_success([0.58], 0.4, 0.6)

  def test_just_below_tolerance_fails(self):
    assert not pickup_success([0.579], 0.4, 0.6)

  def test_min_lift_required(self):
    # Even if max_z >= goal, must lift at least 0.12 from start
    assert not pickup_success([0.6], 0.59, 0.6, min_lift=0.12)

  def test_min_lift_met(self):
    assert pickup_success([0.6], 0.4, 0.6, min_lift=0.12)

  def test_custom_tolerance(self):
    assert pickup_success([0.55], 0.4, 0.6, goal_z_tolerance=0.1)
    assert not pickup_success([0.55], 0.4, 0.6, goal_z_tolerance=0.01)

  def test_defaults_match_stage5_constants(self):
    """Verify defaults match the pickup constants used by checkpoint eval."""
    import inspect
    sig = inspect.signature(pickup_success)
    assert sig.parameters["goal_z_tolerance"].default == PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M
    assert sig.parameters["min_lift"].default == PICKUP_SUCCESS_MIN_LIFT_M

  def test_metrics_reports_original_and_stable_success(self):
    metrics = compute_pickup_success_metrics(
      object_zs=[0.40, 0.58, 0.59, 0.60, 0.60, 0.60],
      object_start_z=0.40,
      goal_z=0.60,
      hold_steps=5,
    )
    assert metrics.success
    assert metrics.stable_success
    assert metrics.max_object_z == 0.60
    assert round(metrics.max_lift_m, 6) == 0.20
    assert metrics.first_success_step == 1
    assert metrics.first_stable_success_step == 1

  def test_metrics_keeps_stable_success_additive(self):
    metrics = compute_pickup_success_metrics(
      object_zs=[0.40, 0.58, 0.55, 0.59],
      object_start_z=0.40,
      goal_z=0.60,
      hold_steps=3,
    )
    assert metrics.success
    assert not metrics.stable_success
    assert metrics.first_success_step == 1
    assert metrics.first_stable_success_step is None

  def test_stable_pickup_success_requires_hold_window(self):
    assert not stable_pickup_success(
      [0.40, 0.58, 0.55, 0.59],
      object_start_z=0.40,
      goal_z=0.60,
      hold_steps=3,
    )
    assert stable_pickup_success(
      [0.40, 0.58, 0.59, 0.60],
      object_start_z=0.40,
      goal_z=0.60,
      hold_steps=3,
    )

  def test_stable_pickup_success_uses_custom_thresholds_for_success(self):
    assert stable_pickup_success(
      [0.40, 0.55, 0.55, 0.55],
      object_start_z=0.40,
      goal_z=0.60,
      goal_z_tolerance=0.05,
      min_lift=0.10,
      hold_steps=3,
    )
    assert not stable_pickup_success(
      [0.40, 0.55, 0.55, 0.55],
      object_start_z=0.40,
      goal_z=0.60,
      goal_z_tolerance=0.01,
      min_lift=0.10,
      hold_steps=3,
    )
