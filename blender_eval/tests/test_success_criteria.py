from blender_eval.success_criteria import pickup_success


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
    """Verify defaults match stage5_collect_dataset constants."""
    import inspect
    sig = inspect.signature(pickup_success)
    assert sig.parameters["goal_z_tolerance"].default == 0.02
    assert sig.parameters["min_lift"].default == 0.12
