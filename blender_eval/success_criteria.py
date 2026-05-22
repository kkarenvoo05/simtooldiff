"""Regression-test mirror of the Stage 5 pickup success criterion.

Runtime evaluators import stage5_collect_dataset._episode_pickup_success()
directly so photoreal evaluation cannot drift from the original checkpoint eval.
This helper remains useful for small unit tests that do not import IsaacGym.
"""

from typing import List


def pickup_success(
  object_zs: List[float],
  object_start_z: float,
  goal_z: float,
  goal_z_tolerance: float = 0.02,
  min_lift: float = 0.12,
) -> bool:
  """Max-height pickup success criterion.

  An episode succeeds if the object's maximum height during the episode:
    1. Reaches within goal_z_tolerance of goal_z, AND
    2. Is at least min_lift above the start height.

  Matches stage5_collect_dataset._episode_pickup_success().
  """
  if not object_zs:
    return False
  max_z = max(object_zs)
  max_lift = max_z - object_start_z
  return bool(
    max_z >= goal_z - goal_z_tolerance
    and max_lift >= min_lift
  )
