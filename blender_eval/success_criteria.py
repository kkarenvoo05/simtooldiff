"""Pickup success criteria used by Blender and lightweight eval tests.

Keep these helpers free of IsaacGym and dataset-collection imports so the
Blender evaluation path can run from a trained checkpoint without pulling in
data collection code.
"""

from dataclasses import dataclass
from typing import List, Optional

PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M = 0.02
PICKUP_SUCCESS_MIN_LIFT_M = 0.12
PICKUP_SUCCESS_HOLD_STEPS = 5


@dataclass(frozen=True)
class PickupSuccessMetrics:
  """Original pickup success plus a stricter stable-hold diagnostic."""

  success: bool
  stable_success: bool
  max_object_z: float
  max_lift_m: float
  first_success_step: Optional[int]
  first_stable_success_step: Optional[int]


def _first_true_step(values: List[bool]) -> Optional[int]:
  for step, value in enumerate(values):
    if value:
      return step
  return None


def _first_hold_step(values: List[bool], hold_steps: int) -> Optional[int]:
  if hold_steps <= 1:
    return _first_true_step(values)
  consecutive = 0
  for step, value in enumerate(values):
    consecutive = consecutive + 1 if value else 0
    if consecutive >= hold_steps:
      return step - hold_steps + 1
  return None


def pickup_success(
  object_zs: List[float],
  object_start_z: float,
  goal_z: float,
  goal_z_tolerance: float = PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M,
  min_lift: float = PICKUP_SUCCESS_MIN_LIFT_M,
) -> bool:
  """Max-height pickup success criterion.

  An episode succeeds if the object's maximum height during the episode:
    1. Reaches within goal_z_tolerance of goal_z, AND
    2. Is at least min_lift above the start height.

  Matches the original Stage 5 max-height pickup success criterion.
  """
  if not object_zs:
    return False
  max_z = max(object_zs)
  max_lift = max_z - object_start_z
  return bool(
    max_z >= goal_z - goal_z_tolerance
    and max_lift >= min_lift
  )


def compute_pickup_success_metrics(
  object_zs: List[float],
  object_start_z: float,
  goal_z: float,
  pickup_gate_history: Optional[List[bool]] = None,
  hold_steps: int = PICKUP_SUCCESS_HOLD_STEPS,
  goal_z_tolerance: float = PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M,
  min_lift: float = PICKUP_SUCCESS_MIN_LIFT_M,
) -> PickupSuccessMetrics:
  """Compute original pickup success and stable-hold diagnostic success.

  The headline/original pickup metric is max-height based. The stable metric is
  additive: it requires the pickup gate to be true for ``hold_steps``
  consecutive timesteps and is used only as a diagnostic.
  """
  if not object_zs:
    return PickupSuccessMetrics(
      success=False,
      stable_success=False,
      max_object_z=float("-inf"),
      max_lift_m=0.0,
      first_success_step=None,
      first_stable_success_step=None,
    )

  max_object_z = max(object_zs)
  max_lift_m = max_object_z - object_start_z
  success = bool(
    max_object_z >= goal_z - goal_z_tolerance
    and max_lift_m >= min_lift
  )

  if pickup_gate_history is None:
    pickup_gate_history = [
      bool(
        z >= goal_z - goal_z_tolerance
        and z - object_start_z >= min_lift
      )
      for z in object_zs
    ]
  first_success_step = _first_true_step(pickup_gate_history)
  first_stable_success_step = _first_hold_step(pickup_gate_history, hold_steps)
  stable_success = bool(success and first_stable_success_step is not None)

  return PickupSuccessMetrics(
    success=success,
    stable_success=stable_success,
    max_object_z=max_object_z,
    max_lift_m=max_lift_m,
    first_success_step=first_success_step,
    first_stable_success_step=first_stable_success_step,
  )


def stable_pickup_success(
  object_zs: List[float],
  object_start_z: float,
  goal_z: float,
  goal_z_tolerance: float = PICKUP_SUCCESS_GOAL_Z_TOLERANCE_M,
  min_lift: float = PICKUP_SUCCESS_MIN_LIFT_M,
  hold_steps: int = PICKUP_SUCCESS_HOLD_STEPS,
) -> bool:
  """Stable pickup success requiring the pickup gate for consecutive steps."""
  gate_history = [
    bool(z >= goal_z - goal_z_tolerance and z - object_start_z >= min_lift)
    for z in object_zs
  ]
  return compute_pickup_success_metrics(
    object_zs=object_zs,
    object_start_z=object_start_z,
    goal_z=goal_z,
    pickup_gate_history=gate_history,
    hold_steps=hold_steps,
    goal_z_tolerance=goal_z_tolerance,
    min_lift=min_lift,
  ).stable_success
