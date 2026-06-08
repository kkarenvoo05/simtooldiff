#!/usr/bin/env python3
"""Shared Stage 5 noisy collection defaults."""

from typing import Dict, Optional


BASE_NOISE_DEFAULTS = {
    "arm_base": 0.005,
    "arm_wrist": 0.01,
    "thumb": 0.015,
    "index": 0.01,
    "middle": 0.01,
    "ring": 0.008,
    "pinky": 0.008,
}


FINGER_GROUPS = ("thumb", "index", "middle", "ring", "pinky")
WRIST_GROUPS = ("arm_wrist",)


def resolve_noise_config(
    *,
    noise_scale: float,
    arm_base_noise: Optional[float],
    arm_wrist_noise: Optional[float],
    thumb_noise: Optional[float],
    index_noise: Optional[float],
    middle_noise: Optional[float],
    ring_noise: Optional[float],
    pinky_noise: Optional[float],
    finger_noise_multiplier: float = 1.0,
    wrist_noise_multiplier: float = 1.0,
) -> Dict[str, float]:
    """Resolve per-group OU noise stds.

    ``noise_scale`` scales every group's base default. Explicit per-group
    overrides (``*_noise``) take precedence over the scaled base. Finally, the
    blunt-fallback multipliers (``finger_noise_multiplier``,
    ``wrist_noise_multiplier``) scale the relevant groups — this is the cheap,
    one-knob alternative to phase-gating that reduces finger/wrist noise
    everywhere (fingers/wrist near contact are the least recoverable, so they
    should carry the least noise).
    """
    scaled = {
        name: base_value * noise_scale
        for name, base_value in BASE_NOISE_DEFAULTS.items()
    }
    overrides = {
        "arm_base": arm_base_noise,
        "arm_wrist": arm_wrist_noise,
        "thumb": thumb_noise,
        "index": index_noise,
        "middle": middle_noise,
        "ring": ring_noise,
        "pinky": pinky_noise,
    }
    for name, value in overrides.items():
        if value is not None:
            scaled[name] = value
    for name in FINGER_GROUPS:
        scaled[name] *= finger_noise_multiplier
    for name in WRIST_GROUPS:
        scaled[name] *= wrist_noise_multiplier
    return scaled
