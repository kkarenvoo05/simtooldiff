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
) -> Dict[str, float]:
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
    return scaled
