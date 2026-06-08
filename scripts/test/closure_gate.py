#!/usr/bin/env python3
"""Phase-gated action noise for grasp closure.

During ``noisy_*`` collection we want the action-noise to be suppressed while
the hand is *closing on the object*, so that the grasp-closure labels are clean
(the policy then learns the expert's firm, deep grip rather than an averaged
timid one). Noise is kept during approach and transport.

The gate is a small, swappable state machine:

  * a **closure detector** decides, per timestep and per env, whether the hand
    is currently in the closure window (``proximity`` | ``palm_z`` | ``off``);
  * a **padding** state machine keeps the gate active for a few extra steps
    after the detector drops out, so a momentary blip out of threshold does not
    re-enable noise mid-grasp;
  * the gate scales the *accumulated* OU noise state (not just the freshly
    sampled increment) on the configured action groups.

CRITICAL: the OU noise is temporally correlated. ``noise_state`` carries
accumulated noise from prior steps, so it is not enough to zero the fresh
sample — the gate must suppress the accumulated ``noise_state`` itself on the
gated groups inside the window. ``ClosureGate.apply`` does exactly that.

The gate only ever touches the *executed* action noise. The stored label
(``clean_action_t`` for ``noisy_clean``) is untouched by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Action-group index map (mirrors stage5/stage6 layout, N_ACT = 29)
# ---------------------------------------------------------------------------

ARM_BASE_IDXS: List[int] = [0, 1, 2]
ARM_WRIST_IDXS: List[int] = [3, 4, 5, 6]
THUMB_IDXS: List[int] = [7, 8, 9, 10, 11]
INDEX_IDXS: List[int] = [12, 13, 14, 15]
MIDDLE_IDXS: List[int] = [16, 17, 18, 19]
RING_IDXS: List[int] = [20, 21, 22, 23]
PINKY_IDXS: List[int] = [24, 25, 26, 27, 28]

GROUP_IDXS: Dict[str, List[int]] = {
    "arm_base": ARM_BASE_IDXS,
    "arm_wrist": ARM_WRIST_IDXS,
    "wrist": ARM_WRIST_IDXS,  # alias — "wrist" reads more naturally in --closure-groups
    "thumb": THUMB_IDXS,
    "index": INDEX_IDXS,
    "middle": MIDDLE_IDXS,
    "ring": RING_IDXS,
    "pinky": PINKY_IDXS,
}

# Default gated groups: everything that closes the grip. arm_base is excluded
# so the gross arm position can still be perturbed while the grip closes
# cleanly.
DEFAULT_CLOSURE_GROUPS: List[str] = [
    "wrist",
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
]


def resolve_group_idxs(groups: List[str]) -> List[int]:
    """Flatten group names to a sorted, de-duplicated list of action indices."""
    idxs: set = set()
    for name in groups:
        if name not in GROUP_IDXS:
            raise ValueError(
                f"Unknown closure group {name!r}; valid: {sorted(GROUP_IDXS)}"
            )
        idxs.update(GROUP_IDXS[name])
    return sorted(idxs)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ClosureGateConfig:
    mode: str = "off"                      # "off" | "proximity" | "palm_z"
    proximity_threshold: float = 0.08      # m; palm-object dist below this = closure
    palm_z_threshold: float = 0.66         # m; palm z below this = closure (palm_z mode)
    noise_scale: float = 0.0               # multiplier on gated-group noise inside window
    groups: List[str] = field(default_factory=lambda: list(DEFAULT_CLOSURE_GROUPS))
    padding: int = 5                       # extra steps to keep gate on after detector drops

    def to_attrs(self) -> Dict[str, object]:
        return {
            "noise_phase_gating": self.mode,
            "closure_proximity_threshold": float(self.proximity_threshold),
            "closure_palm_z_threshold": float(self.palm_z_threshold),
            "closure_noise_scale": float(self.noise_scale),
            "closure_groups": list(self.groups),
            "closure_window_padding": int(self.padding),
        }


# ---------------------------------------------------------------------------
# Closure detector — the swappable piece
# ---------------------------------------------------------------------------

def _palm_center_pos(env) -> torch.Tensor:
    return env.palm_center_pos  # (num_envs, 3) world frame


def _object_pos(env) -> torch.Tensor:
    return env.object_pos       # (num_envs, 3) world frame


def detect_closure(env, cfg: ClosureGateConfig) -> torch.Tensor:
    """Return a bool tensor (num_envs,): True where the hand is closing.

    ``proximity`` (recommended): palm-center to object distance below threshold.
    Object-adaptive — handles differing tool heights automatically.
    ``palm_z``: palm-center z below threshold. Validated fallback when object
    pose is awkward to read mid-rollout (palm z dips to ~0.63 at grasp).
    """
    if cfg.mode == "off":
        palm = _palm_center_pos(env)
        return torch.zeros(palm.shape[0], dtype=torch.bool, device=palm.device)
    if cfg.mode == "proximity":
        dist = torch.linalg.vector_norm(
            _palm_center_pos(env) - _object_pos(env), dim=1
        )
        return dist < cfg.proximity_threshold
    if cfg.mode == "palm_z":
        return _palm_center_pos(env)[:, 2] < cfg.palm_z_threshold
    raise ValueError(f"Unknown closure mode {cfg.mode!r}")


# ---------------------------------------------------------------------------
# The gate (stateful: padding + diagnostics)
# ---------------------------------------------------------------------------

class ClosureGate:
    """Per-env phase gate over the accumulated OU noise state."""

    def __init__(self, cfg: ClosureGateConfig, num_envs: int, device) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._steps_since_in_window = torch.full(
            (num_envs,), 10**9, dtype=torch.long, device=device
        )
        gated = resolve_group_idxs(cfg.groups) if cfg.mode != "off" else []
        # (N_ACT,) multiplier: 1.0 everywhere, noise_scale on gated groups.
        # N_ACT is inferred lazily on first apply from the noise tensor width.
        self._gated_idxs = gated
        self._mult: Optional[torch.Tensor] = None
        # diagnostics
        self.closure_steps_total = 0
        self.gated_steps_total = 0  # env-steps where gate actually scaled noise

    @property
    def enabled(self) -> bool:
        return self.cfg.mode != "off"

    def _build_mult(self, n_act: int) -> torch.Tensor:
        mult = torch.ones(n_act, device=self.device)
        if self._gated_idxs:
            mult[self._gated_idxs] = float(self.cfg.noise_scale)
        return mult

    def apply(self, env, noise_state: torch.Tensor) -> torch.Tensor:
        """Gate ``noise_state`` in place-style and return it.

        Call AFTER the OU update, BEFORE adding to the clean action. The
        accumulated noise on gated groups is multiplied by ``noise_scale`` for
        every env currently in the (padded) closure window.
        """
        if not self.enabled:
            return noise_state
        if self._mult is None:
            self._mult = self._build_mult(noise_state.shape[1])

        raw = detect_closure(env, self.cfg)  # (num_envs,) bool
        self.closure_steps_total += int(raw.sum().item())

        # Padding state machine: in_window if detector fires now OR we are still
        # within `padding` steps of the last firing.
        self._steps_since_in_window = torch.where(
            raw,
            torch.zeros_like(self._steps_since_in_window),
            self._steps_since_in_window + 1,
        )
        in_window = self._steps_since_in_window <= self.cfg.padding  # (num_envs,)
        self.gated_steps_total += int(in_window.sum().item())

        if in_window.any():
            # Where in_window: scale gated groups by mult; else leave as-is.
            gated = noise_state * self._mult                    # (num_envs,N)
            noise_state = torch.where(in_window.unsqueeze(1), gated, noise_state)
        return noise_state

    def diagnostics(self) -> Dict[str, int]:
        return {
            "closure_steps_total": int(self.closure_steps_total),
            "closure_gated_steps_total": int(self.gated_steps_total),
        }
