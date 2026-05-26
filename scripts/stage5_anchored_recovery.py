#!/usr/bin/env python3
"""Shared helpers for anchored-recovery dataset collection."""

from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class AnchoredRecoveryConfig:
    collection_mode: str
    branches_per_rollout: int
    branch_min_step: int
    branch_max_step: int
    branch_gap_override: Optional[int]
    perturb_steps: int
    recovery_steps: int
    min_saved_steps: int
    max_saved_steps: int
    noise_scale: float
    arm_base_noise: float
    arm_wrist_noise: float
    thumb_noise: float
    index_noise: float
    middle_noise: float
    ring_noise: float
    pinky_noise: float
    ou_theta: float
    ou_mu: float
    ou_dt: float
    executed_action_clip: float = 1.0

    @property
    def branch_gap(self) -> int:
        if self.branch_gap_override is not None:
            return int(self.branch_gap_override)
        return self.perturb_steps + self.recovery_steps

    @property
    def saved_branch_steps(self) -> int:
        if self.collection_mode == "verified_full":
            return int(self.max_saved_steps)
        return compute_saved_branch_length(
            perturb_steps=self.perturb_steps,
            recovery_steps=self.recovery_steps,
        )


@dataclass
class AnchoredBranchSnapshot:
    env_state: Dict[str, Any]
    policy_state: Any
    obs: torch.Tensor


@dataclass
class AnchoredBranchBuffer:
    images: List[np.ndarray] = field(default_factory=list)
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)

    def append(self, image: np.ndarray, state: np.ndarray, action: np.ndarray) -> None:
        self.images.append(image)
        self.states.append(state)
        self.actions.append(action)

    @property
    def num_transitions(self) -> int:
        return len(self.images)

    def as_episode(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.images or not self.states or not self.actions:
            raise ValueError("Anchored branch buffer is empty.")
        return (
            np.stack(self.images, axis=0).astype(np.uint8),
            np.stack(self.states, axis=0).astype(np.float32),
            np.stack(self.actions, axis=0).astype(np.float32),
        )

    def slice(self, start: int = 0, end: Optional[int] = None) -> "AnchoredBranchBuffer":
        return AnchoredBranchBuffer(
            images=list(self.images[start:end]),
            states=list(self.states[start:end]),
            actions=list(self.actions[start:end]),
        )


@dataclass
class AnchoredBranchResult:
    success: bool
    aborted_reason: Optional[str]
    buffer: AnchoredBranchBuffer
    noise_metrics: Dict[str, float]
    steps_executed: int
    object_zs: List[float] = field(default_factory=list)

    @property
    def saved_transitions(self) -> int:
        return self.buffer.num_transitions


def compute_saved_branch_length(*, perturb_steps: int, recovery_steps: int) -> int:
    return max(int(perturb_steps) - 1, 0) + int(recovery_steps)


def resolve_branch_max_step(
    *,
    horizon: int,
    perturb_steps: int,
    recovery_steps: int,
    requested_max_step: str,
    collection_mode: str = "fixed_segment",
) -> int:
    if collection_mode == "verified_full":
        auto_max_step = int(horizon) - int(perturb_steps) - 1
    else:
        auto_max_step = int(horizon) - int(perturb_steps) - int(recovery_steps) - 1
    if requested_max_step == "auto":
        return auto_max_step
    return min(int(requested_max_step), auto_max_step)


def build_anchored_recovery_config(args: argparse.Namespace) -> AnchoredRecoveryConfig:
    branch_max_step = resolve_branch_max_step(
        horizon=args.horizon,
        perturb_steps=args.anchored_perturb_steps,
        recovery_steps=args.anchored_recovery_steps,
        requested_max_step=args.anchored_branch_max_step,
        collection_mode=args.anchored_collection_mode,
    )
    return AnchoredRecoveryConfig(
        collection_mode=str(args.anchored_collection_mode),
        branches_per_rollout=int(args.anchored_branches_per_rollout),
        branch_min_step=int(args.anchored_branch_min_step),
        branch_max_step=branch_max_step,
        branch_gap_override=args.anchored_branch_gap,
        perturb_steps=int(args.anchored_perturb_steps),
        recovery_steps=int(args.anchored_recovery_steps),
        min_saved_steps=int(args.anchored_min_saved_steps),
        max_saved_steps=int(args.anchored_max_saved_steps),
        noise_scale=float(args.noise_scale),
        arm_base_noise=float(args.arm_base_noise),
        arm_wrist_noise=float(args.arm_wrist_noise),
        thumb_noise=float(args.thumb_noise),
        index_noise=float(args.index_noise),
        middle_noise=float(args.middle_noise),
        ring_noise=float(args.ring_noise),
        pinky_noise=float(args.pinky_noise),
        ou_theta=float(args.ou_theta),
        ou_mu=float(args.ou_mu),
        ou_dt=float(args.ou_dt),
    )


def validate_anchored_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.noise_strategy != "anchored_recovery":
        return args
    if args.variant != "noisy_clean":
        raise ValueError("--noise-strategy anchored_recovery requires --variant noisy_clean")
    if int(args.num_envs) != 1:
        raise ValueError("--noise-strategy anchored_recovery requires --num-envs=1")
    if bool(getattr(args, "noisy_worker", False)):
        raise ValueError("--noise-strategy anchored_recovery does not support --noisy-worker")
    if int(args.anchored_branches_per_rollout) < 0:
        raise ValueError("--anchored-branches-per-rollout must be >= 0")
    if args.anchored_collection_mode not in {"fixed_segment", "verified_full"}:
        raise ValueError("--anchored-collection-mode must be fixed_segment or verified_full")
    if (
        args.anchored_collection_mode == "verified_full"
        and getattr(args, "pickup_success_mode", None) != "stable_hold"
    ):
        raise ValueError(
            "--anchored-collection-mode verified_full requires "
            "--pickup-success-mode stable_hold"
        )
    if int(args.anchored_branch_min_step) < 0:
        raise ValueError("--anchored-branch-min-step must be >= 0")
    if args.anchored_branch_gap is not None and int(args.anchored_branch_gap) <= 0:
        raise ValueError("--anchored-branch-gap must be positive when set")
    if int(args.anchored_perturb_steps) < 0:
        raise ValueError("--anchored-perturb-steps must be >= 0")
    if int(args.anchored_recovery_steps) < 0:
        raise ValueError("--anchored-recovery-steps must be >= 0")
    if int(args.anchored_perturb_steps) + int(args.anchored_recovery_steps) <= 0:
        raise ValueError(
            "--noise-strategy anchored_recovery requires a positive perturb/recovery horizon"
        )
    if int(args.anchored_min_saved_steps) <= 0:
        raise ValueError("--anchored-min-saved-steps must be positive")
    if int(args.anchored_max_saved_steps) < int(args.anchored_min_saved_steps):
        raise ValueError("--anchored-max-saved-steps must be >= --anchored-min-saved-steps")
    if args.anchored_branch_max_step != "auto":
        int(args.anchored_branch_max_step)
    return args


def sample_branch_trigger_steps(
    *,
    rng: np.random.Generator,
    branch_min_step: int,
    branch_max_step: int,
    branches_per_rollout: int,
    min_gap: int,
) -> List[int]:
    if branches_per_rollout <= 0 or branch_max_step < branch_min_step:
        return []
    if min_gap <= 0:
        min_gap = 1
    candidates = list(range(int(branch_min_step), int(branch_max_step) + 1))
    selected: List[int] = []
    while candidates and len(selected) < int(branches_per_rollout):
        choice_idx = int(rng.integers(0, len(candidates)))
        trigger_step = candidates[choice_idx]
        selected.append(trigger_step)
        candidates = [
            step for step in candidates if abs(step - trigger_step) >= int(min_gap)
        ]
    return sorted(selected)


def anchored_root_attr_values(
    *,
    variant: str,
    config: AnchoredRecoveryConfig,
) -> Dict[str, Any]:
    return {
        "variant": variant,
        "noise_strategy": "anchored_recovery",
        "episode_semantics": "anchored_branch_segment",
        "anchored_collection_mode": config.collection_mode,
        "saved_segment_policy": (
            "verified_recovery_prefix"
            if config.collection_mode == "verified_full"
            else "burst_plus_recovery"
        ),
        "executed_action_clipped": True,
        "anchored_branches_per_rollout": config.branches_per_rollout,
        "anchored_branch_min_step": config.branch_min_step,
        "anchored_branch_max_step": config.branch_max_step,
        "anchored_branch_gap": config.branch_gap,
        "anchored_perturb_steps": config.perturb_steps,
        "anchored_recovery_steps": config.recovery_steps,
        "anchored_min_saved_steps": config.min_saved_steps,
        "anchored_max_saved_steps": config.max_saved_steps,
        "anchored_saved_branch_steps": config.saved_branch_steps,
        "noise_scale": config.noise_scale,
        "ou_theta": config.ou_theta,
        "ou_mu": config.ou_mu,
        "ou_dt": config.ou_dt,
        "arm_base_noise": config.arm_base_noise,
        "arm_wrist_noise": config.arm_wrist_noise,
        "thumb_noise": config.thumb_noise,
        "index_noise": config.index_noise,
        "middle_noise": config.middle_noise,
        "ring_noise": config.ring_noise,
        "pinky_noise": config.pinky_noise,
    }


def validate_anchored_resume_attrs(
    root,
    *,
    variant: str,
    config: AnchoredRecoveryConfig,
) -> None:
    if int(root["meta"]["episode_ends"].shape[0]) <= 0:
        return
    expected = anchored_root_attr_values(variant=variant, config=config)
    optional_fixed_segment_keys = {
        "anchored_collection_mode",
        "anchored_branch_gap",
        "anchored_min_saved_steps",
        "anchored_max_saved_steps",
    }
    mismatches: List[str] = []
    for key, expected_value in expected.items():
        actual_value = root.attrs.get(key, None)
        if (
            config.collection_mode == "fixed_segment"
            and actual_value is None
            and key in optional_fixed_segment_keys
        ):
            continue
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected {expected_value!r}, found {actual_value!r}"
            )
    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(f"Anchored recovery resume attrs mismatch: {joined}")


def update_anchored_root_attrs(
    root,
    *,
    variant: str,
    config: AnchoredRecoveryConfig,
    base_rollouts_attempted: int,
    base_rollouts_succeeded: int,
    branches_attempted: int,
    branches_written: int,
    branches_aborted: int,
    saved_transitions: int,
) -> None:
    for key, value in anchored_root_attr_values(variant=variant, config=config).items():
        root.attrs[key] = value
    root.attrs["anchored_base_rollouts_attempted"] = int(base_rollouts_attempted)
    root.attrs["anchored_base_rollouts_succeeded"] = int(base_rollouts_succeeded)
    root.attrs["anchored_branches_attempted"] = int(branches_attempted)
    root.attrs["anchored_branches_written"] = int(branches_written)
    root.attrs["anchored_branches_aborted"] = int(branches_aborted)
    root.attrs["anchored_saved_transitions"] = int(saved_transitions)


def maybe_fork_rng(device: str):
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return nullcontext()
    device_idx = torch_device.index
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    return torch.random.fork_rng(devices=[device_idx])


def run_anchored_branch(
    *,
    env,
    device: str,
    obs: torch.Tensor,
    config: AnchoredRecoveryConfig,
    active_envs: torch.Tensor,
    build_clean_action: Callable[[torch.Tensor], torch.Tensor],
    sample_group_noise: Callable[[torch.Tensor], torch.Tensor],
    total_steps: Optional[int] = None,
) -> AnchoredBranchResult:
    if env.num_envs != 1:
        raise ValueError("Anchored branch execution currently requires num_envs=1")

    action_dim = int(getattr(env, "num_actions"))
    noise_state = torch.zeros((env.num_envs, action_dim), device=device)
    sqrt_dt = math.sqrt(config.ou_dt)
    if total_steps is None:
        total_steps = config.perturb_steps + config.recovery_steps
    total_steps = int(total_steps)
    buffer = AnchoredBranchBuffer()
    noise_metrics = {"l2_sum": 0.0, "l2_sq_sum": 0.0, "linf_sum": 0.0, "count": 0}
    steps_executed = 0
    object_zs: List[float] = []

    for branch_step in range(total_steps):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            return AnchoredBranchResult(
                success=False,
                aborted_reason="viewer_closed",
                buffer=buffer,
                noise_metrics=noise_metrics,
                steps_executed=steps_executed,
                object_zs=[],
            )
        if not bool(torch.isfinite(obs).all().item()):
            return AnchoredBranchResult(
                success=False,
                aborted_reason="non_finite_obs",
                buffer=buffer,
                noise_metrics=noise_metrics,
                steps_executed=steps_executed,
                object_zs=[],
            )

        clean_action_t = build_clean_action(obs)
        if not bool(torch.isfinite(clean_action_t).all().item()):
            return AnchoredBranchResult(
                success=False,
                aborted_reason="non_finite_clean_action",
                buffer=buffer,
                noise_metrics=noise_metrics,
                steps_executed=steps_executed,
                object_zs=[],
            )

        is_perturb_step = branch_step < config.perturb_steps
        should_save_step = branch_step > 0 or config.perturb_steps <= 0
        if should_save_step:
            image_t = env.render_dataset_camera_rgb(active_envs)
            buffer.append(
                image_t[0].detach().cpu().numpy().astype(np.uint8),
                obs[0].detach().cpu().numpy().astype(np.float32),
                clean_action_t[0].detach().cpu().numpy().astype(np.float32),
            )

        if is_perturb_step:
            sigma_noise = sample_group_noise(clean_action_t)
            noise_state = (
                noise_state
                + config.ou_theta * (config.ou_mu - noise_state) * config.ou_dt
                + sigma_noise * sqrt_dt
            )
            executed_action_t = torch.clamp(
                clean_action_t + noise_state,
                -config.executed_action_clip,
                config.executed_action_clip,
            )
            delta_t = executed_action_t - clean_action_t
            delta_l2_t = torch.linalg.vector_norm(delta_t, dim=1)
            delta_linf_t = torch.abs(delta_t).amax(dim=1)
            noise_metrics["l2_sum"] += float(delta_l2_t.sum().item())
            noise_metrics["l2_sq_sum"] += float(torch.square(delta_l2_t).sum().item())
            noise_metrics["linf_sum"] += float(delta_linf_t.sum().item())
            noise_metrics["count"] += int(delta_l2_t.numel())
        else:
            executed_action_t = clean_action_t

        if not bool(torch.isfinite(executed_action_t).all().item()):
            return AnchoredBranchResult(
                success=False,
                aborted_reason="non_finite_executed_action",
                buffer=buffer,
                noise_metrics=noise_metrics,
                steps_executed=steps_executed,
                object_zs=[],
            )

        obs_dict, _, done, _ = env.step(executed_action_t)
        obs = obs_dict["obs"]
        steps_executed += 1
        object_zs.append(float(env.object_pose[0, 2].item()))

        if not bool(torch.isfinite(obs).all().item()):
            return AnchoredBranchResult(
                success=False,
                aborted_reason="non_finite_next_obs",
                buffer=buffer,
                noise_metrics=noise_metrics,
                steps_executed=steps_executed,
                object_zs=object_zs,
            )

        if bool(done[0].item()) and branch_step + 1 < total_steps:
            if config.collection_mode == "verified_full":
                break
            return AnchoredBranchResult(
                success=False,
                aborted_reason="env_reset",
                buffer=buffer,
                noise_metrics=noise_metrics,
                steps_executed=steps_executed,
                object_zs=object_zs,
            )

    return AnchoredBranchResult(
        success=True,
        aborted_reason=None,
        buffer=buffer,
        noise_metrics=noise_metrics,
        steps_executed=steps_executed,
        object_zs=object_zs,
    )
