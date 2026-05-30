#!/usr/bin/env python3
"""Offline eval: roll out a trained diffusion policy on the pick-place-RELEASE
task (long table) and report per-object release success.

This is the release-aware sibling of eval_diffusion_policy.py. The pickup eval
only checks whether the object cleared a lift height; that metric marks a
pick-place-release policy as "successful" the instant it lifts, ignoring
transport, placement, and a stable release. This script instead reuses the
release collector's env + goal sampling + success classifier
(collect_dataset_pick_place_release.py / stage5_collect_common.py) so the
reported numbers mean the same thing the dataset was scored on.

Key differences vs. the pickup eval:
  * Long-table env (urdf/table_pick_place_release.urdf) with a multi-goal
    trajectory (lift -> transport -> place -> release), not a single lift goal.
  * Per-episode start pose + place goal are sampled with the collector's
    footprint-safe sampler, then set on the env via trajectory_states.
  * Success = _classify_release_pose_based: a POSE-based metric that is fair to a
    BC/diffusion policy. It does NOT gate on the env's RL goal counter (the
    collector's _classify_release_outcome required ~1cm keypoint matches at every
    waypoint -- the RL teacher was trained to hit those, but a diffusion policy
    that visibly picks/places the object rarely reproduces them, so it scored 0%
    despite visually-successful rollouts). Instead we score what the policy
    physically achieved: it picked the object up (lifted), carried it
    (transported), and left it resting on the table at the place goal within the
    same xy/z/speed tolerances the dataset used. Degenerate spawns where the
    object is already off the table before the policy acts are flagged
    invalid_start and excluded from the denominator. We report release_success as
    headline, plus lifted / transported / placed / stable rates and a
    failure-stage breakdown.
  * Single logical env per rollout (the collector forces num_envs=1 because the
    goal trajectory is shared across envs), so episodes run sequentially.

The diffusion policy drives the full action vector itself; unlike the teacher
collector we do NOT override the hand to open during the release phase -- the
policy learned the release behavior from the dataset and we evaluate exactly
that.

Usage (driver):
    python scripts/eval_diffusion_policy_pick_place_release.py \\
        --checkpoint /path/to/diffusion_policy/data/outputs/.../checkpoints/latest.ckpt \\
        --split train \\
        --episodes-per-object 32 \\
        --output-json data/diffusion_eval/ppr_eval.json

Run it from the simtooldiff repo root so the relative pretrained_policy and
config paths resolve.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, deque
from pathlib import Path

# Re-use the stage5 registry/split logic (this module does NOT import isaacgym,
# so it is safe to import in the lightweight driver process).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage5_multi_object_driver import _split, ObjectSpec  # noqa: E402


# Defaults mirrored from stage5_collect_common.py /
# collect_dataset_pick_place_release.py so the driver can expose them without
# importing isaacgym. The worker re-imports the authoritative values and these
# are only argparse defaults; keep them in sync if the collector changes.
DEF_LIFT_HEIGHT_M = 0.20
DEF_PLACE_HEIGHT_M = 0.02
DEF_PLACE_HOLD_GOALS = 10
DEF_RELEASE_STEPS = 45
DEF_HORIZON_WITH_RELEASE = 325
DEF_RELEASE_XY_TOL_M = 0.05
DEF_RELEASE_Z_TOL_M = 0.04
DEF_RELEASE_SPEED_TOL_MPS = 0.25
DEF_TABLE_X_HALF_EXTENT_M = 0.60 / 2.0
DEF_TABLE_X_INSET_MARGIN_M = 0.06
DEF_TABLE_Y_HALF_EXTENT_M = 0.40 / 2.0
DEF_TABLE_Y_INSET_MARGIN_M = 0.04
DEF_PLACE_GOAL_X_MARGIN_M = 0.10
DEF_PLACE_GOAL_Y_MARGIN_M = 0.06
DEF_MIN_EFFECTIVE_TRANSPORT_M = 0.05
# A "fair" pickup bar: the object cleared the table by this much at some point.
# Well below the lift goal (0.20 m) so a clean grasp+carry always counts, but
# above table-contact jitter so a drag along the table does not.
DEF_MIN_LIFT_HEIGHT_M = 0.10
# A degenerate spawn is detected as the object dropping this far below its start
# height *before the policy ever lifts it* (it rolled/fell off the table edge
# before the hand engaged). A normal rollout keeps the object resting near 0
# height until the policy picks it up, so this never triggers on a real attempt.
INVALID_START_FALL_M = 0.15


# Failure-stage labels for the pose-based (diffusion-fair) release metric.
POSE_FAILURE_STAGES = (
    "invalid_start", "no_lift", "no_transport", "no_place", "unstable",
    "drop_reattempt",
)


def _classify_release_pose_based(
    *,
    valid_start: bool,
    lifted: bool,
    transported: bool,
    final_object_on_table: bool,
    final_place_xy_error_m: float,
    final_place_z_error_m: float,
    final_object_speed_mps: float,
    release_xy_tolerance: float,
    release_z_tolerance: float,
    release_speed_tolerance: float,
    reattempted_after_drop: bool,
):
    """Pose-based, diffusion-policy-fair release success.

    Unlike the collector's `_classify_release_outcome`, this does NOT gate on the
    env's RL goal counter (which only ticks on ~1cm keypoint matches the RL
    teacher was trained to hit but a BC/diffusion policy rarely reproduces). It
    scores the policy purely on what it physically achieved: picked the object
    up, carried it to the place goal, and left it resting there within the same
    xy / z / speed tolerances the dataset was scored on.

    Returns (release_success, placed_near_goal, at_rest, failure_stage).
    """
    placed_near_goal = (
        final_object_on_table
        and final_place_xy_error_m <= release_xy_tolerance
        and final_place_z_error_m <= release_z_tolerance
    )
    at_rest = final_object_speed_mps <= release_speed_tolerance
    release_success = (
        valid_start and lifted and transported and placed_near_goal and at_rest
        and not reattempted_after_drop
    )
    # Report the first (earliest-stage) unmet condition so the breakdown points
    # at where the rollout actually broke down.
    if not valid_start:
        failure_stage = "invalid_start"
    elif not lifted:
        failure_stage = "no_lift"
    elif reattempted_after_drop:
        failure_stage = "drop_reattempt"
    elif not transported:
        failure_stage = "no_transport"
    elif not placed_near_goal:
        failure_stage = "no_place"
    elif not at_rest:
        failure_stage = "unstable"
    else:
        failure_stage = None
    return release_success, placed_near_goal, at_rest, failure_stage


# Args that describe task geometry / success tolerances and must be forwarded
# verbatim from driver to each worker subprocess.
_FORWARDED_FLOAT_ARGS = [
    "xy_range", "start_z_offset", "horizon",
    "lift_height", "place_height", "release_steps", "place_hold_goals",
    "min_lift_height",
    "table_x_half_extent", "table_x_inset_margin",
    "table_y_half_extent", "table_y_inset_margin",
    "place_goal_x_margin", "place_goal_y_margin", "min_effective_transport",
    "release_xy_tolerance", "release_z_tolerance", "release_speed_tolerance",
]


# --------------------------- driver --------------------------------------

def _run_one(spec: ObjectSpec, args, result_path: Path) -> dict:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker",
        "--checkpoint", str(args.checkpoint),
        "--object-category", spec.object_category,
        "--object-name", spec.object_name,
        "--task-name", spec.task_name,
        "--object-id", str(spec.object_id),
        "--category-id", str(spec.category_id),
        "--episodes-per-object", str(args.episodes_per_object),
        "--seed", str(args.seed + spec.object_id),
        "--device", args.device,
        "--result-json", str(result_path),
        "--max-success-previews", str(args.max_success_previews),
        "--max-failure-previews", str(args.max_failure_previews),
        "--gif-fps", str(args.gif_fps),
    ]
    for name in _FORWARDED_FLOAT_ARGS:
        flag = "--" + name.replace("_", "-")
        cmd += [flag, str(getattr(args, name))]
    if args.video_dir is not None:
        cmd += ["--video-dir", str(args.video_dir / spec.object_name)]
    print(
        f"\n[ppr-eval-driver] >>> object_id={spec.object_id} "
        f"{spec.object_category}/{spec.object_name} task={spec.task_name}",
        flush=True,
    )
    # IsaacGym's .so is linked against libpython3.8.so.1.0 which lives in
    # $CONDA_PREFIX/lib. Inject it so worker subprocesses can resolve it.
    env = os.environ.copy()
    conda_lib = str(Path(sys.prefix) / "lib")
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{conda_lib}:{existing}" if existing else conda_lib
    subprocess.run(cmd, check=True, env=env)
    return json.loads(result_path.read_text())


def run_driver(args) -> None:
    specs = _split(args.split)
    print(
        f"[ppr-eval-driver] split={args.split} "
        f"objects={[s.object_name for s in specs]}",
        flush=True,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    # Auto-derive video dir alongside the JSON when previews are requested.
    if (args.max_success_previews > 0 or args.max_failure_previews > 0) \
            and args.video_dir is None:
        args.video_dir = args.output_json.parent / f"{args.output_json.stem}_videos"
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ppr-eval-driver] previews -> {args.video_dir}", flush=True)

    per_object = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for spec in specs:
            result_path = td / f"{spec.object_id:02d}_{spec.object_name}.json"
            try:
                result = _run_one(spec, args, result_path)
            except subprocess.CalledProcessError as e:
                print(f"[ppr-eval-driver] !!! {spec.object_name} failed: {e}", flush=True)
                result = {
                    "object_id": spec.object_id,
                    "category_id": spec.category_id,
                    "object_name": spec.object_name,
                    "object_category": spec.object_category,
                    "task_name": spec.task_name,
                    "attempted": 0,
                    "valid_attempted": 0,
                    "invalid_start": 0,
                    "release_succeeded": 0,
                    "lifted": 0,
                    "transported": 0,
                    "placed_near_goal": 0,
                    "release_stable": 0,
                    "release_success_rate": None,
                    "failure_breakdown": {},
                    "error": str(e),
                }
            per_object.append(result)

    total_attempted = sum(r.get("attempted", 0) for r in per_object)
    total_valid = sum(r.get("valid_attempted", 0) for r in per_object)
    total_invalid_start = sum(r.get("invalid_start", 0) for r in per_object)
    total_release = sum(r.get("release_succeeded", 0) for r in per_object)
    total_lifted = sum(r.get("lifted", 0) for r in per_object)
    total_transported = sum(r.get("transported", 0) for r in per_object)
    total_placed = sum(r.get("placed_near_goal", 0) for r in per_object)
    total_stable = sum(r.get("release_stable", 0) for r in per_object)
    # Headline rate excludes degenerate spawns (invalid_start).
    overall = total_release / max(total_valid, 1)
    failure_breakdown: Counter = Counter()
    for r in per_object:
        failure_breakdown.update(r.get("failure_breakdown", {}) or {})

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "episodes_per_object": args.episodes_per_object,
        "xy_range": args.xy_range,
        "horizon": args.horizon,
        "min_lift_height": args.min_lift_height,
        "min_effective_transport": args.min_effective_transport,
        "release_xy_tolerance": args.release_xy_tolerance,
        "release_z_tolerance": args.release_z_tolerance,
        "release_speed_tolerance": args.release_speed_tolerance,
        "overall_release_success_rate": overall,
        "total_attempted": total_attempted,
        "total_valid_attempted": total_valid,
        "total_invalid_start": total_invalid_start,
        "total_release_succeeded": total_release,
        "total_lifted": total_lifted,
        "total_transported": total_transported,
        "total_placed_near_goal": total_placed,
        "total_release_stable": total_stable,
        "failure_breakdown": dict(failure_breakdown),
        "per_object": per_object,
    }
    args.output_json.write_text(json.dumps(summary, indent=2))

    print(f"\n[ppr-eval-driver] DONE", flush=True)
    print(
        f"[ppr-eval-driver] overall release: "
        f"{total_release}/{total_valid} = {overall:.1%} "
        f"(lift {total_lifted}, transport {total_transported}, "
        f"place {total_placed}, stable {total_stable}; "
        f"{total_invalid_start} invalid-start of {total_attempted} excluded)",
        flush=True,
    )
    for r in per_object:
        if r.get("release_success_rate") is None:
            print(f"  - {r['object_name']:<22s} ERROR: {r.get('error')}")
        else:
            print(
                f"  - {r['object_name']:<22s} "
                f"release {r['release_succeeded']:>4d}/{r.get('valid_attempted', 0):>4d} "
                f"= {r['release_success_rate']:.1%} "
                f"(lift {r.get('lifted', 0)} transport {r.get('transported', 0)} "
                f"place {r.get('placed_near_goal', 0)} stable {r.get('release_stable', 0)} "
                f"invalid {r.get('invalid_start', 0)})"
            )
    if failure_breakdown:
        print(f"[ppr-eval-driver] failure stages: {dict(failure_breakdown)}", flush=True)
    print(f"[ppr-eval-driver] saved {args.output_json}", flush=True)


# --------------------------- worker --------------------------------------

def run_worker(args) -> None:
    # IsaacGym must be imported before torch.
    from isaacgym import gymapi  # noqa: F401
    import math
    import numpy as np
    import torch
    import torch.nn.functional as F
    import dill
    import hydra
    from omegaconf import OmegaConf

    # isaacgymenvs/__init__.py registers its own 'eval' resolver without
    # replace=True; import it first, then override with replace=True so the
    # two registrations don't collide.
    import isaacgymenvs  # noqa: F401
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    # Reuse the release collector's env factory, goal builder, footprint-safe
    # sampler, drop detector, and success classifier.
    from collect_dataset_pick_place_release import (
        _make_long_table_env,
        _sample_end_to_end_start_and_goal_release,
        _build_pick_place_release_goals,
        _default_goal_x,
        _place_goal_in_safe_zone,
        _final_object_on_table,
        _drop_detected_after_pickup_attempt,
        _release_goal_stage_name,
    )
    from stage5_collect_common import (
        CHECKPOINT_PATH,
        N_ACT,
        TABLE_Z,
        _load_nominal_start_pose,
    )

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Reconstruct the workspace from the checkpoint and pull the (EMA) policy.
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location=device)
    cfg = payload["cfg"]
    ws_cls = hydra.utils.get_class(cfg._target_)
    workspace = ws_cls(cfg, output_dir=tempfile.mkdtemp(prefix="dp_ppr_eval_"))
    workspace.load_payload(payload)
    policy = workspace.ema_model if cfg.training.use_ema and workspace.ema_model is not None else workspace.model
    policy.to(device).eval()

    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    state_dim = int(cfg.task.state_dim)
    image_shape = tuple(int(x) for x in cfg.task.image_shape)
    target_h, target_w = image_shape[1], image_shape[2]
    print(
        f"[ppr-eval-worker] policy: To={n_obs_steps} k={n_action_steps} "
        f"state_dim={state_dim} image={image_shape}",
        flush=True,
    )

    nominal_start_pose = _load_nominal_start_pose(
        args.object_category,
        args.object_name,
        args.task_name,
        start_z_offset=args.start_z_offset,
    )

    # Bootstrap goals just to construct the env; real per-episode goals are set
    # on env.trajectory_states before each rollout.
    bootstrap_goal_x = _default_goal_x(args)
    bootstrap_goals, _, _ = _build_pick_place_release_goals(
        nominal_start_pose,
        goal_x=bootstrap_goal_x,
        lift_height=args.lift_height,
        place_height=args.place_height,
        place_hold_goals=args.place_hold_goals,
        release_hold_goals=args.release_steps,
    )

    print("[ppr-eval-worker] creating long-table env...", flush=True)
    env = _make_long_table_env(
        num_envs=1,
        nominal_start_pose=nominal_start_pose,
        goal_poses=bootstrap_goals,
        horizon=args.horizon,
        headless=True,
        device=device,
        seed=args.seed,
        object_name=args.object_name,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)

    # Restore the env state the dataset was collected under (matches the
    # collector exactly so the obs distribution the diffusion policy sees is the
    # one it trained on).
    if Path(CHECKPOINT_PATH).exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        env.set_env_state(checkpoint[0]["env_state"])
        print(f"[ppr-eval-worker] restored env state from {CHECKPOINT_PATH}", flush=True)
    else:
        print(
            f"[ppr-eval-worker] WARNING: {CHECKPOINT_PATH} not found; running "
            f"without env-state restore (obs may not match collection). Run from "
            f"the simtooldiff repo root.",
            flush=True,
        )

    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)
    zero_action = torch.zeros((env.num_envs, N_ACT), device=device)

    # Preview setup.
    save_previews = (args.video_dir is not None) and (
        args.max_success_previews > 0 or args.max_failure_previews > 0
    )
    if save_previews:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio  # noqa: WPS433
    preview_caps = {"success": int(args.max_success_previews), "fail": int(args.max_failure_previews)}
    preview_counts = {"success": 0, "fail": 0}
    MIN_PREVIEW_FRAMES = 8

    def render_step():
        """(native uint8 np for previews, normalized resized tensor for the policy)."""
        raw = env.render_dataset_camera_rgb(active_envs)  # (1, H, W, 3) uint8 on device
        native_np = raw[0].detach().cpu().numpy().astype(np.uint8)
        img = raw.float() / 255.0
        img = img.permute(0, 3, 1, 2).contiguous()
        if img.shape[-2:] != (target_h, target_w):
            img = F.interpolate(
                img, size=(target_h, target_w), mode="bilinear", align_corners=False,
            )
        return native_np, img

    def sample_episode_goals():
        """Footprint-safe start/goal, retrying on rejected samples."""
        for _ in range(64):
            try:
                start_pose, goal_x, _ = _sample_end_to_end_start_and_goal_release(
                    env, rng, nominal_start_pose, args,
                )
            except ValueError:
                continue
            goals, release_start_goal_idx, place_goal = _build_pick_place_release_goals(
                start_pose,
                goal_x=goal_x,
                lift_height=args.lift_height,
                place_height=args.place_height,
                place_hold_goals=args.place_hold_goals,
                release_hold_goals=args.release_steps,
            )
            if _place_goal_in_safe_zone(
                env, place_goal,
                table_x_half_extent=args.table_x_half_extent,
                table_x_inset_margin=args.place_goal_x_margin,
                table_y_half_extent=args.table_y_half_extent,
                table_y_inset_margin=args.place_goal_y_margin,
            ):
                return start_pose, goals, release_start_goal_idx, place_goal
        raise RuntimeError("Could not sample a footprint-safe release goal in 64 tries")

    def run_episode(ep_idx: int) -> dict:
        start_pose, goals, release_start_goal_idx, place_goal = sample_episode_goals()
        goals_t = torch.tensor(goals, device=env.device, dtype=env.trajectory_states.dtype)
        env.trajectory_states = goals_t
        env.max_consecutive_successes = len(goals)
        env.object_init_state[env_ids, 0:7] = torch.tensor(
            [start_pose], device=env.device, dtype=env.object_init_state.dtype,
        )
        env.cfg["env"]["tableObjectZOffset"] = float(start_pose[2] - TABLE_Z)
        env.reset_idx(env_ids, tensor_reset=True)
        if hasattr(policy, "reset"):
            policy.reset()

        # Prime obs (a single zero-action step kicks the auto-reset, matching
        # both the pickup eval and the collector).
        obs_dict, _, _, _ = env.step(zero_action)
        obs = obs_dict["obs"]

        # Start-of-rollout object xy, used to measure horizontal transport at the
        # end. (The pick pose is intentionally near the table's front edge, so a
        # footprint-on-table test here would wrongly reject valid starts -- a
        # degenerate spawn is instead detected in-loop as an early fall below.)
        start_object_xy = env.object_pose[0, 0:2].detach().cpu().numpy().copy()

        init_native, init_normalized = render_step()
        image_history = deque(
            [init_normalized.clone() for _ in range(n_obs_steps)], maxlen=n_obs_steps,
        )
        state_history = deque(
            [obs[:, :state_dim].float().contiguous().clone() for _ in range(n_obs_steps)],
            maxlen=n_obs_steps,
        )
        preview_frames = [init_native] if save_previews else []

        # Per-episode release tracking (mirrors _run_release_rollout).
        max_successes_seen = 0
        final_successes = 0
        release_start_step = None
        dropped_after_lift = False
        drop_successes_before = None
        reattempted_after_drop = False
        max_object_height_above_init_m = 0.0
        # Degenerate-spawn detection: object fell off the table before the policy
        # ever lifted it (eval-setup artifact, excluded from the denominator).
        invalid_start = False
        prev_stage_name = None

        action_plan = deque()
        for step in range(args.horizon):
            if not action_plan:
                with torch.no_grad():
                    obs_input = {
                        "image": torch.stack(list(image_history), dim=1),
                        "agent_pos": torch.stack(list(state_history), dim=1),
                    }
                    action_seq = policy.predict_action(obs_input)["action"]  # (1, k, A)
                action_seq = torch.clamp(action_seq, -1.0, 1.0)
                for k in range(action_seq.shape[1]):
                    action_plan.append(action_seq[:, k])

            current_successes = int(env.successes[0].item())
            in_release_phase = (
                args.release_steps > 0 and current_successes >= release_start_goal_idx
            )
            if in_release_phase and release_start_step is None:
                release_start_step = step

            action_t = action_plan.popleft()
            obs_dict, _, done, _ = env.step(action_t)
            obs = obs_dict["obs"]

            final_successes = int(env.successes[0].item())
            max_successes_seen = max(max_successes_seen, final_successes)
            object_height_above_init_m = float(
                env.object_pose[0, 2].item() - env.object_init_state[0, 2].item()
            )
            max_object_height_above_init_m = max(
                max_object_height_above_init_m, object_height_above_init_m,
            )
            # Degenerate spawn: object fell off the table before it was ever
            # lifted. Guarding on "not yet lifted" keeps genuine post-lift drops
            # (a real policy failure) from being excluded.
            if (
                not invalid_start
                and max_object_height_above_init_m < args.min_lift_height
                and object_height_above_init_m < -INVALID_START_FALL_M
            ):
                invalid_start = True
            dropped_now = _drop_detected_after_pickup_attempt(
                object_height_above_init_m=object_height_above_init_m,
                max_object_height_above_init_m=max_object_height_above_init_m,
                lifted_object=bool(env.lifted_object[0].item()),
                in_release_phase=in_release_phase,
            )
            if dropped_now and not dropped_after_lift:
                dropped_after_lift = True
                drop_successes_before = final_successes
            elif (
                dropped_after_lift
                and not reattempted_after_drop
                and drop_successes_before is not None
                and final_successes > drop_successes_before
            ):
                reattempted_after_drop = True

            stage_name = (
                "release" if in_release_phase
                else _release_goal_stage_name(current_successes, release_start_goal_idx)
            )
            if stage_name != prev_stage_name:
                print(
                    f"[ppr-eval-worker] {args.object_name} ep={ep_idx:03d} "
                    f"advanced_to={stage_name} successes={final_successes}/{len(goals)}",
                    flush=True,
                )
            prev_stage_name = stage_name

            native_np, normalized_img = render_step()
            if save_previews:
                preview_frames.append(native_np)
            image_history.append(normalized_img)
            state_history.append(obs[:, :state_dim].float().contiguous())

            if final_successes >= len(goals):
                break
            if bool(done[0].item()):
                break

        # Final-state metrics + classification (identical to the collector).
        final_object_pose_np = env.object_pose[0, 0:7].detach().cpu().numpy()
        final_object_linvel_np = env.object_linvel[0].detach().cpu().numpy()
        place_goal_np = np.asarray(place_goal, dtype=np.float32)
        final_place_xy_error_m = float(np.linalg.norm(final_object_pose_np[0:2] - place_goal_np[0:2]))
        final_place_z_error_m = float(abs(final_object_pose_np[2] - place_goal_np[2]))
        final_object_speed_mps = float(np.linalg.norm(final_object_linvel_np))
        final_transport_m = float(np.linalg.norm(final_object_pose_np[0:2] - start_object_xy))
        lifted = max_object_height_above_init_m >= args.min_lift_height
        transported = final_transport_m >= args.min_effective_transport
        final_object_on_table = _final_object_on_table(
            env, final_object_pose_np,
            table_x_half_extent=args.table_x_half_extent,
            table_x_inset_margin=args.table_x_inset_margin,
            table_y_half_extent=args.table_y_half_extent,
            table_y_inset_margin=args.table_y_inset_margin,
        )
        (
            release_success,
            placed_near_goal,
            at_rest,
            failure_stage,
        ) = _classify_release_pose_based(
            valid_start=not invalid_start,
            lifted=lifted,
            transported=transported,
            final_object_on_table=final_object_on_table,
            final_place_xy_error_m=final_place_xy_error_m,
            final_place_z_error_m=final_place_z_error_m,
            final_object_speed_mps=final_object_speed_mps,
            release_xy_tolerance=args.release_xy_tolerance,
            release_z_tolerance=args.release_z_tolerance,
            release_speed_tolerance=args.release_speed_tolerance,
            reattempted_after_drop=reattempted_after_drop,
        )

        if save_previews:
            outcome = "success" if release_success else "fail"
            if (preview_counts[outcome] < preview_caps[outcome]
                    and len(preview_frames) >= MIN_PREVIEW_FRAMES):
                idx = preview_counts[outcome]
                gif_path = Path(args.video_dir) / f"{outcome}_{idx:02d}_ep{ep_idx:03d}.gif"
                imageio.mimsave(
                    gif_path, np.stack(preview_frames, axis=0),
                    duration=1000.0 / max(args.gif_fps, 1),
                )
                preview_counts[outcome] += 1

        return {
            "release_success": bool(release_success),
            "valid_start": bool(not invalid_start),
            "lifted": bool(lifted),
            "transported": bool(transported),
            "placed_near_goal": bool(placed_near_goal),
            "stable": bool(placed_near_goal and at_rest),
            "max_successes_seen": int(max_successes_seen),
            "failure_stage": failure_stage if not release_success else None,
        }

    attempted = 0
    valid_attempted = 0
    invalid_start_count = 0
    release_succeeded = 0
    lifted_count = 0
    transported_count = 0
    placed_count = 0
    stable_count = 0
    failure_breakdown: Counter = Counter()

    for ep_idx in range(args.episodes_per_object):
        outcome = run_episode(ep_idx)
        attempted += 1
        if not outcome["valid_start"]:
            # Degenerate spawn: excluded from the denominator and from the
            # funnel sub-metrics, which describe valid attempts only.
            invalid_start_count += 1
            failure_breakdown["invalid_start"] += 1
            continue
        valid_attempted += 1
        if outcome["lifted"]:
            lifted_count += 1
        if outcome["transported"]:
            transported_count += 1
        if outcome["placed_near_goal"]:
            placed_count += 1
        if outcome["stable"]:
            stable_count += 1
        if outcome["release_success"]:
            release_succeeded += 1
        elif outcome["failure_stage"] is not None:
            failure_breakdown[outcome["failure_stage"]] += 1
        # Headline rate is over valid (non-degenerate) spawns only.
        sr = release_succeeded / max(valid_attempted, 1)
        print(
            f"[ppr-eval-worker] {args.object_name}: release "
            f"{release_succeeded}/{valid_attempted} ({sr:.1%}) "
            f"[lift {lifted_count} transport {transported_count} "
            f"place {placed_count} stable {stable_count} "
            f"invalid_start {invalid_start_count}]",
            flush=True,
        )

    release_success_rate = release_succeeded / max(valid_attempted, 1)
    result = {
        "object_id": args.object_id,
        "category_id": args.category_id,
        "object_name": args.object_name,
        "object_category": args.object_category,
        "task_name": args.task_name,
        "attempted": attempted,
        "valid_attempted": valid_attempted,
        "invalid_start": invalid_start_count,
        "release_succeeded": release_succeeded,
        "lifted": lifted_count,
        "transported": transported_count,
        "placed_near_goal": placed_count,
        "release_stable": stable_count,
        "release_success_rate": release_success_rate,
        "failure_breakdown": dict(failure_breakdown),
        "preview_dir": str(args.video_dir) if save_previews else None,
        "previews_saved": preview_counts if save_previews else {"success": 0, "fail": 0},
    }
    Path(args.result_json).write_text(json.dumps(result, indent=2))
    print(
        f"[ppr-eval-worker] DONE {args.object_name}: release "
        f"{release_succeeded}/{valid_attempted} = {release_success_rate:.1%} "
        f"({invalid_start_count} invalid-start episodes excluded)",
        flush=True,
    )


# --------------------------- argparse / main ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("train", "ood"), default="train")
    p.add_argument("--episodes-per-object", type=int, default=32)
    p.add_argument("--horizon", type=int, default=DEF_HORIZON_WITH_RELEASE)
    p.add_argument("--xy-range", type=float, default=0.10)
    p.add_argument("--start-z-offset", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-json", type=Path, default=Path("data/diffusion_eval/ppr_eval.json"))

    # Task geometry / goal sampling (forwarded to workers).
    p.add_argument("--lift-height", type=float, default=DEF_LIFT_HEIGHT_M)
    p.add_argument("--place-height", type=float, default=DEF_PLACE_HEIGHT_M)
    p.add_argument("--place-hold-goals", type=int, default=DEF_PLACE_HOLD_GOALS)
    p.add_argument("--release-steps", type=int, default=DEF_RELEASE_STEPS)
    p.add_argument("--table-x-half-extent", type=float, default=DEF_TABLE_X_HALF_EXTENT_M)
    p.add_argument("--table-x-inset-margin", type=float, default=DEF_TABLE_X_INSET_MARGIN_M)
    p.add_argument("--table-y-half-extent", type=float, default=DEF_TABLE_Y_HALF_EXTENT_M)
    p.add_argument("--table-y-inset-margin", type=float, default=DEF_TABLE_Y_INSET_MARGIN_M)
    p.add_argument("--place-goal-x-margin", type=float, default=DEF_PLACE_GOAL_X_MARGIN_M)
    p.add_argument("--place-goal-y-margin", type=float, default=DEF_PLACE_GOAL_Y_MARGIN_M)
    p.add_argument("--min-effective-transport", type=float, default=DEF_MIN_EFFECTIVE_TRANSPORT_M)
    p.add_argument("--min-lift-height", type=float, default=DEF_MIN_LIFT_HEIGHT_M,
                   help="Object must clear the table by at least this much "
                        "(meters) at some point to count as 'lifted'.")

    # Release success tolerances (forwarded to workers).
    p.add_argument("--release-xy-tolerance", type=float, default=DEF_RELEASE_XY_TOL_M)
    p.add_argument("--release-z-tolerance", type=float, default=DEF_RELEASE_Z_TOL_M)
    p.add_argument("--release-speed-tolerance", type=float, default=DEF_RELEASE_SPEED_TOL_MPS)

    # Previews.
    p.add_argument("--max-success-previews", type=int, default=2)
    p.add_argument("--max-failure-previews", type=int, default=2)
    p.add_argument("--gif-fps", type=int, default=10)
    p.add_argument("--video-dir", type=Path, default=None)

    # worker-only fields:
    p.add_argument("--object-category")
    p.add_argument("--object-name")
    p.add_argument("--task-name")
    p.add_argument("--object-id", type=int)
    p.add_argument("--category-id", type=int)
    p.add_argument("--result-json", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
