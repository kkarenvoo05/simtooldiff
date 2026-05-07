#!/usr/bin/env python3
"""Offline eval: roll out a trained diffusion policy in IsaacGym across the
stage5 object set and report per-object pickup success rate.

Subprocess-per-object, mirroring stage5_multi_object_driver.py, because
IsaacGym dislikes recreating a sim with a different object asset in the
same process.

Usage (driver):
    python scripts/eval_diffusion_policy.py \\
        --checkpoint /path/to/diffusion_policy/data/outputs/.../checkpoints/latest.ckpt \\
        --split train \\
        --episodes-per-object 32 \\
        --num-envs 8 \\
        --output-json data/diffusion_eval/eval.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

# Re-use the stage5 registry/split logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage5_multi_object_driver import _split, ObjectSpec  # noqa: E402


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
        "--num-envs", str(args.num_envs),
        "--episodes-per-object", str(args.episodes_per_object),
        "--horizon", str(args.horizon),
        "--xy-range", str(args.xy_range),
        "--seed", str(args.seed + spec.object_id),
        "--device", args.device,
        "--result-json", str(result_path),
        "--max-success-previews", str(args.max_success_previews),
        "--max-failure-previews", str(args.max_failure_previews),
        "--gif-fps", str(args.gif_fps),
    ]
    if args.video_dir is not None:
        cmd += ["--video-dir", str(args.video_dir / spec.object_name)]
    print(
        f"\n[eval-driver] >>> object_id={spec.object_id} "
        f"{spec.object_category}/{spec.object_name} task={spec.task_name}",
        flush=True,
    )
    subprocess.run(cmd, check=True)
    return json.loads(result_path.read_text())


def run_driver(args) -> None:
    specs = _split(args.split)
    print(
        f"[eval-driver] split={args.split} "
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
        print(f"[eval-driver] previews -> {args.video_dir}", flush=True)

    per_object = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for spec in specs:
            result_path = td / f"{spec.object_id:02d}_{spec.object_name}.json"
            try:
                result = _run_one(spec, args, result_path)
            except subprocess.CalledProcessError as e:
                print(f"[eval-driver] !!! {spec.object_name} failed: {e}", flush=True)
                result = {
                    "object_id": spec.object_id,
                    "category_id": spec.category_id,
                    "object_name": spec.object_name,
                    "object_category": spec.object_category,
                    "task_name": spec.task_name,
                    "attempted": 0,
                    "succeeded": 0,
                    "success_rate": None,
                    "error": str(e),
                }
            per_object.append(result)

    total_attempted = sum(r.get("attempted", 0) for r in per_object)
    total_succeeded = sum(r.get("succeeded", 0) for r in per_object)
    overall = total_succeeded / max(total_attempted, 1)
    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "episodes_per_object": args.episodes_per_object,
        "num_envs": args.num_envs,
        "xy_range": args.xy_range,
        "horizon": args.horizon,
        "overall_success_rate": overall,
        "total_attempted": total_attempted,
        "total_succeeded": total_succeeded,
        "per_object": per_object,
    }
    args.output_json.write_text(json.dumps(summary, indent=2))

    print(f"\n[eval-driver] DONE", flush=True)
    print(
        f"[eval-driver] overall: {total_succeeded}/{total_attempted} = {overall:.1%}",
        flush=True,
    )
    for r in per_object:
        if r.get("success_rate") is None:
            print(f"  - {r['object_name']:<22s} ERROR: {r.get('error')}")
        else:
            print(
                f"  - {r['object_name']:<22s} "
                f"{r['succeeded']:>4d}/{r['attempted']:>4d} = {r['success_rate']:.1%}"
            )
    print(f"[eval-driver] saved {args.output_json}", flush=True)


# --------------------------- worker --------------------------------------

def run_worker(args) -> None:
    # IsaacGym must be imported before torch.
    from isaacgym import gymapi  # noqa: F401
    import numpy as np
    import torch
    import torch.nn.functional as F
    import dill
    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    # Lean on stage5's env factory + success criterion + start-pose loader.
    from stage5_collect_dataset import (
        _make_env,
        _load_nominal_start_pose,
        _episode_pickup_success,
        N_ACT,
        LIFT_HEIGHT_M,
    )

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Reconstruct the workspace from the checkpoint and pull the (EMA) policy.
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location=device)
    cfg = payload["cfg"]
    ws_cls = hydra.utils.get_class(cfg._target_)
    workspace = ws_cls(cfg, output_dir=tempfile.mkdtemp(prefix="dp_eval_"))
    workspace.load_payload(payload)
    policy = workspace.ema_model if cfg.training.use_ema and workspace.ema_model is not None else workspace.model
    policy.to(device).eval()

    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    state_dim = int(cfg.task.state_dim)
    image_shape = tuple(int(x) for x in cfg.task.image_shape)
    target_h, target_w = image_shape[1], image_shape[2]
    print(
        f"[eval-worker] policy: To={n_obs_steps} k={n_action_steps} "
        f"state_dim={state_dim} image={image_shape}",
        flush=True,
    )

    # Build env to mirror stage5_collect_dataset.
    nominal_start_pose = _load_nominal_start_pose(
        args.object_category, args.object_name, args.task_name
    )
    nominal_goal_pose = list(nominal_start_pose)
    nominal_goal_pose[2] += LIFT_HEIGHT_M
    nominal_start_z = float(nominal_start_pose[2])
    goal_z = float(nominal_goal_pose[2])

    print("[eval-worker] creating env...", flush=True)
    env = _make_env(
        num_envs=args.num_envs,
        nominal_start_pose=nominal_start_pose,
        nominal_goal_pose=nominal_goal_pose,
        xy_range=args.xy_range,
        horizon=args.horizon,
        headless=True,
        device=device,
        seed=args.seed,
        object_name=args.object_name,
    )
    nominal_init_xy = env.object_init_state[:, 0:2].detach().clone()

    def randomize_object_init_xy(env_id_list):
        if not env_id_list:
            return
        env_ids = torch.tensor(
            env_id_list,
            device=env.object_init_state.device,
            dtype=torch.long,
        )
        deltas = (
            torch.rand(
                (len(env_id_list), 2),
                device=env.object_init_state.device,
                dtype=env.object_init_state.dtype,
            ) * 2.0 - 1.0
        ) * float(args.xy_range)
        env.object_init_state[env_ids, 0:2] = nominal_init_xy[env_ids] + deltas

    randomize_object_init_xy(list(range(env.num_envs)))
    env.gym.refresh_actor_root_state_tensor(env.sim)

    active_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)

    # Prime obs (matches stage5: a single zero-action step kicks the auto-reset).
    zero_action = torch.zeros((env.num_envs, N_ACT), device=device)
    obs_dict, _, _, _ = env.step(zero_action)
    obs = obs_dict["obs"]

    def render_step():
        """Return (native_uint8_np for previews, normalized resized tensor for the policy)."""
        raw = env.render_dataset_camera_rgb(active_envs)  # (B, H, W, 3) uint8 on device
        native_np = raw.detach().cpu().numpy().astype(np.uint8)
        img = raw.float() / 255.0
        img = img.permute(0, 3, 1, 2).contiguous()
        if img.shape[-2:] != (target_h, target_w):
            img = F.interpolate(
                img, size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            )
        return native_np, img

    def current_agent_pos():
        return obs[:, :state_dim].float().contiguous()

    # Preview bookkeeping: per-env list of native-resolution uint8 frames.
    save_previews = (args.video_dir is not None) and (
        args.max_success_previews > 0 or args.max_failure_previews > 0
    )
    if save_previews:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio  # noqa: WPS433
    preview_caps = {
        "success": int(args.max_success_previews),
        "fail": int(args.max_failure_previews),
    }
    preview_counts = {"success": 0, "fail": 0}
    MIN_PREVIEW_FRAMES = 8

    # Seed history buffers by replicating the priming-step obs/image To times.
    init_native, init_normalized = render_step()
    image_history = deque(
        [init_normalized.clone() for _ in range(n_obs_steps)],
        maxlen=n_obs_steps,
    )
    state_history = deque(
        [current_agent_pos().clone() for _ in range(n_obs_steps)],
        maxlen=n_obs_steps,
    )
    per_env_preview_imgs = [
        [init_native[i]] if save_previews else [] for i in range(env.num_envs)
    ]

    per_env_object_zs = [[] for _ in range(env.num_envs)]
    per_env_start_z = [nominal_start_z] * env.num_envs
    just_reset = [False] * env.num_envs

    attempted = 0
    succeeded = 0
    target = args.episodes_per_object

    while attempted < target:
        with torch.no_grad():
            obs_input = {
                "image": torch.stack(list(image_history), dim=1),     # (B, To, 3, H, W)
                "agent_pos": torch.stack(list(state_history), dim=1), # (B, To, state_dim)
            }
            action_seq = policy.predict_action(obs_input)["action"]   # (B, n_action_steps, A)
        # The dataset's actions are already normalized into [-1, 1]; the
        # diffusion policy unnormalizes via the stored LinearNormalizer, so
        # output is back in that same [-1, 1] joint-target space. Clamp for safety.
        action_seq = torch.clamp(action_seq, -1.0, 1.0)

        for k in range(action_seq.shape[1]):
            if attempted >= target:
                break
            action_t = action_seq[:, k]
            obs_dict, _, done, _ = env.step(action_t)
            obs = obs_dict["obs"]

            object_pose_np = env.object_pose[:, 0:7].detach().cpu().numpy()
            for env_i in range(env.num_envs):
                if just_reset[env_i]:
                    per_env_start_z[env_i] = float(object_pose_np[env_i, 2])
                    just_reset[env_i] = False
                    continue
                per_env_object_zs[env_i].append(float(object_pose_np[env_i, 2]))

            done_np = done.detach().cpu().numpy().astype(bool)
            done_indices = [i for i, d in enumerate(done_np) if d]
            # Render once for everyone; reused for both history and preview buffers.
            native_np, normalized_img = render_step()

            for env_i in done_indices:
                if attempted >= target:
                    break
                attempted += 1
                ok = _episode_pickup_success(
                    object_zs=per_env_object_zs[env_i],
                    object_start_z=per_env_start_z[env_i],
                    goal_z=goal_z,
                )
                if ok:
                    succeeded += 1
                per_env_object_zs[env_i].clear()
                just_reset[env_i] = True

                # Save preview if we still have a slot for this outcome.
                if save_previews:
                    outcome = "success" if ok else "fail"
                    frames = per_env_preview_imgs[env_i]
                    if (preview_counts[outcome] < preview_caps[outcome]
                            and len(frames) >= MIN_PREVIEW_FRAMES):
                        idx = preview_counts[outcome]
                        gif_path = Path(args.video_dir) / (
                            f"{outcome}_{idx:02d}_attempt{attempted:03d}.gif"
                        )
                        imageio.mimsave(
                            gif_path,
                            np.stack(frames, axis=0),
                            duration=1000.0 / max(args.gif_fps, 1),
                        )
                        preview_counts[outcome] += 1
                # Reset this env's frame buffer regardless of save outcome.
                if save_previews:
                    per_env_preview_imgs[env_i] = []

            randomize_object_init_xy(done_indices)

            # Append the post-step (post-reset for done envs) frame to all envs.
            if save_previews:
                for env_i in range(env.num_envs):
                    per_env_preview_imgs[env_i].append(native_np[env_i])
            image_history.append(normalized_img)
            state_history.append(current_agent_pos())

        sr = succeeded / max(attempted, 1)
        print(
            f"[eval-worker] {args.object_name}: {succeeded}/{attempted} ({sr:.1%})",
            flush=True,
        )

    success_rate = succeeded / max(attempted, 1)
    result = {
        "object_id": args.object_id,
        "category_id": args.category_id,
        "object_name": args.object_name,
        "object_category": args.object_category,
        "task_name": args.task_name,
        "attempted": attempted,
        "succeeded": succeeded,
        "success_rate": success_rate,
        "preview_dir": str(args.video_dir) if save_previews else None,
        "previews_saved": preview_counts if save_previews else {"success": 0, "fail": 0},
    }
    Path(args.result_json).write_text(json.dumps(result, indent=2))
    print(
        f"[eval-worker] DONE {args.object_name}: "
        f"{succeeded}/{attempted} = {success_rate:.1%}",
        flush=True,
    )


# --------------------------- argparse / main ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("train", "ood"), default="train")
    p.add_argument("--episodes-per-object", type=int, default=32)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=250)
    p.add_argument("--xy-range", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/diffusion_eval/eval.json"),
    )
    p.add_argument(
        "--max-success-previews", type=int, default=2,
        help="Save up to N successful-episode GIFs per object. 0 disables.",
    )
    p.add_argument(
        "--max-failure-previews", type=int, default=2,
        help="Save up to N failed-episode GIFs per object. 0 disables.",
    )
    p.add_argument("--gif-fps", type=int, default=10)
    p.add_argument(
        "--video-dir", type=Path, default=None,
        help="Where to write GIF previews. Auto-derived from --output-json if unset.",
    )
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
