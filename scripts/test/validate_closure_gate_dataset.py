#!/usr/bin/env python3
"""Post-hoc validation of phase-gated-noise collection (V1 + V2).

Reads a collected Zarr and, from the saved per-step state, reconstructs where
the closure window fires and reports whether it lines up with the grasp moment.

State layout (state_list, N_OBS=140, 29 dofs, 1 keypoint):
    palm z              -> idx 89
    keypoints_rel_palm  -> idx 122:125  (palm->object vector; norm = proximity)

Usage:
    conda run -n dp python scripts/validate_closure_gate_dataset.py \
        --zarr data/pick_place_release_gated.zarr --episodes 6 \
        --mode proximity --proximity-threshold 0.08 --padding 5 \
        --plot-dir data/closure_validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

PALM_Z = 89
KP_REL_PALM = slice(122, 125)


def _bounds(ends, i):
    return (0 if i == 0 else int(ends[i - 1])), int(ends[i])


def _window(raw, padding):
    out = np.zeros_like(raw, dtype=bool)
    since = 10**9
    for t, r in enumerate(raw):
        since = 0 if r else since + 1
        out[t] = since <= padding
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--mode", choices=["proximity", "palm_z"], default="proximity")
    ap.add_argument("--proximity-threshold", type=float, default=0.08)
    ap.add_argument("--palm-z-threshold", type=float, default=0.66)
    ap.add_argument("--padding", type=int, default=5)
    ap.add_argument("--plot-dir", type=Path, default=None)
    args = ap.parse_args()

    z = zarr.open(str(args.zarr), "r")
    state, action = z["data/state"], z["data/action"]
    ends = np.asarray(z["meta/episode_ends"][:])
    attrs = dict(z.attrs)

    print(f"Dataset: {args.zarr}  episodes={len(ends)}  variant={attrs.get('variant')}")
    for k in ("noise_phase_gating", "closure_noise_scale", "closure_proximity_threshold",
              "closure_palm_z_threshold", "closure_window_padding", "closure_groups",
              "finger_noise_multiplier", "wrist_noise_multiplier",
              "closure_steps_total", "closure_gated_steps_total"):
        if k in attrs:
            print(f"  attr {k} = {attrs[k]}")
    print()

    tot_gated = tot_steps = 0
    for i in range(min(args.episodes, len(ends))):
        s, e = _bounds(ends, i)
        st = np.asarray(state[s:e])
        palm_z = st[:, PALM_Z]
        prox = np.linalg.norm(st[:, KP_REL_PALM], axis=1)
        raw = (prox < args.proximity_threshold) if args.mode == "proximity" else (palm_z < args.palm_z_threshold)
        win = _window(raw, args.padding)
        tot_gated += int(win.sum()); tot_steps += len(win)
        first = int(np.argmax(win)) if win.any() else -1
        last = int(len(win) - 1 - np.argmax(win[::-1])) if win.any() else -1
        print(f"ep {i:3d}: len={len(win):3d} gated={int(win.sum()):3d} window=[{first},{last}] "
              f"min_palm_z={palm_z.min():.3f} min_prox={prox.min():.3f}")
        if args.plot_dir is not None:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            args.plot_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax[0].plot(palm_z, label="palm z"); ax[0].plot(prox, label="palm->obj dist")
            ax[0].axhline(args.proximity_threshold, ls=":", c="r")
            for t in np.where(win)[0]:
                ax[0].axvspan(t - .5, t + .5, color="orange", alpha=.15)
                ax[1].axvspan(t - .5, t + .5, color="orange", alpha=.15)
            ax[0].legend(); ax[0].set_title(f"ep {i} - closure window shaded")
            ax[1].plot(np.linalg.norm(np.diff(np.asarray(action[s:e]), axis=0), axis=1),
                       label="|d action| step-to-step")
            ax[1].legend(); ax[1].set_xlabel("step")
            fig.tight_layout(); fig.savefig(args.plot_dir / f"ep_{i:03d}.png"); plt.close(fig)

    frac = tot_gated / max(tot_steps, 1)
    print(f"\nGated fraction (sampled): {frac:.1%} ({tot_gated}/{tot_steps})")
    if frac == 0:
        print("  WARNING: gate never fires - threshold too tight, loosen it.")
    elif frac > 0.5:
        print("  WARNING: gate fires most of the episode - threshold too loose, tighten it.")
    else:
        print("  OK: focused window (expect a handful of steps around grasp).")


if __name__ == "__main__":
    main()
