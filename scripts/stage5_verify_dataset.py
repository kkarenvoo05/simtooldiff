#!/usr/bin/env python3
"""Verify Stage 5 Diffusion Policy-style Zarr dataset."""

import argparse
from pathlib import Path

import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    root = zarr.open(str(args.zarr_path), mode="r")
    img = root["data"]["img"]
    state = root["data"]["state"]
    action = root["data"]["action"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)

    print(f"path: {args.zarr_path}")
    print(f"img: {img.shape}, {img.dtype}, chunks={img.chunks}")
    print(f"state: {state.shape}, {state.dtype}, chunks={state.chunks}")
    print(f"action: {action.shape}, {action.dtype}, chunks={action.chunks}")
    print(f"episode_ends: {episode_ends.shape}, last={episode_ends[-1] if len(episode_ends) else None}")
    print(f"attrs: {dict(root.attrs)}")

    assert img.ndim == 4 and img.shape[-1] == 3
    assert img.dtype == np.dtype("uint8")
    assert state.ndim == 2 and state.shape[0] == img.shape[0]
    assert action.ndim == 2 and action.shape[0] == img.shape[0]
    assert len(episode_ends) > 0
    assert episode_ends[-1] == img.shape[0]
    assert np.all(np.diff(episode_ends) > 0)

    starts = np.concatenate([[0], episode_ends[:-1]])
    lengths = episode_ends - starts
    valid_eps = np.where(lengths >= args.horizon)[0]
    assert len(valid_eps) > 0, f"No episodes are at least horizon={args.horizon}"

    rng = np.random.default_rng(0)
    for i in range(min(args.num_samples, len(valid_eps))):
        ep = int(rng.choice(valid_eps))
        start = int(starts[ep])
        end = int(episode_ends[ep])
        t0 = int(rng.integers(start, end - args.horizon + 1))
        img_win = img[t0 : t0 + args.horizon]
        state_win = state[t0 : t0 + args.horizon]
        action_win = action[t0 : t0 + args.horizon]

        assert img_win.shape[0] == args.horizon
        assert state_win.shape[0] == args.horizon
        assert action_win.shape[0] == args.horizon
        print(
            f"sample {i}: ep={ep}, window=[{t0},{t0 + args.horizon}), "
            f"img_mean={float(np.asarray(img_win).mean()):.2f}, "
            f"state_norm={float(np.linalg.norm(np.asarray(state_win)[0])):.2f}, "
            f"action_norm={float(np.linalg.norm(np.asarray(action_win)[0])):.2f}"
        )

    print("PASS: dataset schema and random temporal windows look loadable.")


if __name__ == "__main__":
    main()
