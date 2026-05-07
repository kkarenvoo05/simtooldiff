"""Export image previews from a Diffusion Policy-style Zarr dataset."""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple

import imageio.v3 as iio
import numpy as np
import tyro
import zarr


def log(message: str) -> None:
    print(f"[visualize_dataset] {message}", file=sys.stderr, flush=True)


@dataclass
class VisualizeDatasetArgs:
    dataset_path: Path
    """Path to the dataset Zarr directory."""

    output_dir: Path = Path("data/dataset_previews")
    """Directory where preview frames/videos will be written."""

    num_episodes: int = 2
    """Number of episodes to export."""

    start_episode: int = 0
    """First episode index to export."""

    frame_stride: int = 1
    """Save every Nth frame."""

    fps: int = 15
    """Playback FPS for GIF/MP4 previews."""

    save_frames: bool = True
    """Save PNG frames for each selected episode."""

    save_gif: bool = True
    """Save an animated GIF for each selected episode."""

    save_mp4: bool = False
    """Save an MP4 for each selected episode. Requires imageio ffmpeg support."""


def _episode_bounds(episode_ends: np.ndarray, episode_idx: int) -> Tuple[int, int]:
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return start, end


def main() -> None:
    args = tyro.cli(VisualizeDatasetArgs)
    root = zarr.open(str(args.dataset_path), mode="r")
    images = root["data/img"]
    episode_ends = np.asarray(root["meta/episode_ends"][:])

    log(f"Loaded {args.dataset_path}")
    log(f"data/img shape={images.shape}, dtype={images.dtype}")
    log(f"episode_ends={episode_ends.tolist()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_episode = min(args.start_episode + args.num_episodes, len(episode_ends))

    for episode_idx in range(args.start_episode, last_episode):
        start, end = _episode_bounds(episode_ends, episode_idx)
        frame_indices = np.arange(start, end, args.frame_stride, dtype=np.int64)
        episode_images = np.asarray(images[frame_indices])
        episode_dir = args.output_dir / f"episode_{episode_idx:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        if args.save_frames:
            for local_idx, image in enumerate(episode_images):
                iio.imwrite(episode_dir / f"frame_{local_idx:04d}.png", image)

        if args.save_gif:
            gif_path = episode_dir / f"episode_{episode_idx:04d}.gif"
            iio.imwrite(gif_path, episode_images, duration=1000 / args.fps, loop=0)
            log(f"Wrote {gif_path}")

        if args.save_mp4:
            mp4_path = episode_dir / f"episode_{episode_idx:04d}.mp4"
            iio.imwrite(mp4_path, episode_images, fps=args.fps)
            log(f"Wrote {mp4_path}")

        log(
            f"Exported episode {episode_idx}: dataset[{start}:{end}], "
            f"{len(episode_images)} frames to {episode_dir}"
        )


if __name__ == "__main__":
    main()
