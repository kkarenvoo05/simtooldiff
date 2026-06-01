#!/usr/bin/env python3
"""Compose two rollout GIFs into one labeled side-by-side GIF."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageSequence


def _read_frames(path: Path):
  image = Image.open(path)
  frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(image)]
  if not frames:
    raise ValueError(f"No frames found in {path}")
  return frames


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
  if image.height == height:
    return image
  width = max(1, round(image.width * height / image.height))
  resampling = getattr(Image, "Resampling", Image).LANCZOS
  return image.resize((width, height), resampling)


def _label(draw: ImageDraw.ImageDraw, xy, text: str):
  x, y = xy
  draw.text((x + 1, y + 1), text, fill=(0, 0, 0))
  draw.text((x, y), text, fill=(245, 245, 245))


def compose(left_path: Path, right_path: Path, out_path: Path,
            left_label: str, right_label: str, fps: int) -> None:
  left_frames = _read_frames(left_path)
  right_frames = _read_frames(right_path)

  target_h = max(left_frames[0].height, right_frames[0].height)
  gutter = 8
  label_h = 28
  n_frames = max(len(left_frames), len(right_frames))
  out_frames = []

  for i in range(n_frames):
    left = _resize_to_height(left_frames[min(i, len(left_frames) - 1)], target_h)
    right = _resize_to_height(right_frames[min(i, len(right_frames) - 1)], target_h)
    canvas = Image.new(
      "RGB",
      (left.width + gutter + right.width, label_h + target_h),
      (24, 24, 24),
    )
    canvas.paste(left, (0, label_h))
    canvas.paste(right, (left.width + gutter, label_h))
    draw = ImageDraw.Draw(canvas)
    _label(draw, (8, 7), left_label)
    _label(draw, (left.width + gutter + 8, 7), right_label)
    out_frames.append(canvas)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_frames[0].save(
    out_path,
    save_all=True,
    append_images=out_frames[1:],
    duration=max(1, round(1000 / max(fps, 1))),
    loop=0,
  )
  print(f"[side-by-side] saved {out_path}")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--left", type=Path, required=True)
  parser.add_argument("--right", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--left-label", default="IsaacGym")
  parser.add_argument("--right-label", default="Blender Cycles")
  parser.add_argument("--fps", type=int, default=10)
  args = parser.parse_args()
  compose(args.left, args.right, args.out, args.left_label, args.right_label, args.fps)


if __name__ == "__main__":
  main()
