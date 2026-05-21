"""Renderer protocol and implementations (stub, IsaacGym).

All renderers return batched images: (num_envs, H, W, 3) uint8.
The eval loop calls render() once per step and receives the full batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Protocol

import numpy as np

if TYPE_CHECKING:
  from blender_eval.state_extraction import RenderState


class Renderer(Protocol):
  def render(self, states: Optional[List[RenderState]]) -> np.ndarray:
    """Render one frame for each environment.

    Args:
      states: Per-env RenderState list, or None for renderers that read
        directly from the sim (e.g. IsaacGymRenderer).

    Returns:
      (num_envs, H, W, 3) uint8 RGB array.
    """
    ...

  def close(self) -> None:
    """Clean up resources."""
    ...


class StubRenderer:
  """Returns solid gray images. For plumbing tests without Blender."""

  def __init__(self, num_envs: int, width: int = 512, height: int = 384):
    self.num_envs = num_envs
    self.width = width
    self.height = height

  def render(self, states: Optional[List[RenderState]] = None) -> np.ndarray:
    return np.full((self.num_envs, self.height, self.width, 3), 128, dtype=np.uint8)

  def close(self) -> None:
    pass


class IsaacGymRenderer:
  """Uses IsaacGym's built-in camera for A/B parity testing."""

  def __init__(self, env, active_envs_tensor):
    self.env = env
    self.active_envs = active_envs_tensor

  def render(self, states: Optional[List[RenderState]] = None) -> np.ndarray:
    """Returns (num_envs, H, W, 3) uint8 from the env's dataset camera."""
    img = self.env.render_dataset_camera_rgb(self.active_envs)
    return img.detach().cpu().numpy().astype(np.uint8)

  def close(self) -> None:
    pass
