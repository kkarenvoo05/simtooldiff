"""BlenderRenderer: manages a persistent Blender subprocess for rendering.

IPC design: Blender's stdout is polluted with version banners and render
progress, so we do NOT use stdout for protocol messages. Instead:
  - Commands (JSON pose data, QUIT): sent via stdin pipe.
  - Responses (READY, image paths, ERROR:): sent via a dedicated named pipe
    (FIFO) created by the parent and passed to the script as --response-fifo.
  - Blender's own stdout/stderr: suppressed (sent to /dev/null or stderr).

This keeps the protocol channel free of Blender's log output.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
  from blender_eval.state_extraction import RenderState

from blender_eval.state_extraction import serialize_render_state

SCRIPT_PATH = Path(__file__).resolve().parent / "blender_render_script.py"

_STARTUP_TIMEOUT_S = 120  # Max seconds to wait for Blender to signal READY


class BlenderRenderer:
  """Renderer that delegates to a persistent Blender subprocess.

  Implements the batch Renderer protocol: render() accepts a list of
  RenderState (one per env) and returns (num_envs, H, W, 3) uint8.
  Internally renders one frame at a time (Blender is single-threaded).
  """

  def __init__(
    self,
    num_envs: int,
    width: int = 512,
    height: int = 384,
    manifest_path: str = "",
    camera_path: str = "",
    tool_mesh_path: str = "",
    engine: str = "cycles",
    samples: int = 64,
    blend_file: Optional[str] = None,
    blender_executable: str = "blender",
  ):
    self.num_envs = num_envs
    self.width = width
    self.height = height
    self.tool_mesh_path = tool_mesh_path
    self._closed = False

    # Create a named pipe (FIFO) for protocol responses from Blender.
    self._fifo_dir = tempfile.mkdtemp(prefix="blender_ipc_")
    self._fifo_path = os.path.join(self._fifo_dir, "response.fifo")
    os.mkfifo(self._fifo_path)

    cmd = [
      blender_executable, "--background", "--python", str(SCRIPT_PATH), "--",
      "--manifest", manifest_path,
      "--camera", camera_path,
      "--engine", engine,
      "--width", str(width),
      "--height", str(height),
      "--samples", str(samples),
      "--response-fifo", self._fifo_path,
    ]
    if blend_file:
      cmd += ["--blend-file", blend_file]

    # Blender's stdout goes to /dev/null; stderr to parent for diagnostics.
    self.proc = subprocess.Popen(
      cmd,
      stdin=subprocess.PIPE,
      stdout=subprocess.DEVNULL,
      stderr=sys.stderr,
      text=True,
      bufsize=1,
    )

    # Open the FIFO for reading. This blocks until the child opens the write
    # end. We do this in a thread so we can detect if the child dies first.
    self._fifo_fd = None
    self._fifo_fd = self._open_fifo_with_timeout(_STARTUP_TIMEOUT_S)

    # Wait for READY signal via the FIFO.
    ready_line = self._fifo_fd.readline().strip()
    if ready_line != "READY":
      self.close()
      raise RuntimeError(
        f"Blender subprocess did not signal READY via FIFO, got: {ready_line!r}. "
        f"Check stderr for Blender errors."
      )

  def _open_fifo_with_timeout(self, timeout_s: float):
    """Open the FIFO read end, with a timeout if the child dies before opening it.

    The blocking open("r") will hang forever if the child crashes before it
    opens the write end. We run the open in a thread and poll the child.
    """
    result = [None]
    error = [None]

    def _open():
      try:
        result[0] = open(self._fifo_path, "r")
      except Exception as e:
        error[0] = e

    t = threading.Thread(target=_open, daemon=True)
    t.start()

    deadline = time.monotonic() + timeout_s
    while t.is_alive() and time.monotonic() < deadline:
      # Check if child died
      if self.proc.poll() is not None:
        # Child exited before opening the FIFO — open() will never return.
        # We can't interrupt the blocked open(), but since the thread is
        # daemon it will die with the process. Clean up and raise.
        self._cleanup_fifo()
        raise RuntimeError(
          f"Blender subprocess exited with code {self.proc.returncode} "
          f"before signaling READY. Check stderr for errors."
        )
      t.join(timeout=0.5)

    if t.is_alive():
      self._cleanup_fifo()
      self.proc.kill()
      self.proc.wait()
      raise TimeoutError(
        f"Timed out after {timeout_s}s waiting for Blender to open FIFO. "
        f"Check stderr for errors."
      )

    if error[0] is not None:
      raise error[0]

    return result[0]

  def render(self, states: Optional[List[RenderState]] = None) -> np.ndarray:
    """Render one frame per env. Returns (num_envs, H, W, 3) uint8."""
    if states is None:
      return np.full(
        (self.num_envs, self.height, self.width, 3), 128, dtype=np.uint8
      )

    from PIL import Image

    frames = []
    for state in states:
      payload = serialize_render_state(state, self.tool_mesh_path)
      line = json.dumps(payload)

      self.proc.stdin.write(line + "\n")
      self.proc.stdin.flush()

      # Read response from FIFO. May be an image path or an ERROR: line.
      response = self._fifo_fd.readline().strip()
      if not response:
        raise RuntimeError(
          "Blender subprocess closed the FIFO unexpectedly (empty response). "
          "Check stderr for errors."
        )
      if response.startswith("ERROR:"):
        raise RuntimeError(f"Blender render error: {response}")
      if not os.path.exists(response):
        raise RuntimeError(
          f"Blender returned non-existent path: {response!r}. "
          f"Check stderr for errors."
        )

      img = np.array(Image.open(response).convert("RGB"), dtype=np.uint8)
      frames.append(img)
      os.unlink(response)

    return np.stack(frames, axis=0)

  def _cleanup_fifo(self) -> None:
    """Remove the FIFO file and directory."""
    if hasattr(self, "_fifo_fd") and self._fifo_fd:
      try:
        self._fifo_fd.close()
      except OSError:
        pass
      self._fifo_fd = None
    if hasattr(self, "_fifo_path") and os.path.exists(self._fifo_path):
      os.unlink(self._fifo_path)
    if hasattr(self, "_fifo_dir") and os.path.exists(self._fifo_dir):
      try:
        os.rmdir(self._fifo_dir)
      except OSError:
        pass

  def close(self) -> None:
    """Terminate the Blender subprocess and clean up the FIFO."""
    if self._closed:
      return
    self._closed = True

    if hasattr(self, "proc") and self.proc and self.proc.poll() is None:
      try:
        self.proc.stdin.write("QUIT\n")
        self.proc.stdin.flush()
        self.proc.wait(timeout=10)
      except (BrokenPipeError, subprocess.TimeoutExpired, OSError):
        self.proc.kill()
        self.proc.wait()

    self._cleanup_fifo()

  def __del__(self):
    self.close()
