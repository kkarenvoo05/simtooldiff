import numpy as np
from blender_eval.renderer_interface import StubRenderer


class TestStubRenderer:
  def test_shape_512x384(self):
    r = StubRenderer(num_envs=4, width=512, height=384)
    img = r.render(None)
    assert img.shape == (4, 384, 512, 3)
    assert img.dtype == np.uint8

  def test_shape_512x360(self):
    r = StubRenderer(num_envs=2, width=512, height=360)
    img = r.render(None)
    assert img.shape == (2, 360, 512, 3)
    assert img.dtype == np.uint8

  def test_single_env(self):
    r = StubRenderer(num_envs=1)
    img = r.render(None)
    assert img.shape == (1, 384, 512, 3)

  def test_value_is_gray(self):
    r = StubRenderer(num_envs=1, width=64, height=48)
    img = r.render(None)
    assert (img == 128).all()

  def test_close_is_noop(self):
    r = StubRenderer(num_envs=1)
    r.close()
