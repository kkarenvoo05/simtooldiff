# Blender-in-the-Loop Eval Bridge — Progress & Runbook

## What this branch adds

The `blender-eval` branch adds `blender_eval/`, a Python package that implements
closed-loop photorealistic evaluation of diffusion policies. Physics stays in IsaacGym;
only the image source changes (from IsaacGym's rasterizer to Blender Cycles).

### Package contents

```
blender_eval/
  pose_conversion.py       — quaternion xyzw↔wxyz, 4x4 transforms, URDF origin helper
  camera_params.py         — IsaacGym HFOV -> Blender focal length + look-at rotation
  asset_manifest.py        — URDF parsing: mesh paths, collapsed-link offsets, visual origins
  success_criteria.py      — pickup_success() shared across all eval scripts
  state_extraction.py      — reads rigid_body_states, composes collapsed-link poses
  renderer_interface.py    — Renderer protocol, StubRenderer, IsaacGymRenderer
  blender_renderer.py      — BlenderRenderer: persistent subprocess with FIFO-based IPC
  eval_blender.py          — main eval script (mirrors eval_diffusion_policy.py)
  blender_render_script.py — bpy script that runs inside Blender (UNVERIFIED)
  tests/                   — 77 passing tests (5 GPU tests skip on incompatible hardware)
```

### Key design decisions

- **Renderer protocol**: all renderers return `(B, H, W, 3) uint8` batches. The eval loop
  has no renderer-type branching.
- **Quaternion convention**: `RenderState` stores IsaacGym xyzw throughout.
  `serialize_render_state()` is the single conversion point to Blender wxyz, called by
  `BlenderRenderer` before sending JSON to the subprocess.
- **IPC**: Blender's stdout is polluted with version banners and render logs. Protocol
  messages (READY, image paths, ERROR:) go through a dedicated named pipe (FIFO). Commands
  go via stdin. Startup has a 120s timeout with child-death detection.
- **Collapsed links**: `collapse_fixed_joints=True` merges 5 elastomer links + the palm
  into parent bodies. `asset_manifest.py` records the fixed-joint chain offsets;
  `state_extraction.py` composes `parent_world_pose @ chain_offset @ visual_origin` for each.
- **Image pipeline**: renders at native resolution (512x384 or 512x360), then
  `F.interpolate(bilinear, align_corners=False)` to 192x256. Center crop (168x224) happens
  inside the policy.
- **Render engine**: Cycles (path-tracer) by default. `--engine eevee` for smoke tests.
- **Success criterion**: isolated in `success_criteria.py`. Same defaults as
  `stage5_collect_dataset._episode_pickup_success()`.

## Test status

```
77 passed, 5 skipped (GPU tests -- need compatible CUDA hardware)
```

Tests cover: quaternion math (25), camera params (14), URDF manifest + collapse logic (16),
renderer shapes (5), serialization boundary (7), success criteria (10).

## What works now

- `--renderer stub`: plumbing test, runs end-to-end (gray images, no mesh extraction)
- `--renderer isaacgym`: uses IsaacGym's camera, should match `eval_diffusion_policy.py`
  exactly for A/B parity
- `--renderer blender`: fully wired -- launches Blender subprocess, sends poses via FIFO,
  receives rendered image paths. Requires Blender binary + `.blend` scene template.

## What's NOT done yet

1. **A/B parity test** (`--renderer isaacgym` vs `eval_diffusion_policy.py`) -- needs a
   GPU-compatible machine (not Blackwell + Python 3.8).
2. **Blender installation** -- no `blender` binary on the current box.
3. **`.blend` scene template** -- placeholder lighting exists; needs HDRI, PBR materials,
   matched camera framing. This is GUI work.
4. **Open-loop render sanity check** -- replay ~10 frames in Blender, eyeball the GIF to
   confirm axis conventions and mesh placement. Must pass before trusting closed-loop results.
5. **Photorealistic sim-to-sim comparison** -- clean vs NSCA policies under Blender rendering.

---

## Step-by-step runbook for a compatible machine

Requirements: NVIDIA GPU with sm_70-sm_90 (V100, A100, H100), Python 3.8, Isaac Gym
Preview 4, a trained diffusion policy checkpoint.

### 1. Clone and checkout the branch

```bash
git clone https://github.com/kkarenvoo05/simtooldiff.git
cd simtooldiff
git checkout blender-eval
```

### 2. Set up the Python 3.8 environment

```bash
uv venv --python 3.8
echo 'export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH' >> .venv/bin/activate
source .venv/bin/activate

uv pip install -e .
```

### 3. Install Isaac Gym

If you already have Isaac Gym Preview 4 downloaded somewhere:

```bash
uv pip install -e /path/to/isaacgym/python
```

If not, download from NVIDIA, extract, then install.

### 4. Install rl_games (vendored in the repo)

```bash
cd rl_games && uv pip install -e . && cd -
```

### 5. Install pytest

```bash
uv pip install pytest
```

### 6. Run the unit tests (no GPU needed for most)

```bash
python -m pytest blender_eval/tests/ -v
```

Expected: **77 passed, 5 skipped** if CUDA isn't working yet, or **82 passed** if it is.

### 7. Run the GPU-specific tests

```bash
python -m pytest blender_eval/tests/test_state_extraction.py -v
```

These create a real IsaacGym env with `claw_hammer`, step it once, and verify that
rigid body state extraction, collapsed-link offsets, and object poses are all correct.
All 5 must pass before proceeding.

### 8. Run the A/B parity test

Set the checkpoint path:

```bash
CKPT=/path/to/checkpoint.ckpt
```

Run the reference eval:

```bash
python scripts/eval_diffusion_policy.py \
  --checkpoint $CKPT --split train --episodes-per-object 32 \
  --num-envs 8 --horizon 250 --seed 0 \
  --output-json data/ref_eval/eval.json
```

Run the new eval with `--renderer isaacgym`:

```bash
python blender_eval/eval_blender.py \
  --checkpoint $CKPT --renderer isaacgym --split train \
  --episodes-per-object 32 --num-envs 8 --horizon 250 --seed 0 \
  --output-json data/blender_eval/isaacgym_eval.json
```

Compare results:

```bash
python -c "
import json
ref = json.load(open('data/ref_eval/eval.json'))
new = json.load(open('data/blender_eval/isaacgym_eval.json'))
print(f'Reference: {ref[\"overall_success_rate\"]:.1%}')
print(f'New eval:  {new[\"overall_success_rate\"]:.1%}')
for r, n in zip(ref['per_object'], new['per_object']):
    print(f'  {r[\"object_name\"]:<22s} ref={r[\"success_rate\"]:.1%}  new={n[\"success_rate\"]:.1%}')
"
```

Per-object success rates should match exactly (same seed, env count, horizon, object order).

### 9. Run the stub plumbing test (optional, quick sanity check)

```bash
python blender_eval/eval_blender.py \
  --checkpoint $CKPT --renderer stub --split train \
  --episodes-per-object 4 --num-envs 2 --seed 0 \
  --output-json data/blender_eval/stub_eval.json
```

Should complete without errors. Success rate is meaningless (gray images).

### If the A/B parity test fails

The most likely cause is a subtle divergence in the eval loop. To debug:

1. Add temporary logging to both scripts to dump the first 10 image tensors
   (post-normalize, pre-policy) to disk.
2. Compare element-wise -- the divergence will show up as a specific frame
   where tensors differ.
3. Common causes: history initialization difference, render timing (before vs
   after step), image permutation order, interpolation settings.

### What comes after parity passes

1. **Install Blender 4.x** (`apt install blender` or download portable build)
2. **Author the `.blend` scene template** -- HDRI lighting, PBR materials on
   robot/tools, camera matching the training viewpoint. This is GUI work.
3. **Open-loop render sanity check** -- replay a known rollout in Blender, render
   ~10 frames, make a GIF, eyeball that hand+tool placement and camera framing
   match IsaacGym's viewer. This is where axis convention issues surface.
4. **Run `--renderer blender` closed-loop eval** with the scene template.
5. **Photorealistic sim-to-sim comparison** -- compare clean vs NSCA policy
   success rates under Blender-rendered observations.
