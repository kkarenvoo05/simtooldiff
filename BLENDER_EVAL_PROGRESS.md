# Blender-in-the-Loop Eval Bridge — Progress & Runbook

## What this branch adds

The `blender-eval` branch adds `blender_eval/`, a Python package that implements
closed-loop photorealistic evaluation of diffusion policies. Physics stays in IsaacGym;
only the image source changes (from IsaacGym's rasterizer to Blender Cycles).

### Package contents

```
blender_eval/
  pose_conversion.py       — quaternion xyzw↔wxyz, 4×4 transforms, URDF origin helper
  camera_params.py         — IsaacGym HFOV → Blender focal length + look-at rotation
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
- **Image pipeline**: renders at native resolution (512×384 or 512×360), then
  `F.interpolate(bilinear, align_corners=False)` to 192×256. Center crop (168×224) happens
  inside the policy.
- **Render engine**: Cycles (path-tracer) by default. `--engine eevee` for smoke tests.
- **Success criterion**: isolated in `success_criteria.py`. Same defaults as
  `stage5_collect_dataset._episode_pickup_success()`.

## Test status

```
77 passed, 5 skipped (GPU tests — need compatible CUDA hardware)
```

Tests cover: quaternion math (25), camera params (14), URDF manifest + collapse logic (16),
renderer shapes (5), serialization boundary (7), success criteria (10).

## What works now

- `--renderer stub`: plumbing test, runs end-to-end (gray images, no GPU rendering)
- `--renderer isaacgym`: uses IsaacGym's camera, should match `eval_diffusion_policy.py`
  exactly for A/B parity
- `--renderer blender`: fully wired — launches Blender subprocess, sends poses via FIFO,
  receives rendered image paths. Requires Blender binary + `.blend` scene template.

## What's NOT done yet

1. **A/B parity test** (`--renderer isaacgym` vs `eval_diffusion_policy.py`) — needs a
   GPU-compatible machine (not Blackwell + Python 3.8).
2. **Blender installation** — no `blender` binary on the current box.
3. **`.blend` scene template** — placeholder lighting exists; needs HDRI, PBR materials,
   matched camera framing. This is GUI work.
4. **Open-loop render sanity check** — replay ~10 frames in Blender, eyeball the GIF to
   confirm axis conventions and mesh placement. Must pass before trusting closed-loop results.
5. **Photorealistic sim-to-sim comparison** — clean vs NSCA policies under Blender rendering.

## Runbook: how to run on a compatible machine

### Prerequisites

- NVIDIA GPU with sm_70–sm_90 (V100, A100, H100, etc.)
- Python 3.8 venv with IsaacGym Preview 4 installed
- Diffusion policy checkpoint (e.g. from Christine's training runs)
- Blender 4.x (for `--renderer blender` only)

### Setup

```bash
cd /path/to/simtooldiff
git checkout blender-eval

# Activate your Python 3.8 venv with IsaacGym
source .venv/bin/activate

# Install blender_eval as editable (it's just Python files, no build needed)
# The package is importable from the repo root via blender_eval/

# Run unit tests (no GPU needed for most)
python -m pytest blender_eval/tests/ -v

# Run GPU tests (need CUDA)
python -m pytest blender_eval/tests/test_state_extraction.py -v
```

### Step 1: A/B parity test (isaacgym renderer)

```bash
CKPT=/path/to/checkpoint.ckpt

# Reference eval
python scripts/eval_diffusion_policy.py \
  --checkpoint $CKPT --split train --episodes-per-object 32 \
  --num-envs 8 --horizon 250 --seed 0 \
  --output-json data/ref_eval/eval.json

# New eval with isaacgym renderer (should produce identical results)
python blender_eval/eval_blender.py \
  --checkpoint $CKPT --renderer isaacgym --split train \
  --episodes-per-object 32 --num-envs 8 --horizon 250 --seed 0 \
  --output-json data/blender_eval/isaacgym_eval.json

# Compare
python -c "
import json
ref = json.load(open('data/ref_eval/eval.json'))
new = json.load(open('data/blender_eval/isaacgym_eval.json'))
print(f'Reference: {ref[\"overall_success_rate\"]:.1%}')
print(f'New eval:  {new[\"overall_success_rate\"]:.1%}')
assert ref['overall_success_rate'] == new['overall_success_rate'], 'MISMATCH'
print('PARITY OK')
"
```

### Step 2: Stub plumbing test

```bash
python blender_eval/eval_blender.py \
  --checkpoint $CKPT --renderer stub --split train \
  --episodes-per-object 4 --num-envs 2 --seed 0 \
  --output-json data/blender_eval/stub_eval.json
```

Should complete without errors. Success rate is meaningless (gray images).

### Step 3: Blender eval (once Blender + scene template are ready)

```bash
# Install Blender 4.x and ensure 'blender' is on PATH
blender --version

python blender_eval/eval_blender.py \
  --checkpoint $CKPT --renderer blender --split train \
  --episodes-per-object 32 --num-envs 1 --seed 0 \
  --output-json data/blender_eval/blender_eval.json
```

Note: `--num-envs 1` because Blender renders sequentially (one frame at a time).
With Cycles at 64 samples, expect ~1-5s per frame × 250 steps × 32 episodes ≈ hours.
Use `--engine eevee` for faster smoke tests.
