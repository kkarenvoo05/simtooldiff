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
  success_criteria.py      — regression-test mirror of the Stage 5 pickup metric
  state_extraction.py      — reads rigid_body_states, composes collapsed-link poses
  renderer_interface.py    — Renderer protocol, StubRenderer, IsaacGymRenderer
  blender_renderer.py      — BlenderRenderer: persistent subprocess with FIFO-based IPC
  eval_blender.py          — main eval script (mirrors eval_diffusion_policy.py)
  blender_render_script.py — bpy script that runs inside Blender
  tests/                   — eval/render/camera/state/success regression tests
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
- **Render engine**: Cycles (path-tracer) by default. `--engine eevee` is only for
  renderer debugging, not policy evaluation.
- **Photoreal template**: `--renderer blender` defaults to
  `assets/blender/templates/simtool_lab.blend`, so the driver and worker use the
  same static scene unless `--blend-file` is explicitly overridden.
- **Success criterion**: pickup scoring is implemented in
  `blender_eval/success_criteria.py`, matching the Stage 5 max-height pickup
  criterion while keeping the Blender evaluation package independent of
  dataset-collection success helpers.

## Test status

```
92 passed, 5 skipped on machines without the IsaacGym `gymtorch` JIT toolchain
```

Tests cover: quaternion math, camera params, URDF manifest + collapse logic,
renderer shapes, serialization boundary, IsaacGym state extraction, success
criteria, and the Blender eval CLI defaults.

## What works now

- `--renderer stub`: plumbing test, runs end-to-end (gray images, no mesh extraction)
- `--renderer isaacgym`: uses IsaacGym's camera, should match `eval_diffusion_policy.py`
  exactly for A/B parity
- `--renderer blender`: fully wired -- launches Blender subprocess, loads the
  default photoreal `.blend`, renders policy inputs with Blender Cycles, and saves
  photoreal GIF previews from those same rendered frames.

## Verified (open-loop, no IsaacGym needed)

`blender_eval/open_loop_smoke_test.py` was run on **Blender 4.2.9 LTS (headless, Cycles CPU)**.
It builds the mesh manifest from the real URDF, computes a zero-pose forward-kinematics robot
configuration, and drives the real `BlenderRenderer` IPC. Result: **all 36 robot STL meshes +
the tool OBJ import, the camera configures, and Cycles renders a 256x192 frame** returned over
the FIFO (1012 unique colors). The rendered robot is upright and correctly framed, which
validates the coordinate-frame conversion (Z-up, xyzw->wxyz) and the camera look-at math.
Re-run it on any machine with Blender:

```bash
python blender_eval/open_loop_smoke_test.py --blender /path/to/blender
```

## Verified (IsaacGym + dataset camera)

On the local RTX 4070 laptop, the real SimToolReal env now initializes through
`scripts.stage5_collect_dataset._make_env` for `hammer/claw_hammer/swing_down` with
`num_envs=2`, `headless=True`, and the source-of-truth dataset camera constants from
`stage5_collect_dataset.py`. A zero-action step followed by
`render_dataset_camera_rgb()` returned a CUDA `uint8` tensor with shape
`(2, 384, 512, 3)` and nonzero image statistics (`min=1 max=255 mean=124.94`).

Command shape:

```bash
PATH=$PWD/.venv/bin:$PATH \
LD_LIBRARY_PATH=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH \
.venv/bin/python - <<'PY'
from scripts.stage5_collect_dataset import _load_nominal_start_pose, _make_env, LIFT_HEIGHT_M
# build claw_hammer env, step zero action, call env.render_dataset_camera_rgb(...)
PY
```

Notes from setup: direct `.venv/bin/python` invocations need the Python 3.8 libdir in
`LD_LIBRARY_PATH`, and `PATH=$PWD/.venv/bin:$PATH` is needed so PyTorch can find the
`ninja` binary when loading IsaacGym's `gymtorch` extension.

## Verified (tiny closed loop, untrained policy)

The local laptop now runs the closed loop end-to-end at tiny scale with IsaacGym
physics, a loadable untrained diffusion-policy checkpoint, and Blender Cycles rendering.
The checkpoint used for smoke verification was created at `/tmp/untrained_simtool_tiny.ckpt`
from `train_diffusion_unet_simtool_workspace.yaml` with `task.state_dim=29`,
`task.image_shape=[3,192,256]`, a synthetic normalizer, tiny U-Net dimensions
(`policy.down_dims=[64,128,256]`, `policy.diffusion_step_embed_dim=64`), and
`policy.num_inference_steps=4`. It is not trained and must not be used for success-rate
claims; it only proves the execution path.

Diffusion-policy import setup kept the IsaacGym torch install intact (`torch==2.4.1+cu121`)
by installing only thin packages with `--no-deps`: `diffusers==0.11.1`,
`robomimic==0.2.0`, `einops`, `egl_probe`, `huggingface_hub==0.13.4`, plus missing
workspace import helpers (`dill`, `pandas`, `pytz`). `PYTHONPATH` must include both
the diffusion-policy fork and this repo root:

```bash
PYTHONPATH=/home/takaraet/Projects/cs224r/diffusion-policy:$PWD
```

Stub renderer verification:

```bash
PATH=$PWD/.venv/bin:$PATH \
LD_LIBRARY_PATH=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH \
PYTHONPATH=/home/takaraet/Projects/cs224r/diffusion-policy:$PWD \
.venv/bin/python blender_eval/eval_blender.py --worker \
  --checkpoint /tmp/untrained_simtool_tiny.ckpt \
  --renderer stub \
  --object-category hammer --object-name claw_hammer --task-name swing_down \
  --num-envs 2 --episodes-per-object 1 --horizon 60 \
  --result-json /tmp/r.json
```

Result: exited `0`, wrote `/tmp/r.json`, `attempted=1`, `renderer="stub"`.

Blender bridge verification:

```bash
PATH=$PWD/.venv/bin:$PATH \
LD_LIBRARY_PATH=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH \
PYTHONPATH=/home/takaraet/Projects/cs224r/diffusion-policy:$PWD \
.venv/bin/python blender_eval/eval_blender.py --worker \
  --checkpoint /tmp/untrained_simtool_tiny.ckpt \
  --renderer blender --engine cycles --samples 8 \
  --blender /tmp/blender-4.2.9-linux-x64/blender \
  --object-category hammer --object-name claw_hammer --task-name swing_down \
  --num-envs 2 --episodes-per-object 1 --horizon 60 \
  --result-json /tmp/r_blender.json
```

Result: exited `0`, imported the 36 robot meshes and claw-hammer texture in Blender,
advanced the policy/action/render loop to episode termination, and wrote
`/tmp/r_blender.json` with `attempted=1`, `renderer="blender"`. Success rate was `0.0`,
which is expected for the untrained checkpoint and is not a completion signal.

### Bugs fixed during verification

1. **`--engine` plumbing** in `eval_blender.py`: the flag was read via `getattr` in the worker
   but never defined in `parse_args()` nor forwarded by the driver, so `--engine eevee` / custom
   `--samples` were impossible. Now defined and threaded driver -> worker -> `BlenderRenderer`.
2. **Blender 4.x importer API** in `blender_render_script.py`: used `bpy.ops.import_mesh.stl`
   (legacy add-on, removed in 4.2) and `BLENDER_EEVEE_NEXT` (4.2+ only). Now uses a robust
   `_import_mesh_file()` that prefers `bpy.ops.wm.stl_import` with a legacy fallback, captures
   the imported object by set-difference (not fragile `selected_objects[-1]`), and picks the
   EEVEE engine name valid for the running version. Robot meshes are STL, so this was on the
   critical path.
3. **`--blender` executable plumbing** in `eval_blender.py`: `BlenderRenderer` already supported
   a custom executable path, but the CLI did not expose or forward it. The worker and driver now
   accept `--blender`, allowing the verified portable Blender 4.2.9 binary under `/tmp` to be
   used directly.

## Photoreal Blender scene status

The default Blender evaluation path now uses a committed master template at
`assets/blender/templates/simtool_lab.blend`. The template owns the static scene:

- the `table_narrow_nail.urdf` table as a 0.475 x 0.4 x 0.3 m light-oak box
  centered at the IsaacGym table pose `(0, 0, 0.38)`;
- the small nail/block at the URDF offset `(-0.16, 0.06, 0.175)`, now with a
  light marble material;
- a neutral lab scene instead of a required checkerboard: matte concrete floor,
  concrete back wall with beams, a rear workbench, and a visible overhead strip light;
- deterministic Cycles lighting using the softbox-grid setup, constant seed,
  adaptive sampling, no motion blur, no depth of field, and conservative AgX
  exposure;
- URDF material colors for the robot meshes (KUKA orange/gray, Sharpa hand colors),
  with PBR-ish roughness/specular/coat settings and clear-coated orange paint;
- procedural light-oak, concrete, marble, dark fixture, and laminate materials with
  subtle roughness/normal variation and beveled hard edges.

`blender_render_script.py` still contains a scripted fallback scene, but policy
evaluation should use the template unless you intentionally pass a different
`--blend-file`.

Verified commands:

```bash
PATH=$PWD/.venv/bin:$PATH \
LD_LIBRARY_PATH=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH \
PYTHONPATH=$PWD \
.venv/bin/python blender_eval/open_loop_smoke_test.py \
  --blender /tmp/blender-4.2.9-linux-x64/blender \
  --engine cycles --samples 16 --width 512 --height 384 \
  --out /tmp/lab_lighting_smoke.png
```

Result: PASS, saved `/tmp/lab_lighting_smoke.png` and copied a preview to
`/home/takaraet/Projects/cs224r/blender_scene_previews/lab_cycles_lighting_preview.png`.
The 512x384 smoke frame had 9,618 unique colors, mean pixel value 157.1, and visible
lab wall/workbench, softened shadows, non-checker floor, light-oak table, and URDF
robot colors.

## Verified (trained checkpoint tiny photoreal smoke)

The locally copied epoch-50 checkpoint has been loaded through the real policy
workspace and rolled out for a tiny single-object smoke test with IsaacGym physics
and Blender Cycles frames feeding the policy:

```bash
PATH=$PWD/.venv/bin:$PATH \
LD_LIBRARY_PATH=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH \
PYTHONPATH=/home/takaraet/Projects/cs224r/diffusion-policy:$PWD:$PWD/scripts \
.venv/bin/python blender_eval/eval_blender.py --worker \
  --checkpoint /home/takaraet/Projects/cs224r/checkpoints/epoch=0050-val_loss=0.0465.ckpt \
  --renderer blender --engine cycles --samples 16 --cycles-device cpu \
  --blender /tmp/blender-4.2.9-linux-x64/blender \
  --object-category hammer --object-name claw_hammer --task-name swing_down \
  --object-id 0 --category-id 0 \
  --num-envs 1 --episodes-per-object 1 --horizon 8 \
  --result-json /tmp/photoreal_smoke.json \
  --video-dir /tmp/photoreal_smoke_videos \
  --max-success-previews 1 --max-failure-previews 1 --gif-fps 10 \
  --device cuda:0
```

Result: exited `0`, wrote `/tmp/photoreal_smoke.json`, and saved
`/tmp/photoreal_smoke_videos/fail_00_attempt001.gif`. The JSON recorded
`renderer="blender"`, `engine="cycles"`, the default `simtool_lab.blend`,
`render_width=512`, and `render_height=384`. The short horizon and 16 CPU
samples were only a smoke-test setting; use the normal 250 horizon and 96 samples
for visual evaluation.

## Epoch 50 checkpoint video comparison

The checkpoint Christine evaluated is the best A/B checkpoint because it already
has reference IsaacGym results:

```text
/move/u/chrzhang/diffusion_policy/data/outputs/2026.05.06/22.17.26_train_diffusion_unet_hybrid_simtool_image_state29/checkpoints/epoch=0050-val_loss=0.0465.ckpt
```

Reference result already present locally:

```text
/home/takaraet/Projects/cs224r/diffusion_eval/epoch0050_train.json
overall: 244/288 = 84.7%
```

The checkpoint is now present locally at:

```text
/home/takaraet/Projects/cs224r/checkpoints/epoch=0050-val_loss=0.0465.ckpt
```

If this file is missing after cleanup or on another machine, copy it into the
prepared local directory with:

```bash
scp scdt.stanford.edu:'/move/u/chrzhang/diffusion_policy/data/outputs/2026.05.06/22.17.26_train_diffusion_unet_hybrid_simtool_image_state29/checkpoints/epoch=0050-val_loss=0.0465.ckpt' \
  /home/takaraet/Projects/cs224r/checkpoints/epoch=0050-val_loss=0.0465.ckpt
```

or use resumable copy:

```bash
rsync -avP scdt.stanford.edu:'/move/u/chrzhang/diffusion_policy/data/outputs/2026.05.06/22.17.26_train_diffusion_unet_hybrid_simtool_image_state29/checkpoints/epoch=0050-val_loss=0.0465.ckpt' \
  /home/takaraet/Projects/cs224r/checkpoints/epoch=0050-val_loss=0.0465.ckpt
```

Generate IsaacGym-vs-Blender videos for the default `claw_hammer` train
object:

```bash
cd /home/takaraet/Projects/cs224r/simtooldiff
./blender_eval/run_epoch0050_side_by_side.sh \
  /home/takaraet/Projects/cs224r/checkpoints/epoch=0050-val_loss=0.0465.ckpt
```

The script writes:

```text
/home/takaraet/Projects/cs224r/blender_eval_videos/epoch0050_side_by_side/
  claw_hammer_<timestamp>/
    isaacgym_claw_hammer.json
    blender_cycles_claw_hammer.json
    side_by_side_local_claw_hammer.gif
    side_by_side_christine_claw_hammer.gif
```

For another object, override the object metadata before running, for example:

```bash
OBJ_CATEGORY=spatula OBJ_NAME=flat_spatula TASK_NAME=flip_over \
OBJECT_ID=10 CATEGORY_ID=5 REF_ROOT=/home/takaraet/Projects/cs224r/diffusion_eval/epoch0050_ood_videos \
./blender_eval/run_epoch0050_side_by_side.sh \
  /home/takaraet/Projects/cs224r/checkpoints/epoch=0050-val_loss=0.0465.ckpt
```

## Cluster photoreal evaluation

Use these files to move the photoreal evaluation onto an L40 cluster node:

```text
blender_eval/CLUSTER_EVAL_CONTEXT.md
blender_eval/run_photoreal_eval_l40.sbatch
```

The SLURM script launches Blender itself with
`blender --background --python blender_eval/blender_render_script.py`; Blender
does not need to be open or actively running before the job starts. The L40 path
should use `--cycles-device gpu`; the local 8 GB RTX 4070 path OOMed on GPU
Cycles and had to fall back to CPU Cycles. `MODE=aggregate` only combines JSON
files and does not require Blender.

For broad photoreal eval, use `MODE=array_worker` with a SLURM array and then
`MODE=aggregate` to produce the same split-level JSON shape as the normal eval
driver. Train split uses `--array=0-8`; OOD split uses `--array=0-2`.

## What's NOT done yet

1. **Full split A/B parity test** (`--renderer isaacgym` vs
   `eval_diffusion_policy.py`) -- still the critical gate before any closed-loop
   success number is trusted. Run with the same checkpoint, seed, env count,
   horizon, and split.
2. **Full 250-step photoreal episode review** -- the trained-checkpoint smoke test
   proves the execution path, but a normal-horizon GIF should still be inspected
   before broad eval.
3. **Photorealistic sim-to-sim comparison** -- compare clean vs NSCA policies
   under Blender rendering once the checkpoint set is available locally.

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

Expected on this laptop with `.venv/bin` on `PATH`: **88 passed**.

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
2. **Optional custom scene assets** -- if the procedural lab scene is not enough for the
   paper, add a `.blend` template or external HDRI/PBR textures while preserving the
   training camera geometry.
3. **Open-loop render sanity check** -- replay a known rollout in Blender, render
   ~10 frames, make a GIF, eyeball that hand+tool placement and camera framing
   match IsaacGym's viewer. This is where axis convention issues surface.
4. **Run `--renderer blender` closed-loop eval** with the scene template.
5. **Photorealistic sim-to-sim comparison** -- compare clean vs NSCA policy
   success rates under Blender-rendered observations.
