# Photorealistic Blender Evaluation: Cluster Context

## Objective

Run the diffusion-policy checkpoint evaluation with the same rollout semantics and
outputs as `scripts/eval_diffusion_policy.py`, but replace the policy RGB input
source with Blender Cycles renders from the photorealistic scene template.

The target checkpoint is Christine's epoch-50 clean-data checkpoint:

```text
/move/u/chrzhang/diffusion_policy/data/outputs/2026.05.06/22.17.26_train_diffusion_unet_hybrid_simtool_image_state29/checkpoints/epoch=0050-val_loss=0.0465.ckpt
```

Existing IsaacGym reference result:

```text
/home/takaraet/Projects/cs224r/diffusion_eval/epoch0050_train.json
overall: 244/288 = 84.7%
```

Do not expect the photoreal success rate to match this number. The photoreal
evaluation is intentionally out of distribution relative to the IsaacGym camera
images used for training.

## How The Pipeline Works

`blender_eval/eval_blender.py` mirrors `scripts/eval_diffusion_policy.py`.
The policy, IsaacGym environment, action chunk timing, observation history, reset
logic, success criterion, result JSON layout, and GIF preview behavior are the
same. The only intended behavioral change is the image source:

- `--renderer isaacgym`: calls `env.render_dataset_camera_rgb(...)`, matching the
  original evaluator.
- `--renderer blender`: extracts robot/tool poses from the live IsaacGym env,
  sends them to a persistent Blender subprocess, receives a rendered RGB frame,
  and feeds that frame to the policy.

The Blender renderer is headless. It does not need an already-running Blender GUI
or desktop session. `BlenderRenderer` launches:

```text
blender --background --python blender_eval/blender_render_script.py -- ...
```

and communicates with it through stdin plus a FIFO response pipe.

The default photoreal scene is:

```text
assets/blender/templates/simtool_lab.blend
```

`--renderer blender` now uses that template automatically unless `--blend-file`
is explicitly passed. The template owns static visuals: table, table nail/block,
floor, wall, rear bench, lighting, and static materials. Runtime owns the camera,
robot meshes, tool mesh, and every robot/tool pose. The tool is selected from the
task object name, for example `--object-name claw_hammer` imports the claw hammer
mesh and names it `tool_claw_hammer`.

## Hardware Notes

The local RTX 4070 8 GB was not enough for GPU Cycles while IsaacGym and the
diffusion policy were also resident on CUDA. The failure was:

```text
Blender render error: ERROR: render failed: Error: System is out of GPU memory
```

The CPU-Cycles fallback works but is too slow for broad evaluation. A single
normal-horizon 250-step episode at 512x384 and 96 CPU samples was still running
after many minutes.

An L40 is appropriate for this job. NVIDIA lists the L40 with 48 GB GDDR6 ECC,
which is enough headroom for:

- IsaacGym physics on CUDA;
- the epoch-50 diffusion policy checkpoint;
- one persistent Blender Cycles subprocess;
- 512x384 policy-view renders;
- 96 Cycles samples;
- `num-envs=1`.

Keep `num-envs=1` for Blender rendering. `BlenderRenderer.render()` renders one
frame per env serially, so larger env batches increase render work and memory
without much benefit. Parallelize across SLURM jobs or job arrays instead.

## Environment Requirements

Use Python 3.8. IsaacGym Preview 4 only ships bindings for Python 3.6/3.7/3.8.
The existing MOVE cluster environment appears to be:

```bash
source /nlp/scr/chrzhang/miniconda3/etc/profile.d/conda.sh
conda activate str
```

Required runtime paths:

```bash
export SIMTOOLDIFF_ROOT=/path/to/simtooldiff
export DIFFUSION_POLICY_ROOT=/move/u/chrzhang/diffusion_policy
export PYTHONPATH=$DIFFUSION_POLICY_ROOT:$SIMTOOLDIFF_ROOT:$SIMTOOLDIFF_ROOT/scripts
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Blender 4.x must be available as a binary. It does not need to be running before
the job starts. The sbatch script can use an existing binary via `BLENDER=/path/to/blender`
or download Blender 4.2.9 into scratch if network access is available.

## Recommended Cluster Flow

First submit a one-object smoke test:

```bash
MODE=worker \
EPISODES_PER_OBJECT=1 \
SAMPLES=96 \
CYCLES_DEVICE=gpu \
sbatch blender_eval/run_photoreal_eval_l40.sbatch
```

Expected output:

```text
$OUT_ROOT/worker_claw_hammer_<jobid>/
  blender_cycles_claw_hammer.json
  videos/fail_00_attempt001.gif or videos/success_00_attempt001.gif
```

For a full split, prefer an object-level SLURM array. This runs one object per
L40 job and later aggregates the object JSONs into the same summary shape as
`eval_blender.py`'s driver:

```bash
MODE=array_worker \
SPLIT=train \
EPISODES_PER_OBJECT=32 \
SAMPLES=96 \
CYCLES_DEVICE=gpu \
sbatch --array=0-8 blender_eval/run_photoreal_eval_l40.sbatch
```

After the array finishes, aggregate it. Use the array job id as `ARRAY_RUN_ID`:

```bash
MODE=aggregate \
SPLIT=train \
ARRAY_RUN_ID=<array_job_id> \
EPISODES_PER_OBJECT=32 \
SAMPLES=96 \
CYCLES_DEVICE=gpu \
sbatch blender_eval/run_photoreal_eval_l40.sbatch
```

Expected array output:

```text
$OUT_ROOT/array_train_<array_job_id>/
  object_results/<object_name>.json
  photoreal_train.json
  photoreal_train_videos/<object_name>/*.gif
```

The sequential driver mode is available, but it is likely too slow for broad
photoreal evaluation:

```bash
MODE=driver \
SPLIT=train \
EPISODES_PER_OBJECT=32 \
SAMPLES=96 \
CYCLES_DEVICE=gpu \
sbatch blender_eval/run_photoreal_eval_l40.sbatch
```

Expected output:

```text
$OUT_ROOT/driver_train_<jobid>/
  photoreal_train.json
  photoreal_train_videos/<object_name>/*.gif
```

For OOD:

```bash
MODE=array_worker \
SPLIT=ood \
EPISODES_PER_OBJECT=32 \
SAMPLES=96 \
CYCLES_DEVICE=gpu \
sbatch --array=0-2 blender_eval/run_photoreal_eval_l40.sbatch
```

## What To Check

In the SLURM log, confirm:

- `nvidia-smi` reports an L40 or another GPU with similar VRAM;
- Blender is launched with `--background`;
- Blender receives `--engine cycles --cycles-device gpu`;
- the result JSON has `"renderer": "blender"`;
- the result JSON has `"engine": "cycles"`;
- the result JSON has `"cycles_device": "gpu"`;
- the result JSON has `"blend_file": ".../assets/blender/templates/simtool_lab.blend"`;
- preview GIFs are present and visually photorealistic.

The generated GIFs are the same frames fed to the policy, before the policy's
internal crop/resize transforms. This makes the preview GIFs the correct visual
debug surface for the photoreal policy input.
