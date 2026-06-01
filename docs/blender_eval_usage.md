# Blender Evaluation Usage Guide

This guide describes how to run SimTool diffusion-policy evaluations with
Blender-rendered observations on lab GPUs. It supports both evaluation tasks:

- `pickup`
- `pick_place_release`

The intended workflow is checkpoint-only: teammates should not need training
zarrs, recovery datasets, logs, videos, or committed checkpoints. They need the
repo, the committed Blender assets, a trained policy checkpoint, and a working
IsaacGym/diffusion-policy/Blender environment.

## What Is Committed

The PR includes the reusable evaluation code and visual assets:

- `blender_eval/eval_blender.py`: main closed-loop evaluator
- `blender_eval/aggregate_eval_results.py`: aggregates per-object worker JSONs
- `blender_eval/success_criteria.py`: pickup success metrics
- `blender_eval/blender_renderer.py`: persistent Blender subprocess renderer
- `assets/blender/templates/simtool_lab.blend`: default photoreal scene
- `assets/blender/hdri/` and `assets/blender/textures/`: scene lighting/textures
- `blender_eval/run_photoreal_eval_l40.sbatch`: generic SLURM launcher

Do not commit:

- trained checkpoints
- zarr datasets
- evaluation videos
- SLURM logs
- generated output JSONs

## Runtime Requirements

Use a Python environment that can import:

- `isaacgym`
- `isaacgymenvs`
- `torch`
- `hydra`
- `omegaconf`
- `dill`
- `imageio`
- `diffusion_policy`

Recommended environment assumptions:

- Python 3.8 for IsaacGym Preview 4 compatibility.
- CUDA-capable NVIDIA GPU.
- Blender 4.x with Cycles support. Blender 4.2.9 has been used successfully.
- `ninja` available on `PATH` if IsaacGym needs to JIT the `gymtorch` extension.
- `diffusion_policy` checkout available on `PYTHONPATH`.

The evaluation uses IsaacGym for physics and either Blender, IsaacGym, or a stub
renderer for policy images. Use `--renderer blender` for photoreal evaluation.

## Required Inputs

Set these paths before running:

```bash
export SIMTOOLDIFF_ROOT=/path/to/simtooldiff
export DIFFUSION_POLICY_ROOT=/path/to/diffusion_policy
export BLENDER=/path/to/blender
export CKPT=/path/to/trained_policy.ckpt
export OUT_ROOT=$SIMTOOLDIFF_ROOT/data/blender_eval_runs
```

The checkpoint must match the requested task:

- pickup checkpoint: `--eval-task pickup`
- pick-place-release checkpoint: `--eval-task pick_place_release`

Then configure imports:

```bash
cd "$SIMTOOLDIFF_ROOT"
export PYTHONPATH="$DIFFUSION_POLICY_ROOT:$SIMTOOLDIFF_ROOT:$SIMTOOLDIFF_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

If using a virtualenv, put it on `PATH` and expose its libdir:

```bash
export PATH=/path/to/venv/bin:$PATH
export LD_LIBRARY_PATH="$(python - <<'PY'
import sysconfig
print(sysconfig.get_config_var("LIBDIR") or "")
PY
):${LD_LIBRARY_PATH:-}"
```

Sanity checks:

```bash
nvidia-smi
"$BLENDER" --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import isaacgym, diffusion_policy; print('imports ok')"
```

## Quick Smoke Test

Run one object first. This verifies the checkpoint, IsaacGym, Blender, imports,
and output writing before spending GPU time on a full split.

```bash
mkdir -p "$OUT_ROOT/pickup_smoke"

python blender_eval/eval_blender.py \
  --worker \
  --checkpoint "$CKPT" \
  --renderer blender \
  --eval-task pickup \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down \
  --object-id 4 \
  --category-id 2 \
  --episodes-per-object 2 \
  --num-envs 1 \
  --xy-range 0.10 \
  --seed 0 \
  --device cuda:0 \
  --engine cycles \
  --samples 32 \
  --cycles-device gpu \
  --blender "$BLENDER" \
  --result-json "$OUT_ROOT/pickup_smoke/claw_hammer.json" \
  --video-dir "$OUT_ROOT/pickup_smoke/videos/claw_hammer" \
  --max-success-previews 1 \
  --max-failure-previews 1
```

Expected outputs:

- `$OUT_ROOT/pickup_smoke/claw_hammer.json`
- optional GIF previews under `$OUT_ROOT/pickup_smoke/videos/claw_hammer/`

For PPR, change the checkpoint and task:

```bash
mkdir -p "$OUT_ROOT/ppr_smoke"

python blender_eval/eval_blender.py \
  --worker \
  --checkpoint "$CKPT" \
  --renderer blender \
  --eval-task pick_place_release \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down \
  --object-id 4 \
  --category-id 2 \
  --episodes-per-object 2 \
  --num-envs 1 \
  --xy-range 0.10 \
  --seed 0 \
  --device cuda:0 \
  --engine cycles \
  --samples 32 \
  --cycles-device gpu \
  --blender "$BLENDER" \
  --result-json "$OUT_ROOT/ppr_smoke/claw_hammer.json" \
  --video-dir "$OUT_ROOT/ppr_smoke/videos/claw_hammer" \
  --max-success-previews 1 \
  --max-failure-previews 1
```

## Full Split With The Driver

The simplest full eval runs all objects sequentially in one process:

```bash
python blender_eval/eval_blender.py \
  --checkpoint "$CKPT" \
  --renderer blender \
  --eval-task pickup \
  --split train \
  --episodes-per-object 32 \
  --num-envs 1 \
  --xy-range 0.10 \
  --seed 0 \
  --device cuda:0 \
  --engine cycles \
  --samples 96 \
  --cycles-device gpu \
  --blender "$BLENDER" \
  --output-json "$OUT_ROOT/pickup_train.json"
```

Change `--split ood` for the OOD objects. Change `--eval-task
pick_place_release` and use a PPR checkpoint for PPR.

For Blender evaluation, keep `--num-envs 1`. The renderer processes frames
serially, so object-level parallelism is usually more efficient than larger env
batches.

## Full Split With SLURM

The included SLURM launcher supports one-object workers, object arrays, and
aggregation. It is written for L40-style jobs but can be edited for other lab
partitions/accounts.

Required variables:

```bash
export SIMTOOLDIFF_ROOT=/path/to/simtooldiff
export DIFFUSION_POLICY_ROOT=/path/to/diffusion_policy
export CKPT=/path/to/trained_policy.ckpt
export EVAL_TASK=pickup                  # or pick_place_release
export SPLIT=train                       # or ood
export OUT_ROOT=$SIMTOOLDIFF_ROOT/data/blender_eval_runs
export BLENDER=/path/to/blender          # optional if DOWNLOAD_BLENDER=1
export VENV=/path/to/python38_venv       # or set CONDA_SH and CONDA_ENV
```

One-object smoke on SLURM:

```bash
MODE=worker \
EPISODES_PER_OBJECT=2 \
SAMPLES=32 \
sbatch blender_eval/run_photoreal_eval_l40.sbatch
```

Object-parallel full train split:

```bash
RUN_ID=pickup_train_$(date +%Y%m%d_%H%M%S)

MODE=array_worker \
ARRAY_RUN_ID="$RUN_ID" \
SPLIT=train \
EVAL_TASK=pickup \
EPISODES_PER_OBJECT=32 \
SAMPLES=96 \
sbatch --array=0-8 blender_eval/run_photoreal_eval_l40.sbatch
```

Object-parallel OOD split:

```bash
RUN_ID=pickup_ood_$(date +%Y%m%d_%H%M%S)

MODE=array_worker \
ARRAY_RUN_ID="$RUN_ID" \
SPLIT=ood \
EVAL_TASK=pickup \
EPISODES_PER_OBJECT=32 \
SAMPLES=96 \
sbatch --array=0-2 blender_eval/run_photoreal_eval_l40.sbatch
```

After workers finish, aggregate with the same `RUN_ID`, `SPLIT`, and
`EVAL_TASK` used for that worker batch:

```bash
MODE=aggregate \
ARRAY_RUN_ID="$RUN_ID" \
SPLIT="$SPLIT" \
EVAL_TASK="$EVAL_TASK" \
EPISODES_PER_OBJECT=32 \
SAMPLES=96 \
sbatch blender_eval/run_photoreal_eval_l40.sbatch
```

For PPR, use:

```bash
export EVAL_TASK=pick_place_release
export CKPT=/path/to/ppr_checkpoint.ckpt
```

If `HORIZON` is unset, the evaluator chooses task defaults. Set `HORIZON`
manually only when you intentionally want to override those defaults.

## Result Fields

Pickup JSONs report:

- `attempted`
- `succeeded`
- `success_rate`
- `stable_succeeded`
- `stable_success_rate`

Use `success_rate` as the original pickup success metric. Treat
`stable_success_rate` as a stricter diagnostic.

PPR JSONs report final release success as the headline metric:

- `attempted`
- `succeeded`
- `success_rate`
- `pick_place_succeeded`
- `release_goal_succeeded`
- `release_stable_succeeded`
- `failure_breakdown`

## Common Issues

`ModuleNotFoundError: diffusion_policy`

Check `DIFFUSION_POLICY_ROOT` and `PYTHONPATH`.

`RuntimeError: Ninja is required to load C++ extensions`

Install or load `ninja`, and ensure it is on `PATH`.

Blender cannot find a GPU or runs out of memory

Try `--cycles-device auto`, reduce `--samples`, or run on a larger-memory GPU.
For Blender eval, use `--num-envs 1`.

Checkpoint loads but rollout fails immediately

Confirm the checkpoint task matches `--eval-task`, and that the checkpoint's
diffusion-policy code/config is compatible with the checked-out
`diffusion_policy` repo.

No videos are written

Set `--max-success-previews` or `--max-failure-previews` above zero and provide
`--video-dir`.

Aggregation reports missing objects

Confirm all per-object JSONs exist under the same `ARRAY_RUN_ID` result
directory before running `MODE=aggregate`.
