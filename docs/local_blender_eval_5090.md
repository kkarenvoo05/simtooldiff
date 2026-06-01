# Local RTX 5090 Blender Evaluation Guide

This guide is for running SimTool Blender-rendered diffusion policy evaluations on a local RTX 5090 machine. It covers both supported eval tasks:

- `pickup`
- `pick_place_release`

The main entrypoint is:

```bash
python blender_eval/eval_blender.py
```

The aggregation entrypoint for per-object worker JSONs is:

```bash
python blender_eval/aggregate_eval_results.py
```

## Code To Pull / Push

Use the `simtooldiff` repo branch that contains the local Blender eval changes. In the Stanford workspace this branch is currently:

```bash
cd /move/u/caydengu/Projects/cs224r/simtooldiff
git branch --show-current
# blender-eval
```

The 5090 machine needs these code areas:

- `blender_eval/eval_blender.py`
- `blender_eval/aggregate_eval_results.py`
- `blender_eval/success_criteria.py`
- `blender_eval/tests/test_eval_blender_cli.py`
- `blender_eval/tests/test_aggregate_eval_results.py`
- `scripts/eval_diffusion_policy.py`
- `scripts/stage5_multi_object_driver.py`
- `scripts/stage5_collect_dataset.py`
- `isaacgymenvs/tasks/simtoolreal/env.py`
- `assets/blender/templates/simtool_lab.blend`
- existing PPR helper: `scripts/collect_dataset_pick_place_release.py`

The local 5090 machine does not need cluster-only scripts, logs, zarr datasets, videos, or checkpoints committed into git. Keep checkpoints and output directories as external files.

## Required Local Inputs

Set these paths on the 5090 machine:

```bash
export SIMTOOLDIFF_ROOT=/path/to/simtooldiff
export DIFFUSION_POLICY_ROOT=/path/to/diffusion_policy
export BLENDER=/path/to/blender
export CKPT=/path/to/policy_checkpoint.ckpt
export OUT_ROOT=$SIMTOOLDIFF_ROOT/data/local_blender_eval
```

Recommended Blender version: Blender 4.x. The cluster script has used Blender `4.2.9`, but any compatible Blender 4 build with Cycles GPU support should work.

The policy checkpoint must match the task:

- Pickup checkpoint for `--eval-task pickup`.
- Pick-place-release checkpoint for `--eval-task pick_place_release`.

The checkpoint is not stored in git. Copy or mount it onto the 5090 machine.

## Environment Setup

From the local machine:

```bash
cd "$SIMTOOLDIFF_ROOT"

export PYTHONPATH="$DIFFUSION_POLICY_ROOT:$SIMTOOLDIFF_ROOT:$SIMTOOLDIFF_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$(python - <<'PY'
import sysconfig
print(sysconfig.get_config_var("LIBDIR") or "")
PY
):${LD_LIBRARY_PATH:-}"

nvidia-smi
"$BLENDER" --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The Python environment must be able to import:

- `isaacgym`
- `isaacgymenvs`
- `torch`
- `hydra`
- `omegaconf`
- `dill`
- `imageio`
- `diffusion_policy`

If Blender cannot find CUDA/OptiX devices, start with `--cycles-device auto`, then try `--cycles-device gpu`.

## Object Splits

Pickup and PPR both use the same object registry split:

Train objects:

```text
blue_brush
red_brush
flat_eraser
handle_eraser
claw_hammer
sharpie_marker
staples_marker
long_screwdriver
short_screwdriver
```

OOD objects:

```text
mallet_hammer
flat_spatula
spoon_spatula
```

Full-size evals use `--episodes-per-object 32`, giving 288 train attempts and 96 OOD attempts.

## Quick Smoke Test: One Pickup Object

Use this first to confirm IsaacGym, Blender, and the checkpoint all load.

```bash
cd "$SIMTOOLDIFF_ROOT"
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
  --horizon 250 \
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

Expected output:

- Per-object JSON: `$OUT_ROOT/pickup_smoke/claw_hammer.json`
- Optional GIF previews in: `$OUT_ROOT/pickup_smoke/videos/claw_hammer/`
- Terminal lines like:

```text
[eval-worker] policy: To=2 k=8 state_dim=29 image=(3, 192, 256) renderer=blender
[eval-worker] DONE claw_hammer: 1/2 = 50.0% stable=1/2 = 50.0%
```

Important JSON fields:

```json
{
  "eval_task": "pickup",
  "renderer": "blender",
  "attempted": 2,
  "succeeded": 1,
  "success_rate": 0.5,
  "stable_succeeded": 1,
  "stable_success_rate": 0.5
}
```

For the report, use `succeeded / attempted` and `success_rate` as the original pickup metric. `stable_success_rate` is diagnostic only.

## Full Pickup Eval: Driver Mode

Driver mode runs every object in a split sequentially from one process. This is the simplest local workflow.

Train split:

```bash
cd "$SIMTOOLDIFF_ROOT"
mkdir -p "$OUT_ROOT/pickup_driver"

python blender_eval/eval_blender.py \
  --checkpoint "$CKPT" \
  --renderer blender \
  --eval-task pickup \
  --split train \
  --episodes-per-object 32 \
  --num-envs 1 \
  --horizon 250 \
  --xy-range 0.10 \
  --seed 0 \
  --device cuda:0 \
  --engine cycles \
  --samples 96 \
  --cycles-device gpu \
  --blender "$BLENDER" \
  --output-json "$OUT_ROOT/pickup_driver/photoreal_train.json" \
  --max-success-previews 2 \
  --max-failure-previews 2
```

OOD split:

```bash
python blender_eval/eval_blender.py \
  --checkpoint "$CKPT" \
  --renderer blender \
  --eval-task pickup \
  --split ood \
  --episodes-per-object 32 \
  --num-envs 1 \
  --horizon 250 \
  --xy-range 0.10 \
  --seed 0 \
  --device cuda:0 \
  --engine cycles \
  --samples 96 \
  --cycles-device gpu \
  --blender "$BLENDER" \
  --output-json "$OUT_ROOT/pickup_driver/photoreal_ood.json" \
  --max-success-previews 2 \
  --max-failure-previews 2
```

Expected summary JSON fields:

```json
{
  "eval_task": "pickup",
  "split": "train",
  "episodes_per_object": 32,
  "overall_success_rate": 0.40625,
  "stable_success_rate": 0.3854166666666667,
  "total_attempted": 288,
  "total_succeeded": 117,
  "total_stable_succeeded": 111,
  "per_object": []
}
```

## Full Pickup Eval: Per-Object Worker + Aggregate

Use this if the local 5090 session wants to run one object at a time, resume failed objects manually, or parallelize across multiple local shells.

Example worker for one object:

```bash
RUN_DIR="$OUT_ROOT/pickup_workers/train"
mkdir -p "$RUN_DIR/object_results" "$RUN_DIR/videos/claw_hammer"

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
  --episodes-per-object 32 \
  --num-envs 1 \
  --horizon 250 \
  --xy-range 0.10 \
  --seed 4 \
  --device cuda:0 \
  --engine cycles \
  --samples 96 \
  --cycles-device gpu \
  --blender "$BLENDER" \
  --result-json "$RUN_DIR/object_results/claw_hammer.json" \
  --video-dir "$RUN_DIR/videos/claw_hammer"
```

After every train object has one JSON under `object_results/`, aggregate:

```bash
python blender_eval/aggregate_eval_results.py \
  --split train \
  --eval-task pickup \
  --checkpoint "$CKPT" \
  --result-dir "$RUN_DIR/object_results" \
  --output-json "$RUN_DIR/photoreal_train.json" \
  --renderer blender \
  --engine cycles \
  --samples 96 \
  --cycles-device gpu \
  --episodes-per-object 32 \
  --num-envs 1 \
  --xy-range 0.10 \
  --horizon 250 \
  --video-dir "$RUN_DIR/videos"
```

Aggregation fails with a clear missing-object error if any split object is absent.

## Quick Smoke Test: One Pick-Place-Release Object

Use a PPR checkpoint for this section.

```bash
export PPR_CKPT=/path/to/pick_place_release_checkpoint.ckpt

cd "$SIMTOOLDIFF_ROOT"
mkdir -p "$OUT_ROOT/ppr_smoke"

python blender_eval/eval_blender.py \
  --worker \
  --checkpoint "$PPR_CKPT" \
  --renderer blender \
  --eval-task pick_place_release \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down \
  --object-id 4 \
  --category-id 2 \
  --episodes-per-object 2 \
  --num-envs 1 \
  --horizon 325 \
  --xy-range 0.10 \
  --start-z-offset 0.0 \
  --seed 0 \
  --device cuda:0 \
  --engine cycles \
  --samples 32 \
  --cycles-device gpu \
  --blender "$BLENDER" \
  --release-steps 45 \
  --release-arm-mode hold \
  --release-hand-blend 1.0 \
  --result-json "$OUT_ROOT/ppr_smoke/claw_hammer.json" \
  --video-dir "$OUT_ROOT/ppr_smoke/videos/claw_hammer" \
  --max-success-previews 1 \
  --max-failure-previews 1
```

Expected terminal lines:

```text
[eval-worker-ppr] policy: To=2 state_dim=29 image=(3, 192, 256) renderer=blender
[eval-worker-ppr] DONE claw_hammer: release=1/2 = 50.0% pick_place=2/2 = 100.0%
```

Important JSON fields:

```json
{
  "eval_task": "pick_place_release",
  "attempted": 2,
  "succeeded": 1,
  "success_rate": 0.5,
  "pick_place_succeeded": 2,
  "pick_place_success_rate": 1.0,
  "release_goal_succeeded": 1,
  "release_goal_success_rate": 0.5,
  "release_stable_succeeded": 1,
  "release_stable_success_rate": 0.5,
  "failure_breakdown": {}
}
```

For PPR, `succeeded` / `success_rate` means final release success, not pickup max-height success.

## Full Pick-Place-Release Eval: Driver Mode

Train split:

```bash
mkdir -p "$OUT_ROOT/ppr_driver"

python blender_eval/eval_blender.py \
  --checkpoint "$PPR_CKPT" \
  --renderer blender \
  --eval-task pick_place_release \
  --split train \
  --episodes-per-object 32 \
  --num-envs 1 \
  --horizon 325 \
  --xy-range 0.10 \
  --start-z-offset 0.0 \
  --seed 0 \
  --device cuda:0 \
  --engine cycles \
  --samples 96 \
  --cycles-device gpu \
  --blender "$BLENDER" \
  --release-steps 45 \
  --release-arm-mode hold \
  --release-hand-blend 1.0 \
  --output-json "$OUT_ROOT/ppr_driver/photoreal_ppr_train.json" \
  --max-success-previews 2 \
  --max-failure-previews 2
```

OOD split:

```bash
python blender_eval/eval_blender.py \
  --checkpoint "$PPR_CKPT" \
  --renderer blender \
  --eval-task pick_place_release \
  --split ood \
  --episodes-per-object 32 \
  --num-envs 1 \
  --horizon 325 \
  --xy-range 0.10 \
  --start-z-offset 0.0 \
  --seed 0 \
  --device cuda:0 \
  --engine cycles \
  --samples 96 \
  --cycles-device gpu \
  --blender "$BLENDER" \
  --release-steps 45 \
  --release-arm-mode hold \
  --release-hand-blend 1.0 \
  --output-json "$OUT_ROOT/ppr_driver/photoreal_ppr_ood.json" \
  --max-success-previews 2 \
  --max-failure-previews 2
```

Expected summary JSON fields:

```json
{
  "eval_task": "pick_place_release",
  "overall_success_rate": 0.5,
  "pick_place_success_rate": 0.75,
  "release_goal_success_rate": 0.5,
  "total_attempted": 288,
  "total_succeeded": 144,
  "total_pick_place_succeeded": 216,
  "total_release_goal_succeeded": 144,
  "total_release_stable_succeeded": 140
}
```

## Useful Runtime Knobs

Start with smoke settings:

- `--samples 32`
- `--episodes-per-object 1` or `2`
- `--max-success-previews 1`
- `--max-failure-previews 1`

Use final settings:

- `--samples 96`
- `--episodes-per-object 32`
- `--num-envs 1`
- `--cycles-device gpu`

If Blender OOMs or is unstable:

- reduce `--samples` to `32` or `64`
- set `--max-success-previews 0 --max-failure-previews 0`
- keep `--num-envs 1`
- try `--cycles-device auto`

If you only want speed and not final report-quality renders:

```bash
--engine eevee --samples 16
```

Use Cycles for final comparable photoreal numbers.

## Common Failures

Checkpoint load fails:

- Verify `CKPT` points to the local copied `.ckpt`.
- Verify `DIFFUSION_POLICY_ROOT` is on `PYTHONPATH`.
- Verify the checkpoint task matches `--eval-task`.

`ModuleNotFoundError: isaacgym`:

- Use the IsaacGym-compatible Python environment.
- Make sure `isaacgym_pkg` / local IsaacGym install paths are available as in the cluster environment.

Blender executable fails:

- Run `"$BLENDER" --version`.
- Make sure `BLENDER` points to the actual binary, not the containing directory.

GPU rendering does not engage:

- Run `nvidia-smi` while the worker is rendering.
- Try `--cycles-device gpu` or `--cycles-device auto`.
- Check Blender preferences if the local install does not expose CUDA/OptiX to background renders.

Aggregation fails with missing objects:

- Check which object JSONs exist:

```bash
find "$RUN_DIR/object_results" -maxdepth 1 -name '*.json' -printf '%f\n' | sort
```

- Re-run only the missing object workers.

## Minimal Commands For The Local Codex Session

Give the Codex session on the 5090 these steps:

```bash
cd "$SIMTOOLDIFF_ROOT"
export PYTHONPATH="$DIFFUSION_POLICY_ROOT:$SIMTOOLDIFF_ROOT:$SIMTOOLDIFF_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OUT_ROOT="$SIMTOOLDIFF_ROOT/data/local_blender_eval"

python blender_eval/eval_blender.py --help
python blender_eval/aggregate_eval_results.py --help

# pickup smoke
python blender_eval/eval_blender.py \
  --worker --checkpoint "$CKPT" --renderer blender --eval-task pickup \
  --object-category hammer --object-name claw_hammer --task-name swing_down \
  --object-id 4 --category-id 2 \
  --episodes-per-object 1 --num-envs 1 --horizon 250 \
  --device cuda:0 --engine cycles --samples 32 --cycles-device gpu \
  --blender "$BLENDER" \
  --result-json "$OUT_ROOT/pickup_smoke/claw_hammer.json"

# PPR smoke
python blender_eval/eval_blender.py \
  --worker --checkpoint "$PPR_CKPT" --renderer blender --eval-task pick_place_release \
  --object-category hammer --object-name claw_hammer --task-name swing_down \
  --object-id 4 --category-id 2 \
  --episodes-per-object 1 --num-envs 1 --horizon 325 \
  --device cuda:0 --engine cycles --samples 32 --cycles-device gpu \
  --blender "$BLENDER" \
  --result-json "$OUT_ROOT/ppr_smoke/claw_hammer.json"
```

If both smoke runs complete and write JSON, proceed to the full driver commands above.

