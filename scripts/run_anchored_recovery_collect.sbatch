#!/bin/bash
#SBATCH --account=move
#SBATCH --partition=move --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=anch_collect
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-manual}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-manual}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

if [[ -z "${SIMTOOLDIFF_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/stage5_multi_object_driver.py" ]]; then
    SIMTOOLDIFF_ROOT="$SLURM_SUBMIT_DIR"
  else
    SIMTOOLDIFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi

cd "$SIMTOOLDIFF_ROOT"
mkdir -p logs/slurm data

# uv/venv activation scripts can reference LD_LIBRARY_PATH directly; keep
# nounset enabled while making that reference safe on clean Slurm environments.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  echo "Missing $SIMTOOLDIFF_ROOT/.venv/bin/activate" >&2
  exit 2
fi

DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-/move/u/caydengu/Projects/cs224r/diffusion_policy}"
export PYTHONPATH="$DIFFUSION_POLICY_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all,graphics,utility,compute}"
export LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"

echo "Python: $(which python)"
python -V
echo "SIMTOOLDIFF_ROOT=$SIMTOOLDIFF_ROOT"
echo "DIFFUSION_POLICY_ROOT=$DIFFUSION_POLICY_ROOT"

# -------------------------- user-tunable settings --------------------------

RUN_NAME="${RUN_NAME:-anch_pickup_p4_r24_s1_t05}"
OUT_ZARR="${OUT_ZARR:-data/${RUN_NAME}.zarr}"
export RUN_NAME OUT_ZARR

COLLECTION_TYPE="${COLLECTION_TYPE:-pickup}"
SPLIT="${SPLIT:-train}"
PER_OBJECT_TRANSITIONS="${PER_OBJECT_TRANSITIONS:-10000}"
NUM_ENVS="${NUM_ENVS:-1}"
HORIZON="${HORIZON:-325}"
XY_RANGE="${XY_RANGE:-0.10}"
START_Z_OFFSET="${START_Z_OFFSET:-0.0}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
SAVE_PREVIEW_EVERY="${SAVE_PREVIEW_EVERY:-50}"

VARIANT="${VARIANT:-noisy_clean}"
NOISE_STRATEGY="${NOISE_STRATEGY:-anchored_recovery}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
OU_THETA="${OU_THETA:-0.5}"
OU_MU="${OU_MU:-0.0}"
OU_DT="${OU_DT:-1.0}"

BRANCHES="${BRANCHES:-5}"
ANCH_COLLECTION_MODE="${ANCH_COLLECTION_MODE:-fixed_segment}"
BRANCH_MIN="${BRANCH_MIN:-20}"
BRANCH_MAX="${BRANCH_MAX:-180}"
BRANCH_GAP="${BRANCH_GAP:-}"
PERTURB_STEPS="${PERTURB_STEPS:-4}"
RECOVERY_STEPS="${RECOVERY_STEPS:-24}"
MIN_SAVED_STEPS="${MIN_SAVED_STEPS:-16}"
MAX_SAVED_STEPS="${MAX_SAVED_STEPS:-64}"
PICKUP_SUCCESS_MODE="${PICKUP_SUCCESS_MODE:-max_height}"
PICKUP_SUCCESS_HOLD_STEPS="${PICKUP_SUCCESS_HOLD_STEPS:-5}"

MAX_STEPS="${MAX_STEPS:-}"
MAX_ATTEMPTED_EPISODES="${MAX_ATTEMPTED_EPISODES:-}"

# Optional explicit per-group noise overrides. Leave unset to use
# scripts/stage5_noise_config.py defaults scaled by NOISE_SCALE.
ARM_BASE_NOISE="${ARM_BASE_NOISE:-}"
ARM_WRIST_NOISE="${ARM_WRIST_NOISE:-}"
THUMB_NOISE="${THUMB_NOISE:-}"
INDEX_NOISE="${INDEX_NOISE:-}"
MIDDLE_NOISE="${MIDDLE_NOISE:-}"
RING_NOISE="${RING_NOISE:-}"
PINKY_NOISE="${PINKY_NOISE:-}"

if [[ "$NOISE_STRATEGY" == "anchored_recovery" && "$NUM_ENVS" != "1" ]]; then
  echo "anchored_recovery requires NUM_ENVS=1; got NUM_ENVS=$NUM_ENVS" >&2
  exit 2
fi

echo "RUN_NAME=$RUN_NAME"
echo "OUT_ZARR=$OUT_ZARR"
echo "COLLECTION_TYPE=$COLLECTION_TYPE SPLIT=$SPLIT"
echo "PER_OBJECT_TRANSITIONS=$PER_OBJECT_TRANSITIONS NUM_ENVS=$NUM_ENVS HORIZON=$HORIZON"
echo "VARIANT=$VARIANT NOISE_STRATEGY=$NOISE_STRATEGY NOISE_SCALE=$NOISE_SCALE OU_THETA=$OU_THETA"
echo "ANCH_COLLECTION_MODE=$ANCH_COLLECTION_MODE"
echo "BRANCHES=$BRANCHES BRANCH_WINDOW=[$BRANCH_MIN,$BRANCH_MAX] BRANCH_GAP=${BRANCH_GAP:-auto} PERTURB_STEPS=$PERTURB_STEPS RECOVERY_STEPS=$RECOVERY_STEPS"
echo "MIN_SAVED_STEPS=$MIN_SAVED_STEPS MAX_SAVED_STEPS=$MAX_SAVED_STEPS"
echo "PICKUP_SUCCESS_MODE=$PICKUP_SUCCESS_MODE PICKUP_SUCCESS_HOLD_STEPS=$PICKUP_SUCCESS_HOLD_STEPS"

cmd=(
  python scripts/stage5_multi_object_driver.py
  --collection-type "$COLLECTION_TYPE"
  --split "$SPLIT"
  --output-zarr "$OUT_ZARR"
  --per-object-transitions "$PER_OBJECT_TRANSITIONS"
  --num-envs "$NUM_ENVS"
  --horizon "$HORIZON"
  --xy-range "$XY_RANGE"
  --start-z-offset "$START_Z_OFFSET"
  --seed "$SEED"
  --device "$DEVICE"
  --variant "$VARIANT"
  --noise-strategy "$NOISE_STRATEGY"
  --noise-scale "$NOISE_SCALE"
  --ou-theta "$OU_THETA"
  --ou-mu "$OU_MU"
  --ou-dt "$OU_DT"
  --anchored-branches-per-rollout "$BRANCHES"
  --anchored-collection-mode "$ANCH_COLLECTION_MODE"
  --anchored-branch-min-step "$BRANCH_MIN"
  --anchored-branch-max-step "$BRANCH_MAX"
  --anchored-perturb-steps "$PERTURB_STEPS"
  --anchored-recovery-steps "$RECOVERY_STEPS"
  --anchored-min-saved-steps "$MIN_SAVED_STEPS"
  --anchored-max-saved-steps "$MAX_SAVED_STEPS"
  --pickup-success-mode "$PICKUP_SUCCESS_MODE"
  --pickup-success-hold-steps "$PICKUP_SUCCESS_HOLD_STEPS"
  --save-preview-every "$SAVE_PREVIEW_EVERY"
)

if [[ -n "$MAX_STEPS" ]]; then
  cmd+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$MAX_ATTEMPTED_EPISODES" ]]; then
  cmd+=(--max-attempted-episodes "$MAX_ATTEMPTED_EPISODES")
fi
if [[ -n "$BRANCH_GAP" ]]; then
  cmd+=(--anchored-branch-gap "$BRANCH_GAP")
fi
if [[ -n "$ARM_BASE_NOISE" ]]; then
  cmd+=(--arm-base-noise "$ARM_BASE_NOISE")
fi
if [[ -n "$ARM_WRIST_NOISE" ]]; then
  cmd+=(--arm-wrist-noise "$ARM_WRIST_NOISE")
fi
if [[ -n "$THUMB_NOISE" ]]; then
  cmd+=(--thumb-noise "$THUMB_NOISE")
fi
if [[ -n "$INDEX_NOISE" ]]; then
  cmd+=(--index-noise "$INDEX_NOISE")
fi
if [[ -n "$MIDDLE_NOISE" ]]; then
  cmd+=(--middle-noise "$MIDDLE_NOISE")
fi
if [[ -n "$RING_NOISE" ]]; then
  cmd+=(--ring-noise "$RING_NOISE")
fi
if [[ -n "$PINKY_NOISE" ]]; then
  cmd+=(--pinky-noise "$PINKY_NOISE")
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

python - <<'PY'
import os
import zarr
path = os.environ.get("OUT_ZARR", "")
if not path:
    path = os.environ.get("RUN_NAME", "anch_pickup_p4_r24_s1_t05")
    path = f"data/{path}.zarr"
root = zarr.open(path, mode="r")
print("\n[anchored-collect] summary")
print("zarr", path)
print("transitions", root["data"]["img"].shape[0])
print("episodes", root["meta"]["episode_ends"].shape[0])
for key in [
    "collection_task",
    "variant",
    "noise_strategy",
    "anchored_collection_mode",
    "anchored_base_rollouts_attempted",
    "anchored_base_rollouts_succeeded",
    "anchored_branches_attempted",
    "anchored_branches_written",
    "anchored_branches_aborted",
    "anchored_branches_filtered",
    "anchored_saved_transitions",
    "anchored_branch_metric_count",
    "anchored_branch_stable_success_rate",
    "anchored_branch_max_height_success_rate",
    "anchored_branch_max_consecutive_gate_steps_mean",
    "anchored_verified_branches_attempted",
    "anchored_verified_branches_written",
    "anchored_verified_branches_timeout",
    "anchored_verified_success_rate",
    "anchored_verified_saved_len_mean",
    "anchored_verified_recovery_step_mean",
    "pickup_gate_first_step_mean",
    "pickup_hold_first_step_mean",
    "pickup_success_mode",
    "pickup_success_hold_steps",
    "noise_action_delta_l2_mean",
    "noise_action_delta_linf_mean",
]:
    print(key, root.attrs.get(key))
PY

date
