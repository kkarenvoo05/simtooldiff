#!/bin/bash
#SBATCH --account=move
#SBATCH --partition=move --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40:1
#SBATCH --job-name=photoreal_eval_l40
#SBATCH --output=logs/photoreal_eval_l40-%j.out
#SBATCH --error=logs/photoreal_eval_l40-%j.err

set -euo pipefail

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-manual}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

# -------------------------- user-tunable settings --------------------------

SIMTOOLDIFF_ROOT="${SIMTOOLDIFF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CS224R_ROOT="${CS224R_ROOT:-$(cd "$SIMTOOLDIFF_ROOT/.." && pwd)}"
DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$CS224R_ROOT/diffusion_policy}"

CKPT="${CKPT:-}"
OUT_ROOT="${OUT_ROOT:-$SIMTOOLDIFF_ROOT/data/blender_eval/photoreal_l40}"

# MODE=worker runs one object. MODE=driver runs a full split sequentially.
# MODE=array_worker runs one object selected by SLURM_ARRAY_TASK_ID.
# MODE=aggregate combines completed array_worker JSONs into the driver JSON shape.
MODE="${MODE:-worker}"
SPLIT="${SPLIT:-train}"
ARRAY_RUN_ID="${ARRAY_RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
EVAL_TASK="${EVAL_TASK:-pickup}"

OBJ_CATEGORY="${OBJ_CATEGORY:-hammer}"
OBJ_NAME="${OBJ_NAME:-claw_hammer}"
TASK_NAME="${TASK_NAME:-swing_down}"
OBJECT_ID="${OBJECT_ID:-4}"
CATEGORY_ID="${CATEGORY_ID:-2}"

EPISODES_PER_OBJECT="${EPISODES_PER_OBJECT:-1}"
NUM_ENVS="${NUM_ENVS:-1}"
HORIZON="${HORIZON:-}"
XY_RANGE="${XY_RANGE:-0.10}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"

SAMPLES="${SAMPLES:-96}"
CYCLES_DEVICE="${CYCLES_DEVICE:-gpu}"
ENGINE="${ENGINE:-cycles}"
GIF_FPS="${GIF_FPS:-10}"
MAX_SUCCESS_PREVIEWS="${MAX_SUCCESS_PREVIEWS:-2}"
MAX_FAILURE_PREVIEWS="${MAX_FAILURE_PREVIEWS:-2}"

# Point BLENDER at an existing Blender 4.x binary to skip download.
BLENDER_VERSION="${BLENDER_VERSION:-4.2.9}"
BLENDER_BASE_DIR="${BLENDER_BASE_DIR:-${SLURMTMPDIR:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}}"
BLENDER_DIR="${BLENDER_DIR:-$BLENDER_BASE_DIR/blender-${BLENDER_VERSION}-linux-x64}"
BLENDER="${BLENDER:-$BLENDER_DIR/blender}"
DOWNLOAD_BLENDER="${DOWNLOAD_BLENDER:-1}"
BLENDER_URL="${BLENDER_URL:-https://download.blender.org/release/Blender4.2/blender-${BLENDER_VERSION}-linux-x64.tar.xz}"

# Override these for your environment. If VENV is set, it takes precedence.
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-str}"
VENV="${VENV:-}"

# -------------------------- environment setup ------------------------------

cd "$SIMTOOLDIFF_ROOT"
mkdir -p logs "$OUT_ROOT"

if [[ -z "$CKPT" ]]; then
  echo "Set CKPT=/path/to/trained_policy.ckpt before submitting this job." >&2
  exit 2
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 2
fi

if [[ -n "$VENV" ]]; then
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "VENV was set but $VENV/bin/python is not executable" >&2
    exit 3
  fi
  export PATH="$VENV/bin:$PATH"
  PY="$VENV/bin/python"
  PY_LIBDIR="$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")"
  export LD_LIBRARY_PATH="$PY_LIBDIR:${LD_LIBRARY_PATH:-}"
else
  if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
    echo "Conda activation script not found: ${CONDA_SH:-unset}" >&2
    echo "Set VENV=/path/to/python3.8/venv or CONDA_SH=/path/to/conda.sh." >&2
    exit 3
  fi
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  PY="$(command -v python)"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$DIFFUSION_POLICY_ROOT:$SIMTOOLDIFF_ROOT:$SIMTOOLDIFF_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

if [[ "$MODE" != "aggregate" && ! -x "$BLENDER" ]]; then
  if [[ "$DOWNLOAD_BLENDER" != "1" ]]; then
    echo "Blender binary not found or not executable: $BLENDER" >&2
    echo "Set BLENDER=/path/to/blender or DOWNLOAD_BLENDER=1." >&2
    exit 4
  fi
  mkdir -p "$(dirname "$BLENDER_DIR")"
  TMP_TARBALL="$(dirname "$BLENDER_DIR")/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
  echo "Downloading Blender $BLENDER_VERSION to $BLENDER_DIR"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$BLENDER_URL" -o "$TMP_TARBALL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$TMP_TARBALL" "$BLENDER_URL"
  else
    echo "Neither curl nor wget is available to download Blender." >&2
    exit 4
  fi
  tar -xf "$TMP_TARBALL" -C "$(dirname "$BLENDER_DIR")"
fi

if [[ "$MODE" != "aggregate" && ! -x "$BLENDER" ]]; then
  echo "Blender binary still not executable after setup: $BLENDER" >&2
  exit 4
fi

echo "Python: $PY"
"$PY" --version
if [[ "$MODE" != "aggregate" ]]; then
  echo "Blender: $BLENDER"
  "$BLENDER" --version | head -5
else
  echo "Blender: skipped for aggregate mode"
fi
echo "Checkpoint: $CKPT"
echo "Output root: $OUT_ROOT"
echo "Mode: $MODE"
echo "Eval task: $EVAL_TASK"
nvidia-smi || true

COMMON_ARGS=(
  --checkpoint "$CKPT"
  --renderer blender
  --eval-task "$EVAL_TASK"
  --engine "$ENGINE"
  --samples "$SAMPLES"
  --cycles-device "$CYCLES_DEVICE"
  --blender "$BLENDER"
  --episodes-per-object "$EPISODES_PER_OBJECT"
  --num-envs "$NUM_ENVS"
  --xy-range "$XY_RANGE"
  --device "$DEVICE"
  --max-success-previews "$MAX_SUCCESS_PREVIEWS"
  --max-failure-previews "$MAX_FAILURE_PREVIEWS"
  --gif-fps "$GIF_FPS"
)
if [[ -n "$HORIZON" ]]; then
  COMMON_ARGS+=(--horizon "$HORIZON")
fi

JOB_ID="${SLURM_JOB_ID:-manual}"

case "$MODE" in
  worker)
    RUN_DIR="$OUT_ROOT/worker_${OBJ_NAME}_${JOB_ID}"
    mkdir -p "$RUN_DIR/videos"
    echo "Running one-object photoreal worker: $OBJ_CATEGORY/$OBJ_NAME task=$TASK_NAME"
    "$PY" blender_eval/eval_blender.py \
      --worker \
      "${COMMON_ARGS[@]}" \
      --seed "$SEED" \
      --object-category "$OBJ_CATEGORY" \
      --object-name "$OBJ_NAME" \
      --task-name "$TASK_NAME" \
      --object-id "$OBJECT_ID" \
      --category-id "$CATEGORY_ID" \
      --result-json "$RUN_DIR/blender_cycles_${OBJ_NAME}.json" \
      --video-dir "$RUN_DIR/videos"
    ;;

  array_worker)
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      echo "MODE=array_worker requires sbatch --array. SLURM_ARRAY_TASK_ID is unset." >&2
      exit 6
    fi
    SPEC_LINE="$(
      "$PY" - "$SPLIT" "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "scripts"))
from stage5_multi_object_driver import _split

split = sys.argv[1]
idx = int(sys.argv[2])
specs = _split(split)
if idx < 0 or idx >= len(specs):
    raise SystemExit(f"array index {idx} out of range for split={split} ({len(specs)} objects)")
spec = specs[idx]
print(spec.object_id, spec.category_id, spec.object_category, spec.object_name, spec.task_name)
PY
    )"
    read -r OBJECT_ID CATEGORY_ID OBJ_CATEGORY OBJ_NAME TASK_NAME <<< "$SPEC_LINE"
    OBJECT_SEED=$((SEED + OBJECT_ID))
    RUN_DIR="$OUT_ROOT/array_${SPLIT}_${ARRAY_RUN_ID}"
    RESULT_DIR="$RUN_DIR/object_results"
    VIDEO_ROOT="$RUN_DIR/photoreal_${SPLIT}_videos"
    mkdir -p "$RESULT_DIR" "$VIDEO_ROOT/$OBJ_NAME"
    echo "Running array object ${SLURM_ARRAY_TASK_ID}: $OBJ_CATEGORY/$OBJ_NAME task=$TASK_NAME seed=$OBJECT_SEED"
    "$PY" blender_eval/eval_blender.py \
      --worker \
      "${COMMON_ARGS[@]}" \
      --seed "$OBJECT_SEED" \
      --object-category "$OBJ_CATEGORY" \
      --object-name "$OBJ_NAME" \
      --task-name "$TASK_NAME" \
      --object-id "$OBJECT_ID" \
      --category-id "$CATEGORY_ID" \
      --result-json "$RESULT_DIR/${OBJ_NAME}.json" \
      --video-dir "$VIDEO_ROOT/$OBJ_NAME"
    ;;

  aggregate)
    RUN_DIR="$OUT_ROOT/array_${SPLIT}_${ARRAY_RUN_ID}"
    RESULT_DIR="$RUN_DIR/object_results"
    VIDEO_ROOT="$RUN_DIR/photoreal_${SPLIT}_videos"
    echo "Aggregating array results from $RESULT_DIR"
    AGG_ARGS=(
      --split "$SPLIT"
      --checkpoint "$CKPT"
      --result-dir "$RESULT_DIR"
      --output-json "$RUN_DIR/photoreal_${SPLIT}.json"
      --renderer blender
      --engine "$ENGINE"
      --samples "$SAMPLES"
      --cycles-device "$CYCLES_DEVICE"
      --episodes-per-object "$EPISODES_PER_OBJECT"
      --num-envs "$NUM_ENVS"
      --xy-range "$XY_RANGE"
      --video-dir "$VIDEO_ROOT"
      --eval-task "$EVAL_TASK"
    )
    if [[ -n "$HORIZON" ]]; then
      AGG_ARGS+=(--horizon "$HORIZON")
    fi
    "$PY" blender_eval/aggregate_eval_results.py "${AGG_ARGS[@]}"
    ;;

  driver)
    RUN_DIR="$OUT_ROOT/driver_${SPLIT}_${JOB_ID}"
    mkdir -p "$RUN_DIR"
    echo "Running split photoreal driver: split=$SPLIT episodes_per_object=$EPISODES_PER_OBJECT"
    "$PY" blender_eval/eval_blender.py \
      "${COMMON_ARGS[@]}" \
      --seed "$SEED" \
      --split "$SPLIT" \
      --output-json "$RUN_DIR/photoreal_${SPLIT}.json"
    ;;

  *)
    echo "Unknown MODE=$MODE. Expected worker, array_worker, aggregate, or driver." >&2
    exit 5
    ;;
esac

echo "Outputs written under: $RUN_DIR"
find "$RUN_DIR" -maxdepth 3 -type f -print | sort

echo "Done"
date
h=$((SECONDS / 3600))
m=$((SECONDS % 3600 / 60))
s=$((SECONDS % 60))
echo "Elapsed wall time: ${SECONDS}s (${h}h ${m}m ${s}s)"
