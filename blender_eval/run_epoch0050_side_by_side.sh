#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CS224R_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

REMOTE_CKPT="/move/u/chrzhang/diffusion_policy/data/outputs/2026.05.06/22.17.26_train_diffusion_unet_hybrid_simtool_image_state29/checkpoints/epoch=0050-val_loss=0.0465.ckpt"
CKPT="${CKPT:-${1:-$CS224R_ROOT/checkpoints/epoch=0050-val_loss=0.0465.ckpt}}"
BLENDER="${BLENDER:-/tmp/blender-4.2.9-linux-x64/blender}"
OUT_ROOT="${OUT_ROOT:-$CS224R_ROOT/blender_eval_videos/epoch0050_side_by_side}"
REF_ROOT="${REF_ROOT:-$CS224R_ROOT/diffusion_eval/epoch0050_train_videos}"

OBJ_CATEGORY="${OBJ_CATEGORY:-hammer}"
OBJ_NAME="${OBJ_NAME:-claw_hammer}"
TASK_NAME="${TASK_NAME:-swing_down}"
OBJECT_ID="${OBJECT_ID:-4}"
CATEGORY_ID="${CATEGORY_ID:-2}"

NUM_ENVS="${NUM_ENVS:-1}"
EPISODES="${EPISODES:-1}"
HORIZON="${HORIZON:-250}"
SEED="${SEED:-0}"
SAMPLES="${SAMPLES:-96}"
GIF_FPS="${GIF_FPS:-10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$(dirname "$CKPT")" "$OUT_ROOT"

if [[ ! -f "$CKPT" ]]; then
  cat <<EOF
Checkpoint not found:
  $CKPT

This session cannot SSH to scdt.stanford.edu non-interactively. From this laptop,
copy it with:

  scp scdt.stanford.edu:'$REMOTE_CKPT' '$CKPT'

or, resumable:

  rsync -avP scdt.stanford.edu:'$REMOTE_CKPT' '$CKPT'

Then rerun:
  $0 '$CKPT'
EOF
  exit 2
fi

FREE_KB="$(df -Pk "$CS224R_ROOT" | awk 'NR == 2 {print $4}')"
if (( FREE_KB < 1048576 )); then
  echo "Refusing to run: less than 1 GiB free after checkpoint copy." >&2
  df -h "$CS224R_ROOT" >&2
  exit 3
fi

export PATH="$REPO_ROOT/.venv/bin:$PATH"
PYTHON_LIBDIR="$("$REPO_ROOT/.venv/bin/python" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")"
export LD_LIBRARY_PATH="$PYTHON_LIBDIR:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CS224R_ROOT/diffusion-policy:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PY="$REPO_ROOT/.venv/bin/python"
COMMON_ARGS=(
  --worker
  --checkpoint "$CKPT"
  --object-category "$OBJ_CATEGORY"
  --object-name "$OBJ_NAME"
  --task-name "$TASK_NAME"
  --object-id "$OBJECT_ID"
  --category-id "$CATEGORY_ID"
  --num-envs "$NUM_ENVS"
  --episodes-per-object "$EPISODES"
  --horizon "$HORIZON"
  --seed "$SEED"
  --gif-fps "$GIF_FPS"
  --max-success-previews 1
  --max-failure-previews 1
)

RUN_ROOT="$OUT_ROOT/${OBJ_NAME}_${RUN_ID}"
ISAAC_DIR="$RUN_ROOT/isaacgym"
BLENDER_DIR="$RUN_ROOT/blender_cycles"
mkdir -p "$ISAAC_DIR" "$BLENDER_DIR"

echo "[side-by-side] running IsaacGym renderer preview..."
"$PY" "$REPO_ROOT/blender_eval/eval_blender.py" \
  "${COMMON_ARGS[@]}" \
  --renderer isaacgym \
  --result-json "$RUN_ROOT/isaacgym_${OBJ_NAME}.json" \
  --video-dir "$ISAAC_DIR"

echo "[side-by-side] running Blender Cycles renderer preview..."
"$PY" "$REPO_ROOT/blender_eval/eval_blender.py" \
  "${COMMON_ARGS[@]}" \
  --renderer blender \
  --engine cycles \
  --samples "$SAMPLES" \
  --blender "$BLENDER" \
  --result-json "$RUN_ROOT/blender_cycles_${OBJ_NAME}.json" \
  --video-dir "$BLENDER_DIR"

pick_gif() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name 'success_*.gif' | sort | head -n 1
  find "$dir" -maxdepth 1 -type f -name 'fail_*.gif' | sort | head -n 1
}

ISAAC_GIF="$(pick_gif "$ISAAC_DIR" | head -n 1)"
BLENDER_GIF="$(pick_gif "$BLENDER_DIR" | head -n 1)"
if [[ -z "$ISAAC_GIF" || -z "$BLENDER_GIF" ]]; then
  echo "Could not find both generated GIFs." >&2
  echo "IsaacGym dir: $ISAAC_DIR" >&2
  echo "Blender dir:  $BLENDER_DIR" >&2
  exit 4
fi

"$PY" "$REPO_ROOT/blender_eval/make_side_by_side_gif.py" \
  --left "$ISAAC_GIF" \
  --right "$BLENDER_GIF" \
  --left-label "IsaacGym renderer" \
  --right-label "Blender Cycles" \
  --fps "$GIF_FPS" \
  --out "$RUN_ROOT/side_by_side_local_${OBJ_NAME}.gif"

REF_GIF=""
if [[ -d "$REF_ROOT/$OBJ_NAME" ]]; then
  REF_GIF="$(pick_gif "$REF_ROOT/$OBJ_NAME" | head -n 1 || true)"
fi
if [[ -n "$REF_GIF" ]]; then
  "$PY" "$REPO_ROOT/blender_eval/make_side_by_side_gif.py" \
    --left "$REF_GIF" \
    --right "$BLENDER_GIF" \
    --left-label "Christine IsaacGym eval" \
    --right-label "Blender Cycles" \
    --fps "$GIF_FPS" \
    --out "$RUN_ROOT/side_by_side_christine_${OBJ_NAME}.gif"
fi

echo
echo "[side-by-side] outputs:"
echo "  $RUN_ROOT/isaacgym_${OBJ_NAME}.json"
echo "  $RUN_ROOT/blender_cycles_${OBJ_NAME}.json"
echo "  $RUN_ROOT/side_by_side_local_${OBJ_NAME}.gif"
if [[ -n "$REF_GIF" ]]; then
  echo "  $RUN_ROOT/side_by_side_christine_${OBJ_NAME}.gif"
fi
