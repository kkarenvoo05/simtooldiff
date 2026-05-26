#!/bin/bash
#SBATCH --job-name=collect_pickup_policy
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/collect_pickup_policy_%j.out
#SBATCH --error=logs/slurm/collect_pickup_policy_%j.err

set -euo pipefail

mkdir -p logs/slurm

cd /move/u/karenvo/Projects/simtoolreal   

echo "SCRIPT_MARKER=2026-05-21-path-fix-v1"
echo "SCRIPT_FILE=$(readlink -f "$0")"

# ========================
# ACTIVATE PROJECT ENV
# ========================
# Your venv activation script already prepends the Python 3.8 lib dir to
# LD_LIBRARY_PATH, which Isaac Gym needs to resolve libpython3.8.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source .venv/bin/activate
export PATH="/move/u/karenvo/Projects/simtoolreal/.venv/bin:$PATH"
hash -r

export PYTHONPATH="/move/u/karenvo/Projects/diffusion_policy:${PYTHONPATH:-}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=all,graphics,utility,compute

# (optional but helpful for debugging)
command -v python
command python -V
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

command python scripts/test_pick_place_release_debug.py \
  --collection-type pick_place_release \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down \
  --rollouts 2 \
  --gif-fps 10 \
  --output-dir data/debug_pick_place_release_job
