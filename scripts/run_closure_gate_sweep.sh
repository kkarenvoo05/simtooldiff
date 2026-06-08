#!/bin/bash
#SBATCH --job-name=collect_pickup_policy
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/closure_gate_%j.out
#SBATCH --error=logs/slurm/closure_gate_%j.err


set -euo pipefail

mkdir -p logs/slurm

cd /move/u/karenvo/Projects/simtoolreal   

echo "SCRIPT_MARKER=2026-05-21-stage5-path-fix-v1"
echo "SCRIPT_FILE=$(readlink -f "$0")"

# ========================
# ACTIVATE PROJECT ENV
# ========================
# Activate the conda env with a local Python 3.8 interpreter.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /move/u/karenvo/Projects/simtoolreal/.conda_env

export PYTHONPATH="/move/u/karenvo/Projects/diffusion_policy:${PYTHONPATH:-}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=all,graphics,utility,compute

# (optional but helpful for debugging)
export LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
python -V
which python
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# ========================
python scripts/stage5_multi_object_driver.py \
  --collection-type pick_place_release \
  --split train \
  --output-zarr data/pick_place_release_combined_closure.zarr \
  --per-object-transitions 15000 \
  --num-envs 1 \
  --horizon 325 \
  --xy-range 0.10 \
  --start-z-offset 0.0 \
  --lift-height 0.20 \
  --lateral-offset-range 0.15 \
  --place-height 0.02 \
  --place-hold-goals 10 \
  --table-x-half-extent 0.30 \
  --table-x-inset-margin 0.06 \
  --min-effective-transport 0.05 \
  --variant noisy_clean \
  --seed 1000 \
  --noise-strategy continuous_ou \
  --noise-scale 4 \
  --rollouts-per-init 1 \
  --noise-phase-gating proximity \
  --closure-proximity-threshold 0.08 \
  --closure-noise-scale 0.0 \
  --closure-groups wrist,thumb,index,middle,ring,pinky \
  --closure-window-padding 5 \
  --finger-noise-multiplier 0.5

# Gate kill: 15695002
#   --noise-phase-gating proximity \
#   --closure-proximity-threshold 0.08 \
#   --closure-noise-scale 0.0 \
#   --closure-groups wrist,thumb,index,middle,ring,pinky \
#   --closure-window-padding 5

# Gate reduce: 15695005
#   --noise-phase-gating proximity \
#   --closure-proximity-threshold 0.08 \
#   --closure-noise-scale 0.25 \
#   --closure-groups wrist,thumb,index,middle,ring,pinky \
#   --closure-window-padding 5

# Palmz kill: 15695007
#   --noise-phase-gating palm_z \
#   --closure-palm-z-threshold 0.66 \
#   --closure-noise-scale 0.0 \
#   --closure-groups wrist,thumb,index,middle,ring,pinky \
#   --closure-window-padding 5

# Blunt finger: 15695055
#   --noise-phase-gating off \
#   --finger-noise-multiplier 0.3

# Blunt finger wrist: 15695059
#   --noise-phase-gating off \
#   --finger-noise-multiplier 0.3 \
#   --wrist-noise-multiplier 0.3

# Combined: 15699520
#   --noise-phase-gating proximity \
#   --closure-proximity-threshold 0.08 \
#   --closure-noise-scale 0.0 \
#   --closure-groups wrist,thumb,index,middle,ring,pinky \
#   --closure-window-padding 5 \
#   --finger-noise-multiplier 0.5