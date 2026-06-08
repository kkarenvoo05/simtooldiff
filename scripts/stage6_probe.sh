#!/bin/bash
#SBATCH --job-name=collect_pickup_policy
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/stage6_%j.out
#SBATCH --error=logs/slurm/stage6_%j.err

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

python scripts/stage6_noise_consistency_probe.py \
  --object-category hammer --object-name claw_hammer --task-name swing_down \
  --num-initializations 1 \
  --rollouts-per-init 10 \
  --noise-scale 4 \
  --ou-theta 0.15 \
  --horizon 325 \
  --seed 0 \
  --output-dir data/stage6_probe/claw_hammer_4x