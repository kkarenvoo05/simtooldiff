#!/bin/bash
#SBATCH --job-name=stage5_collect_noisy_test
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/stage5_collect_%j.out
#SBATCH --error=logs/slurm/stage5_collect_%j.err

mkdir -p logs/slurm

cd /move/u/karenvo/Projects/simtoolreal   

# ========================
# ACTIVATE VENV
# ========================
source .venv/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=all,graphics,utility,compute

# (optional but helpful for debugging)
which python
python -V

# ========================
# STAGE 5 DATA COLLECTION
# ========================

# python -u scripts/stage5_collect_dataset.py \
#   --num-envs 1 \
#   --target-transitions 50000 \
#   --xy-range 0.10 \
#   --output-zarr data/stage5_clean_clean_v1_xy01.zarr \
#   --device cuda \
#   --resume \
#   --save-preview-every 25

python scripts/stage5_collect_dataset.py \
  --num-envs 4 \
  --target-transitions 1000 \
  --max-steps 2000 \
  --output-zarr data/stage5_smoke.zarr

# python -u scripts/stage5_collect_noisy_dataset.py \
#   --num-envs 32 \
#   --target-transitions 50000 \
#   --xy-range 0.10 \
#   --variant noisy_clean \
#   --noise-level 0.02 \
#   --output-zarr data/stage5_noisy_clean_v1_xy01.zarr \
#   --device cuda \
#   --resume \
#   --save-preview-every 25

# ========================
# VERIFY DATASET
# ========================

# python stage5_verify_dataset.py \
#     data/stage5_claw_hammer_v1.zarr