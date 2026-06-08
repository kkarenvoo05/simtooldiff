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

cd /move/u/chrzhang/simtooldiff

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

# python scripts/stage5_multi_object_driver.py \
#   --collection-type pick_place_release \
#   --split train \
#   --output-zarr data/anchored_pick_place_release_train.zarr \
#   --per-object-transitions 15000 \
#   --num-envs 1 \
#   --horizon 325 \
#   --xy-range 0.10 \
#   --start-z-offset 0.0 \
#   --lift-height 0.20 \
#   --lateral-offset-range 0.15 \
#   --place-height 0.02 \
#   --place-hold-goals 10 \
#   --table-x-half-extent 0.30 \
#   --table-x-inset-margin 0.06 \
#   --min-effective-transport 0.05 \
#   --variant noisy_clean \
#   --noise-strategy anchored_recovery \
#   --noise-scale 1.0 \
#   --anchored-branches-per-rollout 3 \
#   --anchored-perturb-steps 3 \
#   --anchored-recovery-steps 15 \
#   --save-preview-every 0 \
#   --seed 1000

# python scripts/stage5_multi_object_driver.py \
#   --collection-type pick_place_release \
#   --split train \
#   --output-zarr data/anchored_pick_place_release_train.zarr \
#   --per-object-transitions 15000 \
#   --num-envs 1 \
#   --horizon 325 \
#   --xy-range 0.10 \
#   --start-z-offset 0.0 \
#   --lift-height 0.20 \
#   --lateral-offset-range 0.15 \
#   --place-height 0.02 \
#   --place-hold-goals 10 \
#   --table-x-half-extent 0.30 \
#   --table-x-inset-margin 0.06 \
#   --min-effective-transport 0.05 \
#   --variant noisy_clean \
#   --noise-strategy anchored_recovery \
#   --noise-scale 1.0 \
#   --anchored-branches-per-rollout 3 \
#   --anchored-perturb-steps 3 \
#   --anchored-recovery-steps 15 \
#   --seed 0

# python scripts/eval_diffusion_policy.py \
#   --checkpoint /move/u/karenvo/Projects/diffusion_policy/data/outputs/2026.05.14/11.21.05_train_diffusion_unet_hybrid_simtool_image_state29/checkpoints/epoch=0050-val_loss=0.0951.ckpt \
#   --split train \
#   --episodes-per-object 32 \
#   --num-envs 8 \
#   --output-json data/diffusion_eval/epoch0050_noisy_train.json

# python scripts/eval_diffusion_policy.py \
#   --checkpoint /move/u/karenvo/Projects/diffusion_policy/data/outputs/2026.05.29/06.04.30_train_diffusion_unet_hybrid_simtool_image_state29/checkpoints/epoch=0010-val_loss=0.0529.ckpt \
#   --split ood \
#   --episodes-per-object 32 \
#   --num-envs 8 \
#   --output-json data/diffusion_eval/eval_pick_place_release_noisy_rollout10.json \
#   --collection-type pick_place_release

python scripts/eval_diffusion_policy_pick_place_release.py \
  --checkpoint /move/u/karenvo/Projects/diffusion_policy/data/outputs/2026.05.29/06.04.30_train_diffusion_unet_hybrid_simtool_image_state29/checkpoints/epoch=0010-val_loss=0.0529.ckpt \
  --split ood \
  --episodes-per-object 32 \
  --output-json data/diffusion_eval/new_eval_pick_place_release_noisy_rollout10_ood_32.json

# python scripts/test_pick_place_release_debug.py \
#   --collection-type pick_place_release \
#   --object-category hammer \
#   --object-name claw_hammer \
#   --task-name swing_down \
#   --rollouts 10 \
#   --release-steps 1 \
#   --variant clean \
#   --noise-scale 4 \
#   --gif-fps 10 \
#   --output-dir data/debug_pick_place_release_clean

# python scripts/stage5_multi_object_driver.py \
#   --collection-type pick_place_release \
#   --split train \
#   --output-zarr data/pick_place_release_noisy_rollout1.zarr \
#   --per-object-transitions 15000 \
#   --num-envs 1 \
#   --horizon 325 \
#   --xy-range 0.10 \
#   --start-z-offset 0.0 \
#   --lift-height 0.20 \
#   --lateral-offset-range 0.15 \
#   --place-height 0.02 \
#   --place-hold-goals 10 \
#   --table-x-half-extent 0.30 \
#   --table-x-inset-margin 0.06 \
#   --min-effective-transport 0.05 \
#   --variant noisy_clean \
#   --seed 1000 \
#   --noise-strategy continuous_ou \
#   --noise-scale 4 \
#   --rollouts-per-init 1

#   --anchored-branches-per-rollout 3 \
#   --anchored-perturb-steps 3 \
#   --anchored-recovery-steps 15 \
