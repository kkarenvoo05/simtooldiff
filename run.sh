#!/bin/bash
#SBATCH --account=move
#SBATCH --partition=move --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:a5000:1

#SBATCH --job-name="eval_diffusion_policy"
#SBATCH --output=logs/eval_diffusion_policy-%j.out
#SBATCH --error=logs/eval_diffusion_policy-%j.err

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# not needed if already in the conda environment when running this script
source /nlp/scr/chrzhang/miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH="/move/u/chrzhang/conda/envs/str/lib:$LD_LIBRARY_PATH"
conda activate str

# ------ TRAIN ------

# i used seed 0 the first time i ran this, and then 1000 for the second run because i wanted to double the amount of data
# python scripts/stage5_multi_object_driver.py \
#     --split train \
#     --output-zarr data/stage5_train.zarr \
#     --per-object-transitions 15000 \
#     --num-envs 8 \
#     --horizon 250 \
#     --xy-range 0.10 \
#     --seed 1000 \
#     2>&1 | tee -a data/stage5_train_run.log

# python scripts/stage5_multi_object_driver.py \
#     --split ood \
#     --output-zarr data/stage5_ood.zarr \
#     --per-object-transitions 2000 \
#     --num-envs 8 \
#     --horizon 250 \
#     --xy-range 0.10 \
#     --seed 100 \
#     2>&1 | tee data/stage5_ood_run.log

# ------ EVAL ------

# python scripts/eval_diffusion_policy.py \
#     --checkpoint /move/u/chrzhang/diffusion_policy/data/outputs/2026.05.06/22.17.26_train_diffusion_unet_hybrid_simtool_image_state29/checkpoints/epoch=0050-val_loss=0.0465.ckpt \
#     --split train \
#     --episodes-per-object 32 \
#     --num-envs 8 \
#     --output-json data/diffusion_eval/epoch0050_train.json

python scripts/eval_diffusion_policy.py \
    --checkpoint /move/u/chrzhang/diffusion_policy/data/outputs/2026.05.06/22.17.26_train_diffusion_unet_hybrid_simtool_image_state29/checkpoints/epoch=0050-val_loss=0.0465.ckpt \
    --split ood \
    --episodes-per-object 32 \
    --num-envs 8 \
    --output-json data/diffusion_eval/epoch0050_ood.json

echo "Done"
h=$((SECONDS / 3600))
m=$((SECONDS % 3600 / 60))
s=$((SECONDS % 60))
echo "Elapsed wall time: ${SECONDS}s (${h}h ${m}m ${s}s)"
