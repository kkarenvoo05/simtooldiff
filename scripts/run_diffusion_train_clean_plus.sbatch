#!/usr/bin/env bash
#SBATCH --partition=move
#SBATCH --account=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/dptrain_%x_%j.out
#SBATCH --error=logs/slurm/dptrain_%x_%j.err

set -euo pipefail

SIMTOOL_ROOT="${SIMTOOL_ROOT:-/move/u/caydengu/Projects/cs224r/simtooldiff}"
DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-/move/u/caydengu/Projects/cs224r/diffusion_policy}"
CONDA_ENV="${CONDA_ENV:-dp}"
CLEAN_ZARR="${CLEAN_ZARR:-/move/u/chrzhang/simtooldiff/data/stage5_train.zarr}"
ANCHOR_ZARR="${ANCHOR_ZARR:?Set ANCHOR_ZARR to the anchored recovery zarr path}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME for this training run}"

STATE_DIM="${STATE_DIM:-29}"
IMAGE_H="${IMAGE_H:-192}"
IMAGE_W="${IMAGE_W:-256}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.02}"
CLEAN_REPEAT="${CLEAN_REPEAT:-1}"
ANCHOR_REPEAT="${ANCHOR_REPEAT:-1}"
CLEAN_COPY_TO_MEMORY="${CLEAN_COPY_TO_MEMORY:-false}"
ANCHOR_COPY_TO_MEMORY="${ANCHOR_COPY_TO_MEMORY:-true}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DIFFUSION_POLICY_ROOT}/data/outputs_anchored}"
WANDB_MODE="${WANDB_MODE:-offline}"

source /move/u/caydengu/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${SIMTOOL_ROOT}:${DIFFUSION_POLICY_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"

mkdir -p logs/slurm "${OUTPUT_ROOT}"

echo "RUN_NAME=${RUN_NAME}"
echo "CLEAN_ZARR=${CLEAN_ZARR}"
echo "ANCHOR_ZARR=${ANCHOR_ZARR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "REPEAT_FACTORS=[${CLEAN_REPEAT},${ANCHOR_REPEAT}]"
echo "COPY_TO_MEMORY=[${CLEAN_COPY_TO_MEMORY},${ANCHOR_COPY_TO_MEMORY}]"

cd "${DIFFUSION_POLICY_ROOT}"
python train.py --config-name=train_diffusion_unet_hybrid_workspace task=simtool_image \
  task.dataset._target_=simtooldiff_diffusion_dataset.ConcatSimtoolImageDataset \
  task.dataset.zarr_path="${CLEAN_ZARR}" \
  +task.dataset.zarr_paths="[${CLEAN_ZARR},${ANCHOR_ZARR}]" \
  +task.dataset.repeat_factors="[${CLEAN_REPEAT},${ANCHOR_REPEAT}]" \
  +task.dataset.copy_to_memory="[${CLEAN_COPY_TO_MEMORY},${ANCHOR_COPY_TO_MEMORY}]" \
  task.dataset.state_dim="${STATE_DIM}" \
  task.dataset.val_ratio="${VAL_RATIO}" \
  task.dataset.seed="${SEED}" \
  task.image_shape="[3,${IMAGE_H},${IMAGE_W}]" \
  dataloader.batch_size="${BATCH_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  val_dataloader.batch_size="${BATCH_SIZE}" \
  val_dataloader.num_workers="${NUM_WORKERS}" \
  training.seed="${SEED}" \
  training.num_epochs="${EPOCHS}" \
  training.checkpoint_every=10 \
  training.rollout_every=1000000 \
  checkpoint.topk.monitor_key=val_loss \
  checkpoint.topk.mode=min \
  'checkpoint.topk.format_str="epoch={epoch:04d}-val_loss={val_loss:.4f}.ckpt"' \
  logging.project=diffusion_policy_simtool \
  logging.mode="${WANDB_MODE}" \
  logging.name="${RUN_NAME}" \
  exp_name="${RUN_NAME}" \
  hydra.run.dir="${OUTPUT_ROOT}/${RUN_NAME}"
