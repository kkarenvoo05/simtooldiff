#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

submit_job() {
  local run_name="$1"
  local anchor_zarr="$2"
  shift 2

  env \
  RUN_NAME="${run_name}" \
  ANCHOR_ZARR="${ROOT}/${anchor_zarr}" \
  "$@" \
  sbatch --job-name="dp_${run_name}" scripts/run_diffusion_train_clean_plus.sbatch
}

submit_job anch_clean_plus_p4_r24_s1_t05 data/anch_pickup_p4_r24_s1_t05.zarr "$@"
submit_job anch_clean_plus_p4_r24_s05_t05 data/anch_pickup_p4_r24_s05_t05.zarr "$@"
submit_job anch_clean_plus_p3_r15_s1_t015 data/anch_pickup_p3_r15_s1_t015.zarr "$@"
submit_job anch_clean_plus_p6_r24_s1_t05 data/anch_pickup_p6_r24_s1_t05.zarr "$@"
