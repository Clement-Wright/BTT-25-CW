#!/usr/bin/env bash
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
campaign="$project/v4-rate-20261127"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$campaign/src"
study=${1:?Supply the completed two-rate study directory}
out="$campaign/runs/laptop-two-rate-export-${SLURM_JOB_ID:?}"
"$project/environments/baseline/bin/python" "$campaign/scripts/export_rate_delta.py" "$study" "$out"
