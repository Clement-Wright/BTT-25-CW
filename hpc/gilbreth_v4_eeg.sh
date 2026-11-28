#!/usr/bin/env bash
#SBATCH --job-name=btt-four-eeg
#SBATCH --account=ieee-smcs
#SBATCH --partition=a10,a30,a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=9000M
#SBATCH --time=00:18:00
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}/btt-25-cw"
run="$project/v4-rate-20261127"
out="$run/runs/eeg-four-${SLURM_JOB_ID:?}"
mkdir "$out"
export PYTHONPATH="$run/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
scontrol show job -o "$SLURM_JOB_ID" > "$out/scheduler-start.txt"
"$project/environments/baseline/bin/python" "$run/cadence-code-v1/scalp_eeg.py" \
    --root "$project/datasets/ds003626-2.0.0/sub-01" --output "$out/study" \
    --registration "$run/cadence-code-v1/SCALP_EEG_CHECK.md" \
    --acquisition "$run/runs/eeg-setup-${1:?Pass verified acquisition job}/setup/acquisition.json"
