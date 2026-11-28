#!/usr/bin/env bash
#SBATCH --job-name=btt-eeg-setup
#SBATCH --account=ieee-smcs
#SBATCH --partition=a10,a30,a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=00:22:00
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}/btt-25-cw"
run="$project/v4-rate-20261127"
out="$run/runs/eeg-setup-${SLURM_JOB_ID:?}"
mkdir "$out"
scontrol show job -o "$SLURM_JOB_ID" > "$out/scheduler-start.txt"
"$project/environments/baseline/bin/python" "$run/cadence-code-v1/scalp_eeg_setup.py" \
    --root "$project/datasets/ds003626-2.0.0/sub-01" --output "$out/setup"
