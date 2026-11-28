#!/usr/bin/env bash
#SBATCH --job-name=btt-replay-score
#SBATCH --account=ieee-smcs
#SBATCH --partition=a10,a30,a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=9000M
#SBATCH --time=00:16:00
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}/btt-25-cw"
campaign="$project/v4-rate-20261127"
out="$campaign/runs/cadence-score-${SLURM_JOB_ID:?}"
mkdir "$out"
export PYTHONPATH="$campaign/src"
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$project/workspace"
scontrol show job -o "$SLURM_JOB_ID" > "$out/scheduler-start.txt"
"$project/environments/baseline/bin/python" "$campaign/cadence-score-v1/v4_rate_replay_audit.py" \
    --config configs/local.yaml --corpus-metadata assets/local/corpus_metadata.csv \
    --blind "$campaign/runs/cadence-${1:?Blind trial job}/study" \
    --gru40 "$campaign/runs/gru40-${2:?GRU40 job}/study/gru40" \
    --two-rate "$campaign/runs/two_rate-${3:?Two-rate job}/study/two_rate" --output "$out/study"
