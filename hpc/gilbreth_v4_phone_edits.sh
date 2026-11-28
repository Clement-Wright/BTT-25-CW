#!/usr/bin/env bash
#SBATCH --job-name=btt-word-propose
#SBATCH --account=ieee-smcs
#SBATCH --partition=a10,a30,a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=9000M
#SBATCH --time=00:17:00
set -eu -o pipefail
umask 077
project="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}/btt-25-cw"
campaign="$project/v4-rate-20261127"
root="$campaign/runs/phone-edits-${SLURM_JOB_ID:?}"
code="$campaign/phone-robustness-code"
mkdir -p "$root"
scontrol show job -o "$SLURM_JOB_ID" > "$root/scheduler-start.txt"
export PYTHONPATH="$code/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$project/workspace"
"$project/environments/baseline/bin/python" -m btt25.phone_edit_candidates \
    --config configs/local.yaml --registration "$code/configs/phone_edit_candidates_v4.json" \
    --arpa runs/compact-a100/lm/train-3gram.arpa --lexicon assets/local/lexicon.txt \
    --cache runs/baseline-val-a100 --unigrams "$project/v3-smoothing-20261127/assets/train_unigrams.json" \
    --candidates "$campaign/runs/phone-robustness-${1:?source job}/study/candidates.jsonl" --output "$root/study"
