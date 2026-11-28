#!/usr/bin/env bash
#SBATCH --job-name=btt-cached-priors
#SBATCH --account=ieee-smcs
#SBATCH --partition=a10,a30,a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=9000M
#SBATCH --time=00:15:00
set -eu -o pipefail
umask 077
project="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}/btt-25-cw"
campaign="$project/v4-rate-20261127"
job="${SLURM_JOB_ID:?}"
root="$campaign/runs/phone-priors-$job"
code="$campaign/phone-robustness-code"
mkdir -p "$root"
scontrol show job -o "$job" > "$root/scheduler-start.txt"
export PYTHONPATH="$code/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$project/workspace"
base="$project/v3-smoothing-20261127/runs"
"$project/environments/baseline/bin/python" "$code/scripts/compare_cached_phone_priors.py" \
    --config configs/local.yaml --acoustics "$campaign/runs/phone-robustness-${1:?source job}/study/candidates.jsonl" \
    --score "distilgpt2=$project/v2/runs/v2-fixed-12523725/v2-f1-distilgpt2/decode/candidates.jsonl" \
    --score "qwen36=$base/frontier-Qwen3.6-35B-A3B-12620738/candidate-scores.jsonl" \
    --score "qwen38=$base/frontier-Qwen3.8-27B-12621131/candidate-scores.jsonl" \
    --output "$root/study"
