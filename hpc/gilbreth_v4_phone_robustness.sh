#!/usr/bin/env bash
#SBATCH --job-name=btt-word-risk
#SBATCH --account=ieee-smcs
#SBATCH --partition=a10,a30,a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=9000M
#SBATCH --time=00:35:00
set -eu -o pipefail
umask 077
project="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}/btt-25-cw"
campaign="$project/v4-rate-20261127"
job="${SLURM_JOB_ID:?}"
root="$campaign/runs/phone-robustness-$job"
mkdir -p "$root"
cd "$project/workspace"
decoder="$project/environments/decoder/bin/python"
python="$project/environments/baseline/bin/python"
prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
export LD_LIBRARY_PATH="$prefix/lib:$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$project/v3-smoothing-20261127/src"
export BTT_FLASHLIGHT_VARIANT=unlimited-label-patch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BTT_STOP_FILE="$root/stop"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
scontrol show job -o "$job" > "$root/scheduler-start.txt"
"$python" "$campaign/phone-robustness-code/src/btt25/phone_robustness.py" \
    --config configs/local.yaml --lexicon assets/local/lexicon.txt \
    --kenlm-model runs/compact-a100/lm/train-3gram.binary --cache runs/baseline-val-a100 \
    --control "$project/v2/runs/v2-search-attempt-12523602/v2-s6-repaired-250/decode/predictions.jsonl" \
    --decoder-python "$decoder" --decoder-module-path "$project/v2/cache/flashlight-unlimited-12523602/site-packages" \
    --output "$root/study"
