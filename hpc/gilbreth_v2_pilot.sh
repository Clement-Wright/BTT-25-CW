#!/usr/bin/env bash
# Stage the v2 tree beside v1, then submit this file from the Gilbreth login shell.
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
old="$project/workspace"
new="$project/v2"
python="$project/environments/baseline/bin/python"
decoder="$project/environments/decoder/bin/python"
cd "$old"
export PYTHONPATH="$new/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
export BTT_STOP_FILE="$project/slurm/v2-stop-${SLURM_JOB_ID:?}"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
run="$new/runs/v2-pilot-${SLURM_JOB_ID}"
mkdir -p "$new/runs"
"$python" -m btt25.benchmark --config configs/local.yaml --output "$run" \
    --decoder-python "$decoder" --lexicon assets/local/lexicon.txt \
    --kenlm-model runs/compact-a100/lm/train-3gram.binary --lm-weight 2 \
    --logit-cache runs/baseline-val-a100 --beam 50 100 250 --nbest 50 \
    --passes 3 --trials 96 --pilot-seconds 900 --max-seconds 1140 &
child=$!
while true; do
    code=0
    wait "$child" || code=$?
    if ! kill -0 "$child" 2>/dev/null; then exit "$code"; fi
done
