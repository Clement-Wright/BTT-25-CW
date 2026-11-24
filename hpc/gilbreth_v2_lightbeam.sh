#!/usr/bin/env bash
# Compare the pinned private LightBeam runtime with and without periodic Qwen fusion.
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
old="$project/workspace"
stage="$project/v2"
source="$stage/vendor/brainaudio"
python="$project/environments/baseline/bin/python"
cd "$old"
export PYTHONPATH="$stage/src:$source/src:$stage/vendor/lightbeam-deps"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export BTT_STOP_FILE="$project/slurm/v2-stop-${SLURM_JOB_ID:?}"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
out="$stage/runs/v2-lightbeam-${SLURM_JOB_ID}"
"$python" -m btt25.lightbeam_adapter --registration "$stage/configs/lightbeam_v2.json" \
    --config configs/local.yaml --upstream "$source" --logits runs/baseline-val-a100 \
    --kenlm runs/compact-a100/lm/train-3gram.binary --lexicon assets/local/lexicon.txt \
    --model-dir "$stage/assets/qwen3-1.7b-base" --output "$out" --max-seconds 1580 &
child=$!
while true; do
    code=0
    wait "$child" || code=$?
    if ! kill -0 "$child" 2>/dev/null; then exit "$code"; fi
done
