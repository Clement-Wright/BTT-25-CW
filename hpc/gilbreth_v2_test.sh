#!/usr/bin/env bash
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
old="$project/workspace"
new="$project/v2"
report="${1:?Provide the original v2 report job ID}"
repair="${2:?Provide the Flashlight repair build job ID}"
decoder="$project/environments/decoder/bin/python"
cd "$old"
prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
export PYTHONPATH="$new/src:$new/vendor/lightbeam-deps"
export LD_LIBRARY_PATH="$prefix/lib:$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export BTT_STOP_FILE="$project/slurm/v2-stop-${SLURM_JOB_ID:?}"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
"$project/environments/baseline/bin/python" -m btt25.export_v2_test --config configs/local.yaml \
    --freeze "$new/runs/v2-report-$report/freeze.json" --lexicon assets/local/lexicon.txt --decoder-python "$decoder" \
    --decoder-module-path "$new/cache/flashlight-unlimited-$repair/site-packages" --distil assets/local/distilgpt2 \
    --output "$new/runs/v2-test-$SLURM_JOB_ID" --max-seconds 1300 &
child=$!
while true; do
    code=0
    wait "$child" || code=$?
    if ! kill -0 "$child" 2>/dev/null; then exit "$code"; fi
done
