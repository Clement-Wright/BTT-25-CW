#!/usr/bin/env bash
# Compare two pinned CUDA scorers on the same completed validation candidate set.
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
old="$project/workspace"
new="$project/v2"
python="$project/environments/baseline/bin/python"
cd "$old"
export PYTHONPATH="$new/src:$new/vendor/lightbeam-deps"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export BTT_STOP_FILE="$project/slurm/v2-stop-${SLURM_JOB_ID:?}"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
sources=()
for path in "$@"; do sources+=(--source "$path"); done
additional=()
if [[ -f "$new/configs/search_expanded_v2.json" ]]; then
    additional=(--search-registration "$new/configs/search_expanded_v2.json")
fi
registration="$new/configs/fixed_candidates_v2.json"
model="$new/assets/qwen3-1.7b-base"
prefix=v2-fixed
deadline=2850
if [[ ${BTT_QWEN_4B_PRIOR:-} ]]; then
    registration="$new/configs/fixed_candidates_4b_v2.json"
    model="$new/assets/qwen3-4b-base"
    prefix=v2-fixed-4b
    deadline=5100
    additional+=(--prior "$BTT_QWEN_4B_PRIOR")
fi
"$python" -m btt25.fixed_candidates_v2 --config configs/local.yaml --registration "$registration" \
    --distil assets/local/distilgpt2 --qwen "$model" "${sources[@]}" "${additional[@]}" \
    --output "$new/runs/$prefix-$SLURM_JOB_ID" --max-seconds "$deadline" &
child=$!
while true; do
    code=0
    wait "$child" || code=$?
    if ! kill -0 "$child" 2>/dev/null; then exit "$code"; fi
done
