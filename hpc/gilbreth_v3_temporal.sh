#!/usr/bin/env bash
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
campaign="${1:?Supply campaign source root}"
group="${2:?Use smoothing, sampling, reference, reference-spike, pilot-control, pilot-fusion or pilot-dense}"
workspace="$project/workspace"
decoder="$project/environments/decoder/bin/python"
python="$project/environments/baseline/bin/python"
job="${SLURM_JOB_ID:?}"
root="$campaign/runs/$group-$job"
mkdir -p "$root"
cd "$workspace"
code_root="${3:-$campaign}"
export PYTHONPATH="$code_root/src"
export BTT_FLASHLIGHT_VARIANT=unlimited-label-patch
export BTT_STOP_FILE="$root/stop"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
export LD_LIBRARY_PATH="$prefix/lib:$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
scontrol show job -o "$job" > "$root/scheduler-start.txt"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT

run() {
    local method="$1" stride="$2" seconds="$3" max_seconds="$4" child code
    "$python" -m btt25.temporal_study --config configs/local.yaml \
        --lexicon assets/local/lexicon.txt --kenlm-model runs/compact-a100/lm/train-3gram.binary \
        --corpus-metadata assets/local/corpus_metadata.csv \
        --decoder-python "$decoder" --output "$root/$method-s$stride" \
        --decoder-module-path "$project/v2/cache/flashlight-unlimited-12523602/site-packages" \
        --method "$method" --stride "$stride" --pilot-seconds "$seconds" --max-seconds "$max_seconds" \
        --lm-weight 2 &
    child=$!
    while true; do
        code=0
        wait "$child" || code=$?
        if ! kill -0 "$child" 2>/dev/null; then return "$code"; fi
    done
}

if [[ "$group" == smoothing ]]; then
    run baseline 4 0 420
    run raw 4 0 420
    run gaussian-1 4 0 420
    run box-5 4 0 420
elif [[ "$group" == sampling ]]; then
    run baseline 2 0 420
    run baseline 1 0 420
    run coarse-40 4 0 420
    run coarse-80 4 0 420
elif [[ "$group" == reference || "$group" == reference-spike ]]; then
    alignment=dwell
    if [[ "$group" == reference-spike ]]; then alignment=spike; fi
    "$python" -m btt25.reference_phone_diagnostic \
        --data-manifest assets/local/data_manifest.json --split-manifest assets/local/splits.json \
        --corpus-metadata assets/local/corpus_metadata.csv \
        --training-unigrams "$campaign/assets/train_unigrams.json" \
        --neural-predictions runs/baseline-val-a100/predictions.jsonl \
        --word-predictions "$project/v2/runs/v2-search-attempt-12523602/v2-s6-repaired-250/decode/predictions.jsonl" \
        --lexicon assets/local/lexicon.txt --kenlm-model runs/compact-a100/lm/train-3gram.binary \
        --decoder-python "$decoder" --decoder-module-path "$project/v2/cache/flashlight-unlimited-12523602/site-packages" \
        --lm-weight 2 --beam 250 --nbest 50 --alignment "$alignment" --output "$root/diagnostic"
elif [[ "$group" == pilot-control ]]; then
    run baseline 4 900 1500
elif [[ "$group" == pilot-fusion ]]; then
    run channel-residual 4 900 1500
elif [[ "$group" == pilot-dense ]]; then
    run dense-residual 4 900 1500
else
    exit 2
fi
