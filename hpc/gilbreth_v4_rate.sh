#!/usr/bin/env bash
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
campaign="$project/v4-rate-20261127"
group="${1:?Specify clock or a training group}"
shift
out="$campaign/runs/$group-${SLURM_JOB_ID:?}"
mkdir -p "$out"
cd "$project/workspace"
export PYTHONPATH="$campaign/src"
export BTT_FLASHLIGHT_VARIANT=unlimited-label-patch
export BTT_STOP_FILE="$out/stop"
export MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
decoder="$project/environments/decoder/bin/python"
python="$project/environments/baseline/bin/python"
prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
export LD_LIBRARY_PATH="$prefix/lib:$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
scontrol show job -o "$SLURM_JOB_ID" > "$out/scheduler-start.txt"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
args=(--config configs/local.yaml --lexicon assets/local/lexicon.txt
      --kenlm-model runs/compact-a100/lm/train-3gram.binary --decoder-python "$decoder"
      --decoder-module-path "$project/v2/cache/flashlight-unlimited-12523602/site-packages")
case "$group" in
    clock) "$python" -m btt25.clock_diagnostic "${args[@]}" --output "$out/clock" & ;;
    bench*)
        # Each block starts a fresh process; one persistent model serves its 24 requests.
        # Rotate which condition goes first so timing does not follow one fixed ordering.
        run_benchmark() {
            local entries=("$@") pass shift_by j entry name source
            for pass in 0 1 2; do
                shift_by=$((pass * ${#entries[@]} / 3))
                for ((j=0; j<${#entries[@]}; j++)); do
                    entry="${entries[$(((j + shift_by) % ${#entries[@]}))]}"
                    name="${entry%%=*}"
                    source="${entry#*=}"
                    "$python" -m btt25.rate_benchmark "${args[@]}" \
                        --output "$out/benchmark/$name/$pass" --pass-index "$pass" \
                        --condition "$name" --study-dir "$source" || return $?
                done
            done
        }
        run_benchmark "$@" & ;;
    *) "$python" -m btt25.v4_rate_study "${args[@]}" --corpus-metadata assets/local/corpus_metadata.csv \
        --lm-weight 2 --output "$out/study" "$@" & ;;
esac
child=$!
while true; do
    code=0
    wait "$child" || code=$?
    if ! kill -0 "$child" 2>/dev/null; then exit "$code"; fi
done
