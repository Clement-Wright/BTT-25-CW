#!/usr/bin/env bash
set -euo pipefail
umask 077
p="${RCAC_SCRATCH:?}/btt-25-cw"
c="$p/v4-rate-20261127"
mode="${1:-strong}"
out="$c/runs/$mode-${SLURM_JOB_ID:?}"
mkdir -p "$out"
scontrol show job -o "$SLURM_JOB_ID" > "$out/scheduler-start.txt"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$c/src:$p/v2/vendor/lightbeam-deps"
export BTT_FLASHLIGHT_VARIANT=unlimited-label-patch BTT_STOP_FILE="$out/stop"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
decoder="$p/environments/decoder/bin/python"
prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
export LD_LIBRARY_PATH="$prefix/lib:$p/environments/kenlm/lib:$p/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
cd "$p/workspace"
common=(--config configs/local.yaml --output "$out/results" --freeze "$p/v2/runs/v2-report-12523758/freeze.json"
        --distil assets/local/distilgpt2 --lexicon assets/local/lexicon.txt --decoder-python "$decoder"
        --decoder-module-path "$p/v2/cache/flashlight-unlimited-12523602/site-packages")
if [[ "$mode" == alignment-benchmark ]]; then
    exec "$p/environments/baseline/bin/python" -m btt25.alignment_benchmark "${common[@]}" \
        --expected "$c/runs/phone-priors-12625459/study/predictions.jsonl"
fi
[[ "$mode" == strong ]]
arms=()
for name in gru80 gru40 two_rate; do
    matches=()
    for f in "$c"/runs/"$name"-*/study/"$name"/metrics.json; do
        if [[ -f "$f" ]] && jq -e '.status == "complete" and .updates == 2048' "$f" >/dev/null; then
            matches+=("$(dirname "$f")")
        fi
    done
    [[ ${#matches[@]} -eq 1 ]]
    arms+=("${matches[0]}")
done
exec "$p/environments/baseline/bin/python" -m btt25.strong_rate_decode "${common[@]}" "${arms[@]}"
