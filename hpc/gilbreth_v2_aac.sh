#!/usr/bin/env bash
# Evaluate a single predeclared external AAC four-gram arm.
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
old="$project/workspace"
new="$project/v2"
python="$project/environments/baseline/bin/python"
decoder="$project/environments/decoder/bin/python"
root="$new/assets/imagineville-dec19-small"
condition=v2-s7-aac-fourgram
registration="$new/configs/search_v2.json"
beam=100
fastlight=v0.0.7
if [[ ${1:-} ]]; then
    condition=v2-s8-repaired-aac-250
    registration="$new/configs/search_expanded_v2.json"
    beam=250
    fastlight=unlimited-label-patch
    target="$new/cache/flashlight-unlimited-$1/site-packages"
    test -d "$target"
fi
run="$new/runs/$condition-${SLURM_JOB_ID:?}"
url=https://data.imagineville.org/lm/dec19
cd "$old"
export PYTHONPATH="$new/src"
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH="$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
if [[ ${target:-} ]]; then
    export PYTHONPATH="$target:$PYTHONPATH"
    prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
    export LD_LIBRARY_PATH="$prefix/lib:$LD_LIBRARY_PATH"
fi
export BTT_FLASHLIGHT_VARIANT="$fastlight"
export BTT_STOP_FILE="$project/slurm/v2-stop-$SLURM_JOB_ID"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
wait_stage() {
    "$@" &
    local child=$! code
    while true; do
        code=0
        wait "$child" || code=$?
        if ! kill -0 "$child" 2>/dev/null; then return "$code"; fi
    done
}
mkdir -p "$root" "$run"
started=$SECONDS
for name in lm_dec19_small_4gram.kenlm.gz vocab_lower_100k.txt; do
    if [[ ! -s "$root/$name" ]]; then
        curl --fail --location --retry 2 --connect-timeout 15 --max-time 360 "$url/$name" -o "$root/$name.partial"
        mv "$root/$name.partial" "$root/$name"
    fi
done
lm="$root/dec19-small-4gram.binary"
if [[ ! -s "$lm" ]]; then
    gzip -dc "$root/lm_dec19_small_4gram.kenlm.gz" > "$lm.partial"
    mv "$lm.partial" "$lm"
fi
{
    printf 'official_source=%s\nlicense=CC-BY-4.0\nsetup_wall_seconds=%s\n' "$url" "$((SECONDS-started))"
    sha256sum "$lm" "$root/lm_dec19_small_4gram.kenlm.gz" "$root/vocab_lower_100k.txt"
} > "$run/acquisition.txt"
lexicon="$old/assets/local/lexicon.txt"
wait_stage "$python" -m btt25.benchmark --config configs/local.yaml --output "$run/pilot" \
    --decoder-python "$decoder" --lexicon "$lexicon" --kenlm-model "$lm" --lm-weight 2 \
    --logit-cache runs/baseline-val-a100 --beam "$beam" --nbest 50 --passes 3 --trials 96 \
    --pilot-seconds 900 --max-seconds 1300
"$python" - "$run/pilot" "$beam" <<'PY'
import json
from pathlib import Path
import sys

pilot = Path(sys.argv[1])
evidence = json.loads((pilot / "metrics.json").read_text())
resources = json.loads((pilot / "resources.json").read_text())
row = evidence["by_beam"][sys.argv[2]]
forecast = 2 * 1426 * max(g["latency_seconds"]["p95"] for g in row["by_length"].values()) + 60
if (evidence["state"] != "complete" or evidence["cold_setup"]["sustained_warm_s"] < 900
    or row["failures"] or resources["outcome"] != "complete" or forecast > 690):
    raise ValueError("AAC search failed sustained admission")
(pilot / "admission.json").write_text(json.dumps({"forecast_seconds": forecast}))
PY
wait_stage "$decoder" "$new/scripts/decode_worker.py" --manifest assets/local/data_manifest.json \
    --logit-run runs/baseline-val-a100 --lexicon "$lexicon" --output-dir "$run/decode" \
    --kenlm-model "$lm" --lm-weight 2 --nbest 50 --beam "$beam" --max-seconds 690 > "$run/decode.log" 2>&1
wait_stage "$decoder" "$new/scripts/audit_trie.py" --lexicon "$lexicon" --output-dir "$run/lexicon" > "$run/lexicon.log" 2>&1
wait_stage "$python" -m btt25.search_diagnostic --registration "$registration" \
    --condition "$condition" --data-manifest assets/local/data_manifest.json --split-manifest assets/local/splits.json \
    --corpus-metadata assets/local/corpus_metadata.csv --predictions "$run/decode/predictions.jsonl" \
    --candidates "$run/decode/candidates.jsonl" --candidate-cache "$run/decode/cache_manifest.json" \
    --lexicon "$run/lexicon/searchable_lexicon.txt" --lm-vocab "$root/vocab_lower_100k.txt" --output "$run/dev-tune"
