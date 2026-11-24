#!/usr/bin/env bash
# Run one admitted set of three cached-logit comparisons.
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
old="$project/workspace"
new="$project/v2"
python="$project/environments/baseline/bin/python"
decoder="$project/environments/decoder/bin/python"
mode="${1:?Use original or repaired}"
pilot="${2:?Supply completed pilot Slurm job ID}"
cd "$old"
export PYTHONPATH="$new/src"
export LD_LIBRARY_PATH="$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export BTT_STOP_FILE="$project/slurm/v2-stop-${SLURM_JOB_ID:?}"
export CUDA_VISIBLE_DEVICES=
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
pilot_dir="$new/runs/v2-pilot-$pilot"
receipt="$new/runs/v2-search-attempt-${SLURM_JOB_ID}"
mkdir -p "$receipt"
lexicon="$old/assets/local/lexicon.txt"
effective="$receipt/lexicon/searchable_lexicon.txt"
case "$mode" in
    original)
        first=1
        export BTT_FLASHLIGHT_VARIANT=v0.0.7
        ;;
    repaired)
        first=4
        export BTT_FLASHLIGHT_VARIANT=unlimited-label-patch
        build="$new/cache/flashlight-unlimited-${SLURM_JOB_ID}"
        wait_stage bash "$new/hpc/gilbreth_v2_flashlight.sh" "$build"
        export PYTHONPATH="$build/site-packages:$PYTHONPATH"
        prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
        export LD_LIBRARY_PATH="$prefix/lib:$LD_LIBRARY_PATH"
        cp "$build/build_receipt.txt" "$receipt/native-build.txt"
        pilot_dir="$receipt/repaired-pilot"
        wait_stage "$python" -m btt25.benchmark --config configs/local.yaml --output "$pilot_dir" \
            --decoder-python "$decoder" --lexicon "$lexicon" \
            --kenlm-model runs/compact-a100/lm/train-3gram.binary --lm-weight 2 \
            --logit-cache runs/baseline-val-a100 --beam 50 100 250 --nbest 50 \
            --passes 3 --trials 96 --pilot-seconds 900 --max-seconds 1140
        ;;
    *) exit 2 ;;
esac
"$python" - "$pilot_dir" "$receipt" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

pilot, receipt = map(Path, sys.argv[1:])
metrics = json.loads((pilot / "metrics.json").read_text())
resources = json.loads((pilot / "resources.json").read_text())
if (metrics["state"] != "complete" or metrics["cold_setup"]["sustained_warm_s"] < 900
        or resources["outcome"] != "complete"
        or max(resources["peak_accounted_host_bytes"], resources.get("peak_cgroup_bytes") or 0)
           >= resources["host_target_bytes"]):
    raise SystemExit("Matching search study requires a successful sustained pilot")
settings = json.loads((pilot / "config.resolved.yaml").read_text())
if (settings["mode"] != "cached-logit-to-text" or settings["beam"] != [50, 100, 250]
        or settings["nbest"] != 50 or settings["lm_weight"] != 2 or settings["beam_threshold"] != 25
        or set(metrics["by_beam"]) != {"50", "100", "250"}):
    raise SystemExit("Pilot settings differ from the registered arms")
for name, path in {"config": "configs/local.yaml", "lexicon": "assets/local/lexicon.txt",
                   "kenlm": "runs/compact-a100/lm/train-3gram.binary"}.items():
    if settings["hashes"][name] != hashlib.sha256(Path(path).read_bytes()).hexdigest():
        raise SystemExit(f"Pilot asset mismatch: {name}")
if any(row["failures"] for row in metrics["by_beam"].values()):
    raise SystemExit("Pilot recorded failed or timed-out requests; inspect evidence")
cases = {}
for beam, row in metrics["by_beam"].items():
    worst = max(group["latency_seconds"]["p95"] for group in row["by_length"].values())
    cold = metrics["cold_setup"][beam]["process_ready_s"]
    # Include coverage scoring, duplicate file reads and result writing.
    cases[beam] = {"p95_longest_bin_or_worse_s": worst, "cold_s": cold,
                   "forecast_seconds": 2.0 * (worst * 1426 + cold) + 60}
if sum(row["forecast_seconds"] for row in cases.values()) > 23 * 60:
    raise SystemExit("Conservative comparison forecast exceeds this scheduler window")
(receipt / "admission.json").write_text(json.dumps({"pilot": str(pilot), "arms": cases}, indent=2))
PY
wait_stage "$decoder" "$new/scripts/audit_trie.py" --lexicon "$lexicon" --output-dir "$receipt/lexicon" \
    > "$receipt/lexicon-audit.log" 2>&1
"$python" - "$receipt/lexicon/trie_audit.json" "$mode" <<'PY'
import json
from pathlib import Path
import sys
audit = json.loads(Path(sys.argv[1]).read_text())
lost = audit["dictionary_pronunciations"] - audit["searchable_pronunciations"]
expected = 0 if sys.argv[2] == "repaired" else 224
if lost != expected:
    raise SystemExit(f"Unexpected native trie behavior: {lost} lost pronunciations; expected {expected}")
PY
for beam in 50 100 250; do
    test ! -e "$BTT_STOP_FILE"
    id="v2-s${first}-$mode-$beam"
    run="$receipt/$id"
    wait_stage "$decoder" "$new/scripts/decode_worker.py" --manifest assets/local/data_manifest.json \
        --logit-run runs/baseline-val-a100 --lexicon "$lexicon" --output-dir "$run/decode" \
        --kenlm-model runs/compact-a100/lm/train-3gram.binary --lm-weight 2 --nbest 50 --beam "$beam" \
        --max-seconds 660 > "$receipt/$id.log" 2>&1
    wait_stage "$python" -m btt25.search_diagnostic --registration "$new/configs/search_v2.json" \
        --condition "$id" --data-manifest assets/local/data_manifest.json --split-manifest assets/local/splits.json \
        --corpus-metadata assets/local/corpus_metadata.csv --predictions "$run/decode/predictions.jsonl" \
        --candidates "$run/decode/candidates.jsonl" --candidate-cache "$run/decode/cache_manifest.json" --lexicon "$effective" \
        --lm-vocab runs/compact-a100/lm/train-vocabulary.txt --output "$run/dev-tune"
    first=$((first+1))
done
