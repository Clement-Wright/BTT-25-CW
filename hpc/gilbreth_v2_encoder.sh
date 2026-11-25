#!/usr/bin/env bash
# Pilot and conditionally admit one v2 encoder using its measured throughput.
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
old="$project/workspace"
new="$project/v2"
python="$project/environments/baseline/bin/python"
decoder="$project/environments/decoder/bin/python"
mode="${1:?Choose transformer, monophone or diphone}"
registration="$new/configs/encoders_v2.json"
case "$mode" in
    transformer) id=v2-e1-transformer ;;
    monophone) id=v2-e2-monophone ;;
    diphone) id=v2-e3-diphone ;;
    *) exit 2 ;;
esac
run="$new/runs/$id-${SLURM_JOB_ID:?}"
resume_dir=${2:-}
if [[ "$resume_dir" ]]; then
    admission_job=${3:-$resume_dir}
    [[ "$mode" = transformer && "$resume_dir" =~ ^[0-9]+$ && "$admission_job" =~ ^[0-9]+$ ]]
    previous="$new/runs/$id-$resume_dir"
    original_admission="$new/runs/$id-$admission_job/pilot/admission.json"
    test -s "$previous/train/checkpoints/last.pt.sha256.json"
    test -s "$original_admission"
fi
mkdir -p "$run"
cd "$old"
export PYTHONPATH="$new/src"
export LD_LIBRARY_PATH="$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export BTT_STOP_FILE="$project/slurm/v2-stop-${SLURM_JOB_ID}"
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
lexicon="$old/assets/local/lexicon.txt"
lm="$old/runs/compact-a100/lm/train-3gram.binary"
flags=(--config configs/local.yaml --decoder-python "$decoder" --lexicon "$lexicon"
       --kenlm-model "$lm" --lm-weight 2 --mode "$mode")
if [[ ! "$resume_dir" ]]; then
    wait_stage "$python" -m btt25.train_v2 "${flags[@]}" --output "$run/pilot" --pilot-seconds 900 --max-seconds 1320
    "$python" - "$run/pilot" "$mode" "$registration" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

pilot = Path(sys.argv[1])
mode = sys.argv[2]
registration = json.loads(Path(sys.argv[3]).read_text())
metrics = json.loads((pilot / 'metrics.json').read_text())
resources = json.loads((pilot / 'resources.json').read_text())
config = json.loads((pilot / 'config.resolved.yaml').read_text())
peak = max(resources['peak_accounted_host_bytes'], resources.get('peak_cgroup_bytes') or 0)
if (metrics['status'] != 'pilot_complete' or metrics['sustained_seconds'] < registration['pilot_seconds']
    or not metrics['updates'] or metrics['trials'] != 8 or resources['outcome'] != 'complete'
    or peak >= resources['host_target_bytes']):
    raise ValueError('Missing matching sustained encoder admission')
updates = registration['training']['transformer_updates' if mode == 'transformer' else 'head_updates']
resume = mode == 'transformer' and metrics['updates'] < updates
remaining = updates - metrics['updates'] if resume else updates
forecast = 1.5 * remaining * metrics['sustained_seconds'] / metrics['updates'] + 600
if forecast > 5100:
    raise ValueError(f'Conservative encoder forecast {forecast:.1f}s exceeds this scheduler window')
(pilot / 'admission.json').write_text(json.dumps({'identity': config['identity'], 'forecast_seconds': forecast,
    'admitted_updates': updates, 'resume_pilot': resume,
    'registration_sha256': hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()}, indent=2))
PY
    read -r updates resume < <("$python" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["admitted_updates"],int(p["resume_pilot"]))' "$run/pilot/admission.json")
    checkpoint=()
    if [[ "$resume" = 1 ]]; then checkpoint=(--resume "$run/pilot/checkpoints/last.pt"); fi
    admission="$run/pilot/admission.json"
else
    admission="$original_admission"
    updates=$("$python" - "$admission" "$previous/train/run_manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    admission = json.load(stream)
with open(sys.argv[2]) as stream:
    prior = json.load(stream)
if prior["status"] != "budget_stopped" or not 0 < prior["updates"] < admission["admitted_updates"]:
    raise ValueError("Expected a budget-stopped run below its registered endpoint")
print(admission["admitted_updates"])
PY
)
    checkpoint=(--resume "$previous/train/checkpoints/last.pt")
    printf '%s\n' "$resume_dir" > "$run/resumed-job-id.txt"
    mkdir "$run/pilot"
    cp "$admission" "$run/pilot/admission.json"
fi
wait_stage "$python" -m btt25.train_v2 "${flags[@]}" --output "$run/train" --pilot-seconds 0 --max-seconds 5100 \
    --updates "$updates" --admission "$admission" "${checkpoint[@]}"
wait_stage "$decoder" "$new/scripts/decode_worker.py" --manifest assets/local/data_manifest.json \
    --logit-run "$run/train/inference" --lexicon "$lexicon" --output-dir "$run/decode" \
    --kenlm-model "$lm" --lm-weight 2 --nbest 50 --beam 50 --max-seconds 480 > "$run/decode.log" 2>&1
wait_stage "$decoder" "$new/scripts/audit_trie.py" --lexicon "$lexicon" --output-dir "$run/lexicon" > "$run/lexicon.log" 2>&1
wait_stage "$python" -m btt25.search_diagnostic --registration "$registration" --condition "$id" \
    --data-manifest assets/local/data_manifest.json --split-manifest assets/local/splits.json \
    --corpus-metadata assets/local/corpus_metadata.csv --predictions "$run/decode/predictions.jsonl" \
    --candidates "$run/decode/candidates.jsonl" --candidate-cache "$run/decode/cache_manifest.json" \
    --lexicon "$run/lexicon/searchable_lexicon.txt" --lm-vocab runs/compact-a100/lm/train-vocabulary.txt --output "$run/dev-tune"
