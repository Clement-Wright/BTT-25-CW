#!/usr/bin/env bash
# Freeze the exact admitted comparisons, benchmark them, and score the exposed benchmark.
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
old="$project/workspace"
new="$project/v2"
python="$project/environments/baseline/bin/python"
decoder="$project/environments/decoder/bin/python"
unlimited="$new/cache/flashlight-unlimited-${1:?Provide repaired search job ID}/site-packages"
shift
out="$new/runs/v2-report-${SLURM_JOB_ID:?}"
mkdir -p "$out"
cd "$old"
export PYTHONPATH="$new/src:$new/vendor/lightbeam-deps"
prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
export LD_LIBRARY_PATH="$prefix/lib:$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
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
common=(--config configs/local.yaml --freeze "$out/freeze.json" --registration "$new/configs/search_v2.json"
        --extra-registration "$new/configs/fixed_candidates_v2.json" --extra-registration "$new/configs/encoders_v2.json"
        --extra-registration "$new/configs/search_expanded_v2.json" --extra-registration "$new/configs/fixed_candidates_4b_v2.json")
runs=()
for entry in "$@"; do
    path="${entry#*=}"
    if [[ -f "$path/decode/run_manifest.json" && -f "$path/dev-tune/metrics.json" && -f "$path/dev-tune/resources.json" ]]; then
        runs+=(--run "$entry")
    else
        printf '%s\n' "$entry" >> "$out/unavailable-comparisons.txt"
    fi
done
wait_stage "$python" -m btt25.report_v2 seal "${common[@]}" --control v2-s1-original-50 "${runs[@]}"
"$python" - "$out" "$new" "$old" "$unlimited" <<'PY'
import json
from pathlib import Path
import sys

out, new, old, unlimited = map(Path, sys.argv[1:])
freeze = json.loads((out / 'freeze.json').read_text())
sets = freeze['conditions']

def row(name):
    root = Path(sets[name]['directory'])
    cache = json.loads((root / 'decode/cache_manifest.json').read_text())
    tune = json.loads((root / 'dev-tune/metrics.json').read_text())
    data = {'condition': name, 'beam': cache['decoder']['beam'], 'nbest': cache['decoder']['nbest'],
            'kenlm_model': cache['kenlm_model_path'], 'lm_weight': cache['decoder']['lm_weight'],
            'beam_threshold': cache['decoder']['beam_threshold'], 'native_revision': cache['decoder']['variant']}
    if cache['decoder']['variant'] == 'unlimited-label-patch':
        if not unlimited.is_dir():
            raise FileNotFoundError(unlimited)
        data['decoder_module_path'] = str(unlimited)
    if name.startswith('v2-f'):
        model = name.split('-')[-1]
        paths = {'distilgpt2': old / 'assets/local/distilgpt2', 'qwen17b': new / 'assets/qwen3-1.7b-base',
                 'qwen4b': new / 'assets/qwen3-4b-base'}
        data['neural'] = {'model': model, 'path': str(paths[model]),
            'revision': tune['model']['revision'], 'weight': tune['chosen_weight'],
            'batch_size': 8, 'candidate_limit': cache['hypothesis_limit']}
    if name.startswith('v2-e'):
        data['encoder_checkpoint'] = str(root / 'train/checkpoints/last.pt')
        data['encoder_sha256'] = cache['source_cache_manifest']['checkpoint_hash']
    return tune['selected_errors'], data

selected = [freeze['baseline_condition'], freeze['selected_condition']]
for prefix in ('v2-s', 'v2-f', 'v2-e'):
    names = [name for name in sets if name.startswith(prefix)]
    if names:
        selected.append(min(names, key=lambda name: (row(name)[0], name)))
repaired = [name for name in sets if '-repaired-' in name]
if repaired:
    selected.append(min(repaired, key=lambda name: (row(name)[0], name)))
cases = [row(name)[1] for name in dict.fromkeys(selected)]
(out / 'benchmark-cases.json').write_text(json.dumps(cases, indent=2) + '\n')
PY
if ! wait_stage "$python" -m btt25.benchmark --config configs/local.yaml --output "$out/feature-benchmark" \
    --decoder-python "$decoder" --lexicon assets/local/lexicon.txt --cases "$out/benchmark-cases.json" \
    --passes 3 --trials 96 --max-seconds 1350; then
    printf '%s\n' 'Matched timing attempt failed. See the feature benchmark manifest and scheduler log.' >> "$out/unavailable-comparisons.txt"
fi
for split in dev_report official_val; do
    wait_stage "$python" -m btt25.report_v2 report "${common[@]}" --score-split "$split" --output "$out/$split"
done
