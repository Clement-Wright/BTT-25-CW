#!/usr/bin/env bash
# Freeze and time the compact control and repaired search on one allocation.
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
old="$project/workspace"
new="$project/v2"
out="$new/runs/v2-search-timing-${SLURM_JOB_ID:?}"
python="$project/environments/baseline/bin/python"
decoder="$project/environments/decoder/bin/python"
repaired="$new/cache/flashlight-unlimited-${1:?Supply the repaired build job ID}/site-packages"
test -d "$repaired"
mkdir -p "$out"
cd "$old"
export PYTHONPATH="$new/src"
prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
export LD_LIBRARY_PATH="$prefix/lib:$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export BTT_STOP_FILE="$project/slurm/v2-stop-${SLURM_JOB_ID}"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
"$python" - "$out" "$repaired" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

out, repaired = map(Path, sys.argv[1:])
common = {'kenlm_model': 'runs/compact-a100/lm/train-3gram.binary', 'lm_weight': 2, 'beam_threshold': 25}
cases = [
    {**common, 'condition': 'v2-s0-archived', 'beam': 50, 'nbest': 10, 'native_revision': 'v0.0.7'},
    {**common, 'condition': 'v2-s3-original-250', 'beam': 250, 'nbest': 50, 'native_revision': 'v0.0.7'},
    {**common, 'condition': 'v2-s6-repaired-250', 'beam': 250, 'nbest': 50,
     'native_revision': 'unlimited-label-patch', 'decoder_module_path': str(repaired)},
]
data = (json.dumps(cases, indent=2) + '\n').encode()
(out / 'cases.json').write_bytes(data)
(out / 'design.sha256').write_text(hashlib.sha256(data).hexdigest() + '\n')
PY
"$python" -m btt25.benchmark --config configs/local.yaml --output "$out/features" \
    --decoder-python "$decoder" --lexicon assets/local/lexicon.txt --cases "$out/cases.json" \
    --passes 3 --trials 96 --max-seconds 720 &
child=$!
while true; do
    code=0
    wait "$child" || code=$?
    if ! kill -0 "$child" 2>/dev/null; then exit "$code"; fi
done
