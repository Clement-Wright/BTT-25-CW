#!/usr/bin/env bash
# Private, pinned upstream checkout and minimal dependencies for the LightBeam trial.
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
stage="$project/v2"
source="$stage/vendor/brainaudio"
baseline="$project/environments/baseline/bin/python"
base_python=$(readlink -f "$baseline")
deps="$stage/vendor/lightbeam-deps"
qwen="$stage/assets/qwen3-1.7b-base"
out="$stage/runs/v2-lightbeam-setup-${SLURM_JOB_ID:?}"
mkdir -p "$stage/vendor" "$out"
export PYTHONPATH="$stage/src"
export HF_HOME="$project/cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export PIP_CACHE_DIR="$project/cache/pip"
export TMPDIR="$project/cache/tmp"
mkdir -p "$TMPDIR"
rev=90e0ac086d4dee643be154f446a6c374600575d7
if [[ ! -d "$source/.git" ]]; then
    git init -q "$source"
    git -C "$source" remote add origin https://github.com/ebrahimfeghhi/brainaudio.git
    git -C "$source" config remote.origin.promisor true
    git -C "$source" config remote.origin.partialclonefilter blob:none
    git -C "$source" sparse-checkout set --no-cone /src/ /scripts/run_decoder.py /scripts/decoder_config.py /README.md
    git -C "$source" fetch --depth=1 --filter=blob:none origin "$rev"
    git -C "$source" checkout --detach FETCH_HEAD
fi
test "$(git -C "$source" rev-parse HEAD)" = "$rev"
"$base_python" -m pip install --target "$deps" --no-deps --disable-pip-version-check \
    pandas==2.3.3 python-dateutil==2.9.0.post0 pytz==2025.2 tzdata==2025.2 six==1.17.0 \
    edit-distance==1.0.7 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 \
    accelerate==1.10.1 peft==0.17.1 kenlm==0.3.0
# Replace only the two NeMo utility imports with the matching local classes.
# Keep the patches and upstream code in private scratch.
"$baseline" - "$source" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1]) / "src/brainaudio/inference/decoder/batched_beam_decoding_utils.py"
old = (
    "from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses\n"
    "from nemo.utils.enum import PrettyStrEnum"
)
new = "from .rnnt_utils import Hypothesis, NBestHypotheses\nfrom .enum import PrettyStrEnum"
text = p.read_text(encoding="utf-8")
if old in text:
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
elif new not in text:
    raise ValueError("Unexpected LightBeam NeMo import contract")
PY
git -C "$source" diff --check
git -C "$source" diff > "$out/upstream-runtime.diff"
"$base_python" -m pip list --path "$deps" --format=freeze > "$out/python-packages.txt"
export PYTHONPATH="$stage/src:$source/src:$deps"
"$baseline" -c 'from brainaudio.inference.decoder import BatchedBeamCTCComputer; import peft; import kenlm'
"$baseline" - "$qwen" "$out/model" <<'PY'
import sys
from pathlib import Path
from btt25.qwen_scorer import acquire
acquire(*(Path(arg) for arg in sys.argv[1:]))
PY
