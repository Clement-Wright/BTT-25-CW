#!/usr/bin/env bash
# Acquire and verify the publisher's three Qwen3-4B shards in private scratch.
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
new="$project/v2"
python="$project/environments/baseline/bin/python"
out="$new/runs/v2-qwen4-setup-${SLURM_JOB_ID:?}"
export PYTHONPATH="$new/src:$new/vendor/lightbeam-deps"
export HF_HOME="$project/cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES=
"$python" - "$new/assets/qwen3-4b-base" "$out" <<'PY'
from pathlib import Path
import sys
from btt25.qwen_scorer import acquire
model, output = map(Path, sys.argv[1:])
acquire(model, output, size="4b")
PY
