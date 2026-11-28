#!/usr/bin/env bash
set -euo pipefail
umask 077
p="${RCAC_SCRATCH:?}/btt-25-cw"
c="$p/v4-rate-20261127"
out="$c/runs/gru20-${SLURM_JOB_ID:?}"
mkdir -p "$out"
cd "$p/workspace"
export PYTHONPATH="$c/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BTT_FLASHLIGHT_VARIANT=unlimited-label-patch
export BTT_STOP_FILE="$out/stop"
dec="$p/environments/decoder/bin/python"
prefix=$(dirname "$(dirname "$(readlink -f "$dec")")")
export LD_LIBRARY_PATH="$prefix/lib:$p/environments/kenlm/lib:$p/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
scontrol show job -o "$SLURM_JOB_ID" > "$out/scheduler-start.txt"
sha256sum "$c/src/btt25/v4_dense_clock.py" > "$out/extension-code.sha256"
exec "$p/environments/baseline/bin/python" -m btt25.v4_dense_clock --config configs/local.yaml \
    --lexicon assets/local/lexicon.txt --kenlm-model runs/compact-a100/lm/train-3gram.binary \
    --decoder-python "$dec" --decoder-module-path "$p/v2/cache/flashlight-unlimited-12523602/site-packages" \
    --corpus-metadata assets/local/corpus_metadata.csv --lm-weight 2 --conditions gru20 --updates 2048 \
    --pilot-seconds 900 --max-seconds 5400 --output "$out/study"
