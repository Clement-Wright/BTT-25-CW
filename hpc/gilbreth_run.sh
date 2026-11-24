#!/usr/bin/env bash
# Submit this from a Gilbreth login shell with a bounded Slurm time limit.
# Arguments are passed literally to the requested project entry point.
set -eo pipefail
umask 077
source /etc/profile
set -u
project="${RCAC_SCRATCH:?}/btt-25-cw"
cd "$project/workspace"
export PYTHONPATH="$PWD/src"
export HF_HOME="$project/cache/huggingface"
export TMPDIR="$project/cache/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export LD_LIBRARY_PATH="$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
export BTT_STOP_FILE="$project/slurm/stop-${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR"
if [[ -e "$BTT_STOP_FILE" ]]; then
    printf 'Previous signal for this Slurm ID requires an explicit new attempt\n' >&2
    exit 2
fi
stop() { printf 'Slurm requested a safe stop\n' > "$BTT_STOP_FILE"; }
trap stop USR1 TERM INT
python="$project/environments/baseline/bin/python"
case "${1:?Missing stage}" in
    prepare)
        shift
        data_root="${1:?Missing verified expanded data root}"
        shift
        "$python" scripts/verify_release.py --data-root "$data_root" \
            --hashes assets/t15_expanded_sha256.json --output assets/local/gilbreth_release.json
        "$python" scripts/prepare_local.py --data-root "$data_root" "$@"
        ;;
    python)
        shift
        "$python" "$@" &
        child=$!
        while true; do
            code=0
            wait "$child" || code=$?
            if ! kill -0 "$child" 2>/dev/null; then exit "$code"; fi
        done
        ;;
    *) printf 'Unknown Gilbreth stage: %s\n' "$1" >&2; exit 2 ;;
esac
