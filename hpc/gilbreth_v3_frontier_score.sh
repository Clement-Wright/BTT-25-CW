#!/usr/bin/env bash
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
campaign="${1:?Supply campaign path}"
name="${2:?Supply pinned Qwen model name}"
setup="${3:?Supply acquisition job ID}"
max_seconds="${4:-5100}"
prior=()
if [[ -n "${5:-}" ]]; then
    prior=(--prior "$campaign/runs/frontier-$name-$5")
fi
job="${SLURM_JOB_ID:?}"
work="$project/workspace"
source="$campaign/vendor/llama.cpp-28ff0958291ce3465fabd7bd679d4b0edd742bd9"
bin="$source/build/bin"
export PYTHONPATH="$campaign/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export BTT_STOP_FILE="$campaign/logs/frontier-$job.stop"
set +u
source /etc/profile.d/modules.sh
module load cuda/12.6.0
set -u
export LD_LIBRARY_PATH="$bin:${LD_LIBRARY_PATH:-}"
scontrol show job -o "$job" > "$campaign/logs/frontier-$job.scheduler.txt"
if [[ -n "${5:-}" ]]; then
    [[ "$5" =~ ^[0-9]+$ ]] || exit 2
    scorer="$campaign/vendor/gguf-candidate-score-$5"
    [[ -x "$scorer" ]] || { echo "Missing exact pilot scorer binary" >&2; exit 2; }
else
    scorer="$campaign/vendor/gguf-candidate-score-$job"
    g++ -O3 -std=c++17 -I"$source/include" -I"$source/ggml/include" "$campaign/hpc/gguf_candidate_score.cpp" \
        "$bin/libllama.so" "$bin/libggml.so" "$bin/libggml-base.so" -Wl,-rpath-link,"$bin" -o "$scorer"
fi
cd "$work"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
"$project/environments/baseline/bin/python" -m btt25.gguf_study --config configs/local.yaml \
    --source "$project/v2/runs/v2-search-attempt-12523602/v2-s6-repaired-250/decode" \
    --neural runs/baseline-val-a100/predictions.jsonl --binary "$scorer" \
    --model "$campaign/assets/$name-Q4_K_M.gguf" \
    --acquisition "$campaign/runs/frontier-setup-$setup/metrics.json" \
    --output "$campaign/runs/frontier-$name-$job" --max-seconds "$max_seconds" "${prior[@]}" &
child=$!
while true; do
    code=0
    wait "$child" || code=$?
    if ! kill -0 "$child" 2>/dev/null; then exit "$code"; fi
done
