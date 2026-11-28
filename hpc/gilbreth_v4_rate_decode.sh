#!/usr/bin/env bash
#SBATCH --job-name=btt-cadence
#SBATCH --account=ieee-smcs
#SBATCH --partition=a100-40gb,a100-80gb,a30,a10
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=01:45:00
#SBATCH --signal=B:USR1@180
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}/btt-25-cw"
campaign="$project/v4-rate-20261127"
code="$campaign/cadence-code-v1"
out="$campaign/runs/cadence-${SLURM_JOB_ID:?}"
mkdir "$out"
cd "$project/workspace"
export PYTHONPATH="$campaign/src"
export BTT_FLASHLIGHT_VARIANT=unlimited-label-patch
export BTT_STOP_FILE="$out/stop"
export MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
decoder="$project/environments/decoder/bin/python"
python="$project/environments/baseline/bin/python"
prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
export LD_LIBRARY_PATH="$prefix/lib:$project/environments/kenlm/lib:$project/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
scontrol show job -o "$SLURM_JOB_ID" > "$out/scheduler-start.txt"
find_arm() {
    local name="$1" task="$2" line
    line=$(sacct -X -nP -j "$task" -o JobID,JobIDRaw,State | awk -F'|' -v id="$task" '$1==id && $3=="COMPLETED" {print $2}')
    if [[ ! "$line" =~ ^[0-9]+$ ]]; then echo "Missing completed receipt for $task" >&2; return 2; fi
    printf '%s/runs/%s-%s/study/%s' "$campaign" "$name" "$line" "$name"
}
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
two=$(find_arm two_rate "${1:?Pass two_rate Slurm array ID}")
fast=$(find_arm gru40 "${2:?Pass gru40 Slurm array ID}")
"$python" "$code/v4_rate_decode.py" \
    --config configs/local.yaml --lexicon assets/local/lexicon.txt \
    --kenlm-model runs/compact-a100/lm/train-3gram.binary --decoder-python "$decoder" \
    --decoder-module-path "$project/v2/cache/flashlight-unlimited-12523602/site-packages" \
    --corpus-metadata assets/local/corpus_metadata.csv --gru40 "$fast" --two-rate "$two" \
    --output "$out/study" --pilot-seconds 900 --max-seconds 6000 &
child=$!
while true; do
    status=0
    wait "$child" || status=$?
    if ! kill -0 "$child" 2>/dev/null; then exit "$status"; fi
done
