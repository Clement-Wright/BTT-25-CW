#!/usr/bin/env bash
#SBATCH --job-name=btt-rate-edits
#SBATCH --account=ieee-smcs
#SBATCH --partition=a10,a30,a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7000M
#SBATCH --time=00:25:00
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}/btt-25-cw"
campaign="$project/v4-rate-20261127"
out="$campaign/runs/rate-edits-${SLURM_JOB_ID:?}"
mkdir "$out"
cd "$project/workspace"
export PYTHONPATH="$campaign/src"
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python="$project/environments/baseline/bin/python"
scontrol show job -o "$SLURM_JOB_ID" > "$out/scheduler-start.txt"
if [[ "${1:-}" == "pair" ]]; then
    shift
    if [[ "$#" -ne 2 ]]; then echo "Pass the matched control and comparison arm run names" >&2; exit 2; fi
    files=()
    for run_name in "$@"; do
        if [[ ! "$run_name" =~ ^(gru80|gru20)-[0-9]+$ ]]; then echo "Unexpected pair run" >&2; exit 2; fi
        files+=("$campaign/runs/$run_name/study/${run_name%-*}/predictions.jsonl")
    done
    "$python" "$campaign/cadence-code-v1/v4_rate_analysis.py" --config configs/local.yaml \
        --corpus-metadata assets/local/corpus_metadata.csv --output "$out/diagnostics" "${files[@]}"
    exit 0
fi
primary="${1:?Pass the registered primary array ID}"
supplemental="${2:?Pass the registered secondary array ID}"
readarray -t entries < <(sacct -nPX -j "$primary,$supplemental" -o JobID,JobIDRaw,State)
declare -A jobs
for entry in "${entries[@]}"; do
    IFS='|' read -r task job state <<< "$entry"
    if [[ "$state" == "COMPLETED" ]]; then jobs[$task]=$job; fi
done
files=()
names=(gru80 two_rate gru40 adaptive_filter causal_conv patch10 patch20 bin40 bin10_hold activation_only transform_only input_transform)
tasks=("${primary}_1" "${primary}_0" "${primary}_2" "${primary}_3" "${primary}_4"
       "${supplemental}_0" "${supplemental}_1" "${supplemental}_2" "${supplemental}_3"
       "${supplemental}_4" "${supplemental}_5" "${supplemental}_6")
for i in "${!names[@]}"; do
    task="${tasks[$i]}" name="${names[$i]}"
    job="${jobs[$task]:-}"
    if [[ "$job" =~ ^[0-9]+$ ]]; then
        files+=("$campaign/runs/$name-$job/study/$name/predictions.jsonl")
    else
        echo "Unfinished comparison: $name $task" >> "$out/exclusions.txt"
        if [[ "$name" == "gru80" ]]; then echo "Missing comparison control" >&2; exit 2; fi
    fi
done
"$python" "$campaign/cadence-code-v1/v4_rate_analysis.py" --config configs/local.yaml \
    --corpus-metadata assets/local/corpus_metadata.csv --output "$out/diagnostics" "${files[@]}"
