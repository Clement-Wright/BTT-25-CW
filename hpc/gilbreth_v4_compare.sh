#!/usr/bin/env bash
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
campaign="$project/v4-rate-20261127"
cd "$campaign"
names=(gru80 gru40 gru20 two_rate patch10 patch20 bin40 bin10_hold adaptive_filter
       activation_only transform_only input_transform causal_conv)
entries=("official=$project/v3-smoothing-20261127/runs/smoothing-12618760/baseline-s4")
out="$campaign/runs/bench-all-${SLURM_JOB_ID:?}"
mkdir -p "$out"
: > "$out/unavailable.txt"
for name in "${names[@]}"; do
    matches=()
    for f in "$campaign"/runs/"$name"-*/study/"$name"/metrics.json; do
        if [[ -f "$f" ]] && jq -e '.status == "complete" and .updates == 2048' "$f" >/dev/null; then
            matches+=("$(dirname "$(dirname "$f")")")
        fi
    done
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "$name" >> "$out/unavailable.txt"
        continue
    elif [[ ${#matches[@]} -gt 1 ]]; then
        echo "Ambiguous completed runs for $name" >&2
        exit 2
    fi
    entries+=("$name=${matches[0]}")
done
printf '%s\n' "${entries[@]}" > "$out/sources.txt"
bash hpc/gilbreth_v4_rate.sh bench-all "${entries[@]}"
export PYTHONPATH="$campaign/src"
cd "$project/workspace"
"$project/environments/baseline/bin/python" -m btt25.v4_summary \
    --config configs/local.yaml --benchmark "$out" --clock "$campaign/runs/clock-12625303/clock" \
    --output "$out/rate_v4.json"
