#!/usr/bin/env bash
set -euo pipefail
cd "${RCAC_SCRATCH:?}/btt-25-cw/v4-rate-20261127"
case "${1:?Expected primary or supplemental}" in
    primary) conditions=(two_rate gru80 gru40 adaptive_filter causal_conv) ;;
    supplemental) conditions=(patch10 patch20 bin40 bin10_hold activation_only transform_only input_transform) ;;
    *) exit 2 ;;
esac
condition="${conditions[${SLURM_ARRAY_TASK_ID:?}]}"
exec bash hpc/gilbreth_v4_rate.sh "$condition" --conditions "$condition" \
    --updates 2048 --pilot-seconds 900 --max-seconds 5400
