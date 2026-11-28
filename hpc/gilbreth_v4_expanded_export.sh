#!/usr/bin/env bash
#SBATCH --job-name=btt-union-export
#SBATCH --account=ieee-smcs
#SBATCH --partition=a100-80gb,a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M
#SBATCH --time=00:27:00
set -eu -o pipefail
umask 077
export RCAC_SCRATCH="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}"
p="$RCAC_SCRATCH/btt-25-cw"
c="$p/v4-rate-20261127"
code="$c/phone-robustness-code"
root="$c/runs/phone-expanded-export-${SLURM_JOB_ID:?}"
mkdir -p "$root"
scontrol show job -o "$SLURM_JOB_ID" > "$root/scheduler-start.txt"
export PYTHONPATH="$code/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BTT_FLASHLIGHT_VARIANT=unlimited-label-patch
decoder="$p/environments/decoder/bin/python"
prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
export LD_LIBRARY_PATH="$prefix/lib:$p/environments/kenlm/lib:$p/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
cd "$p/workspace"
timing="$c/runs/phone-expanded-timing-${1:?timing job ID}/study/metrics.json"
count=$(jq '[.[] | select(.split=="test")] | length' assets/local/data_manifest.json)
jq -e --argjson n "$count" '
  select(.status=="complete" and .timing.expanded_alignment.requests==72) |
  {p95_s:.timing.expanded_alignment.p.p95, n:$n, margin:1.35, setup_save_allowance_s:120} |
  .forecast_s=(.p95_s*.n*.margin+.setup_save_allowance_s) |
  .admitted=(.forecast_s < 1350) | select(.admitted)
' "$timing" > "$root/admission.json"
"$p/environments/baseline/bin/python" -m btt25.phone_word_inference \
    --config configs/local.yaml --registration "$code/configs/phone_edit_neural_v4.json" \
    --proposal-config "$code/configs/phone_edit_candidates_v4.json" --lexicon assets/local/lexicon.txt \
    --verify-counts "$p/v3-smoothing-20261127/assets/train_unigrams.json" \
    --arpa runs/compact-a100/lm/train-3gram.arpa --training-text runs/compact-a100/lm/train-normalized.txt \
    --train-lm-manifest runs/compact-a100/lm/manifest.json --kenlm-model runs/compact-a100/lm/train-3gram.binary \
    --decoder-python "$decoder" --decoder-module-path "$p/v2/cache/flashlight-unlimited-12523602/site-packages" \
    --distil assets/local/distilgpt2 --partition test --output "$root/study"
