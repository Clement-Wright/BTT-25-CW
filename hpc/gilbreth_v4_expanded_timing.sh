#!/usr/bin/env bash
#SBATCH --job-name=btt-union-timing
#SBATCH --account=ieee-smcs
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M
#SBATCH --time=00:17:00
set -eu -o pipefail
umask 077
export RCAC_SCRATCH="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}"
p="$RCAC_SCRATCH/btt-25-cw"
c="$p/v4-rate-20261127"
code="$c/phone-robustness-code"
root="$c/runs/phone-expanded-timing-${SLURM_JOB_ID:?}"
mkdir -p "$root"
scontrol show job -o "$SLURM_JOB_ID" > "$root/scheduler-start.txt"
export PYTHONPATH="$code/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BTT_FLASHLIGHT_VARIANT=unlimited-label-patch
decoder="$p/environments/decoder/bin/python"
prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
export LD_LIBRARY_PATH="$prefix/lib:$p/environments/kenlm/lib:$p/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
cd "$p/workspace"
"$p/environments/baseline/bin/python" "$code/scripts/benchmark_expanded_phone_words.py" \
    --config configs/local.yaml --registration "$code/configs/phone_edit_neural_v4.json" \
    --study "$c/runs/phone-expanded-priors-${1:?source job}/study" --distil assets/local/distilgpt2 \
    --proposal-config "$code/configs/phone_edit_candidates_v4.json" --lexicon assets/local/lexicon.txt \
    --unigrams "$p/v3-smoothing-20261127/assets/train_unigrams.json" --arpa runs/compact-a100/lm/train-3gram.arpa \
    --kenlm-model runs/compact-a100/lm/train-3gram.binary --decoder-python "$decoder" \
    --decoder-module-path "$p/v2/cache/flashlight-unlimited-12523602/site-packages" --output "$root/study"
