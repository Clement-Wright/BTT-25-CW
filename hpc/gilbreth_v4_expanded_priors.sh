#!/usr/bin/env bash
#SBATCH --job-name=btt-distil-union
#SBATCH --account=ieee-smcs
#SBATCH --partition=a10,a30,a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M
#SBATCH --time=00:18:00
set -eu -o pipefail
umask 077
export RCAC_SCRATCH="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}"
p="$RCAC_SCRATCH/btt-25-cw"
c="$p/v4-rate-20261127"
code="$c/phone-robustness-code"
root="$c/runs/phone-expanded-priors-${SLURM_JOB_ID:?}"
mkdir -p "$root"
scontrol show job -o "$SLURM_JOB_ID" > "$root/scheduler-start.txt"
export PYTHONPATH="$code/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$p/workspace"
"$p/environments/baseline/bin/python" "$code/scripts/score_expanded_phone_words.py" \
    --registration "$code/configs/phone_edit_neural_v4.json" --config configs/local.yaml \
    --cache runs/baseline-val-a100 --source "$c/runs/phone-robustness-12625458/study/candidates.jsonl" \
    --proposals "$c/runs/phone-edits-12625561/study/predictions.jsonl" \
    --proposal-config "$code/configs/phone_edit_candidates_v4.json" --arpa runs/compact-a100/lm/train-3gram.arpa \
    --lexicon assets/local/lexicon.txt --unigrams "$p/v3-smoothing-20261127/assets/train_unigrams.json" \
    --distil assets/local/distilgpt2 \
    --old-scores "$p/v2/runs/v2-fixed-12523725/v2-f1-distilgpt2/decode/candidates.jsonl" \
    --output "$root/study"
