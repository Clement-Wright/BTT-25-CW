#!/usr/bin/env bash
#SBATCH --job-name=btt-two-rate-words
#SBATCH --account=ieee-smcs
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=13000M
#SBATCH --time=00:46:00
set -eu -o pipefail
umask 077
export RCAC_SCRATCH="${RCAC_SCRATCH:-/scratch/gilbreth/${USER:?}}"
p="$RCAC_SCRATCH/btt-25-cw"
c="$p/v4-rate-20261127"
code="${BTT_TWO_RATE_SOURCE:-$c/two-rate-expanded-source-final}"
root="$c/runs/two-rate-expanded-${SLURM_JOB_ID:?}"
mkdir -p "$root"
scontrol show job -o "$SLURM_JOB_ID" > "$root/scheduler-start.txt"
export PYTHONPATH="$code/src:$p/v2/vendor/lightbeam-deps"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BTT_FLASHLIGHT_VARIANT=unlimited-label-patch
export BTT_STOP_FILE="$root/stop"
decoder="$p/environments/decoder/bin/python"
prefix=$(dirname "$(dirname "$(readlink -f "$decoder")")")
export LD_LIBRARY_PATH="$prefix/lib:$p/environments/kenlm/lib:$p/environments/kenlm/lib64:${LD_LIBRARY_PATH:-}"
cd "$p/workspace"
"$p/environments/baseline/bin/python" -m btt25.two_rate_expanded \
    --config configs/local.yaml --trained "$c/runs/two_rate-12625370/study/two_rate" \
    --strong "$c/runs/strong-12625552/results" --prior-timing "$c/runs/phone-expanded-timing-12626215/study" \
    --registration "$code/configs/phone_edit_neural_v4.json" --proposal-config "$code/configs/phone_edit_candidates_v4.json" \
    --lexicon assets/local/lexicon.txt --distil assets/local/distilgpt2 \
    --unigrams "$p/v3-smoothing-20261127/assets/train_unigrams.json" --arpa runs/compact-a100/lm/train-3gram.arpa \
    --kenlm-model runs/compact-a100/lm/train-3gram.binary --decoder-python "$decoder" \
    --decoder-module-path "$p/v2/cache/flashlight-unlimited-12523602/site-packages" --output "$root/study"
