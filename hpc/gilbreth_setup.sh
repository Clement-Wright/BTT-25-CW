#!/usr/bin/env bash
# Run inside a bounded Slurm allocation. No dataset labels are read here.
set -eo pipefail
umask 077
source /etc/profile
set -u
project="${RCAC_SCRATCH:?}/btt-25-cw"
cd "$project/workspace"
export UV_CACHE_DIR="$project/cache/uv"
export HF_HOME="$project/cache/huggingface"
export TMPDIR="$project/cache/tmp"
export PYTHONPATH="$PWD/src"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TMPDIR" "$project/environments" assets/local
module load anaconda/2024.10-py312 uv/0.8.0
baseline="$project/environments/baseline"
decoder="$project/environments/decoder"
if [[ ! -x "$baseline/bin/python" ]]; then
    uv venv --python "$(command -v python)" "$baseline"
fi
if [[ ! -x "$decoder/bin/python" ]]; then
    uv venv --python "$baseline/bin/python" "$decoder"
fi
uv pip install --python "$baseline/bin/python" --index-url https://download.pytorch.org/whl/cu126 'torch==2.8.0+cu126' 'torchvision==0.23.0+cu126'
uv pip install --python "$baseline/bin/python" -r configs/requirements-baseline.txt \
    'transformers==4.56.2' 'tokenizers==0.22.2' 'safetensors==0.6.2' 'huggingface-hub==0.35.3'
uv pip install --python "$decoder/bin/python" 'numpy==2.2.6' 'psutil==7.0.0'
wheel=flashlight_text-0.0.7-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
"$baseline/bin/python" scripts/acquire_linux_decoder.py --output "$project/cache/$wheel"
uv pip install --python "$decoder/bin/python" "$project/cache/$wheel"
"$baseline/bin/python" -m btt25.cli acquire-model
"$baseline/bin/python" scripts/acquire_extras.py --with-scorer
"$decoder/bin/python" scripts/decode_worker.py --build-lexicon-only --cmudict assets/local/cmudict/cmudict.dict --lexicon assets/local/lexicon.txt
{
    date -Is
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    "$baseline/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
    uv pip freeze --python "$baseline/bin/python"
    uv pip freeze --python "$decoder/bin/python"
} > "$project/receipts/setup-${SLURM_JOB_ID:?}.txt"
