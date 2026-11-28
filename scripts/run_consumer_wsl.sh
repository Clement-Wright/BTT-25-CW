#!/usr/bin/env bash
# Run inside WSL from the checkout. The Python env stays read-only.
set -euo pipefail
umask 077
python=$(realpath -s "${1:?Existing CPython 3.12 environment with torch 2.8, scipy, numpy, yaml, psutil}")
native=$(realpath "${2:?Directory staged by hpc/stage_consumer_decoder.sh}")
campaign=${3:?New campaign directory under runs}
pilot="${campaign}-pilot"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$native/packages:$native/hdf5"
export LD_LIBRARY_PATH="$native/lib:$native/packages/flashlight/lib/text:${LD_LIBRARY_PATH:-}"
config="$native/config.wsl.yaml"
(cd "$native" && sha256sum --check --quiet transfer.sha256)
echo "6676cddc47120fb5d463c606fbbe0728ed45deaedf4895db723d087a74b40034  $native/lm/train-3gram.binary" | sha256sum --check --quiet
"$python" - "$config" <<'PY'
import json
from pathlib import Path, PureWindowsPath
import sys
import yaml
from flashlight.lib.text.decoder import Trie
from flashlight.lib.text.decoder.kenlm import KenLM

trie = Trie(41,40)
for i in range(7):
    trie.insert([1], i, 0.)
if len(trie.search([1]).labels) != 7:
    raise RuntimeError("Expected the unlimited-label Flashlight build")
dest = Path(sys.argv[1])
config = yaml.safe_load(Path("configs/local.yaml").read_text())
records = json.loads(Path(config["data_manifest"]).read_text())
mapped = {}
for record in records:
    original = record["source_file"]
    if original not in mapped:
        win = PureWindowsPath(original)
        if not win.drive or not win.is_absolute():
            raise ValueError("Expected an absolute Windows path in this manifest")
        path = Path("/mnt", win.drive[0].lower(), *win.parts[1:])
        if not path.is_file():
            raise FileNotFoundError(path)
        mapped[original] = str(path)
    record["source_file"] = mapped[original]
manifest = dest.parent/"manifest.wsl.json"
manifest.write_text(json.dumps(records))
config.update(data_manifest=str(manifest), device="cpu")
dest.write_text(yaml.safe_dump(config))
PY
args=(--config "$config" --decoder-python "$python" --kenlm-model "$native/lm/train-3gram.binary")
if [[ -n "${4:-}" ]]; then
  delta_dir=$(realpath "$4")
  "$python" -m btt25.consumer_replay "${args[@]}" --precision fp32 \
    --two-rate-delta "$delta_dir/two_rate.pt" --delta-receipt "$delta_dir/delta.json" \
    --out "${campaign}-two-rate-replay"
  exit
fi
if [[ ! -f "$pilot/report.md" ]]; then
  "$python" -m btt25.consumer_v4 --mode pilot --out "$pilot" "${args[@]}"
fi
"$python" -u scripts/run_consumer_v4.py --pilot "$pilot" --campaign "$campaign" "${args[@]}"
