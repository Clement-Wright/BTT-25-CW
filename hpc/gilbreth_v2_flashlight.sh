#!/usr/bin/env bash
# Build the pinned MIT-licensed Flashlight dependency with all homophones retained.
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
python="$project/environments/decoder/bin/python"
kenlm="$project/environments/kenlm"
root="${1:?Provide a new private build directory}"
rev=bbe9e3c201f5c9c3f3c0d553f0ea73af5e0a5209
pybind_rev=v2.13.6
mkdir -p "$root"
git init -q "$root/source"
git -C "$root/source" fetch --depth=1 https://github.com/flashlight/text.git "$rev"
git -C "$root/source" checkout --detach FETCH_HEAD
test "$(git -C "$root/source" rev-parse HEAD)" = "$rev"
git clone --depth 1 --branch "$pybind_rev" https://github.com/pybind/pybind11.git "$root/pybind11"
test "$(git -C "$root/pybind11" rev-parse HEAD)" = a2e59f0e7065404b44dfe92a28aca47ba1378dc4
"$python" - "$root/source" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]) / "flashlight/lib/text/decoder/Trie.cpp"
text = source.read_text(encoding="utf-8")
original = """  if (node->labels.size() < kTrieMaxLabel) {
    node->labels.push_back(label);
    node->scores.push_back(score);
  } else {
    std::cerr << "[Trie] Trie label number reached limit: " << kTrieMaxLabel
              << "\\n";
  }"""
if text.count(original) != 1:
    raise ValueError("Pinned Flashlight insertion code changed")
source.write_text(text.replace(original, """  node->labels.push_back(label);
  node->scores.push_back(score);"""), encoding="utf-8")
PY
git -C "$root/source" diff --check
git -C "$root/source" diff > "$root/unlimited-labels.diff"
# Keep the v1 wheel and its version/ABI wrapper intact. Only the new job's search
# path starts with these isolated Python bindings and the matching shared libs.
target="$root/site-packages"
site=$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
mkdir -p "$target"
cp -a "$site/flashlight" "$site/flashlight_text-0.0.7.dist-info" "$target/"
prefix=$(dirname "$(dirname "$(readlink -f "$python")")")
export LD_LIBRARY_PATH="$kenlm/lib:$prefix/lib:${LD_LIBRARY_PATH:-}"
cmake -S "$root/source" -B "$root/build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DFL_TEXT_BUILD_TESTS=OFF \
  -DFL_TEXT_BUILD_STANDALONE=OFF -DFL_TEXT_USE_KENLM=ON -DFL_TEXT_BUILD_PYTHON=ON \
  -DFL_TEXT_BUILD_PYTHON_PACKAGE=ON -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$root/pybind11" \
  -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE="$python" -DPython3_EXECUTABLE="$python" -DKENLM_ROOT="$kenlm" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$target/flashlight/lib/text"
cmake --build "$root/build" --parallel "${SLURM_CPUS_PER_TASK:-4}"
# Original wheel libraries use auditwheel-renamed dependency names.
export PYTHONPATH="$target:${PYTHONPATH:-}"
"$python" - <<'PY'
from flashlight.lib.text.decoder import Trie
from flashlight.lib.text.decoder.kenlm import KenLM
trie = Trie(41, 40)
for value in range(32):
    node = trie.insert([1, 1], value, 0.)
assert list(node.labels) == list(range(32))
PY
{
  printf 'flashlight_revision=%s\n' "$rev"
  git -C "$root/pybind11" rev-parse HEAD
  sha256sum "$root/unlimited-labels.diff"
  find "$target/flashlight/lib/text" -maxdepth 1 -type f -name '*.so*' -exec sha256sum '{}' ';'
  grep -E '(^Python3_EXECUTABLE|^Python_EXECUTABLE|^PYBIND11_FINDPYTHON)' "$root/build/CMakeCache.txt"
} > "$root/build_receipt.txt"
