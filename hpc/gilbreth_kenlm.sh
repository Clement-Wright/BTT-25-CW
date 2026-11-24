#!/usr/bin/env bash
# Build the pinned KenLM estimator CLI and native libraries in a Slurm allocation.
set -eo pipefail
umask 077
source /etc/profile
module load anaconda/2024.10-py312
set -u
anaconda_root="$(dirname "$(dirname "$(command -v python)")")"
test -f "$anaconda_root/lib/libboost_program_options.so"
project="${RCAC_SCRATCH:?}/btt-25-cw"
scratch_build="$project/cache/kenlm-build"
prefix="$project/environments/kenlm"
archive="$scratch_build/kenlm-0.3.0.tar.gz"
digest=c4628bb9fb63c8a6f9240035b8b037385cfc404cb72e933cf48878291edac1e8
url=https://files.pythonhosted.org/packages/02/6e/f4e159cef39ea4ef2244b4aa32857d447d60a5d0ee269e1b1f8d3723c03e/kenlm-0.3.0.tar.gz
mkdir -p "$scratch_build" "$prefix"
if [[ ! -f "$archive" ]]; then
    curl -fL --retry 2 --connect-timeout 30 -o "$archive.partial" "$url"
    printf '%s  %s\n' "$digest" "$archive.partial" | sha256sum --check
    mv "$archive.partial" "$archive"
fi
printf '%s  %s\n' "$digest" "$archive" | sha256sum --check
tar -xzf "$archive" -C "$scratch_build"
build="$scratch_build/build-anaconda"
cmake -S "$scratch_build/kenlm-0.3.0" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_PYTHON_STANDALONE=OFF \
    -DCMAKE_PREFIX_PATH="$anaconda_root" -DBoost_ROOT="$anaconda_root" \
    -DKENLM_MAX_ORDER=6 -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_INSTALL_RPATH="$prefix/lib;$prefix/lib64;$anaconda_root/lib"
cmake --build "$build" --parallel "${SLURM_CPUS_PER_TASK:-4}"
cmake --install "$build"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib64:$anaconda_root/lib:${LD_LIBRARY_PATH:-}"
# Some KenLM source releases leave command-line tools in the build tree.
for tool in lmplz build_binary; do
    test -x "$prefix/bin/$tool" || install -D "$build/bin/$tool" "$prefix/bin/$tool"
done
{
    date -Is
    sha256sum "$archive" "$prefix/bin/lmplz" "$prefix/bin/build_binary"
    find "$prefix" -name 'libkenlm*.so*' -type f -exec sha256sum '{}' ';'
    ldd "$prefix/bin/lmplz"
    "$prefix/bin/lmplz" --help 2>&1 || :
} > "$project/receipts/kenlm-build-${SLURM_JOB_ID:?}.txt"
