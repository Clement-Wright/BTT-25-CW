#!/usr/bin/env bash
# Acquire and verify the official archive in an allocated node.
set -eo pipefail
umask 077
source /etc/profile
set -u
project="${RCAC_SCRATCH:?}/btt-25-cw"
cd "$project/workspace"
archive="$project/data/t15_copyTask_neuralData.zip"
url=https://huggingface.co/datasets/yw784/brain-to-text-25-competition/resolve/2f2fabacbc9af637da3bac483bf14e1231fb4e8a/t15_copyTask_neuralData.zip
echo "fetch_started=$(date -Is)"
if [[ ! -f "$archive" ]]; then
    curl --fail --location --retry 2 --connect-timeout 30 --continue-at - --output "$archive.partial" "$url"
    printf '%s  %s\n' fb828fa7ef937aca405d6912addc876f6747f75a8a1711f6d2052aacf34ec5b5 "$archive.partial" | sha256sum --check
    mv "$archive.partial" "$archive"
else
    printf '%s  %s\n' fb828fa7ef937aca405d6912addc876f6747f75a8a1711f6d2052aacf34ec5b5 "$archive" | sha256sum --check
fi
echo "verified_archive=$(date -Is)"
python3 scripts/expand_release.py --archive "$archive" --data-root "$project/data/archive-expanded/hdf5_data_final" \
    --hashes assets/t15_expanded_sha256.json --receipt "$project/receipts/archive-expanded.json"
echo "verified_release=$(date -Is)"
