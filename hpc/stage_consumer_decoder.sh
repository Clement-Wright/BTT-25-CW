#!/usr/bin/env bash
# Run inside WSL, in the project checkout. Stage only the matched native decoder.
set -euo pipefail
umask 077
socket=${1:?Supply the existing control socket}
host=${2:?Supply the authorized user and host}
project=${3:?Supply the cluster project directory}
boostlib=${4:?Supply the cluster Boost library directory}
dest=${5:?Supply a new private staging directory}
mkdir -p "$dest"
dest=$(realpath "$dest")
if [[ -s "$dest/transfer.sha256" ]]; then
  (cd "$dest" && sha256sum --check --quiet transfer.sha256)
  printf 'Reusing verified private decoder in %s\n' "$dest"
  exit 0
fi
mkdir -p "$dest/packages" "$dest/lib" "$dest/lm"
source="${BTT_REPAIRED_PACKAGES:-$project/v2/cache/flashlight-unlimited-12523602/site-packages}"
ssh_args=(-S "$socket" -o BatchMode=yes)
export RSYNC_RSH="ssh -S $socket -o BatchMode=yes"
started=$SECONDS
rsync -rLt --checksum --exclude='__pycache__' --exclude='*.pyc' "$host:$source/flashlight" "$host:$source/flashlight_text-0.0.7.dist-info" "$dest/packages/"
for lib in libkenlm.so libkenlm_util.so; do
  rsync -Lt --checksum "$host:$project/environments/kenlm/lib/$lib" "$dest/lib/"
done
for name in program_options system thread unit_test_framework; do
  rsync -Lt --checksum "$host:$boostlib/libboost_$name.so.1.82.0" "$dest/lib/"
done
lm="${BTT_TRAIN_TRIGRAM:-$project/workspace/runs/compact-a100/lm/train-3gram.binary}"
rsync -Lt --checksum "$host:$lm" "$dest/lm/train-3gram.binary"
printf -v quoted_lm '%q' "$lm"
remote_hash=$(ssh "${ssh_args[@]}" "$host" "sha256sum -- $quoted_lm" | cut -d' ' -f1)
test "$(sha256sum "$dest/lm/train-3gram.binary" | cut -d' ' -f1)" = "$remote_hash"
(cd "$dest" && find packages lib lm -type f -print0 | sort -z | xargs -0 sha256sum > transfer.sha256)
printf 'elapsed_seconds=%s\n' "$((SECONDS-started))" > "$dest/transfer.time"
