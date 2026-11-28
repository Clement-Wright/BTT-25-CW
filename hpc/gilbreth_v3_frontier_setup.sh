#!/usr/bin/env bash
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
campaign="${1:?Supply campaign path}"
out="$campaign/runs/frontier-setup-${SLURM_JOB_ID:?}"
mkdir -p "$out" "$campaign/assets" "$campaign/vendor"
export PYTHONPATH="$campaign/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
scontrol show job -o "$SLURM_JOB_ID" > "$out/scheduler-start.txt"
set +u
source /etc/profile.d/modules.sh
module load cuda/12.6.0
set -u

"$project/environments/baseline/bin/python" - "$campaign" "$out" <<'PY'
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from btt25.data import write_json
from btt25.resources import ResourceMonitor

root, out = map(Path, sys.argv[1:])
models = [
    ("Qwen3.8-27B", "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
     "c1869413dece09109e55b7dcc2c77ef382b0d0bd",
     "d15d4b6de55f662ac40e867702d31afe738d032fa06b829b2b1a21a49d4a48ee"),
    ("Qwen3.6-35B-A3B", "345d467b117d1fd5eda9b879a5988de9ae5f47a7",
     "52f9f9b8052fb9c8b69bba4a95eda0f319ac3d2a",
     "f4ffe03c9139e6a905a5d346f77d9fabdd179367c3280dfd005d1e75ca5ed8c8"),
]
commit = "28ff0958291ce3465fabd7bd679d4b0edd742bd9"
config = {"runtime": {"repo": "https://github.com/ggml-org/llama.cpp.git", "commit": commit},
          "models": models, "license": "Apache-2.0", "precision": "Q4_K_M"}
write_json(out / "config.resolved.yaml", config)
status = {"state": "started", "models": {}}
started = time.monotonic()
try:
    with ResourceMonitor(out / "resources.json", max_seconds=12500) as monitor, \
            (out / "setup.log").open("w", encoding="utf-8") as log:
        def run(command):
            with subprocess.Popen(command, stdout=log, stderr=log) as child:
                try:
                    while child.poll() is None:
                        monitor.check()
                        time.sleep(1)
                    if child.returncode:
                        raise RuntimeError(f"{command[0]} returned {child.returncode}")
                finally:
                    if child.poll() is None:
                        child.terminate()

        source = root / f"vendor/llama.cpp-{commit}"
        if not (source / ".git").exists():
            run(["git", "clone", "--depth", "1", "--branch", "b10901", config["runtime"]["repo"], str(source)])
        if subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() != commit:
            raise RuntimeError("Unexpected llama.cpp commit")
        build = source / "build"
        run(["cmake", "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
             "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=80", "-DBUILD_SHARED_LIBS=ON",
             "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=OFF"])
        run(["cmake", "--build", build, "--parallel", "4", "--target", "llama"])
        for name, base_revision, revision, expected in models:
            then = time.monotonic()
            filename = f"{name}-Q4_K_M.gguf"
            model = root / "assets" / filename
            partial = model.with_suffix(".part")
            repo = f"ggml-org/{name}-GGUF"
            if not model.is_file():
                offset = partial.stat().st_size if partial.exists() else 0
                request = Request(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
                                  headers={"Range": f"bytes={offset}-"} if offset else {})
                with urlopen(request, timeout=90) as response, partial.open("ab") as stream:
                    content_range = response.headers.get("Content-Range", "")
                    if offset and (response.status != 206 or not content_range.startswith(f"bytes {offset}-")):
                        raise ValueError("The model host did not honor the resume range")
                    dirty_start = offset
                    while chunk := response.read(8 * 1024 * 1024):
                        stream.write(chunk)
                        if stream.tell() - dirty_start >= 64 * 1024 * 1024:
                            stream.flush()
                            os.fdatasync(stream.fileno())
                            os.posix_fadvise(stream.fileno(), dirty_start, stream.tell() - dirty_start,
                                             os.POSIX_FADV_DONTNEED)
                            dirty_start = stream.tell()
                        monitor.check()
            digest = hashlib.sha256()
            downloaded = model if model.exists() else partial
            with downloaded.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
                    os.posix_fadvise(stream.fileno(), stream.tell() - len(chunk), len(chunk), os.POSIX_FADV_DONTNEED)
                    monitor.check()
            if digest.hexdigest() != expected:
                raise ValueError(f"Unexpected weight digest for {filename}")
            if downloaded == partial:
                partial.rename(model)
            status["models"][name] = {"base": f"Qwen/{name}", "base_revision": base_revision,
                                     "repo": repo, "revision": revision, "sha256": expected,
                                     "bytes": model.stat().st_size, "setup_s": time.monotonic() - then}
            write_json(out / "provenance.json", status)
        status["state"] = "complete"
except BaseException as exc:
    status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    raise
finally:
    status["elapsed_s"] = time.monotonic() - started
    write_json(out / "metrics.json", status)
    (out / "report.md").write_text(
        f"# Frontier language model setup\n\nStatus: {status['state']}. "
        f"Verified assets: {', '.join(status['models'])}. No model execution.\n", encoding="utf-8")
PY
