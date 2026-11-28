#!/usr/bin/env bash
set -euo pipefail
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
campaign="${1:?Supply campaign path}"
job="${SLURM_JOB_ID:?}"
out="$campaign/runs/frontier-matched-timing-$job"
mkdir "$out"
set +u
source /etc/profile.d/modules.sh
module load cuda/12.6.0
set -u
export LD_LIBRARY_PATH="$campaign/vendor/llama.cpp-28ff0958291ce3465fabd7bd679d4b0edd742bd9/build/bin:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$campaign/src" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export BTT_STOP_FILE="$out/stop"
trap 'touch "$BTT_STOP_FILE"' USR1 TERM INT
scontrol show job -o "$job" > "$out/scheduler-start.txt"
"$project/environments/baseline/bin/python" - "$project" "$campaign" "$out" "${2:?Supply Qwen3.6 job ID}" "${3:?Supply Qwen3.8 pilot job ID}" <<'PY' &
import json
from pathlib import Path
import sys
from time import perf_counter

from btt25.assets import sha256
from btt25.benchmark import timing_summary
from btt25.data import write_json
from btt25.gguf_study import backend
from btt25.inference import load_json, read_jsonl
from btt25.resources import ResourceMonitor

project, campaign, out = map(Path, sys.argv[1:4])
arms = dict(zip(("Qwen3.6-35B-A3B", "Qwen3.8-27B"), sys.argv[4:], strict=True))
decoding = project / "v2/runs/v2-search-attempt-12523602/v2-s6-repaired-250/decode/candidates.jsonl"
candidate_hash = sha256(decoding)
configs = {name: load_json(campaign / f"runs/frontier-{name}-{job}/config.resolved.yaml")
           for name, job in arms.items()}
ids = configs[next(iter(arms))]["pilot_trial_ids"]
if any(c["pilot_trial_ids"] != ids or c["candidate_limit"] != 25
       or c["inputs"]["candidates.jsonl"] != candidate_hash for c in configs.values()):
    raise ValueError("Require matched pilot IDs and the unchanged candidate cache")
lengths = {row["trial_id"]: row["input_length"] for row in
           load_json(project / "workspace/assets/local/data_manifest.json") if row["trial_id"] in ids}
packets = {r["trial_id"]: r["candidates"][:25] for r in read_jsonl(decoding) if r["trial_id"] in lengths}
if len(ids) != 24 or set(packets) != set(ids):
    raise ValueError("Require all 24 original pilot trials")
write_json(out / "config.resolved.yaml", {"candidate_sha256": candidate_hash, "pilot_ids": ids,
            "source_jobs": arms, "scope": "cached candidates only", "orders": ["AB", "BA", "AB"],
            "model_revisions": {name: c["model"] for name, c in configs.items()}})
write_json(out / "provenance.json", {"candidate_sha256": candidate_hash,
           "scorers": {name: c["inputs"] for name, c in configs.items()}})
results = {name: [] for name in arms}
cold = {name: [] for name in arms}
state = "failed"
try:
    with ResourceMonitor(out / "resources.json", max_seconds=1200) as monitor, \
            (out / "timing.jsonl").open("w", buffering=1, encoding="utf-8") as log:
        order = list(arms)
        for repeat in range(3):
            for name in order if repeat != 1 else order[::-1]:
                gpu_binary = campaign / f"vendor/gguf-candidate-score-{arms[name]}"
                if sha256(gpu_binary) != configs[name]["inputs"][gpu_binary.name]:
                    raise ValueError(f"Scorer binary hash changed for {name}")
                now = perf_counter()
                with backend(gpu_binary, campaign / f"assets/{name}-Q4_K_M.gguf",
                             out / f"{name}-{repeat}.log", monitor) as (score, info):
                    cold[name].append(perf_counter() - now)
                    if info["mode"] != "serial":
                        raise ValueError("Timing requires the same serial mode as the accuracy runs")
                    for trial in ids:
                        start = perf_counter()
                        choices = score(packets[trial])
                        elapsed = perf_counter() - start
                        row = {"model": name, "pass": repeat, "trial_id": trial, "latency_s": elapsed,
                               "input_seconds": lengths[trial] * .020, "status": "ok" if choices else "empty",
                               "stages": {"gguf_likelihood": elapsed}, "candidates": len(choices),
                               "tokens": sum(c["neural_tokens"] for c in choices)}
                        log.write(json.dumps(row) + "\n")
                        results[name].append(row)
                monitor.check()
    state = load_json(out / "resources.json")["outcome"]
finally:
    summary = {name: timing_summary(rows) for name, rows in results.items()}
    write_json(out / "metrics.json", {"state": state, "cold_load_seconds": cold, "warm": summary})
    (out / "report.md").write_text("# Matched cached-candidate timing\n\n"
        + "\n".join(f"{name} completed {m['requests']} timed requests." for name, m in summary.items())
        + f"\n\nState `{state}`.\n", encoding="utf-8")
if state != "complete":
    raise RuntimeError(f"Timing did not complete: {state}")
PY
child=$!
while true; do
    code=0
    wait "$child" || code=$?
    if ! kill -0 "$child" 2>/dev/null; then exit "$code"; fi
done
