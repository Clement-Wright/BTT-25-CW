"""Score frozen T15 candidates with a pinned GGUF model inside Gilbreth."""
import argparse
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
from math import isfinite
from pathlib import Path
import re
import subprocess
from time import perf_counter

import numpy as np

from .assets import sha256
from .benchmark import timing_summary
from .data import load_split_manifest, reference_only, write_json
from .evaluation import _by_id, _selection, edit_counts, normalize_text
from .fixed_candidates_v2 import rank, set_weights
from .inference import load_config, load_json, read_jsonl
from .qwen_scorer import cluster_guard
from .resources import BudgetExceeded, ResourceMonitor


@contextmanager
def backend(binary, model, log, monitor):
    with log.open("w", encoding="utf-8") as stderr, subprocess.Popen(
            [binary, model], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            text=True, encoding="utf-8", bufsize=1,
    ) as child, ThreadPoolExecutor(max_workers=1) as reader:
        def receive(timeout):
            deadline = perf_counter() + timeout
            pending = reader.submit(child.stdout.readline)
            while True:
                monitor.check()
                if perf_counter() >= deadline:
                    raise TimeoutError("GGUF backend response deadline")
                try:
                    response = pending.result(timeout=.5)
                except TimeoutError:
                    continue
                if not response:
                    raise RuntimeError("GGUF backend stopped; inspect backend.log")
                return response.split()

        try:
            bos, vocabulary, serial, batch, repeat, mode, weight_bytes = receive(360)
            if int(bos) <= 0 or int(vocabulary) < 100_000 or int(weight_bytes) < 10_000_000_000:
                raise ValueError("Unexpected GGUF model or tokenizer")
            evidence = log.read_text(encoding="utf-8")
            layers = re.search(r"offloaded (\d+)/(\d+) layers to GPU", evidence)
            if not layers or layers[1] != layers[2]:
                raise RuntimeError("Model did not verify complete GPU layer placement")
            buffers = {name: float(mib) for name, mib in re.findall(
                r"([\w-]+) model buffer size\s*=\s*([\d.]+) MiB", evidence)}
            if not any(name.startswith("CUDA") and "Host" not in name for name in buffers):
                raise RuntimeError("No CUDA model-weight allocation was recorded")
            host_mib = sum(size for name, size in buffers.items() if not name.startswith("CUDA") or "Host" in name)
            if host_mib > 1.0:
                raise RuntimeError(f"Unexpected host-resident model weights: {host_mib} MiB")

            def score_candidates(choices):
                scores = []
                for choice in choices:
                    text = choice["text"]
                    if any(char in text for char in "\n\r\x00"):
                        raise ValueError("Candidate contains an unexpected record delimiter")
                    child.stdin.write(text + "\n")
                    child.stdin.flush()
                    logprob, token_count = receive(20)
                    if not isfinite(float(logprob)) or not 1 <= int(token_count) <= 193:
                        raise ValueError("Invalid candidate likelihood from GGUF backend")
                    scores.append({**choice, "neural_logprob": float(logprob), "neural_tokens": int(token_count)})
                return scores

            yield score_candidates, {"delimiter": int(bos), "vocabulary": int(vocabulary),
                                     "serial_fixture_nats": float(serial), "batch_fixture_nats": float(batch),
                                     "repeat_fixture_nats": float(repeat), "mode": mode,
                                     "weight_bytes": int(weight_bytes), "gpu_layers": int(layers[1]),
                                     "model_buffer_mib": buffers}
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()


def score_predictions(output, packets, predictions, references, weight):
    counts = Counter()
    with output.open("w", encoding="utf-8", buffering=1) as stream:
        for trial, words in references.items():
            choices = packets[trial]
            best = max(choices, key=lambda c: rank(c, weight), default=None)
            text = best["text"] if best else ""
            changes = edit_counts(words, normalize_text(text).split())
            baseline = edit_counts(words, normalize_text(predictions[trial]["text"]).split())
            counts.update(changes)
            counts.update(baseline_errors=baseline["errors"], trials=1, empty=int(best is None),
                          improved=int(changes["errors"] < baseline["errors"]),
                          worsened=int(changes["errors"] > baseline["errors"]))
            stream.write(json.dumps({"trial_id": trial, "text": text, "counts": changes,
                                     "status": "ok" if best else "empty"}) + "\n")
    return counts


def prior_pilot(directory, contract, source):
    previous = load_json(directory / "config.resolved.yaml")
    for key in ("candidate_limit", "weights", "model", "split_hash", "scope", "pilot_trial_ids", "policy", "fusion"):
        if previous[key] != contract[key]:
            raise ValueError(f"Previous pilot differs in {key}")
    source_files = ("candidates.jsonl", "predictions.jsonl")
    if any(previous["inputs"][name] != contract["inputs"][name] for name in source_files):
        raise ValueError("Previous pilot used a different candidate cache")
    old_binaries = [digest for name, digest in previous["inputs"].items() if name.startswith("gguf-candidate-score-")]
    new_binaries = [digest for name, digest in contract["inputs"].items() if name.startswith("gguf-candidate-score-")]
    if len(old_binaries) != 1 or old_binaries != new_binaries:
        raise ValueError("Previous pilot used a different scoring binary")
    receipt = load_json(directory / "metrics.json")
    resources = load_json(directory / "resources.json")
    if (receipt["state"] != "budget_stopped" or receipt.get("pilot_passes", 0) < 3
            or receipt.get("error") != "BudgetExceeded: Pilot forecast exceeds the remaining study budget"
            or resources["peak_accounted_host_bytes"] >= resources["host_target_bytes"]):
        raise ValueError("Previous pilot lacks a completed throughput and resource receipt")
    packets = {trial: row["candidates"] for trial, row in
               _by_id(read_jsonl(directory / "candidate-scores.jsonl"), "prior candidates").items()}
    if set(packets) != set(contract["pilot_trial_ids"]):
        raise ValueError("Previous pilot has incomplete or additional candidate scores")
    for trial, candidates in packets.items():
        original = source[trial]["candidates"][:contract["candidate_limit"]]
        if len(candidates) != len(original):
            raise ValueError(f"Previous pilot candidate coverage differs for {trial}")
        for scored, submitted in zip(candidates, original, strict=True):
            if (any(scored[key] != value for key, value in submitted.items())
                    or not isfinite(scored["neural_logprob"]) or not 1 <= scored["neural_tokens"] <= 193):
                raise ValueError(f"Invalid previous pilot candidate for {trial}")
    return receipt, packets


def run(args):
    cluster_guard()
    if not 0 < args.max_seconds <= 28_200:
        raise ValueError("Use an application deadline of at most 28,200 seconds")
    config = load_config(args.config)
    records = load_json(config["data_manifest"])
    split = load_split_manifest(config["split_manifest"], records)
    selected = _selection(_by_id(records, "trial"), split, "dev_tune")
    length_sorted = sorted(selected, key=lambda r: r["input_length"])
    pilots = [length_sorted[i] for i in np.linspace(0, len(selected) - 1, 24, dtype=int)]
    source_rows = _by_id(read_jsonl(args.source / "candidates.jsonl"), "candidates")
    predictions = _by_id(read_jsonl(args.source / "predictions.jsonl"), "saved text")
    neural = _by_id(read_jsonl(args.neural), "neural predictions")
    for row in selected:
        trial = row["trial_id"]
        identities = (source_rows[trial], predictions[trial], neural[trial])
        if len({r["logit_sha256"] for r in identities}) != 1:
            raise ValueError(f"Emission identity mismatch for {trial}")
    args.output.mkdir(parents=True, exist_ok=False)
    model_name = args.model.name.removesuffix("-Q4_K_M.gguf")
    acquisition = load_json(args.acquisition)
    identity = acquisition["models"][model_name]
    if acquisition["state"] != "complete" or args.model.stat().st_size != identity["bytes"]:
        raise ValueError("Model lacks a verified acquisition receipt")
    contract = {"candidate_limit": 25, "weights": [.25, .5, .75, 1., 1.25],
                "model": identity, "split_hash": split["manifest_hash"], "scope": "dev_tune",
                "max_seconds": args.max_seconds,
                "pilot_seconds": 900, "pilot_fixed_weight": 1.25,
                "pilot_trial_ids": [r["trial_id"] for r in pilots],
                "policy": "Exact lowercase candidate tokens, endoftext at both ends, FP32 sum, no chat template",
                "fusion": "acoustic + weight * Qwen likelihood", "general_pretraining_overlap": "unknown",
                "inputs": {
                    **{p.name: sha256(p) for p in (args.binary, args.acquisition,
                        args.source / "candidates.jsonl", args.source / "predictions.jsonl", Path(__file__))},
                    "neural-predictions.jsonl": sha256(args.neural),
                }}
    previous, cached = (None, {}) if args.prior is None else prior_pilot(args.prior, contract, source_rows)
    if previous is not None:
        contract["prior_pilot"] = {
            "run": str(args.prior),
            "sha256": {name: sha256(args.prior / name) for name in (
                "metrics.json", "config.resolved.yaml", "resources.json", "candidate-scores.jsonl", "timing.jsonl")},
        }
    write_json(args.output / "config.resolved.yaml", contract)
    write_json(args.output / "provenance.json", contract["inputs"])
    status = {"state": "started"}
    started = perf_counter()
    try:
        with (
            ResourceMonitor(args.output / "resources.json", max_seconds=args.max_seconds) as monitor,
            backend(args.binary, args.model, args.output / "backend.log", monitor) as (score, info),
        ):
            status.update(backend=info, load_seconds=perf_counter() - started)
            history, packets = [], {}
            with (
                (args.output / "timing.jsonl").open("w", encoding="utf-8", buffering=1) as trace,
                (args.output / "candidate-scores.jsonl").open("w", encoding="utf-8", buffering=1) as raw,
            ):
                def request(row, phase):
                    monitor.check()
                    trial = row["trial_id"]
                    prior = source_rows[trial]["candidates"][:contract["candidate_limit"]]
                    now = perf_counter()
                    candidates = score(prior)
                    elapsed = perf_counter() - now
                    receipt = {"trial_id": trial, "phase": phase, "latency_s": elapsed,
                               "input_seconds": row["input_length"] * .020, "status": "ok" if candidates else "empty",
                               "stages": {"gguf_likelihood": elapsed}, "candidates": len(candidates),
                               "tokens": sum(c["neural_tokens"] for c in candidates)}
                    trace.write(json.dumps(receipt, allow_nan=False) + "\n")
                    if trial not in packets:
                        raw.write(json.dumps({"trial_id": trial, "candidates": candidates}, allow_nan=False) + "\n")
                        packets[trial] = candidates
                    return receipt

                if previous is None:
                    pilot_start = perf_counter()
                    passes = 0
                    while passes < 3 or perf_counter() - pilot_start < contract["pilot_seconds"]:
                        order = pilots if passes % 2 == 0 else pilots[::-1]
                        history.extend(request(row, "pilot") for row in order)
                        passes += 1
                        if passes == 1:
                            pilot_references = {
                                r["trial_id"]: normalize_text(reference_only(r)["reference_text"]).split()
                                for r in pilots}
                            counts = score_predictions(args.output / "pilot-predictions.jsonl", packets, predictions,
                                                       pilot_references, contract["pilot_fixed_weight"])
                            status["pilot_fixed"] = {"counts": counts, "weight": contract["pilot_fixed_weight"],
                                                     "wer": counts["errors"] / counts["reference_length"]}
                            write_json(args.output / "metrics.json", status)
                    status.update(pilot=timing_summary(history), pilot_passes=passes,
                                  candidates=sum(row["candidates"] for row in history),
                                  tokens=sum(row["tokens"] for row in history))
                else:
                    stable = ("delimiter", "vocabulary", "weight_bytes", "gpu_layers", "model_buffer_mib", "mode")
                    if any(previous["backend"][key] != info[key] for key in stable):
                        raise ValueError("Backend identity or GPU placement differs from the prior pilot")
                    if abs(previous["backend"]["repeat_fixture_nats"] - info["repeat_fixture_nats"]) > .005:
                        raise ValueError("Backend fixture differs from the prior pilot")
                    first = max(pilots, key=lambda row: len(cached[row["trial_id"]]))
                    request(first, "resume_check")
                    for new, old in zip(packets[first["trial_id"]], cached[first["trial_id"]], strict=True):
                        if (new["neural_tokens"] != old["neural_tokens"]
                                or abs(new["neural_logprob"] - old["neural_logprob"]) > .005):
                            raise ValueError("Pilot scoring changed in the resumed allocation")
                    for trial, candidates in cached.items():
                        if trial not in packets:
                            raw.write(json.dumps({"trial_id": trial, "candidates": candidates}, allow_nan=False) + "\n")
                            packets[trial] = candidates
                    status.update({key: previous[key] for key in ("pilot", "pilot_fixed", "pilot_passes", "candidates", "tokens")})
                    status["prior_pilot"] = contract["prior_pilot"]
                remaining = [row for row in selected if row["trial_id"] not in packets]
                p90 = status["pilot"]["latency_seconds"]["p90"]
                status["forecast_seconds"] = len(remaining) * p90 * 1.25 + 90.
                if status["forecast_seconds"] + perf_counter() - started > args.max_seconds:
                    raise BudgetExceeded("Pilot forecast exceeds the remaining study budget")
                study = [request(row, "tune") for row in remaining]
            status["timing_tune"] = timing_summary(study)
            references = {row["trial_id"]: normalize_text(reference_only(row)["reference_text"]).split()
                          for row in selected}
            weight, grid = set_weights(packets, references, contract["weights"])
            counts = score_predictions(args.output / "predictions.jsonl", packets, predictions, references, weight)
            monitor.check()
            status.update(state="complete", counts=counts, weight=weight, weight_grid_edits=grid,
                          wer=counts["errors"] / counts["reference_length"])
        if load_json(args.output / "resources.json")["outcome"] != "complete":
            raise BudgetExceeded("Resource guard stopped the run")
    except BaseException as exc:
        status.update(state="budget_stopped" if isinstance(exc, BudgetExceeded) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status["elapsed_s"] = perf_counter() - started
        write_json(args.output / "metrics.json", status)
        (args.output / "report.md").write_text(
            f"# {model_name} fixed-candidate scoring\n\nStatus: {status['state']}. "
            f"Tuning WER: {status.get('wer')}. See timing and candidate scores for the measured work.\n",
            encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "source", "neural", "binary", "model", "acquisition", "output"):
        cli.add_argument(f"--{key}", required=True, type=Path)
    cli.add_argument("--max-seconds", type=int, default=5100)
    cli.add_argument("--prior", type=Path)
    run(cli.parse_args())
