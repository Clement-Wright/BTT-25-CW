"""Offline batch-one benchmark from features to text or cached logits to text."""
from base64 import b64encode
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import argparse
from argparse import Namespace
import json
from math import ceil
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
from threading import Thread
from time import perf_counter

import numpy as np
import torch

from .assets import sha256
from .data import load_split_manifest, object_hash, read_trial, write_json
from .inference import load_config, load_json, read_jsonl
from .model import gaussian_kernel, load_encoder, output_length, smooth_valid
import yaml
from .resources import BudgetExceeded, ResourceMonitor


SERVER = Path(__file__).resolve().parents[2] / "scripts/decoder_server.py"


def _receive(queue, deadline):
    try:
        line = queue.get(timeout=max(0, deadline - perf_counter()))
    except Empty as error:
        raise TimeoutError("Native decoder response timed out") from error
    if not line:
        raise RuntimeError("Native decoder exited; see its stderr artifact")
    return json.loads(line)


@contextmanager
def native_decoder(args, beam, log):
    cmd = [str(args.decoder_python), "-u", str(SERVER), "--lexicon", str(args.lexicon),
           "--beam", str(beam), "--nbest", str(args.nbest),
           "--beam-threshold", str(args.beam_threshold)]
    if args.kenlm_model:
        cmd += ["--kenlm-model", str(args.kenlm_model), "--lm-weight", str(args.lm_weight)]
    started = perf_counter()
    env = os.environ.copy()
    if getattr(args, "decoder_module_path", None):
        env["PYTHONPATH"] = str(args.decoder_module_path) + os.pathsep + env.get("PYTHONPATH", "")
    with log.open("w", encoding="utf-8") as error_log, subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error_log,
        text=True, encoding="utf-8", bufsize=1, env=env,
    ) as proc:
        queue = Queue()

        def read():
            for line in proc.stdout:
                queue.put(line)
            queue.put(None)

        reader = Thread(target=read, daemon=True)
        reader.start()
        try:
            started_service = _receive(queue, started + args.start_timeout)

            def decode(trial_id, array):
                deadline = perf_counter() + args.request_timeout
                payload = b64encode(np.asarray(array, dtype="<f4", order="C").tobytes()).decode("ascii")
                proc.stdin.write(json.dumps({"trial_id": trial_id, "output_length": len(array),
                                             "logits": payload}) + "\n")
                proc.stdin.flush()
                reply = _receive(queue, deadline)
                if reply["trial_id"] != trial_id:
                    raise RuntimeError("Native decoder response trial mismatch")
                return reply

            yield decode, {**started_service, "process_ready_s": perf_counter() - started}
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            reader.join(timeout=2)


def feature_encoder(config):
    device = torch.device(config.get("device", "cuda"))
    torch.set_num_threads(config.get("cpu_threads", 4))
    model, upstream = load_encoder(config["checkpoint"], config["model_args"], str(device),
                                   config["checkpoint_hash"])
    if model.contract_hash != config["contract_hash"]:
        raise ValueError("Checkpoint preprocessing contract mismatch")
    transforms = upstream["dataset"]["data_transforms"]
    kernel = gaussian_kernel(transforms["smooth_kernel_std"], transforms["smooth_kernel_size"])

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @torch.inference_mode()
    def encode(record):
        start = perf_counter()
        features = read_trial(record, include_labels=False)["features"]
        read_done = perf_counter()
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        day = torch.tensor([record["day_index"]], device=device)
        sync()
        upload_done = perf_counter()
        x = smooth_valid(x, kernel)
        sync()
        prep_done = perf_counter()
        logits = model(x, day)[0]
        sync()
        network_done = perf_counter()
        array = logits.cpu().numpy()
        expected = output_length(record["input_length"], len(kernel), model.patch_size, model.patch_stride)
        if array.shape != (expected, 41) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("Invalid output length, dtype or emission values")
        return array, {"feature_read_s": read_done - start, "upload_s": upload_done - read_done,
                       "preprocess_s": prep_done - upload_done, "encoder_s": network_done - prep_done,
                       "download_and_verify_s": perf_counter() - network_done}

    sync()
    return encode


def alternate_feature_encoder(config, checkpoint, expected_sha256):
    from .train_v2 import Encoder
    from .alternative_encoders import marginal_log_probs

    checkpoint = Path(checkpoint)
    if sha256(checkpoint) != expected_sha256:
        raise ValueError("Alternative benchmark checkpoint hash differs from its freeze")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    encoder = Encoder(config, saved["identity"]["mode"], config["device"])
    if encoder.contract != saved["contract"]:
        raise ValueError("Alternative benchmark checkpoint contract mismatch")
    encoder.model.load_state_dict(saved["model_state_dict"], strict=True)
    encoder.model.eval()
    device = torch.device(config["device"])

    @torch.inference_mode()
    def encode(record):
        began = perf_counter()
        features = read_trial(record, include_labels=False)["features"]
        read = perf_counter()
        logits = encoder.forward(features, record["day_index"], training=False)
        if encoder.mode == "diphone":
            logits = marginal_log_probs(logits.float().log_softmax(-1))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encoded = perf_counter()
        array = logits.cpu().numpy()
        length = record["input_length"] // 4 if encoder.mode == "transformer" else output_length(record["input_length"])
        if array.shape != (length, 41) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("Invalid alternative encoder output")
        return array, {"feature_read_s": read - began, "upload_prep_encoder_s": encoded - read,
                       "download_and_verify_s": perf_counter() - encoded}

    return encode


def cached_encoder(cache, config, records, split):
    meta = load_json(cache / "cache_manifest.json")
    for key in ("checkpoint_hash", "contract_hash"):
        if meta[key] != config[key]:
            raise ValueError(f"Logit cache differs from verified {key}")
    if (meta["partition"] != "val" or meta["references_loaded"] is not False
            or meta["manifest_hash"] != object_hash(records) or meta["split_hash"] != split["manifest_hash"]):
        raise ValueError("Expected reference-free validation logits from the verified data and split")
    args = yaml.safe_load(Path(config["model_args"]).read_text(encoding="utf-8"))
    params = args["model"]
    transforms = args["dataset"]["data_transforms"]
    kernel_length = len(gaussian_kernel(transforms["smooth_kernel_std"], transforms["smooth_kernel_size"]))
    source = read_jsonl(cache / "predictions.jsonl")
    index = {row["trial_id"]: row for row in source}
    if len(source) != len(index):
        raise ValueError("Duplicate cached prediction IDs")

    def encode(record):
        start = perf_counter()
        row = index[record["trial_id"]]
        path = (cache / row["logit_path"]).resolve()
        if (row["status"] != "ok" or row["input_length"] != record["input_length"]
                or not path.is_relative_to(cache.resolve()) or sha256(path) != row["logit_sha256"]):
            raise ValueError("Unverified cached emissions")
        logits = np.load(path, allow_pickle=False)
        length = output_length(record["input_length"], kernel_length, params["patch_size"], params["patch_stride"])
        if row["output_length"] != length or logits.shape != (length, 41) or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise ValueError("Cached emissions violate shape/dtype contract")
        return logits, {"cache_read_verify_s": perf_counter() - start}

    return encode


def timing_summary(rows):
    values = [r["latency_s"] for r in rows]
    success = [r for r in rows if r["status"] == "ok"]
    def dist(samples):
        return {f"p{q}": float(np.percentile(samples, q)) for q in (50, 90, 95, 99)} | {
            "max": max(samples)} if samples else None

    acoustic_seconds = sum(row["input_seconds"] for row in success)
    stages = defaultdict(list)
    for row in success:
        for key, value in row["stages"].items():
            stages[key].append(value)
    return {"requests": len(rows), "successes": len(success), "failures": len(rows) - len(success),
            "timeouts": sum(r["status"] == "timeout" for r in rows),
            "latency_seconds": dist(values), "successful_latency_seconds": dist([r["latency_s"] for r in success]),
            "stage_latency_seconds": {k: dist(v) for k, v in stages.items()},
            "successful_real_time_factor": sum(r["latency_s"] for r in success) / acoustic_seconds
            if acoustic_seconds else None, "successful_trials_per_request_second": len(success) / sum(values)
            if sum(values) else None}


def neural_ranker(settings):
    from .qwen_scorer import QwenScorer, cluster_guard
    from .rescoring import DistilGPT2Scorer

    cluster_guard()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Neural benchmark requires allocated BF16 CUDA")
    if settings["model"] in ("qwen17b", "qwen4b"):
        scorer = QwenScorer(Path(settings["path"]), size="1.7b" if settings["model"] == "qwen17b" else "4b")
        score = lambda texts: scorer.score(texts, batch_size=settings["batch_size"])
    elif settings["model"] == "distilgpt2":
        scorer = DistilGPT2Scorer(Path(settings["path"]), device="cuda:0", dtype=torch.bfloat16)

        def score(texts):
            return [(x["log_likelihood"], x["token_count"]) for x in scorer.score_many(
                texts, batch_size=settings["batch_size"])]
    else:
        raise ValueError("Unknown registered benchmark scorer")
    if scorer.metadata["revision"] != settings["revision"]:
        raise ValueError("Neural benchmark scorer revision mismatch")

    def rank(choices):
        subset = choices[:settings["candidate_limit"]]
        values = score([c["text"] for c in subset])
        torch.cuda.synchronize()
        return max(zip(subset, values, strict=True), key=lambda item: item[0]["component_scores"]["acoustic"] +
                   settings["weight"] * item[1][0])[0]["text"]

    return rank, scorer.metadata


def run(args):
    config = load_config(args.config)
    records = load_json(config["data_manifest"])
    split = load_split_manifest(config["split_manifest"], records)
    eligible = set(split["dev_tune"])
    selected = sorted((r for r in records if r["trial_id"] in eligible),
                      key=lambda r: (r["input_length"], r["official_order_index"]))
    if args.trials:
        if not 1 <= args.trials <= len(selected):
            raise ValueError("Benchmark scope requires a positive eligible trial count")
        selected = [selected[i] for i in np.linspace(0, len(selected) - 1, args.trials, dtype=int)]
    length_bins = {row["trial_id"]: ["short", "middle", "long"][min(2, 3 * i // len(selected))]
                   for i, row in enumerate(selected)}
    selected.sort(key=lambda r: r["official_order_index"])
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    mode = "cached-logit-to-text" if args.logit_cache else "released-features-to-text"
    cases = {}
    if args.cases:
        for case in load_json(args.cases):
            name = case["condition"]
            if name in cases or not name or not name.replace("-", "").isalnum():
                raise ValueError("Expected distinct filesystem-safe condition IDs")
            if not 1 <= case["nbest"] <= case["beam"]:
                raise ValueError("Candidate limit must fit the beam")
            cases[name] = Namespace(**{**vars(args), **case})
        if not cases:
            raise ValueError("No benchmark condition specified")
    else:
        cases = {beam: Namespace(**{**vars(args), "beam": beam}) for beam in args.beam}
    hashes = {k: sha256(v) for k, v in {"config": args.config, "lexicon": args.lexicon,
               "kenlm": args.kenlm_model, "data_manifest": config["data_manifest"],
               "server": SERVER, "cases": args.cases}.items() if v}
    case_identity = {str(key): {"beam": cfg.beam, "nbest": cfg.nbest, "lm_weight": cfg.lm_weight,
                     "kenlm_sha256": sha256(cfg.kenlm_model) if cfg.kenlm_model else None,
                     "native_revision": getattr(cfg, "native_revision", None),
                     "model": getattr(cfg, "neural", None)} for key, cfg in cases.items()}
    write_json(output / "config.resolved.yaml", {"mode": mode, "passes_minimum": args.passes, "pilot_seconds": args.pilot_seconds, "beam": args.beam,
               "nbest": args.nbest, "beam_threshold": args.beam_threshold, "lm_weight": args.lm_weight,
               "scope": "tuning features, references unopened", "request_timeout_s": args.request_timeout,
               "input_ids": [r["trial_id"] for r in selected], "source": config, "hashes": hashes,
               "decoder_python": str(args.decoder_python), "torch": str(torch.__version__), "cases": case_identity})
    rows = []
    state = "complete"
    setup = {}
    i = 0
    try:
        with ResourceMonitor(output / "resources.json", max_seconds=args.max_seconds) as monitor, ExitStack() as stack:
            started = perf_counter()
            encode = cached_encoder(args.logit_cache, config, records, split) if args.logit_cache else feature_encoder(config)
            setup["encoder_s"] = perf_counter() - started
            services = {}
            rankers = {}
            encoders = {}
            for name, case in cases.items():
                encoder_setup = None
                if getattr(case, "encoder_checkpoint", None):
                    if args.logit_cache:
                        raise ValueError("A feature model cannot replace the verified cached-logit encoder")
                    began = perf_counter()
                    encoders[name] = alternate_feature_encoder(config, case.encoder_checkpoint, case.encoder_sha256)
                    encoder_setup = perf_counter() - began
                else:
                    encoders[name] = encode
                decoder, receipt = stack.enter_context(native_decoder(case, case.beam, output / f"decoder-{name}.log"))
                if encoder_setup is not None:
                    receipt["encoder_setup_s"] = encoder_setup
                services[name], setup[str(name)] = decoder, receipt
                if getattr(case, "neural", None):
                    started = perf_counter()
                    rankers[name], metadata = neural_ranker(case.neural)
                    receipt.update(neural_setup_s=perf_counter() - started, neural_model=metadata)
                warm = perf_counter()
                array, _ = encoders[name](selected[len(selected) // 2])
                choices = decoder(selected[len(selected) // 2]["trial_id"], array)["candidates"]
                if name in rankers and choices:
                    rankers[name](choices)
                receipt["warmup_s"] = perf_counter() - warm
            with (output / "timing.jsonl").open("w", encoding="utf-8") as timings, (
                output / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
                i = 0
                pilot_start = perf_counter()
                case_order = list(cases)
                while i < args.passes or perf_counter() - pilot_start < args.pilot_seconds:
                    offset = i % len(case_order)
                    order = case_order[offset:] + case_order[:offset]
                    for record in selected:
                        for name in order:
                            monitor.check()
                            trial = record["trial_id"]
                            start = perf_counter()
                            stages = {}
                            failure = None
                            choices, text = [], ""
                            try:
                                array, stages = encoders[name](record)
                                encoded = perf_counter()
                                reply = services[name](trial, array)
                                searched = perf_counter()
                                choices = reply["candidates"]
                                if choices:
                                    text = rankers[name](choices) if name in rankers else choices[0]["text"]
                                finished = perf_counter()
                                stages.update(native_search_s=reply["search_s"], ipc_search_s=searched - encoded,
                                              neural_and_selection_s=finished - searched)
                            except Exception as error:
                                failure = error
                                finished = perf_counter()
                            status = ("timeout" if isinstance(failure, TimeoutError) else "failed" if failure
                                      else "ok" if choices else "decoder_error")
                            row = {"trial_id": trial, "beam": cases[name].beam, "condition": str(name),
                                   "pass": i + 1, "status": status,
                                   "latency_s": finished - start, "input_seconds": record["input_length"] * .020,
                                   "length_bin": length_bins[trial], "stages": stages}
                            rows.append(row)
                            timings.write(json.dumps(row) + "\n")
                            timings.flush()
                            predictions.write(json.dumps({"trial_id": trial, "beam": cases[name].beam,
                                                          "condition": str(name), "pass": i + 1,
                                                          "status": status, "text": text}) + "\n")
                            predictions.flush()
                            if failure:
                                state = "timeout" if isinstance(failure, TimeoutError) else "failed"
                                raise failure
                    i += 1
                setup["sustained_warm_s"] = perf_counter() - pilot_start
                setup["completed_passes"] = i
            monitor.check()
    except BaseException as error:
        if state != "timeout":
            state = "budget_stopped" if isinstance(error, (BudgetExceeded, KeyboardInterrupt)) else "failed"
        raise
    finally:
        requests_per_pass = len(selected) * len(cases)
        expected_passes = max(args.passes, ceil(len(rows) / requests_per_pass))
        metrics = {"state": state, "mode": mode, "streaming": "unsupported", "references_loaded": False,
                   "cold_setup": setup, "minimum_requested": requests_per_pass * args.passes,
                   "attempted_requests": len(rows), "completed_passes": i,
                   "expected_requests": requests_per_pass * expected_passes,
                   "memory_scope": "combined persistent benchmark workers, no concurrent requests",
                   "by_condition" if args.cases else "by_beam": {}}
        for name in cases:
            subset = [r for r in rows if r["condition"] == str(name)]
            metrics["by_condition" if args.cases else "by_beam"][str(name)] = {**timing_summary(subset), "by_length": {
                group: timing_summary([r for r in subset if r["length_bin"] == group])
                for group in ("short", "middle", "long")}}
        write_json(output / "metrics.json", metrics)
        (output / "report.md").write_text(f"# Offline pipeline benchmark\n\nMode `{mode}`. "
            f"State `{state}`. Saved {len(rows)} of {metrics['expected_requests']} requests. "
            "See metrics.json for latency, throughput and real-time factor. "
            "The trace includes per-request feature/cache I/O, transfer, encoder and native search where applicable.\n",
            encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "output", "decoder-python", "lexicon"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--kenlm-model", type=Path)
    parser.add_argument("--logit-cache", type=Path)
    parser.add_argument("--cases", type=Path, help="JSON array of conditions for interleaved comparisons")
    parser.add_argument("--lm-weight", type=float, default=2)
    parser.add_argument("--beam", type=int, nargs="+", default=[50])
    parser.add_argument("--beam-threshold", type=float, default=25)
    parser.add_argument("--nbest", type=int, default=10)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--start-timeout", type=float, default=120)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--pilot-seconds", type=int, default=0)
    args = parser.parse_args()
    if len(args.beam) != len(set(args.beam)) or not args.passes >= 3 or not 1 <= args.nbest <= min(args.beam):
        parser.error("Require distinct beams, nbest within beam width, and at least three warm passes")
    if args.pilot_seconds != 0 and not 900 <= args.pilot_seconds <= 1800:
        parser.error("A sustained pilot must request 900 to 1800 seconds")
    run(args)


if __name__ == "__main__":
    main()
