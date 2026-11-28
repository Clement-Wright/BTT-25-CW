"""CPU T15 deployment benchmark. Run from the repository root."""

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import yaml

from .assets import sha256
from .benchmark import native_decoder, timing_summary
from .ctc import greedy_ctc
from .data import (
    attach_corpus_metadata,
    load_split_manifest,
    object_hash,
    read_trial,
    reference_only,
    write_json,
)
from .evaluation import _aggregate, _by_id, _score_row, _selection
from .model import gaussian_kernel, load_encoder, output_length, smooth_valid
from .resources import ResourceMonitor


def cpu_name():
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    try:
        return next(
            line.partition(":")[2].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        return platform.processor() or "unavailable"


def load_cpu(config, precision):
    model, upstream = load_encoder(
        config["checkpoint"], config["model_args"], "cpu", config["checkpoint_hash"]
    )
    if model.contract_hash != config["contract_hash"]:
        raise ValueError("Unexpected preprocessing contract")
    transforms = upstream["dataset"]["data_transforms"]
    kernel = gaussian_kernel(transforms["smooth_kernel_std"], transforms["smooth_kernel_size"])
    if precision == "int8":
        torch.backends.quantized.engine = "x86"
        torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )
        if not isinstance(model.gru, torch.ao.nn.quantized.dynamic.GRU):
            raise ValueError("Requested dynamic quantization was not applied")
    return model, kernel


def decoder_args(args):
    return argparse.Namespace(
        decoder_python=args.decoder_python,
        lexicon=args.lexicon,
        nbest=1,
        beam_threshold=25,
        kenlm_model=args.kenlm_model,
        lm_weight=args.lm_weight,
        start_timeout=90,
        request_timeout=30,
    )


@torch.inference_mode()
def predict(record, model, kernel, decode):
    trial = record["trial_id"]
    t0 = time.perf_counter()
    features = torch.from_numpy(read_trial(record)["features"]).unsqueeze(0)
    t1 = time.perf_counter()
    x = smooth_valid(features, kernel)
    t2 = time.perf_counter()
    logits = model(x, [record["day_index"]])[0].numpy()
    if logits.shape != (output_length(record["input_length"]), 41) or not np.isfinite(logits).all():
        raise ValueError("Invalid emissions")
    t3 = time.perf_counter()
    reply = decode(trial, logits)
    candidate = next(iter(reply["candidates"]), None)
    t4 = time.perf_counter()
    return {
        "trial_id": trial,
        "status": "ok" if candidate else "empty_search",
        "phoneme_status": "ok",
        "text": candidate["text"] if candidate else "",
        "phoneme_ids": greedy_ctc(logits.argmax(-1)),
        "input_seconds": record["input_length"] * 0.020,
        "latency_s": t4 - t0,
        "stages": {"read": t1 - t0, "smoothing": t2 - t1, "encoder": t3 - t2, "search_ipc": t4 - t3},
    }


def sample_lengths(records, n):
    ordered = sorted(records, key=lambda r: (r["input_length"], r["trial_id"]))
    return [ordered[int(i)] for i in np.linspace(0, len(ordered) - 1, n)]


def evidence_inputs(args):
    config = yaml.safe_load(args.config.read_text(encoding="utf8"))
    records = json.loads(Path(config["data_manifest"]).read_text(encoding="utf8"))
    split = load_split_manifest(config["split_manifest"], records)
    tune = _selection(_by_id(records, "data"), split, "dev_tune")
    if config.get("corpus_metadata"):
        tune = attach_corpus_metadata(tune, config["corpus_metadata"])
    return config, split, tune


def score(rows, records):
    by_id = _by_id(rows, "prediction")
    if set(by_id) != {r["trial_id"] for r in records}:
        raise ValueError("Unmatched prediction IDs")
    all_rows, counts, groups = [], [], {"per_session": defaultdict(list), "per_corpus": defaultdict(list)}
    for record in records:
        ref = reference_only(record)
        scored = dict(
            by_id[record["trial_id"]],
            reference_text=ref["reference_text"],
            reference_phoneme_ids=ref["phoneme_ids"],
        )
        index = len(all_rows)
        groups["per_session"][record["session_id"]].append(index)
        groups["per_corpus"][record.get("corpus_id") or "unknown"].append(index)
        all_rows.append(scored)
        counts.append(_score_row(scored))
    grouped = {
        name: {
            key: _aggregate([all_rows[i] for i in ids], [counts[i] for i in ids])
            for key, ids in group.items()
        }
        for name, group in groups.items()
    }
    return {"overall": _aggregate(all_rows, counts), "trial_edits": counts, **grouped}


def run(args):
    if os.getenv("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=-1 for this CPU benchmark")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA device visible in CPU-only run")
    args.out.mkdir(parents=True, exist_ok=False)
    config, split, tune = evidence_inputs(args)
    precisions = ["fp32", "int8"] if args.mode in {"pilot", "timing"} else [args.precision]
    records = sample_lengths(tune, 12) if args.mode == "pilot" else tune
    if args.mode == "timing":
        records = sample_lengths(tune, 96)
    decoder_version = subprocess.check_output(
        [
            str(args.decoder_python),
            "-c",
            "import importlib.metadata as m; print(m.version('flashlight-text'))",
        ],
        text=True,
    ).strip()
    root = Path(__file__).resolve().parents[2]
    source_hashes = {
        str(file.relative_to(root)).replace("\\", "/"): sha256(file)
        for file in (
            Path(__file__).resolve(),
            root / "scripts/decoder_server.py",
            root / "scripts/decode_worker.py",
            root / "src/btt25/model.py",
            root / "src/btt25/benchmark.py",
        )
    }
    resolved = {
        "mode": args.mode,
        "precision": precisions,
        "scope": "dev_tune",
        "split_hash": split["manifest_hash"],
        "checkpoint_hash": config["checkpoint_hash"],
        "contract_hash": config["contract_hash"],
        "ordered_id_hash": object_hash([r["trial_id"] for r in records]),
        "lexicon_sha256": sha256(args.lexicon),
        "lm": "Flashlight KenLM" if args.kenlm_model else "Flashlight ZeroLM",
        "lm_sha256": sha256(args.kenlm_model) if args.kenlm_model else None,
        "lm_weight_natural_log": args.lm_weight if args.kenlm_model else 0.0,
        "beam": 50,
        "nbest": 1,
        "threshold": 25,
        "cpu_threads": args.threads,
        "backend": "torch.ao.quantization.quantize_dynamic, CPU x86, qint8 GRU and Linear; day transforms FP32",
        "pilot_seconds": args.pilot_seconds,
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "flashlight_text": decoder_version,
        "cpu": cpu_name(),
        "data_manifest_sha256": sha256(config["data_manifest"]),
        "source_sha256": source_hashes,
    }
    (args.out / "config.resolved.yaml").write_text(yaml.safe_dump(resolved), encoding="utf8")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with ResourceMonitor(args.out / "resources.json", args.watchdog) as monitor, ExitStack() as stack:
        models, setup, rows = {}, {}, []
        for precision in precisions:
            started = time.perf_counter()
            models[precision] = load_cpu(config, precision)
            setup[precision] = {"load_convert_s": time.perf_counter() - started}
            if args.mode == "pilot":
                weights = args.out / f"{precision}.pt"
                torch.save(models[precision][0].state_dict(), weights)
                setup[precision].update(serialized_bytes=weights.stat().st_size, sha256=sha256(weights))
            gc.collect()
            monitor.sample()
        decode, setup["decoder"] = stack.enter_context(
            native_decoder(decoder_args(args), 50, args.out / "decoder.stderr.log")
        )
        for precision in precisions:
            started = time.perf_counter()
            predict(records[len(records) // 2], *models[precision], decode)
            setup[precision]["warmup_s"] = time.perf_counter() - started
        setup["measured_requests_start_elapsed_s"] = monitor.sample()["elapsed_seconds"]
        write_json(args.out / "setup.json", setup)
        started = time.perf_counter()
        with (args.out / "predictions.jsonl").open("x", encoding="utf8") as out:
            pass_index = 0
            while True:
                for record in records:
                    for precision in precisions if pass_index % 2 == 0 else reversed(precisions):
                        monitor.check()
                        row = predict(record, *models[precision], decode)
                        row.update(precision=precision, pass_index=pass_index)
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        rows.append(row)
                pass_index += 1
                if (
                    args.mode == "score"
                    or args.mode == "timing"
                    and pass_index >= 3
                    or args.mode == "pilot"
                    and time.perf_counter() - started >= args.pilot_seconds
                ):
                    break
        setup["measured_requests_end_elapsed_s"] = monitor.sample()["elapsed_seconds"]
        write_json(args.out / "setup.json", setup)
        metrics = {p: timing_summary([r for r in rows if r["precision"] == p]) for p in precisions}
        if args.mode == "score":
            metrics[args.precision].update(score(rows, records))
        if args.mode == "timing":
            for precision in precisions:
                ordered = sorted(records, key=lambda r: (r["input_length"], r["trial_id"]))
                for bin_index, group in enumerate(("short", "middle", "long")):
                    ids = {r["trial_id"] for r in ordered[bin_index * 32 : (bin_index + 1) * 32]}
                    metrics[precision][group] = timing_summary(
                        [r for r in rows if r["precision"] == precision and r["trial_id"] in ids]
                    )
        if args.mode == "pilot":
            for p in precisions:
                metrics[p]["quality_first_pass"] = score(
                    [r for r in rows if r["precision"] == p and r["pass_index"] == 0], records
                )["overall"]
        write_json(args.out / "metrics.json", metrics)
    result = [
        f"# CPU {args.mode} run",
        "",
        f"{len(rows):,} complete-pipeline requests on {len(records)} T15 tuning trials. "
        f"{args.threads} CPU threads, beam 50 with {resolved['lm']}.",
        "",
    ]
    for precision, item in metrics.items():
        result.append(
            f"{precision.upper()} p95 was {1000*item['latency_seconds']['p95']:.1f} ms. "
            f"Real-time factor was {item['successful_real_time_factor']:.4f}."
        )
        if "overall" in item:
            q = item["overall"]
            result.append(
                f"Word errors {q['word']['errors']}/{q['word']['reference_length']} "
                f"({q['wer']:.2%}). Phone errors {q['phoneme']['errors']}/{q['phoneme']['reference_length']} "
                f"({q['per']:.2%})."
            )
        result.append("")
    resources = json.loads((args.out / "resources.json").read_text(encoding="utf8"))
    result.append(f"The sampled process tree peaked at {resources['peak_tree_rss_bytes']/1e9:.3f} GB RSS.")
    (args.out / "report.md").write_text("\n".join(result) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "score", "timing"), required=True)
    parser.add_argument("--precision", choices=("fp32", "int8"), default="fp32")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--lexicon", type=Path, default=Path("assets/local/lexicon.txt"))
    parser.add_argument("--kenlm-model", type=Path)
    parser.add_argument("--lm-weight", type=float, default=2.0)
    parser.add_argument("--decoder-python", type=Path, default=Path(".decoder-venv/Scripts/python.exe"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pilot-seconds", type=int, default=900)
    parser.add_argument("--watchdog", type=int, default=28200)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
