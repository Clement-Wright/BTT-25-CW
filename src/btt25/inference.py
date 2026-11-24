"""Incremental reference-free logits and phoneme predictions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .assets import sha256
from .ctc import greedy_ctc
from .data import TOKENS, read_trial, object_hash, write_json,load_split_manifest
from .model import load_encoder, gaussian_kernel, smooth_valid, output_length
from .resources import ResourceMonitor, BudgetExceeded


def validate_cache_metadata(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Logit cache mismatch: {key}")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("checkpoint", "model_args", "data_manifest", "split_manifest"):
        if key not in config:
            raise ValueError(f"Configuration missing {key}")
    return config


def recover_cached_predictions(run_dir,expected,allowed):
    run_dir=Path(run_dir)
    cache=run_dir/"cache_manifest.json"
    output=run_dir/"predictions.jsonl"
    if output.exists() and not cache.exists():
        raise ValueError("Saved predictions have no original cache manifest")
    if cache.exists(): validate_cache_metadata(load_json(cache),expected)
    rows=read_jsonl(output) if output.exists() else []
    if len({r["trial_id"] for r in rows})!=len(rows):
        raise ValueError("Duplicate cached inference IDs")
    for row in rows:
        if row["trial_id"] not in allowed: raise ValueError("Cached prediction has unrelated ID")
        path=(run_dir/row["logit_path"]).resolve()
        if not path.is_relative_to(run_dir.resolve()) or not path.is_file(): raise ValueError("Missing/invalid cached logit path")
        if sha256(path)!=row.get("logit_sha256"): raise ValueError("Cached logit hash mismatch")
        logits=np.load(path,allow_pickle=False)
        if logits.dtype!=np.float32 or logits.shape!=(row["output_length"],41) or not np.isfinite(logits).all():
            raise ValueError("Cached logit length/dtype/value mismatch")
        if row.get("phoneme_ids")!=greedy_ctc(logits.argmax(-1).tolist()):
            raise ValueError("Cached phoneme prediction disagrees with logits")
    return rows


def inference(config, split, run_dir, limit=None):
    if limit is not None and limit<1:
        raise ValueError("Diagnostic limit must be positive")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logits").mkdir(exist_ok=True)
    records = [r for r in load_json(config["data_manifest"]) if r["split"] == split]
    if not records:
        raise ValueError(f"Missing requested partition {split}; no fallback")
    if limit is not None:
        records = records[:limit]
    splits = load_split_manifest(config["split_manifest"],load_json(config["data_manifest"]))
    expected = {"checkpoint_hash": config["checkpoint_hash"], "contract_hash": config["contract_hash"], "split_hash": splits["manifest_hash"], "dtype": "float32", "allow_tf32": False, "partition": split, "manifest_hash": object_hash(load_json(config["data_manifest"]))}
    cache_path = run_dir / "cache_manifest.json"
    output = run_dir / "predictions.jsonl"
    existing = recover_cached_predictions(run_dir,expected,{r["trial_id"] for r in records})
    if len({r["trial_id"] for r in existing}) != len(existing):
        raise ValueError("Duplicate trial in existing inference output")
    allowed = {r["trial_id"] for r in records}
    if any(r["trial_id"] not in allowed for r in existing):
        raise ValueError("Inference resume IDs do not match partition")
    for r in existing:
        if not (run_dir / r["logit_path"]).is_file():
            raise ValueError("Prediction has no saved logits; cannot resume")
    complete = {r["trial_id"] for r in existing}
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump({**config, "partition": split, "limit": limit}, sort_keys=True), encoding="utf-8")
    status = {"state": "created", "started_at": datetime.now(timezone.utc).isoformat(), "expected_trials": len(records), "checkpoint_hash": expected["checkpoint_hash"], "contract_hash": expected["contract_hash"], "split_hash": expected["split_hash"], "references_loaded": False, "text_available": False}
    write_json(run_dir / "run_manifest.json", status)
    write_json(cache_path, {**expected, "expected_trials": len(records), "references_loaded": False})
    try:
        with ResourceMonitor(run_dir / "resources.json", max_seconds=config.get("max_seconds", 28200)) as monitor:
            torch.set_num_threads(config.get("cpu_threads", 4))
            model, args = load_encoder(config["checkpoint"], config["model_args"], config.get("device", "cuda"), config["checkpoint_hash"])
            if model.contract_hash != config["contract_hash"]:
                raise ValueError("Model contract hash differs from resolved configuration")
            status["state"] = "inferencing"
            write_json(run_dir / "run_manifest.json", status)
            transforms = args["dataset"]["data_transforms"]
            kernel = gaussian_kernel(transforms["smooth_kernel_std"], transforms["smooth_kernel_size"])
            durations = []
            with output.open("a", encoding="utf-8") as stream, torch.inference_mode():
                for record in records:
                    if record["trial_id"] in complete:
                        continue
                    monitor.check()
                    start = time.monotonic()
                    item = read_trial(record, include_labels=False)
                    x = torch.from_numpy(item["features"]).unsqueeze(0).to(config.get("device", "cuda"))
                    smoothed = smooth_valid(x, kernel)
                    logits = model(smoothed, torch.tensor([record["day_index"]], device=x.device))[0]
                    expected_length = output_length(record["input_length"], len(kernel), model.patch_size, model.patch_stride)
                    if logits.shape != (expected_length, 41) or not torch.isfinite(logits).all():
                        raise ValueError(f"Invalid logits or output length: {record['trial_id']}")
                    array = logits.cpu().numpy()
                    name = f"logits/{record['official_order_index']:06d}.npy"
                    np.save(run_dir / name, array, allow_pickle=False)
                    phones = greedy_ctc(array.argmax(-1).tolist(), output_length=expected_length)
                    elapsed = time.monotonic()-start
                    durations.append(elapsed)
                    row = {"trial_id": record["trial_id"], "official_order_index": record["official_order_index"], "session_id": record["session_id"], "split": split, "input_length": record["input_length"], "output_length": expected_length, "phoneme_ids": phones, "phonemes": [TOKENS[i] for i in phones], "text": "", "text_available": False, "logit_path": name, "logit_sha256": sha256(run_dir / name), "elapsed_seconds": elapsed, "status": "ok"}
                    stream.write(json.dumps(row)+"\n")
                    stream.flush()
                    complete.add(record["trial_id"])
                    if len(complete) % 100 == 0:
                        print(json.dumps({"stage": "inference", "saved": len(complete), "expected": len(records)}), flush=True)
            status.update(state="complete" if limit is None else "diagnostic_partial", saved_trials=len(complete), stage="phoneme_inference", elapsed_seconds=time.monotonic()-monitor.start, p50_trial_seconds=float(np.median(durations)) if durations else None, p95_trial_seconds=float(np.quantile(durations,.95)) if durations else None)
    except BaseException as error:
        status.update(state="budget_stopped" if isinstance(error,(BudgetExceeded,KeyboardInterrupt)) else "failed", saved_trials=len(complete), error=f"{type(error).__name__}: {error}")
        write_json(run_dir / "run_manifest.json", status)
        raise
    write_json(run_dir / "run_manifest.json", status)
    # This artifact describes coverage only. Scoring is a separate selected-label operation.
    write_json(run_dir / "metrics.json", {"scope": "feature-only coverage", "expected_trials": len(records), "saved_trials": len(complete), "per": None, "wer": None})
    (run_dir / "report.md").write_text(f"# Phoneme inference\n\nStatus: {status['state']}. Saved {len(complete)} / {len(records)} trials. References were not loaded. Word decoding and scored evaluation are separate stages.\n", encoding="utf-8")
    return status
