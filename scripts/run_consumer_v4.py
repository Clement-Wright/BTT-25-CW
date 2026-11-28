"""Admit and run the laptop comparisons sequentially after the sustained CPU pilot."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import yaml


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7 or os.name != "nt":
                raise
            time.sleep(0.1 * (attempt + 1))


def verify_completed(path, mode, precision, matched):
    """Refuse to count a completed run from a different pilot as this condition."""
    receipt = yaml.safe_load((path / "config.resolved.yaml").read_text(encoding="utf8"))
    expected = dict(matched)
    if mode == "replay":
        names = {
            "score_split": "scope", "split_sha256": "split_hash", "checkpoint": "checkpoint_hash",
            "contract": "contract_hash", "lexicon": "lexicon_sha256", "torch": "torch",
            "lm_sha256": "lm_sha256", "lm_weight_natural_log": "lm_weight_natural_log",
            "cpu_threads": "cpu_threads", "beam": "beam",
        }
        expected = {field: matched[source] for field, source in names.items()}
        expected["precision"] = precision
        if receipt.get("delta_sha256"):
            raise ValueError(f"{path} used adapted encoder weights")
    else:
        expected.update(mode=mode, precision=[precision] if precision else ["fp32", "int8"])
    differences = [field for field, value in expected.items() if receipt.get(field) != value]
    if differences:
        raise ValueError(f"{path} differs from this campaign in {', '.join(differences)}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    p.add_argument("--decoder-python", type=Path, default=Path(".decoder-venv/Scripts/python.exe"))
    p.add_argument("--lexicon", type=Path, default=Path("assets/local/lexicon.txt"))
    p.add_argument("--kenlm-model", type=Path)
    p.add_argument("--wait-pilot", action="store_true")
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Preserve the failed attempt and start a separate directory",
    )
    args = p.parse_args()
    wait_start = time.monotonic()
    while args.wait_pilot and not (args.pilot / "resources.json").is_file():
        if time.monotonic() - wait_start >= 1800:
            raise TimeoutError("Pilot has no final resource receipt after 30 minutes")
        time.sleep(10)
    resources = read(args.pilot / "resources.json")
    metrics = read(args.pilot / "metrics.json")
    pilot_config = yaml.safe_load((args.pilot / "config.resolved.yaml").read_text(encoding="utf8"))
    if (
        resources["outcome"] != "complete"
        or resources["elapsed_seconds"] < 900
        or pilot_config["pilot_seconds"] < 900
        or resources["peak_accounted_host_bytes"] > 12e9
    ):
        raise ValueError("Sustained pilot duration, resource or completion check failed")
    config = yaml.safe_load(args.config.read_text(encoding="utf8"))
    split = read(Path(config["split_manifest"]))
    # Probe in a short-lived child to keep a second Torch runtime out of timed runs.
    runtime = json.loads(subprocess.check_output([
        sys.executable, "-c",
        "import json, torch; from btt25.consumer_v4 import cpu_name; "
        "print(json.dumps([cpu_name(), str(torch.__version__)]))",
    ], text=True))
    flashlight_version = subprocess.check_output([
        str(args.decoder_python), "-c",
        "import importlib.metadata as m; print(m.version('flashlight-text'))",
    ], text=True).strip()
    matched = {
        "scope": "dev_tune",
        "split_hash": split["manifest_hash"],
        "checkpoint_hash": config["checkpoint_hash"],
        "contract_hash": config["contract_hash"],
        "lexicon_sha256": digest(args.lexicon),
        "beam": 50,
        "nbest": 1,
        "cpu_threads": 4,
        "threshold": 25,
        "lm_sha256": digest(args.kenlm_model) if args.kenlm_model else None,
        "lm_weight_natural_log": 2.0 if args.kenlm_model else 0.0,
        "data_manifest_sha256": digest(config["data_manifest"]),
        "cpu": runtime[0], "torch": runtime[1], "python": platform.python_version(),
        "flashlight_text": flashlight_version,
    }
    missing_identity = matched.keys() - pilot_config.keys()
    if any(pilot_config[key] != matched[key] for key in matched.keys() & pilot_config.keys()):
        raise ValueError("Pilot and campaign configurations differ")
    n = len(split["dev_tune"])
    configurations = [
        ("score-fp32", "score", "fp32"),
        ("score-int8", "score", "int8"),
        ("timing", "timing", None),
        ("replay-fp32", "replay", "fp32"),
        ("replay-int8", "replay", "int8"),
    ]
    p95_sum = sum(metrics[precision]["latency_seconds"]["p95"] for precision in ("fp32", "int8"))
    longest_seconds = max(
        row["input_seconds"]
        for row in (json.loads(line) for line in (args.pilot / "predictions.jsonl").open(encoding="utf8"))
    )
    forecast = math.ceil(1.5 * ((n + 3 * 96) * p95_sum + 16 * longest_seconds) + 120)
    if forecast > 7.5 * 3600:
        raise ValueError(f"Conservative forecast exceeds budget: {forecast} s")
    args.campaign.mkdir(parents=True, exist_ok=True)
    schedule = {
        "pilot": args.pilot.name,
        "pilot_p95_seconds_sum": p95_sum,
        "longest_replay_seconds": longest_seconds,
        "forecast_seconds": forecast,
        "margin": "1.5 times p95 per request and worst-case replay duration, plus 120 s for setup and reporting",
        "conditions": [c[0] for c in configurations],
        "score_split": "dev_tune",
        "status": "admitted",
    }
    registry_file = args.campaign / "conditions.json"
    registry = read(registry_file) if registry_file.exists() else {}
    completed_count = 0
    for name, mode, precision in configurations:
        existing = args.campaign / registry.get(name, name)
        if (existing / "report.md").is_file() and read(existing / "resources.json")["outcome"] == "complete":
            verify_completed(existing, mode, precision, matched)
            completed_count += 1
    if missing_identity and completed_count < len(configurations):
        raise ValueError(f"A new job requires a pilot with complete runtime identity: {sorted(missing_identity)}")
    state_path = args.campaign / "state.json"
    if completed_count == len(configurations) and state_path.is_file() and read(state_path).get("status") == "complete":
        print(f"Verified all {completed_count} completed conditions.", flush=True)
        return
    save(args.campaign / "admission.json", schedule)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    deadline = time.monotonic() + min(7 * 3600 + 50 * 60, forecast)
    for name, mode, precision in configurations:
        out = args.campaign / registry.get(name, name)
        if out.exists():
            if (out / "report.md").is_file() and read(out / "resources.json")["outcome"] == "complete":
                registry[name] = out.name
                save(registry_file, registry)
                continue
            receipt = out / "resources.json"
            if (
                not args.retry_failed
                or not receipt.is_file()
                or read(receipt)["outcome"] not in {"failed", "budget_stopped"}
            ):
                raise ValueError(f"Unfinished run preserved in {out}. Inspect its receipt before retrying")
            index = 2
            while (args.campaign / f"{name}-r{index}").exists():
                index += 1
            out = args.campaign / f"{name}-r{index}"
        registry[name] = out.name
        save(registry_file, registry)
        module = "btt25.consumer_replay" if mode == "replay" else "btt25.consumer_v4"
        cmd = [
            sys.executable,
            "-u",
            "-m",
            module,
            "--out",
            str(out),
            "--config",
            str(args.config),
            "--lexicon",
            str(args.lexicon),
            "--decoder-python",
            str(args.decoder_python),
        ]
        if mode != "replay":
            cmd.extend(["--mode", mode])
        if precision:
            cmd.extend(["--precision", precision])
        if args.kenlm_model:
            cmd.extend(["--kenlm-model", str(args.kenlm_model)])
        remaining = deadline - time.monotonic()
        if remaining < 30:
            save(args.campaign / "state.json", {"condition": name, "status": "budget_stopped"})
            raise RuntimeError("Insufficient time remains to launch this condition")
        cmd.extend(["--watchdog", str(int(remaining) - 15)])
        save(args.campaign / "state.json", {"condition": name, "command": cmd, "status": "running"})
        print(name, flush=True)
        completed = subprocess.run(cmd, env=env, timeout=max(1, deadline - time.monotonic()))
        if completed.returncode:
            save(
                args.campaign / "state.json",
                {"condition": name, "command": cmd, "status": "failed", "returncode": completed.returncode},
            )
            raise SystemExit(completed.returncode)
    save(args.campaign / "state.json", {"status": "complete"})


if __name__ == "__main__":
    main()
