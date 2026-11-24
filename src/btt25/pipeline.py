"""One supervised, resumable train -> val inference -> words -> dev_tune job.

Use --dry-run to inspect every command before launching. Stage subprocesses
release model memory on exit; the decoder may use a separate Python environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from .runner import supervise


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file(path):
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _python(value):
    # Resolving a Linux venv's Python symlink would select the base environment.
    absolute = Path(os.path.abspath(os.path.expanduser(str(value))))
    if absolute.is_file():
        return str(absolute)
    located = shutil.which(str(value))
    if located:
        return located
    raise FileNotFoundError(f"Python interpreter not found: {value}")


def build_plan(config, lexicon, run_dir, *, decoder_python=None, decoder_script=None,
               mode=None, profile_dir=None, updates=720, unigrams=None,
               kenlm_model=None, lm_weight=None, target_seconds=None, hard_seconds=None):
    """Resolve commands and input hashes without loading a model or HDF5 file.

    Configuration paths follow the existing CLI convention: relative paths are
    relative to the caller's working directory. Explicit interpreters work on
    both Windows and Linux; omit decoder_python for the adjacent decoder venv
    when present, or the current interpreter otherwise.
    """
    config_path, lexicon_path = _file(config), _file(lexicon)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a mapping")
    required = ("checkpoint", "model_args", "data_manifest", "split_manifest")
    if any(key not in settings for key in required):
        raise ValueError("Configuration requires checkpoint, model_args, data_manifest and split_manifest")
    if mode not in (None, "head", "session_bias"):
        raise ValueError("Training mode must be head or session_bias")
    if mode and not profile_dir:
        raise ValueError("Training requires a measured profile directory")
    if profile_dir and not mode:
        raise ValueError("A profile directory requires a training mode")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("updates must be a positive integer")
    target = settings.get("target_seconds", 27000) if target_seconds is None else target_seconds
    hard = settings.get("absolute_watchdog_seconds", 28200) if hard_seconds is None else hard_seconds
    numeric = lambda value: not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    if not all(numeric(value) for value in (target, hard)) or not 0 < target < hard <= 28200 or target > 27000:
        raise ValueError("Whole-job target must be <=7.5h and precede the <=7h50m watchdog")
    if lm_weight is not None and (not kenlm_model or not numeric(lm_weight) or lm_weight < 0):
        raise ValueError("lm_weight requires a kenlm model and a finite nonnegative value")
    root = Path(__file__).resolve().parents[2]
    adjacent = root / ".decoder-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    decoder_python = _python(decoder_python or (adjacent if adjacent.is_file() else sys.executable))
    worker = _file(decoder_script or root / "scripts/decode_worker.py")
    run_dir = Path(run_dir).expanduser().resolve()
    main_python = _python(sys.executable)
    inputs = {"config": config_path, "lexicon": lexicon_path, "decoder_script": worker,
              **{key: _file(settings[key]) for key in required}}
    profile = Path(profile_dir).expanduser().resolve() if profile_dir else None
    if profile:
        for name in ("profile.json", "resources.json", "config.resolved.yaml"):
            inputs[f"profile/{name}"] = _file(profile / name)
    if unigrams:
        inputs["unigrams"] = _file(unigrams)
    if kenlm_model:
        inputs["kenlm_model"] = _file(kenlm_model)
    cli = [main_python, "-m", "btt25.cli"]
    stages = []
    inference_config = str(run_dir / "train/inference.yaml") if mode else str(config_path)

    def add(name, command, evidence):
        stages.append({"name": name, "command": command, "run_dir": str(run_dir / name),
                       "evidence": evidence})

    if mode:
        add("train", [*cli, "train", "--config", str(config_path), "--run-dir", str(run_dir / "train"),
                      "--mode", mode, "--profile-dir", str(profile), "--updates", str(updates),
                      "--max-hours", str(target / 3600)],
            ["run_manifest.json", "checkpoints/last.pt", "inference.yaml"])
    add("infer", [*cli, "infer", "--config", inference_config, "--split", "val", "--run-dir", str(run_dir / "infer")],
        ["run_manifest.json", "cache_manifest.json", "predictions.jsonl"])
    decode = [decoder_python, str(worker), "--logit-run", str(run_dir / "infer"),
              "--lexicon", str(lexicon_path), "--manifest", str(inputs["data_manifest"]),
              "--output-dir", str(run_dir / "decode"), "--beam", "50", "--nbest", "10"]
    for option, key in (("--unigrams", "unigrams"), ("--kenlm-model", "kenlm_model")):
        if key in inputs:
            decode.extend([option, str(inputs[key])])
    if lm_weight is not None:
        decode.extend(["--lm-weight", str(lm_weight)])
    add("decode", decode, ["run_manifest.json", "cache_manifest.json", "predictions.jsonl", "candidates.jsonl"])
    add("evaluate", [*cli, "evaluate", "--config", inference_config,
                     "--predictions", str(run_dir / "decode/predictions.jsonl"),
                     "--output-dir", str(run_dir / "evaluate"), "--score-split", "dev_tune"], ["metrics.json"])
    plan = {"version": 1, "run_dir": str(run_dir), "working_directory": str(Path.cwd()),
            "decoder_python": decoder_python, "decoder_script": str(worker), "mode": mode,
            "updates": updates, "target_seconds": target, "hard_seconds": hard,
            "input_files": {key: {"path": str(path), "sha256": file_hash(path)} for key, path in inputs.items()},
            "stages": stages, "reference_access": "optional official train labels; evaluation dev_tune only"}
    plan["plan_hash"] = _object_hash(plan)
    return plan


def _check_stop():
    stop = os.environ.get("BTT_STOP_FILE")
    if stop and Path(stop).exists():
        raise TimeoutError("Whole-job supervisor requested a safe stop between stages")


def _measured(stage, updates):
    directory = Path(stage["run_dir"])
    path = directory / ("metrics.json" if stage["name"] == "evaluate" else "run_manifest.json")
    measured = _read(path) if path.is_file() else {"state": "missing"}
    if stage["name"] == "evaluate":
        state = "complete" if measured.get("score_split") == "dev_tune" and measured.get("evaluated_count", 0) > 0 else "incomplete"
    else:
        state = measured.get("state", "missing")
    if stage["name"] == "train" and state == "training_complete_inference_pending":
        if measured.get("step") != updates or file_hash(directory / "checkpoints/last.pt") != measured.get("checkpoint_hash"):
            raise ValueError("Completed training checkpoint/update evidence differs")
        state = "complete"
    if stage["name"] in ("infer", "decode") and state == "complete":
        if not measured.get("expected_trials") or measured.get("expected_trials") != measured.get("saved_trials"):
            raise ValueError("Completed stage has incomplete coverage evidence")
    return state, measured


def _fingerprint(stage):
    directory = Path(stage["run_dir"])
    paths = [_file(directory / name) for name in stage["evidence"]]
    fingerprints = {str(path): file_hash(path) for path in paths}
    # Verify saved logit bytes too, without loading arrays or reference labels.
    if stage["name"] == "infer":
        rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
        for row in rows:
            path = (directory / row["logit_path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Inference logit evidence differs")
            digest = file_hash(path)
            if digest != row.get("logit_sha256"):
                raise ValueError("Inference logit evidence differs")
            fingerprints[str(path)] = digest
    return fingerprints


def execute_plan(plan):
    """Execute sequential stages once, retaining attempts and verified outputs.

    A nonzero exit or incomplete stage manifest stops the pipeline. A subsequent
    call skips content-identical completed stages and uses the existing resume
    CLI for interrupted training; inference and decoding retain their own cache
    checks. This function deliberately never starts its own supervisor.
    """
    if plan.get("plan_hash") != _object_hash({key: value for key, value in plan.items() if key != "plan_hash"}):
        raise ValueError("Pipeline plan hash differs")
    for source in plan["input_files"].values():
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError("Pipeline input changed; build a new plan and run directory")
    directory = Path(plan["run_dir"])
    manifest = directory / "pipeline_manifest.json"
    state = _read(manifest) if manifest.is_file() else {"version": 1, "plan_hash": plan["plan_hash"], "stages": {}}
    if state.get("plan_hash") != plan["plan_hash"]:
        raise ValueError("Existing pipeline plan differs; use a new run directory")
    directory.mkdir(parents=True, exist_ok=True)
    state.update(state="running", started_at=datetime.now(timezone.utc).isoformat())
    state.pop("error", None)
    _write(manifest, state)
    started = time.monotonic()
    try:
        for stage in plan["stages"]:
            _check_stop()
            name = stage["name"]
            state["active_stage"] = name
            record = state["stages"].setdefault(name, {"state": "pending", "attempts": []})
            if record["state"] == "complete":
                if _fingerprint(stage) != record["evidence"]:
                    raise ValueError(f"Completed {name} evidence changed; use a new run directory")
                continue
            command = stage["command"][:]
            stage_manifest = Path(stage["run_dir"]) / "run_manifest.json"
            if name == "train" and stage_manifest.is_file():
                measured_state, measured = _measured(stage, plan["updates"])
                if measured_state == "complete":
                    record.update(state="complete", recovered_stage_manifest=measured, evidence=_fingerprint(stage))
                    _write(manifest, state)
                    continue
                command[command.index("train")] = "resume"
            number = len(record["attempts"]) + 1
            log = directory / "logs" / f"{name}-attempt-{number:03d}.log"
            log.parent.mkdir(exist_ok=True)
            archive = log.with_suffix("")
            for filename in ("run_manifest.json", "resources.json", "resources.samples.jsonl", "metrics.json"):
                previous = Path(stage["run_dir"]) / filename
                if previous.is_file():
                    archive.mkdir(exist_ok=True)
                    shutil.copy2(previous, archive / filename)
            attempt = {"command": command, "log": str(log), "started_at": datetime.now(timezone.utc).isoformat(), "state": "running"}
            record["attempts"].append(attempt)
            record["state"] = "running"
            _write(manifest, state)
            print(json.dumps({"stage": name, "command": command, "log": str(log)}), flush=True)
            stage_started = time.monotonic()
            with log.open("w", encoding="utf-8") as output:
                result = subprocess.run(command, cwd=plan["working_directory"], stdout=output, stderr=subprocess.STDOUT, check=False)
            measured_state, measured = _measured(stage, plan["updates"])
            outcome = "complete" if result.returncode == 0 and measured_state == "complete" else "budget_stopped" if measured_state == "budget_stopped" else "failed"
            attempt.update(state=outcome, returncode=result.returncode, measured_state=measured_state,
                           stage_manifest=measured, elapsed_seconds=time.monotonic() - stage_started)
            if outcome == "complete":
                record["evidence"] = _fingerprint(stage)
            else:
                state.update(state=outcome, error=f"{name} stopped: exit {result.returncode}, measured state {measured_state}")
            record["state"] = outcome
            _write(manifest, state)
            if outcome != "complete":
                break
        else:
            _check_stop()
            state.update(state="complete", active_stage=None)
    except BaseException as error:
        state.update(state="budget_stopped" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "failed",
                     error=f"{type(error).__name__}: {error}")
        active = state["stages"].get(state.get("active_stage"))
        if active and active["state"] == "running":
            active["state"] = state["state"]
            active["attempts"][-1].update(state=state["state"], error=state["error"])
    state.update(attempt_elapsed_seconds=time.monotonic() - started, finished_at=datetime.now(timezone.utc).isoformat())
    _write(manifest, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Existing local configuration; paths follow the caller's working directory")
    parser.add_argument("--lexicon", help="Validated phone lexicon")
    parser.add_argument("--run-dir", help="New pipeline root, or the same root to resume verified stages")
    parser.add_argument("--decoder-python", help="Decoder Python path or executable; adjacent decoder venv or current Python by default")
    parser.add_argument("--decoder-script", help="Native worker path; scripts/decode_worker.py by default")
    parser.add_argument("--mode", choices=("head", "session_bias"), help="Optional bounded calibration before inference")
    parser.add_argument("--profile-dir", help="Required measured sustained pilot evidence when training")
    parser.add_argument("--updates", type=int, default=720)
    parser.add_argument("--unigrams", help="Optional official-training unigram counts")
    parser.add_argument("--kenlm-model", help="Optional native KenLM backend asset; omitted for the existing fallback")
    parser.add_argument("--lm-weight", type=float, help="Optional KenLM worker weight; worker default when omitted")
    parser.add_argument("--target-seconds", type=float, help="Whole-job cooperative target, <=27000 seconds")
    parser.add_argument("--hard-seconds", type=float, help="Absolute whole-job watchdog, <=28200 seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and commands without launching or writing outputs")
    parser.add_argument("--execute-plan", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.execute_plan:
        result = execute_plan(_read(args.execute_plan))
    else:
        if not all((args.config, args.lexicon, args.run_dir)):
            parser.error("--config, --lexicon and --run-dir are required")
        options = vars(args).copy()
        for key in ("dry_run", "execute_plan"):
            options.pop(key)
        plan = build_plan(**options)
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        plan_path = Path(plan["run_dir"]) / "pipeline_plan.json"
        if plan_path.is_file() and _read(plan_path) != plan:
            raise ValueError("Existing pipeline plan differs; use a new run directory")
        _write(plan_path, plan)
        command = [sys.executable, "-m", "btt25.pipeline", "--execute-plan", str(plan_path)]
        print(json.dumps({"supervised_command": command, "stages": plan["stages"]}), flush=True)
        watchdog = supervise(command, plan["run_dir"], target_seconds=plan["target_seconds"], hard_seconds=plan["hard_seconds"])
        status_path = Path(plan["run_dir"]) / "pipeline_manifest.json"
        result = _read(status_path) if status_path.is_file() else {"state": "failed", "error": "Sequential child produced no pipeline manifest"}
        result = {**result, "supervisor": watchdog}
        if watchdog["state"] != "complete" and result["state"] == "complete":
            result["state"] = watchdog["state"]
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["state"] == "complete" else 75 if result["state"] == "budget_stopped" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
