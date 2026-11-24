"""CPU-only orchestration contracts; fake stages never load participant data."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from btt25 import pipeline


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    paths = {}
    for key in ("checkpoint", "model_args", "data_manifest", "split_manifest"):
        path = root / key
        path.write_text("fixture only", encoding="utf-8")
        paths[key] = str(path)
    paths.update(checkpoint_hash="fixture-checkpoint", contract_hash="fixture-contract")
    config = root / "local.yaml"
    config.write_text(yaml.safe_dump(paths), encoding="utf-8")
    lexicon = root / "lexicon.txt"
    lexicon.write_text("fixture F IH K S CH ER\n", encoding="utf-8")
    worker = root / "decode_worker.py"
    worker.write_text("# fixture; never executed\n", encoding="utf-8")
    profile = root / "profile"
    for name in ("profile.json", "resources.json", "config.resolved.yaml"):
        write_json(profile / name, {})
    return SimpleNamespace(root=root, config=config, lexicon=lexicon, worker=worker,
                           profile=profile, paths=paths)


def make_plan(inputs, **options):
    return pipeline.build_plan(inputs.config, inputs.lexicon, inputs.root / "run",
                               decoder_python=sys.executable, decoder_script=inputs.worker,
                               **options)


def stage_outputs(stage, *, state="complete"):
    directory = Path(stage["run_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    if stage["name"] == "train":
        checkpoint = directory / "checkpoints/last.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"synthetic checkpoint; never loaded")
        write_json(directory / "run_manifest.json", {
            "state": "training_complete_inference_pending" if state == "complete" else state,
            "step": 720, "checkpoint_hash": pipeline.file_hash(checkpoint),
        })
        (directory / "inference.yaml").write_text("checkpoint: synthetic\n", encoding="utf-8")
    elif stage["name"] in {"infer", "decode"}:
        write_json(directory / "run_manifest.json", {
            "state": state, "expected_trials": 1, "saved_trials": 1,
        })
        write_json(directory / "cache_manifest.json", {"fixture": stage["name"]})
        row = {"trial_id": "fixture"}
        if stage["name"] == "infer":
            logits = directory / "logits/000000.npy"
            logits.parent.mkdir(exist_ok=True)
            logits.write_bytes(b"synthetic bytes; never loaded")
            row.update(logit_path="logits/000000.npy", logit_sha256=pipeline.file_hash(logits))
        (directory / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        if stage["name"] == "decode":
            (directory / "candidates.jsonl").write_text('{"trial_id":"fixture","candidates":[]}\n', encoding="utf-8")
    else:
        write_json(directory / "metrics.json", {"score_split": "dev_tune", "evaluated_count": 1})


def fake_runner(plan, calls, states=None, returncodes=None):
    states, returncodes = states or {}, returncodes or {}

    def run(command, **kwargs):
        stage = next(stage for stage in plan["stages"]
                     if stage["name"] == ("train" if "resume" in command else
                                          "decode" if str(plan["decoder_script"]) in command else
                                          "evaluate" if "evaluate" in command else
                                          "train" if "train" in command else "infer"))
        calls.append((stage["name"], command))
        stage_outputs(stage, state=states.get(stage["name"], "complete"))
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["cwd"] == plan["working_directory"]
        return SimpleNamespace(returncode=returncodes.get(stage["name"], 0))

    return run


def test_baseline_plan_uses_val_features_and_only_tuning_references(inputs):
    plan = make_plan(inputs)
    assert [stage["name"] for stage in plan["stages"]] == ["infer", "decode", "evaluate"]
    infer, decode, evaluate = [stage["command"] for stage in plan["stages"]]
    assert infer[infer.index("--split") + 1] == "val"
    assert decode[0] == sys.executable
    assert decode[decode.index("--beam") + 1] == "50"
    assert decode[decode.index("--nbest") + 1] == "10"
    assert evaluate[evaluate.index("--score-split") + 1] == "dev_tune"
    assert "dev_report" not in json.dumps(plan)
    assert "--kenlm-model" not in decode


def test_optional_training_and_kenlm_arguments_are_explicit(inputs):
    model = inputs.root / "lm.binary"
    model.write_bytes(b"fixture language model")
    plan = make_plan(inputs, mode="session_bias", profile_dir=inputs.profile,
                     kenlm_model=model, lm_weight=2.5)
    assert plan["stages"][0]["name"] == "train"
    assert "--updates" in plan["stages"][0]["command"]
    assert str(inputs.root / "run/train/inference.yaml") in plan["stages"][1]["command"]
    decode = plan["stages"][2]["command"]
    assert decode[decode.index("--kenlm-model") + 1] == str(model)
    assert decode[decode.index("--lm-weight") + 1] == "2.5"


def test_profile_is_required_for_training_and_lm_weight_requires_model(inputs):
    with pytest.raises(ValueError, match="profile"):
        make_plan(inputs, mode="head")
    with pytest.raises(ValueError, match="kenlm"):
        make_plan(inputs, lm_weight=2)
    with pytest.raises(ValueError, match="watchdog"):
        make_plan(inputs, target_seconds=30000)


def test_sequential_execution_resumes_verified_stages_without_repeating_work(inputs, monkeypatch):
    plan = make_plan(inputs)
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "run", fake_runner(plan, calls, {"decode": "budget_stopped"}))
    stopped = pipeline.execute_plan(plan)
    assert stopped["state"] == "budget_stopped"
    assert [name for name, _ in calls] == ["infer", "decode"]
    assert stopped["stages"]["decode"]["attempts"][0]["measured_state"] == "budget_stopped"
    monkeypatch.setattr(pipeline.subprocess, "run", fake_runner(plan, calls))
    complete = pipeline.execute_plan(plan)
    assert complete["state"] == "complete"
    assert [name for name, _ in calls] == ["infer", "decode", "decode", "evaluate"]
    assert len(complete["stages"]["decode"]["attempts"]) == 2
    archived = inputs.root / "run/logs/decode-attempt-002/run_manifest.json"
    assert json.loads(archived.read_text())["state"] == "budget_stopped"
    pipeline.execute_plan(plan)
    assert len(calls) == 4


def test_interrupted_training_uses_existing_resume_api(inputs, monkeypatch):
    plan = make_plan(inputs, mode="head", profile_dir=inputs.profile)
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "run", fake_runner(plan, calls, {"train": "budget_stopped"}))
    assert pipeline.execute_plan(plan)["state"] == "budget_stopped"
    assert [name for name, _ in calls] == ["train"]
    monkeypatch.setattr(pipeline.subprocess, "run", fake_runner(plan, calls))
    result = pipeline.execute_plan(plan)
    assert result["state"] == "complete"
    assert "resume" in calls[1][1]
    assert "--profile-dir" in calls[1][1]


def test_failed_exit_and_incomplete_measured_state_never_advance(inputs, monkeypatch):
    plan = make_plan(inputs)
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "run", fake_runner(plan, calls, returncodes={"infer": 7}))
    result = pipeline.execute_plan(plan)
    assert result["state"] == "failed"
    assert [name for name, _ in calls] == ["infer"]
    assert result["stages"]["infer"]["attempts"][0]["returncode"] == 7


def test_changed_inputs_and_completed_evidence_are_rejected(inputs, monkeypatch):
    plan = make_plan(inputs)
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "run", fake_runner(plan, calls))
    assert pipeline.execute_plan(plan)["state"] == "complete"
    prediction = Path(plan["stages"][0]["run_dir"]) / "predictions.jsonl"
    changed = json.loads(prediction.read_text())
    changed["trial_id"] = "changed"
    prediction.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    result = pipeline.execute_plan(plan)
    assert result["state"] == "failed"
    assert "evidence" in result["error"]
    inputs.lexicon.write_text("changed AA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="plan"):
        pipeline.execute_plan(make_plan(inputs))


def test_cooperative_stop_is_checked_before_launching_any_stage(inputs, monkeypatch):
    plan = make_plan(inputs)
    stop = inputs.root / "STOP_REQUESTED"
    stop.write_text("stop\n", encoding="utf-8")
    monkeypatch.setenv("BTT_STOP_FILE", str(stop))
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "run", fake_runner(plan, calls))
    result = pipeline.execute_plan(plan)
    assert result["state"] == "budget_stopped"
    assert calls == []


def test_invalid_stage_artifacts_preserve_failure_and_measured_state(inputs, monkeypatch):
    plan = make_plan(inputs)
    calls = []
    generate = fake_runner(plan, calls)

    def incomplete(command, **kwargs):
        result = generate(command, **kwargs)
        (Path(plan["stages"][0]["run_dir"]) / "predictions.jsonl").unlink()
        return result

    monkeypatch.setattr(pipeline.subprocess, "run", incomplete)
    result = pipeline.execute_plan(plan)
    assert result["state"] == "failed"
    stage = result["stages"]["infer"]
    assert stage["state"] == "failed"
    assert stage["attempts"][0]["state"] == "failed"
    assert stage["attempts"][0]["measured_state"] == "complete"
    assert len(calls) == 1


def cli_args(inputs):
    return ["--config", str(inputs.config), "--lexicon", str(inputs.lexicon),
            "--decoder-python", sys.executable, "--decoder-script", str(inputs.worker),
            "--run-dir", str(inputs.root / "run")]


def test_outer_cli_supervises_exactly_one_sequential_child(inputs, monkeypatch, capsys):
    calls = []

    def supervise(command, run_dir, **kwargs):
        calls.append((command, run_dir, kwargs))
        write_json(Path(run_dir) / "pipeline_manifest.json", {"state": "complete"})
        return {"state": "complete", "returncode": 0}

    monkeypatch.setattr(pipeline, "supervise", supervise)
    assert pipeline.main(cli_args(inputs)) == 0
    assert len(calls) == 1
    assert calls[0][0][:3] == [sys.executable, "-m", "btt25.pipeline"]
    assert "--execute-plan" in calls[0][0]
    assert calls[0][2] == {"target_seconds": 27000, "hard_seconds": 28200}
    assert '"state": "complete"' in capsys.readouterr().out


def test_dry_run_exposes_commands_without_subprocesses_or_output_writes(inputs, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run launched work")

    monkeypatch.setattr(pipeline, "supervise", forbidden)
    assert pipeline.main([*cli_args(inputs), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["stages"]) == 3
    assert not (inputs.root / "run").exists()


def test_supervised_child_does_not_start_a_second_supervisor(inputs, monkeypatch):
    plan = make_plan(inputs)
    path = inputs.root / "plan.json"
    write_json(path, plan)

    def forbidden(*args, **kwargs):
        raise AssertionError("nested supervisor")

    monkeypatch.setattr(pipeline, "supervise", forbidden)
    monkeypatch.setattr(pipeline, "execute_plan", lambda actual: {"state": "complete"})
    assert pipeline.main(["--execute-plan", str(path)]) == 0


def test_python_symlink_keeps_the_virtual_environment_location(tmp_path):
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    try:
        interpreter.symlink_to(sys.executable)
    except OSError:
        pytest.skip("Creating a Python symlink requires Windows developer mode")
    assert pipeline._python(interpreter) == str(interpreter.absolute())
