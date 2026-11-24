"""Metadata-only checks for the locked two-study reporting path."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from btt25.assets import sha256
from btt25.data import object_hash
from btt25.evaluation import corpus_metrics, edit_counts, normalize_text
from btt25.reporting import write_evaluation
from btt25.training import calibration_manifest


@pytest.fixture
def reporting_script(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("report_studies_review", scripts / "report_studies.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value, hashed=False):
    value = copy.deepcopy(value)
    if hashed:
        value.pop("manifest_hash", None)
        value["manifest_hash"] = object_hash(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")


@pytest.fixture
def calibrated(tmp_path):
    records = [{"trial_id": f"train-{i}", "split": "train", "session_id": "s1", "official_order_index": i}
               for i in range(2)]
    records += [{"trial_id": f"val-{i}", "split": "val", "session_id": "s1", "official_order_index": i,
                 "input_length": 20 + i} for i in range(2)]
    data = tmp_path / "data.json"
    save(data, records)
    split_path = tmp_path / "split.json"
    split = save(split_path, {"dev_tune": ["val-0"], "dev_report": ["val-1"]}, hashed=True)
    source = tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint never loaded")
    model_args = tmp_path / "model.json"
    save(model_args, {"tiny": True})
    config = {"data_manifest": str(data), "split_manifest": str(split_path), "checkpoint": str(source),
              "checkpoint_hash": sha256(source), "model_args": str(model_args), "contract_hash": "contract"}
    protocol = {"decoder": {"lm": "KenLM", "lm_weight": 0.3, "beam": 50, "nbest": 10},
                "lexicon_sha256": "lexicon", "tokens_hash": "tokens", "unigrams_sha256": None,
                "kenlm_model_sha256": "language model"}
    base = tmp_path / "compact"
    save(base / "cache_manifest.json", protocol, hashed=True)
    roots = {}
    for name, mode in (("head", "head"), ("bias", "session_bias")):
        root = tmp_path / name
        roots[name] = root
        save(root / "pipeline_manifest.json", {"state": "complete"})
        checkpoint = root / "train/checkpoints/last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"{mode} checkpoint never loaded".encode())
        executed = dict(config, mode=mode, updates=3, lr=1e-5, per_session=2, seed=20261116,
                        regularization=1.0, backbone_dropout=0,
                        selection="last checkpoint at fixed matched update count; no report selection")
        (root / "train/config.resolved.yaml").write_text(yaml.safe_dump(executed))
        inference = dict(config, checkpoint=str(checkpoint), checkpoint_hash=sha256(checkpoint))
        (root / "train/inference.yaml").write_text(yaml.safe_dump(inference))
        save(root / "train/run_manifest.json", {"state": "training_complete_inference_pending", "diagnostic": False,
             "mode": mode, "step": 3, "source_checkpoint_hash": config["checkpoint_hash"],
             "checkpoint_hash": sha256(checkpoint), "cumulative_training_seconds": 5.0})
        save(root / "train/calibration_manifest.json", calibration_manifest(records, per_session=2))
        names = ["out.weight", "out.bias"] + (["day_biases.0"] if mode == "session_bias" else [])
        save(root / "train/trainable_parameters.json", {"names": names, "count": 10 if mode == "head" else 14,
             "backprop_through_frozen_gru": mode == "session_bias", "gru_training_mode": mode == "session_bias",
             "gru_dropout": 0.0 if mode=="session_bias" else 0.4})
        neural = {"checkpoint_hash": sha256(checkpoint), "contract_hash": "contract", "split_hash": split["manifest_hash"],
                  "manifest_hash": object_hash(records), "partition": "val", "dtype": "float32", "allow_tf32": False,
                  "references_loaded": False, "expected_trials": 2}
        save(root / "infer/cache_manifest.json", neural)
        save(root / "infer/run_manifest.json", {"state": "complete", "references_loaded": False,
             "expected_trials": 2, "saved_trials": 2, "checkpoint_hash": sha256(checkpoint),
             "split_hash": split["manifest_hash"], "contract_hash": "contract"})
        predictions = [{"trial_id": f"val-{i}", "official_order_index": i, "split": "val", "status": "ok",
                        "phoneme_ids": [1, 1, 2], "logit_sha256": f"{mode}-logits-{i}", "text": ""} for i in range(2)]
        rows(root / "infer/predictions.jsonl", predictions)
        cache = save(root / "decode/cache_manifest.json", dict(protocol, source_cache_manifest=neural,
            source_predictions_path=str(root / "infer/predictions.jsonl"),
            source_predictions_sha256=sha256(root / "infer/predictions.jsonl"),
            data_manifest_sha256=sha256(data), references_loaded=False), hashed=True)
        save(root / "decode/run_manifest.json", {"state": "complete", "references_loaded": False,
             "expected_trials": 2, "saved_trials": 2, "decoder_cache_hash": cache["manifest_hash"]})
        rows(root / "decode/predictions.jsonl", [dict(row, text="two words", decoder_cache_hash=cache["manifest_hash"],
                                                     phoneme_status="ok") for row in predictions])
    return {"args": SimpleNamespace(**roots), "config": config, "records": records, "split": split, "base": base}


def test_complete_calibration_checks_checkpoint_to_decoded_prediction_chain(reporting_script, calibrated):
    result = reporting_script.complete_calibration(calibrated["args"].head, "head")
    assert result["steps"] == 3
    assert result["trainable_count"] == 10
    assert result["calibration_count"] == 2


@pytest.mark.parametrize("changed", ["inference_checkpoint", "decode_checkpoint", "updates", "calibration_id", "trainable", "phone_identity"])
def test_completed_marker_cannot_override_wrong_condition_evidence(reporting_script, calibrated, changed):
    root = calibrated["args"].head
    if changed == "inference_checkpoint":
        path = root / "train/inference.yaml"
        value = yaml.safe_load(path.read_text())
        value["checkpoint"] = calibrated["config"]["checkpoint"]
        value["checkpoint_hash"] = calibrated["config"]["checkpoint_hash"]
        path.write_text(yaml.safe_dump(value))
    elif changed == "decode_checkpoint":
        path = root / "decode/cache_manifest.json"
        value = json.loads(path.read_text())
        value["source_cache_manifest"]["checkpoint_hash"] = "B2 checkpoint"
        save(path, value, hashed=True)
    elif changed == "updates":
        path = root / "train/config.resolved.yaml"
        value = yaml.safe_load(path.read_text())
        value["updates"] = 4
        path.write_text(yaml.safe_dump(value))
    elif changed == "calibration_id":
        path = root / "train/calibration_manifest.json"
        value = json.loads(path.read_text())
        value["trial_ids"][0] = "val-1"
        save(path, value, hashed=True)
    elif changed == "trainable":
        path = root / "train/trainable_parameters.json"
        value = json.loads(path.read_text())
        value["names"].append("day_biases.0")
        save(path, value)
    else:
        path = root / "decode/predictions.jsonl"
        value = [json.loads(line) for line in path.read_text().splitlines()]
        value[0]["phoneme_ids"] = [9]
        rows(path, value)
    with pytest.raises(ValueError):
        reporting_script.complete_calibration(root, "head")


def test_calibration_comparison_locks_updates_ids_and_compact_decoder(reporting_script, calibrated):
    args = calibrated["args"]
    conditions = reporting_script._calibrations(args, calibrated["config"], calibrated["records"],
                                               calibrated["split"], calibrated["base"])
    assert conditions["B1"]["calibration_count"] == conditions["B2"]["calibration_count"]
    path = calibrated["base"] / "cache_manifest.json"
    value = json.loads(path.read_text())
    value["decoder"]["lm_weight"] = 1.0
    save(path, value, hashed=True)
    with pytest.raises(ValueError, match="decoder"):
        reporting_script._calibrations(args, calibrated["config"], calibrated["records"], calibrated["split"], calibrated["base"])


def test_calibration_comparison_rejects_different_update_protocol(reporting_script, calibrated):
    path = calibrated["args"].bias / "train/config.resolved.yaml"
    value = yaml.safe_load(path.read_text())
    value["lr"] = 1e-3
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="matched|protocol"):
        reporting_script._calibrations(calibrated["args"], calibrated["config"], calibrated["records"],
                                      calibrated["split"], calibrated["base"])


def test_reusing_scored_a_report_requires_the_exact_live_outputs(reporting_script, calibrated, tmp_path):
    frozen = {"manifest_hash": "frozen"}
    split = calibrated["split"]
    trace = {"score_split": "dev_report", "split_manifest_hash": split["manifest_hash"], "freeze_manifest_hash": "frozen"}
    row = dict(calibrated["records"][-1], **trace, reference_text="two words", reference_phoneme_ids=[1, 1, 2],
               text="two", phoneme_ids=[1, 1, 2], status="ok", phoneme_status="ok")
    edits = {"trial_id": row["trial_id"], **trace, "status": "ok",
             "word": edit_counts(normalize_text(row["reference_text"]).split(), row["text"].split()),
             "phoneme": edit_counts(row["reference_phoneme_ids"], row["phoneme_ids"])}
    directory = tmp_path / "scored"
    write_evaluation({"metrics": dict(corpus_metrics([row]), **trace), "scored_rows": [row], "per_trial": [edits]}, directory)
    prediction = {key: row[key] for key in ("trial_id", "text", "phoneme_ids", "status", "phoneme_status")}
    result = reporting_script._reuse_report(directory, [prediction], calibrated["records"], split, frozen)
    assert result["metrics"]["word"]["errors"] == 1
    assert result["metrics"]["wer"] == 0.5
    prediction["text"] = "a different live deployment"
    with pytest.raises(ValueError, match="output|prediction"):
        reporting_script._reuse_report(directory, [prediction], calibrated["records"], split, frozen)


def test_resource_summary_keeps_archived_peaks_and_all_measured_stage_attempts(reporting_script, tmp_path):
    save(tmp_path / "supervisor.json", {"elapsed_seconds": 9})
    save(tmp_path / "pipeline_manifest.json", {"stages": {
        "train": {"attempts": [{"elapsed_seconds": 12}, {"elapsed_seconds": 8}]},
        "infer": {"attempts": [{"elapsed_seconds": 3}]},
        "decode": {"attempts": [{"elapsed_seconds": 5}]},
        "evaluate": {"attempts": [{"elapsed_seconds": 2}]}}})
    save(tmp_path / "train/resources.json", {"peak_tree_rss_bytes": 30, "peak_device_used_bytes": 10})
    save(tmp_path / "train/attempts/0001/resources.json", {"peak_tree_rss_bytes": 90, "peak_device_used_bytes": 20})
    result = reporting_script.measured_resources(tmp_path)
    assert result["elapsed_seconds"] == 30
    assert result["stage_elapsed_seconds"]["train"] == 20
    assert result["latest_supervisor_elapsed_seconds"] == 9
    assert result["peak_tree_rss_bytes"] == 90
    assert result["peak_device_used_bytes"] == 20
