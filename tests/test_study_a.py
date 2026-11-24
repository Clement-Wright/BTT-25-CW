import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from btt25 import study_a
from btt25.assets import sha256
from btt25.data import object_hash
from btt25.rescoring import MODEL_ID, MODEL_REVISION


def write_json(path, value, hashed=False):
    value = copy.deepcopy(value)
    if hashed:
        value["manifest_hash"] = object_hash(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FakeScorer:
    calls = []
    change_candidates = False
    fail_candidates = False

    def __init__(self, model_dir, device="cpu", allow_cpu_fallback=False):
        self.metadata = {"device": device, "asset_hashes_verified": True}

    def score_candidates(self, candidates):
        self.calls.append([row["text"] for row in candidates])
        if self.fail_candidates:
            raise RuntimeError("synthetic scorer failure")
        result = [dict(row, lm_log_likelihood=-10.0 if row["text"].startswith("bad") else -1.0,
                       lm_token_count=3, lm_elapsed_seconds=0.001) for row in candidates]
        if self.change_candidates:
            result[0]["text"] = "a newly generated answer"
        return result


@pytest.fixture
def source(tmp_path, monkeypatch):
    FakeScorer.calls = []
    FakeScorer.change_candidates = False
    FakeScorer.fail_candidates = False
    monkeypatch.setattr(study_a, "DistilGPT2Scorer", FakeScorer)
    records, packets, predictions = [], [], []
    ids = [f"trial-{letter}" for letter in "abcdefghijkl"]
    tune_ids, report_ids = ids[:8], ids[8:]
    for i, trial_id in enumerate(ids):
        records.append({"trial_id": trial_id, "split": "val", "official_order_index": i,
                        "session_id": f"session-{i // 2}", "block_id": 1,
                        "input_length": 100 + i, "corpus_id": "Random" if i >= 8 else "sentences"})
        margin = 0.02 if i >= 8 else (i + 1) * 0.04
        choices = [
            {"text": f"bad {trial_id}", "rank": 1, "phoneme_ids": [5],
             "component_scores": {"rank_score": 0.0}},
            {"text": f"good {trial_id}", "rank": 2, "phoneme_ids": [6],
             "component_scores": {"rank_score": -margin}},
        ]
        packets.append({"trial_id": trial_id, "input_id": trial_id, "output_id": trial_id,
                        "logit_sha256": f"logit-{trial_id}", "candidates": choices})
        predictions.append({"trial_id": trial_id, "input_id": trial_id, "output_id": trial_id,
                            "official_order_index": i, "session_id": records[-1]["session_id"],
                            "split": "val", "input_length": 100 + i,
                            "text": choices[0]["text"], "phoneme_ids": [1, 1, 2],
                            "status": "ok", "logit_sha256": f"logit-{trial_id}",
                            "elapsed_seconds": (i + 1) * 0.001})
    manifest_path = tmp_path / "data.json"
    write_json(manifest_path, records)
    split_path = tmp_path / "splits.json"
    split = write_json(split_path, {"dev_tune": tune_ids, "dev_report": report_ids}, hashed=True)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint, never loaded")
    candidate_run = tmp_path / "compact"
    candidate_run.mkdir()
    cache = write_json(candidate_run / "cache_manifest.json", {
        "references_loaded": False, "data_manifest_sha256": sha256(manifest_path),
        "source_cache_manifest": {"checkpoint_hash": sha256(checkpoint), "split_hash": split["manifest_hash"],
                                  "manifest_hash": object_hash(records), "references_loaded": False,
                                  "partition": "val", "dtype": "float32", "expected_trials": len(records)},
        "decoder": {"candidate_count": 10, "lm": "ZeroLM"},
    }, hashed=True)
    for row in [*packets, *predictions]:
        row["decoder_cache_hash"] = cache["manifest_hash"]
    write_rows(candidate_run / "candidates.jsonl", packets)
    write_rows(candidate_run / "predictions.jsonl", predictions)
    write_json(candidate_run / "run_manifest.json", {"state": "complete", "references_loaded": False, "elapsed_seconds": 2.5})
    scorer_dir = tmp_path / "scorer"
    scorer_dir.mkdir()
    files = []
    for name in ("model.safetensors", "config.json", "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"):
        (scorer_dir / name).write_text("synthetic " + name)
        files.append({"file": name, "sha256": sha256(scorer_dir / name)})
    write_json(scorer_dir / "manifest.json", {"model_id": MODEL_ID, "revision": MODEL_REVISION, "files": files})
    config = {"data_manifest": str(manifest_path), "split_manifest": str(split_path),
              "checkpoint": str(checkpoint), "checkpoint_hash": sha256(checkpoint),
              "neural_lambda_grid": [0, 0.1, 0.3, 1], "call_fraction": 0.5,
              "random_seeds": [20261116, 20261117, 20261118], "max_seconds": 60}
    return {"config": config, "run": candidate_run, "scorer": scorer_dir, "records": records,
            "packets": packets, "predictions": predictions, "split": split,
            "tune_ids": tune_ids, "report_ids": report_ids}


def tune_source(source, output, monkeypatch, selection_bad=False):
    memberships = study_a.grouped_gate_split(source["records"], source["tune_ids"])

    def reader(record):
        assert (output / "gate_memberships.json").is_file()
        trial_id = record["trial_id"]
        assert trial_id in source["tune_ids"], "report labels opened during tuning"
        word = "bad" if selection_bad and trial_id in memberships["gate_selection"] else "good"
        return {"reference_text": f"{word} {trial_id}", "phoneme_ids": [1, 1, 2]}

    monkeypatch.setattr(study_a, "reference_only", reader)
    return study_a.tune(source["config"], source["run"], output, source["scorer"], device="cpu")


def joint_freeze(source, selected_path, tmp_path):
    from btt25.freeze import current_hashes

    config_file, asset_file, b_checkpoint = tmp_path / "baseline.json", tmp_path / "contract.json", tmp_path / "b.pt"
    config_file.write_text('{"baseline": true}')
    asset_file.write_text('{"contract": "synthetic"}')
    b_checkpoint.write_bytes(b"synthetic study B checkpoint")
    registry = {"schema_version": 1, "split_manifest": source["config"]["split_manifest"], "artifacts": {
        "baseline": {"kind": "files", "config": {"baseline": str(config_file)},
                     "checkpoint": {"model": source["config"]["checkpoint"]}, "assets": {"contract": str(asset_file)}},
        "studies": {
            "confidence_lm": {"kind": "confidence_lm", "config": source["config"],
                              "candidate_run": str(source["run"]), "scorer_dir": str(source["scorer"]),
                              "selected_config": str(selected_path)},
            "session_calibration": {"kind": "files", "config": {"b": str(config_file)},
                                    "checkpoint": {"calibration": str(b_checkpoint)}, "assets": {"contract": str(asset_file)}},
        },
    }}
    expected = current_hashes(registry)
    freeze = {**copy.deepcopy(expected), "registry_hash": object_hash(registry),
              "selection_rules": "Both studies selected on dev_tune, then jointly frozen."}
    freeze_path, expected_path = tmp_path / "joint_freeze.json", tmp_path / "artifact_registry.json"
    write_json(freeze_path, freeze, hashed=True)
    write_json(expected_path, registry)
    return freeze_path, expected_path


def test_gate_memberships_are_grouped_disjoint_and_independent_of_input_order(source):
    memberships = study_a.grouped_gate_split(source["records"], source["tune_ids"])
    assert memberships == study_a.grouped_gate_split(list(reversed(source["records"])), list(reversed(source["tune_ids"])))
    fit, selection = set(memberships["gate_fit"]), set(memberships["gate_selection"])
    assert fit and selection and not fit & selection
    assert fit | selection == set(source["tune_ids"])
    for group in {row["session_id"] for row in source["records"] if row["trial_id"] in fit}:
        assert all(row["trial_id"] not in selection for row in source["records"] if row["session_id"] == group)


def test_tuning_saves_memberships_before_labels_and_uses_only_gate_fit_for_lambda(source, tmp_path, monkeypatch):
    original = (source["run"] / "candidates.jsonl").read_bytes()
    output = tmp_path / "tune"
    selected = tune_source(source, output, monkeypatch, selection_bad=True)
    assert selected["selected_lambda"] == 0.1
    margins = [study_a.cheap_features(study_a.compact_candidates(packet["candidates"]), 100)["top_two_margin_per_word"]
               for packet in source["packets"] if packet["trial_id"] in selected["gate_fit_ids"]]
    assert selected["gate_threshold"] == pytest.approx(float(np.median(margins)))
    assert selected["source_hashes"]["candidates_sha256"] == sha256(source["run"] / "candidates.jsonl")
    assert selected["oracle"]["baseline_minus_oracle_wer"] > 0
    assert len(FakeScorer.calls) == len(source["tune_ids"])
    assert (output / "prep_resources.json").is_file()
    assert (output / "selected_config.json").is_file()
    assert (source["run"] / "candidates.jsonl").read_bytes() == original
    second = tune_source(source, tmp_path / "second", monkeypatch, selection_bad=False)
    assert second["selected_lambda"] == selected["selected_lambda"]


def test_random_controls_match_actual_gated_count_within_reference_free_length_bins():
    features = [{"trial_id": f"id-{i}", "word_count": words, "candidate_count": 2}
                for i, words in enumerate([1, 2, 5, 6, 9, 10, 11, 12, 30])]
    gated = ["id-0", "id-2", "id-4", "id-8"]
    chosen = study_a.length_matched_random(features, gated, seed=20261116)
    expected = Counter(study_a.candidate_length_bin(row["word_count"]) for row in features if row["trial_id"] in gated)
    actual = Counter(study_a.candidate_length_bin(row["word_count"]) for row in features if row["trial_id"] in chosen)
    assert actual == expected
    assert len(chosen) == len(gated)
    assert chosen == study_a.length_matched_random(list(reversed(features)), gated, seed=20261116)


def test_random_controls_match_each_partition_when_gated_length_distributions_differ():
    rows = []
    for prefix, words, margins in (("tune", [2, 2, 8, 8, 12, 12], [0, 0, 1, 1, 1, 1]),
                                   ("report", [2, 2, 8, 8, 12, 12], [1, 1, 0, 1, 0, 0])):
        rows.extend({"trial_id": f"{prefix}-{i}", "word_count": word, "candidate_count": 2,
                     "top_two_margin_per_word": margin} for i, (word, margin) in enumerate(zip(words, margins)))
    context = {"features": {row["trial_id"]: row for row in rows}, "ordered_ids": [row["trial_id"] for row in rows],
               "split": {f"dev_{prefix}": [row["trial_id"] for row in rows if row["trial_id"].startswith(prefix)]
                         for prefix in ("tune", "report")}}
    gated, controls, _ = study_a._decisions(context, 0.5)
    for members in context["split"].values():
        def counts(ids):
            return Counter(study_a.candidate_length_bin(context["features"][trial_id]["word_count"])
                           for trial_id in ids if trial_id in members)
        for chosen in controls.values():
            assert counts(chosen) == counts(gated)


def test_tuning_rejects_candidates_from_another_encoder_before_labels(source, tmp_path, monkeypatch):
    path = source["run"] / "cache_manifest.json"
    cache = json.loads(path.read_text())
    del cache["manifest_hash"]
    cache["source_cache_manifest"]["checkpoint_hash"] = "another checkpoint"
    cache = write_json(path, cache, hashed=True)
    for name in ("candidates.jsonl", "predictions.jsonl"):
        rows = [json.loads(line) for line in (source["run"] / name).read_text().splitlines()]
        for row in rows:
            row["decoder_cache_hash"] = cache["manifest_hash"]
        write_rows(source["run"] / name, rows)
    monkeypatch.setattr(study_a, "reference_only", lambda row: pytest.fail("wrong encoder opened labels"))
    with pytest.raises(ValueError, match="source|checkpoint|encoder"):
        study_a.tune(source["config"], source["run"], tmp_path / "tune", source["scorer"], device="cpu")
    assert not FakeScorer.calls


def test_report_rejects_an_unsealed_registry_even_with_identical_file_hashes(source, tmp_path, monkeypatch):
    tune_source(source, tmp_path / "tune", monkeypatch)
    selected_path = tmp_path / "tune" / "selected_config.json"
    freeze, registry_path = joint_freeze(source, selected_path, tmp_path)
    registry = json.loads(registry_path.read_text())
    original = Path(registry["artifacts"]["baseline"]["config"]["baseline"])
    copy_path = tmp_path / "unsealed-baseline.json"
    copy_path.write_bytes(original.read_bytes())
    registry["artifacts"]["baseline"]["config"]["baseline"] = str(copy_path)
    write_json(registry_path, registry)
    monkeypatch.setattr(study_a, "reference_only", lambda row: pytest.fail("unsealed registry opened labels"))
    with pytest.raises(ValueError, match="registry"):
        study_a.report(source["config"], source["run"], tmp_path / "report", source["scorer"],
                       selected_path, freeze, registry_path, device="cpu")


def test_tuning_rejects_changed_candidates_returned_by_the_scorer(source, tmp_path, monkeypatch):
    FakeScorer.change_candidates = True
    with pytest.raises(ValueError, match="candidate"):
        tune_source(source, tmp_path / "changed", monkeypatch)


@pytest.mark.parametrize("changed", ["candidate_file", "scorer_asset", "frozen_hash"])
def test_report_rejects_changed_assets_before_labels_or_scorer_calls(source, tmp_path, monkeypatch, changed):
    tune_source(source, tmp_path / "tune", monkeypatch)
    selected_path = tmp_path / "tune" / "selected_config.json"
    freeze, expected = joint_freeze(source, selected_path, tmp_path)
    if changed == "candidate_file":
        with (source["run"] / "candidates.jsonl").open("a") as stream:
            stream.write("\n")
    elif changed == "scorer_asset":
        (source["scorer"] / "config.json").write_text("changed")
    else:
        value = json.loads(freeze.read_text())
        del value["manifest_hash"]
        value["studies"]["confidence_lm"]["config_hash"] = "changed"
        write_json(freeze, value, hashed=True)
    FakeScorer.calls = []

    def forbidden(record):
        raise AssertionError("invalid frozen state opened references")

    monkeypatch.setattr(study_a, "reference_only", forbidden)
    with pytest.raises(ValueError, match="hash|checksum|changed"):
        study_a.report(source["config"], source["run"], tmp_path / "report", source["scorer"],
                       selected_path, freeze, expected, device="cpu")
    assert FakeScorer.calls == []


def test_report_recomputes_study_b_artifacts_instead_of_accepting_stale_expected_values(source, tmp_path, monkeypatch):
    tune_source(source, tmp_path / "tune", monkeypatch)
    selected_path = tmp_path / "tune" / "selected_config.json"
    freeze, registry_path = joint_freeze(source, selected_path, tmp_path)
    registry = json.loads(registry_path.read_text())
    b_checkpoint = registry["artifacts"]["studies"]["session_calibration"]["checkpoint"]["calibration"]
    Path(b_checkpoint).write_bytes(b"changed after the joint freeze")

    def forbidden(record):
        raise AssertionError("Changed study B assets must fail before label access")

    monkeypatch.setattr(study_a, "reference_only", forbidden)
    with pytest.raises(ValueError, match="session_calibration"):
        study_a.report(source["config"], source["run"], tmp_path / "report", source["scorer"],
                       selected_path, freeze, registry_path, device="cpu")


def test_report_requires_both_frozen_studies_before_reference_access(source, tmp_path, monkeypatch):
    tune_source(source, tmp_path / "tune", monkeypatch)
    selected_path = tmp_path / "tune" / "selected_config.json"
    freeze, expected = joint_freeze(source, selected_path, tmp_path)
    value = json.loads(freeze.read_text())
    del value["manifest_hash"]
    del value["studies"]["session_calibration"]
    write_json(freeze, value, hashed=True)

    def forbidden(record):
        raise AssertionError("joint freeze was not checked before labels")

    monkeypatch.setattr(study_a, "reference_only", forbidden)
    with pytest.raises(ValueError, match="stud"):
        study_a.report(source["config"], source["run"], tmp_path / "report", source["scorer"],
                       selected_path, freeze, expected, device="cpu")


def test_deployment_really_calls_each_policy_and_preserves_negative_results_and_phones(source, tmp_path, monkeypatch):
    selected = tune_source(source, tmp_path / "tune", monkeypatch)
    selected_path = tmp_path / "tune" / "selected_config.json"
    freeze, expected = joint_freeze(source, selected_path, tmp_path)
    FakeScorer.calls = []
    opened = []

    def report_reader(record):
        assert record["trial_id"] in source["report_ids"]
        opened.append(record["trial_id"])
        return {"reference_text": f"bad {record['trial_id']}", "phoneme_ids": [1, 1, 2]}

    monkeypatch.setattr(study_a, "reference_only", report_reader)
    output = tmp_path / "report"
    summary = study_a.report(source["config"], source["run"], output, source["scorer"],
                             selected_path, freeze, expected, device="cpu")
    assert sorted(opened) == sorted(source["report_ids"])
    assert len(summary["conditions"]) == 6
    gated_calls = summary["conditions"]["A2"]["lm_calls"]
    assert len(FakeScorer.calls) == len(source["records"]) + 4 * gated_calls
    assert summary["conditions"]["A1"]["word_effect"]["delta_b_minus_a"] > 0
    assert summary["conditions"]["A1"]["word_effect"]["harm_count"] == len(source["report_ids"])
    for name, condition in summary["conditions"].items():
        saved = [json.loads(line) for line in (output / name / "full_predictions.jsonl").read_text().splitlines()]
        assert all(row["phoneme_ids"] == [1, 1, 2] for row in saved)
        assert condition["per"] == 0
        if name.startswith("A3_"):
            assert condition["lm_calls"] == gated_calls
            assert condition["call_bins"] == summary["conditions"]["A2"]["call_bins"]
            assert condition["scored_call_bins"] == summary["conditions"]["A2"]["scored_call_bins"]
            assert condition["partition_calls"] == summary["conditions"]["A2"]["partition_calls"]
        assert "Random" in condition["corpus_effects"]
    assert summary["selected_lambda"] == selected["selected_lambda"]
    assert summary["conditions"]["A0"]["trial_latency_p50_seconds"] == pytest.approx(0.0065)
    assert summary["shared_compact_decode"]["recorded_source_run_elapsed_seconds"] == 2.5
    assert summary["shared_compact_decode"]["sum_trial_decode_seconds"] == pytest.approx(0.078)
    assert (output / "A2" / "resources.json").is_file()
    assert (output / "study_summary.json").is_file()


def test_failed_lm_calls_count_word_failures_without_erasing_valid_phone_predictions(source, tmp_path, monkeypatch):
    tune_source(source, tmp_path / "tune", monkeypatch)
    selected_path = tmp_path / "tune" / "selected_config.json"
    freeze, expected = joint_freeze(source, selected_path, tmp_path)
    FakeScorer.fail_candidates = True
    monkeypatch.setattr(study_a, "reference_only", lambda row: {
        "reference_text": f"bad {row['trial_id']}", "phoneme_ids": [1, 1, 2]
    })
    output = tmp_path / "failures"
    summary = study_a.report(source["config"], source["run"], output, source["scorer"],
                             selected_path, freeze, expected, device="cpu")
    metrics = json.loads((output / "A1" / "metrics.json").read_text())
    assert metrics["failed_count"] == len(source["report_ids"])
    assert metrics["wer"] == 1
    assert metrics["per"] == 0
    assert summary["conditions"]["A1"]["lm_failures"] == len(source["records"])
