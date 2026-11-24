import copy
import csv

import pytest

from btt25.evaluation import (
    corpus_metrics,
    edit_counts,
    evaluate_predictions,
    export_submission,
    normalize_text,
)


def test_edit_counts_preserve_repeated_reference_labels():
    assert edit_counts([1, 1, 2], [1, 2]) == {
        "substitutions": 0,
        "deletions": 1,
        "insertions": 0,
        "reference_length": 3,
        "errors": 1,
    }


def test_edit_alignment_prefers_substitutions_on_equal_cost_paths():
    assert edit_counts(["a", "b"], ["b", "a"]) == {
        "substitutions": 2,
        "deletions": 0,
        "insertions": 0,
        "reference_length": 2,
        "errors": 2,
    }


def test_empty_reference_retains_insertions():
    assert edit_counts([], ["a", "b"])["insertions"] == 2
    assert edit_counts(["a", "b"], [])["deletions"] == 2


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("  DON'T, stop!  ", "don't stop"),
        ("well-known a- b--c 'd' 123", "well-known a bc'd'"),
        ("Word,word\tX\nY café", "wordwordxy caf"),
    ],
)
def test_normalization_matches_the_pinned_official_helper(original, expected):
    assert normalize_text(original) == expected


def test_corpus_rates_use_summed_edits_and_reference_lengths():
    rows = [
        {
            "reference_text": "one",
            "text": "wrong",
            "reference_phoneme_ids": [1, 1],
            "phoneme_ids": [1],
            "status": "ok",
        },
        {
            "reference_text": "a b c d e f g h i",
            "text": "a b c d e f g h i",
            "reference_phoneme_ids": [2] * 8,
            "phoneme_ids": [2] * 8,
            "status": "ok",
        },
    ]
    metrics = corpus_metrics(rows)
    assert metrics["wer"] == pytest.approx(0.1)
    assert metrics["per"] == pytest.approx(0.1)
    assert metrics["word"]["reference_length"] == 10
    assert metrics["phoneme"]["deletions"] == 1


def test_corpus_rate_can_exceed_one_and_empty_denominator_is_undefined():
    row = {
        "reference_text": "one",
        "text": "one two three four",
        "reference_phoneme_ids": [],
        "phoneme_ids": [2, 3],
        "status": "ok",
    }
    metrics = corpus_metrics([row])
    assert metrics["wer"] == 3.0
    assert metrics["per"] is None
    assert metrics["phoneme"]["insertions"] == 2
    assert corpus_metrics([])["wer"] is None


def validation_fixture():
    records = [
        {"trial_id": "report", "split": "val", "official_order_index": 0},
        {"trial_id": "tune-b", "split": "val", "official_order_index": 2},
        {"trial_id": "tune-a", "split": "val", "official_order_index": 1},
    ]
    manifest = {
        "dev_tune": ["tune-b", "tune-a"],
        "dev_report": ["report"],
        "manifest_hash": "split-v1",
    }
    predictions = [
        {"trial_id": "report", "text": "locked", "phoneme_ids": [9], "status": "ok"},
        {"trial_id": "tune-a", "text": "a a", "phoneme_ids": [1, 1], "status": "ok"},
        {"trial_id": "tune-b", "text": "b", "phoneme_ids": [2], "status": "ok"},
    ]
    return records, manifest, predictions


def tune_only_reference(record):
    if record["trial_id"] == "report":
        raise AssertionError("locked reporting labels were opened")
    if record["trial_id"] == "tune-a":
        return {"reference_text": "a a", "phoneme_ids": [1, 1]}
    return {"reference_text": "b", "phoneme_ids": [2]}


def freeze_fixture(split_hash="split-v1"):
    baseline = {"config_hash": "b-c", "checkpoint_hash": "b-p", "assets_hash": "b-a"}
    studies = {
        "confidence_lm": {"config_hash": "l-c", "checkpoint_hash": "l-p", "assets_hash": "l-a"},
        "session_calibration": {
            "config_hash": "s-c", "checkpoint_hash": "s-p", "assets_hash": "s-a"
        },
    }
    expected = {"baseline": baseline, "studies": studies, "split_hash": split_hash}
    freeze = {
        **copy.deepcopy(expected),
        "selection_rules": "Select on dev_tune only, then freeze both primary studies.",
        "manifest_hash": "freeze-v1",
    }
    return freeze, expected


def test_default_selection_never_opens_reporting_references():
    records, manifest, predictions = validation_fixture()
    result = evaluate_predictions(predictions, records, manifest, tune_only_reference)
    assert result["metrics"]["wer"] == 0
    assert result["metrics"]["score_split"] == "dev_tune"
    assert [row["trial_id"] for row in result["scored_rows"]] == ["tune-a", "tune-b"]
    assert result["scored_rows"][0]["score_split"] == "dev_tune"
    assert result["scored_rows"][0]["split_manifest_hash"] == "split-v1"
    assert [row["trial_id"] for row in result["per_trial"]] == ["tune-a", "tune-b"]
    assert result["metrics"]["evaluated_count"] == 2


@pytest.mark.parametrize("score_split", ["dev_report", "official_val"])
def test_reporting_is_rejected_before_any_label_reader_call(score_split):
    records, manifest, predictions = validation_fixture()

    def forbidden_reader(record):
        raise AssertionError("freeze validation must precede all reference access")

    with pytest.raises(ValueError, match="freeze"):
        evaluate_predictions(predictions, records, manifest, forbidden_reader, score_split)


def test_missing_and_failed_predictions_remain_in_both_denominators():
    records, manifest, predictions = validation_fixture()
    predictions = [
        predictions[0],
        {"trial_id": "tune-b", "text": "b", "phoneme_ids": [2], "status": "decoder_error"},
    ]
    result = evaluate_predictions(predictions, records, manifest, tune_only_reference)
    metrics = result["metrics"]
    assert metrics["evaluated_count"] == 2
    assert metrics["failed_count"] == 2
    assert metrics["missing_count"] == 1
    assert metrics["skipped_count"] == 0
    assert metrics["prediction_coverage"] == 0.5
    assert metrics["word"]["reference_length"] == 3
    assert metrics["phoneme"]["reference_length"] == 3
    assert metrics["wer"] == 1.0
    assert metrics["per"] == 1.0
    assert [row["text"] for row in result["scored_rows"]] == ["", ""]
    assert [row["status"] for row in result["scored_rows"]] == ["missing", "decoder_error"]


def test_joint_freeze_allows_report_and_full_validation():
    records, manifest, predictions = validation_fixture()
    freeze, expected = freeze_fixture()

    def reader(record):
        if record["trial_id"] == "report":
            return {"reference_text": "locked", "phoneme_ids": [9]}
        return tune_only_reference(record)

    report = evaluate_predictions(
        predictions, records, manifest, reader, "dev_report", freeze, expected
    )
    assert report["metrics"]["evaluated_count"] == 1
    assert report["metrics"]["freeze_manifest_hash"] == "freeze-v1"
    assert report["scored_rows"][0]["trial_id"] == "report"
    assert report["scored_rows"][0]["freeze_manifest_hash"] == "freeze-v1"
    assert report["per_trial"][0]["score_split"] == "dev_report"
    full = evaluate_predictions(
        predictions, records, manifest, reader, "official_val", freeze, expected
    )
    assert full["metrics"]["evaluated_count"] == 3
    assert full["metrics"]["wer"] == 0


@pytest.mark.parametrize("change", ["config_hash", "checkpoint_hash", "assets_hash", "split_hash"])
def test_changed_frozen_hashes_fail_before_reference_access(change):
    records, manifest, predictions = validation_fixture()
    freeze, expected = freeze_fixture()
    if change == "split_hash":
        expected[change] = "changed"
    else:
        expected["studies"]["confidence_lm"][change] = "changed"

    def forbidden_reader(record):
        raise AssertionError("hash validation must precede all reference access")

    with pytest.raises(ValueError, match="hash"):
        evaluate_predictions(
            predictions, records, manifest, forbidden_reader, "dev_report", freeze, expected
        )


def test_freeze_requires_both_studies_and_live_expected_hashes():
    records, manifest, predictions = validation_fixture()
    freeze, expected = freeze_fixture()
    del freeze["studies"]["session_calibration"]
    with pytest.raises(ValueError, match="stud"):
        evaluate_predictions(
            predictions, records, manifest, tune_only_reference, "dev_report", freeze, expected
        )
    freeze, _ = freeze_fixture()
    with pytest.raises(ValueError, match="expected_hashes"):
        evaluate_predictions(
            predictions, records, manifest, tune_only_reference, "dev_report", freeze
        )


def test_paired_dropout_replacement_requires_a_recorded_reason():
    records, manifest, predictions = validation_fixture()
    freeze, expected = freeze_fixture()
    freeze["studies"]["paired_dropout"] = freeze["studies"].pop("confidence_lm")
    expected["studies"]["paired_dropout"] = expected["studies"].pop("confidence_lm")
    with pytest.raises(ValueError, match="replacement_reason"):
        evaluate_predictions(
            predictions, records, manifest, tune_only_reference, "dev_report", freeze, expected
        )
    freeze["replacement_reason"] = "The confidence LM assets are unavailable; predeclared fallback."
    result = evaluate_predictions(
        predictions, records, manifest,
        lambda record: {"reference_text": "locked", "phoneme_ids": [9]},
        "dev_report", freeze, expected,
    )
    assert result["metrics"]["evaluated_count"] == 1


@pytest.mark.parametrize("problem", ["duplicate_prediction", "unknown_prediction", "overlap", "unknown_trial", "wrong_split"])
def test_invalid_metadata_is_rejected_before_label_access(problem):
    records, manifest, predictions = validation_fixture()
    if problem == "duplicate_prediction":
        predictions.append(dict(predictions[0]))
    elif problem == "unknown_prediction":
        predictions.append({"trial_id": "unknown", "text": "", "phoneme_ids": [], "status": "ok"})
    elif problem == "overlap":
        manifest["dev_report"].append("tune-a")
    elif problem == "unknown_trial":
        manifest["dev_tune"].append("unknown")
    else:
        records[1]["split"] = "test"

    def forbidden_reader(record):
        raise AssertionError("metadata validation must precede reference access")

    with pytest.raises(ValueError):
        evaluate_predictions(predictions, records, manifest, forbidden_reader)


def test_submission_uses_captured_order_and_sequential_ids(tmp_path):
    records = [
        {"trial_id": "session-z/2", "split": "test", "official_order_index": 1},
        {"trial_id": "not-test", "split": "val", "official_order_index": 0},
        {"trial_id": "session-z/10", "split": "test", "official_order_index": 0},
    ]
    predictions = [
        {"trial_id": "session-z/2", "text": "second, sentence", "status": "ok"},
        {"trial_id": "session-z/10", "text": "first sentence", "status": "ok"},
    ]
    output = tmp_path / "submission.csv"
    assert export_submission(predictions, records, output) == output
    with output.open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["id", "text"], ["0", "first sentence"], ["1", "second, sentence"]
        ]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "extra", "failure", "order_collision"])
def test_submission_rejects_incomplete_ambiguous_or_failed_rows(tmp_path, problem):
    records = [
        {"trial_id": "a", "split": "test", "official_order_index": 0},
        {"trial_id": "b", "split": "test", "official_order_index": 1},
    ]
    predictions = [
        {"trial_id": "a", "text": "a", "status": "ok"},
        {"trial_id": "b", "text": "b", "status": "ok"},
    ]
    if problem == "missing":
        predictions.pop()
    elif problem == "duplicate":
        predictions.append(dict(predictions[0]))
    elif problem == "extra":
        predictions.append({"trial_id": "other", "text": "x", "status": "ok"})
    elif problem == "failure":
        predictions[1]["status"] = "failed"
    else:
        records[1]["official_order_index"] = 0
    output = tmp_path / "submission.csv"
    with pytest.raises(ValueError):
        export_submission(predictions, records, output)
    assert not output.exists()
