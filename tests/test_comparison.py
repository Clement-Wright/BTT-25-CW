import copy
import csv
import json

import pytest

from btt25.comparison import compare_runs
from btt25.evaluation import evaluate_predictions
from btt25.reporting import write_evaluation


def result_fixture(second_run=False):
    references = {
        "short": {"reference_text": "one", "phoneme_ids": [1]},
        "long": {"reference_text": "a b c d e f g h i", "phoneme_ids": [2] * 9},
    }
    records = [
        {"trial_id": "short", "split": "val", "official_order_index": 0,
         "session_id": "s1", "block_id": 1, "corpus_id": None,
         "source_file": "C:/private/neural/data_val.hdf5"},
        {"trial_id": "long", "split": "val", "official_order_index": 1,
         "session_id": "s1", "block_id": 2, "corpus_id": None,
         "source_file": "C:/private/neural/data_val.hdf5"},
    ]
    if second_run:
        predictions = [
            {"trial_id": "short", "text": "one", "phoneme_ids": [1], "status": "ok"},
            {"trial_id": "long", "text": "x y z d e f g h i", "phoneme_ids": [3] * 3 + [2] * 6, "status": "ok"},
        ]
    else:
        predictions = [
            {"trial_id": "short", "text": "wrong", "phoneme_ids": [3], "status": "ok"},
            {"trial_id": "long", "text": "a b c d e f g h i", "phoneme_ids": [2] * 9, "status": "ok"},
        ]
    manifest = {"dev_tune": ["short", "long"], "dev_report": [], "manifest_hash": "split-v1"}
    return evaluate_predictions(predictions, records, manifest, lambda row: references[row["trial_id"]])


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_report_exports_only_scored_rows_and_sums_subgroup_denominators(tmp_path):
    result = result_fixture()
    output = write_evaluation(result, tmp_path / "run", {"checkpoint_hash": "sha256-checkpoint"})
    for name in (
        "metrics.json", "predictions.jsonl", "per_trial.jsonl", "by_session.csv",
        "by_corpus.csv", "length_bins.csv", "report.md",
    ):
        assert (output / name).is_file()
    saved = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [row["trial_id"] for row in saved] == ["short", "long"]
    assert all("source_file" not in row for row in saved)
    assert "source_file" in result["scored_rows"][0]
    session = read_csv(output / "by_session.csv")[0]
    assert session["session_id"] == "s1"
    assert int(session["word_reference_length"]) == 10
    assert float(session["wer"]) == pytest.approx(0.1)
    corpus = read_csv(output / "by_corpus.csv")[0]
    assert corpus["corpus_id"] == "unknown"
    assert float(corpus["per"]) == pytest.approx(0.1)
    bins = {row["length_bin"]: row for row in read_csv(output / "length_bins.csv")}
    assert int(bins["1-5"]["word_reference_length"]) == 1
    assert int(bins["6-10"]["word_reference_length"]) == 9


def test_phone_only_report_marks_word_metrics_unmeasured(tmp_path):
    result = result_fixture()
    output = write_evaluation(result, tmp_path / "phones", {"text_available": False})
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["wer"] is None
    assert metrics["text_available"] is False
    assert metrics["per"] == pytest.approx(0.1)
    assert read_csv(output / "by_session.csv")[0]["wer"] == ""
    report = (output / "report.md").read_text()
    assert "WER: not measured" in report
    assert "phonemes only" in report
    assert "100.00%" not in report


def test_report_selects_examples_by_error_then_trial_id_and_keeps_failures(tmp_path):
    records = [
        {"trial_id": trial_id, "split": "val", "official_order_index": order}
        for order, trial_id in enumerate(["z", "b", "a"])
    ]
    predictions = [
        {"trial_id": "z", "text": "plausible stale output", "phoneme_ids": [1], "status": "failed"},
        {"trial_id": "b", "text": "yes", "phoneme_ids": [1], "status": "ok"},
        {"trial_id": "a", "text": "yes", "phoneme_ids": [1], "status": "ok"},
    ]
    result = evaluate_predictions(
        predictions, records,
        {"dev_tune": ["z", "b", "a"], "dev_report": [], "manifest_hash": "split-v1"},
        lambda row: {"reference_text": "yes", "phoneme_ids": [1]},
    )
    report = (write_evaluation(result, tmp_path / "examples") / "report.md").read_text()
    assert "Best example: `a`" in report
    assert "Median example: `b`" in report
    assert "Worst example: `z`" in report
    assert "Status: failed" in report
    assert "plausible stale output" not in report


def test_report_rejects_reporting_rows_without_a_freeze_before_writing(tmp_path):
    result = result_fixture()
    for row in [result["metrics"], *result["scored_rows"], *result["per_trial"]]:
        row["score_split"] = "dev_report"
    output = tmp_path / "blocked"
    with pytest.raises(ValueError, match="freeze"):
        write_evaluation(result, output)
    assert not output.exists()


@pytest.mark.parametrize("changed", ["score_split", "split_manifest_hash", "trial_ids"])
def test_comparison_rejects_mismatched_scoring_populations(tmp_path, changed):
    first, second = result_fixture(), result_fixture(True)
    if changed == "trial_ids":
        second["scored_rows"][0]["trial_id"] = "different"
        second["per_trial"][0]["trial_id"] = "different"
    else:
        for row in [second["metrics"], *second["scored_rows"], *second["per_trial"]]:
            row[changed] = "official_val" if changed == "score_split" else "split-v2"
            if changed == "score_split":
                row["freeze_manifest_hash"] = "freeze-v1"
    run_a = write_evaluation(first, tmp_path / "a")
    run_b = write_evaluation(second, tmp_path / "b")
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="split|trial"):
        compare_runs(run_a, run_b, output, replicates=50)
    assert not output.exists()


def test_comparison_uses_corpus_ratios_and_reports_rescue_and_harm(tmp_path):
    run_a = write_evaluation(result_fixture(), tmp_path / "a")
    run_b = write_evaluation(result_fixture(True), tmp_path / "b")
    resources = {"elapsed_seconds": 12.0, "peak_tree_rss_bytes": 1234}
    (run_b / "resources.json").write_text(json.dumps(resources))
    output = tmp_path / "comparison"
    result = compare_runs(run_a, run_b, output, replicates=200)
    for metric in ("per", "wer"):
        values = result[metric]
        assert values["rate_a"] == pytest.approx(0.1)
        assert values["rate_b"] == pytest.approx(0.3)
        assert values["delta_b_minus_a"] == pytest.approx(0.2)
        assert values["rescue_count"] == 1
        assert values["harm_count"] == 1
        assert values["rescue_rate"] == 0.5
        assert values["harm_rate"] == 0.5
        assert values["bootstrap"]["units"] == 2
        assert values["session_cluster_bootstrap"]["units"] == 1
        assert values["session_block_cluster_bootstrap"]["units"] == 2
    assert result["runs"]["b"]["resources"] == resources
    assert json.loads((output / "comparison.json").read_text())["per"]["delta_b_minus_a"] == pytest.approx(0.2)
    assert (output / "paired_trials.csv").is_file()


def test_paired_self_comparison_has_exactly_zero_intervals(tmp_path):
    run = write_evaluation(result_fixture(), tmp_path / "a")
    result = compare_runs(run, run, tmp_path / "self", seed=4, replicates=200)
    for metric in ("per", "wer"):
        values = result[metric]
        assert values["delta_b_minus_a"] == 0
        assert values["rescue_count"] == 0
        assert values["harm_count"] == 0
        assert values["unchanged_count"] == 2
        for interval in ("bootstrap", "session_cluster_bootstrap", "session_block_cluster_bootstrap"):
            assert values[interval]["lower"] == 0
            assert values[interval]["upper"] == 0
    report = (tmp_path / "self" / "report.md").read_text()
    assert "single-participant dataset" in report


def test_phone_only_comparison_does_not_turn_empty_text_into_a_word_baseline(tmp_path):
    run_a = write_evaluation(result_fixture(), tmp_path / "a", {"text_available": False})
    run_b = write_evaluation(result_fixture(True), tmp_path / "b")
    result = compare_runs(run_a, run_b, tmp_path / "comparison", replicates=50)
    assert result["per"]["measured"] is True
    assert result["wer"]["measured"] is False
    assert "WER: not measured" in (tmp_path / "comparison" / "report.md").read_text()


def test_session_cluster_resampling_preserves_correlated_trial_changes(tmp_path):
    records, references, predictions_a, predictions_b = [], {}, [], []
    for i in range(10):
        trial_id = f"trial-{i:02d}"
        records.append({"trial_id": trial_id, "split": "val", "official_order_index": i,
                        "session_id": "s1" if i < 5 else "s2", "block_id": 1})
        references[trial_id] = {"reference_text": "yes", "phoneme_ids": [1]}
        for predictions, correct in ((predictions_a, i >= 5), (predictions_b, i < 5)):
            predictions.append({"trial_id": trial_id, "text": "yes" if correct else "no",
                                "phoneme_ids": [1] if correct else [2], "status": "ok"})
    manifest = {"dev_tune": list(references), "dev_report": [], "manifest_hash": "split-v1"}
    run_a, run_b = [
        write_evaluation(evaluate_predictions(predictions, records, manifest,
                         lambda row: references[row["trial_id"]]), tmp_path / name)
        for predictions, name in ((predictions_a, "a"), (predictions_b, "b"))
    ]
    result = compare_runs(run_a, run_b, tmp_path / "cluster", seed=7, replicates=1000)
    values = result["per"]
    assert values["delta_b_minus_a"] == 0
    assert values["session_cluster_bootstrap"]["units"] == 2
    assert values["session_cluster_bootstrap"]["lower"] == -1
    assert values["session_cluster_bootstrap"]["upper"] == 1
    assert values["session_cluster_bootstrap"]["lower"] < values["bootstrap"]["lower"]
    assert values["session_cluster_bootstrap"]["upper"] > values["bootstrap"]["upper"]


def test_comparison_is_reproducible_for_the_same_seed(tmp_path):
    run_a = write_evaluation(result_fixture(), tmp_path / "a")
    run_b = write_evaluation(result_fixture(True), tmp_path / "b")
    first = compare_runs(run_a, run_b, tmp_path / "first", seed=42, replicates=100)
    second = compare_runs(run_a, run_b, tmp_path / "second", seed=42, replicates=100)
    assert first == second


def test_comparison_rejects_changed_reference_lengths_even_if_split_hash_matches(tmp_path):
    run_a = write_evaluation(result_fixture(), tmp_path / "a")
    second = copy.deepcopy(result_fixture(True))
    second["per_trial"][0]["phoneme"]["reference_length"] = 2
    run_b = write_evaluation(second, tmp_path / "b")
    with pytest.raises(ValueError, match="reference"):
        compare_runs(run_a, run_b, tmp_path / "bad", replicates=50)
