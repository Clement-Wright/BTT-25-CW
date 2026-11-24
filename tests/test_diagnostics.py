import json
from pathlib import Path

import pytest

from btt25.diagnostics import fixed_examples, write_diagnostics
from btt25.evaluation import evaluate_predictions
from btt25.reporting import write_evaluation


def scored_run(tmp_path, name="a", improved=False, text_available=True):
    records = [
        {"trial_id": "short", "split": "val", "official_order_index": 0,
         "session_id": "s1", "corpus_id": "Random words", "source_file": "never-open-dataset.hdf5"},
        {"trial_id": "long", "split": "val", "official_order_index": 1,
         "session_id": "s1", "corpus_id": None, "source_file": "never-open-dataset.hdf5"},
        {"trial_id": "empty", "split": "val", "official_order_index": 2,
         "session_id": None, "corpus_id": None, "source_file": "never-open-dataset.hdf5"},
    ]
    references = {
        "short": {"reference_text": "Known known MYSTERY.", "phoneme_ids": [1, 1, 2]},
        "long": {"reference_text": "known rare rare other", "phoneme_ids": [3] * 4},
        "empty": {"reference_text": "", "phoneme_ids": []},
    }
    predictions = [
        {"trial_id": "short", "text": "known known mystery" if improved else "known known wrong",
         "phoneme_ids": [1, 1, 2], "status": "ok"},
        {"trial_id": "empty", "text": "", "phoneme_ids": [], "status": "ok"},
    ]
    if improved:
        predictions.append({"trial_id": "long", "text": "known rare rare other", "phoneme_ids": [3] * 4, "status": "ok"})
    result = evaluate_predictions(
        predictions, records,
        {"dev_tune": ["short", "long", "empty"], "dev_report": [], "manifest_hash": "split-v1"},
        lambda row: references[row["trial_id"]],
    )
    return write_evaluation(result, tmp_path / name, {"text_available": text_available})


@pytest.fixture
def lexicon(tmp_path):
    path = tmp_path / "lexicon.tsv"
    path.write_text(";;; alternate pronunciations do not add types\nKNOWN K N OW N\nknown K N OW N\nother AH DH ER\n", encoding="utf-8")
    return path


def test_oov_token_and_type_coverage_use_reference_denominators_and_keep_failures(tmp_path, lexicon):
    run = scored_run(tmp_path)
    diagnostics = write_diagnostics(run, lexicon)
    overall = diagnostics["overall"]
    oov = overall["lexicon_oov"]
    assert overall["trial_count"] == 3
    assert overall["failed_count"] == 1
    assert overall["missing_count"] == 1
    assert oov["reference_token_count"] == 7
    assert oov["reference_type_count"] == 4
    assert oov["oov_token_count"] == 3
    assert oov["oov_type_count"] == 2
    assert oov["token_coverage"] == pytest.approx(4 / 7)
    assert oov["type_coverage"] == 0.5
    assert oov["sentence_oov_rate"] == pytest.approx(2 / 3)
    assert oov["empty_reference_count"] == 1
    assert oov["oov_counts"] == {"mystery": 1, "rare": 2}
    assert "wrong" not in oov["oov_counts"]
    assert diagnostics["lexicon_type_count"] == 2
    assert json.loads((run / "diagnostics.json").read_text())["overall"]["lexicon_oov"] == oov


def test_grouped_diagnostics_preserve_unknown_metadata_and_sum_edit_types(tmp_path, lexicon):
    result = write_diagnostics(scored_run(tmp_path), lexicon, tmp_path / "diagnostics")
    assert result["overall"]["word"] == {
        "substitutions": 1, "deletions": 4, "insertions": 0,
        "errors": 5, "reference_length": 7, "error_rate": pytest.approx(5 / 7),
    }
    assert result["overall"]["phoneme"]["deletions"] == 4
    assert result["overall"]["per"] == pytest.approx(4 / 7)
    assert result["by_session"]["s1"]["word"]["reference_length"] == 7
    assert result["by_session"]["unknown"]["trial_count"] == 1
    unknown = result["by_corpus"]["unknown"]
    assert unknown["trial_count"] == 2
    assert unknown["lexicon_oov"]["oov_token_count"] == 2
    assert unknown["lexicon_oov"]["sentence_oov_rate"] == 0.5
    assert result["unusual_random_corpora"]["Random words"]["word"]["reference_length"] == 3


def test_empty_reference_denominators_are_undefined_and_phone_only_wer_is_unmeasured(tmp_path, lexicon):
    result = write_diagnostics(scored_run(tmp_path, text_available=False), lexicon)
    empty = result["by_session"]["unknown"]
    assert empty["lexicon_oov"]["token_coverage"] is None
    assert empty["lexicon_oov"]["type_coverage"] is None
    assert empty["lexicon_oov"]["sentence_oov_rate"] == 0
    assert result["overall"]["wer"] is None
    assert result["overall"]["word"]["error_rate"] is None
    assert result["overall"]["per"] == pytest.approx(4 / 7)


def test_unauthorized_reporting_trace_is_rejected_before_cached_reference_access(tmp_path, lexicon, monkeypatch):
    run = scored_run(tmp_path)
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["score_split"] = "dev_report"
    metrics_path.write_text(json.dumps(metrics))
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if path.name in {"predictions.jsonl", "per_trial.jsonl"}:
            raise AssertionError("Locked cached references were opened before the scoring gate")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    with pytest.raises(ValueError, match="freeze"):
        write_diagnostics(run, lexicon)


def test_fixed_examples_preserve_declared_ids_across_conditions_and_do_not_rank_by_errors(tmp_path):
    first, second = scored_run(tmp_path, "a"), scored_run(tmp_path, "b", improved=True)
    result = fixed_examples({"A0": first, "A1": second}, ["long", "short"], tmp_path / "examples")
    assert result["ids"] == ["long", "short"]
    assert [row["trial_id"] for row in result["rows"]] == ["long", "short"]
    long = result["rows"][0]
    assert long["reference_text"] == "known rare rare other"
    assert long["predictions"]["A0"]["status"] == "missing"
    assert long["predictions"]["A0"]["text"] == ""
    assert long["predictions"]["A1"]["text"] == "known rare rare other"
    assert result["markdown"].index("long") < result["markdown"].index("short")
    assert json.loads((tmp_path / "examples" / "fixed_examples.json").read_text())["ids"] == ["long", "short"]
    assert (tmp_path / "examples" / "fixed_examples.md").read_text() == result["markdown"]


def test_fixed_examples_display_phone_predictions_for_phone_only_conditions(tmp_path):
    first = scored_run(tmp_path, "phones", text_available=False)
    second = scored_run(tmp_path, "words", improved=True)
    result = fixed_examples({"phones": first, "words": second}, ["short"])
    assert result["rows"][0]["predictions"]["phones"]["text"] is None
    assert result["rows"][0]["predictions"]["phones"]["phoneme_ids"] == [1, 1, 2]
    assert "phones (phonemes)" in result["markdown"]
    assert "Reference phones" in result["markdown"]


@pytest.mark.parametrize("ids", [["short", "short"], ["locked-report-trial"]])
def test_fixed_examples_reject_duplicate_or_unscored_ids_without_writing(tmp_path, ids):
    run = scored_run(tmp_path)
    output = tmp_path / "blocked"
    with pytest.raises(ValueError, match="ID|allowlist"):
        fixed_examples({"A0": run}, ids, output)
    assert not output.exists()


@pytest.mark.parametrize("changed", ["split", "allowlist", "reference"])
def test_fixed_examples_require_identical_scoring_identity_allowlists_and_references(tmp_path, changed):
    first, second = scored_run(tmp_path, "a"), scored_run(tmp_path, "b", improved=True)
    if changed == "split":
        metrics_path = second / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["split_manifest_hash"] = "different-split"
        metrics_path.write_text(json.dumps(metrics))
        for filename in ("predictions.jsonl", "per_trial.jsonl"):
            rows = [json.loads(line) for line in (second / filename).read_text().splitlines()]
            for row in rows:
                row["split_manifest_hash"] = "different-split"
            (second / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif changed == "allowlist":
        for filename in ("predictions.jsonl", "per_trial.jsonl"):
            rows = [json.loads(line) for line in (second / filename).read_text().splitlines()]
            rows = [row for row in rows if row["trial_id"] != "empty"]
            (second / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        rows = [json.loads(line) for line in (second / "predictions.jsonl").read_text().splitlines()]
        rows[1]["reference_text"] = "changed words of sentence"
        (second / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="split|allowlist|reference"):
        fixed_examples({"A0": first, "A1": second}, ["short"])
