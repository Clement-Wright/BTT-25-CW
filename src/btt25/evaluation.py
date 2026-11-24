"""Corpus error counts, reference-access gating, and ordered test export.

Reporting freeze schema::

    {
      "baseline": {"config_hash": str, "checkpoint_hash": str, "assets_hash": str},
      "studies": {
        "confidence_lm": {"config_hash": str, "checkpoint_hash": str, "assets_hash": str},
        "session_calibration": {same three hash fields}
      },
      "split_hash": str, "selection_rules": str, "manifest_hash": str
    }

The documented fallback replaces confidence_lm with paired_dropout and adds a
nonempty replacement_reason. expected_hashes has baseline, studies, split_hash
with the same structure, computed by the caller from the current run's assets
and configurations rather than copied from the freeze. Hash validation happens
before reference_loader is called. manifest_hash identifies the saved manifest;
the caller is responsible for computing and checking its content hash on read.
"""

import csv
from collections.abc import Callable, Iterable, Sequence
from operator import index
from pathlib import Path
import re


_COUNT_KEYS = ("substitutions", "deletions", "insertions", "reference_length", "errors")
_HASH_KEYS = ("config_hash", "checkpoint_hash", "assets_hash")
_TRACE_KEYS = ("score_split", "split_manifest_hash", "freeze_manifest_hash")


def edit_counts(reference: Sequence, hypothesis: Sequence) -> dict:
    """Unit-cost Levenshtein counts; ties prefer diagonal, deletion, insertion.

    Repeated reference tokens are preserved. Empty references retain insertion
    counts; error rates with a zero corpus reference length are undefined.
    """
    reference, hypothesis = list(reference), list(hypothesis)
    # Each cell carries (errors, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, token in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, predicted in enumerate(hypothesis, 1):
            diagonal, above, left = previous[j - 1], previous[j], current[j - 1]
            mismatch = int(token != predicted)
            candidates = [
                (diagonal[0] + mismatch, diagonal[1] + mismatch, diagonal[2], diagonal[3]),
                (above[0] + 1, above[1], above[2] + 1, above[3]),
                (left[0] + 1, left[1], left[2], left[3] + 1),
            ]
            current.append(min(candidates, key=lambda candidate: candidate[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_length": len(reference),
        "errors": errors,
    }


def normalize_text(s: str) -> str:
    """Match official helper at upstream commit 980a5b7e96392210edf976d76bdb5d15d2e0e62b.

    ASCII letters, hyphens, spaces and straight apostrophes survive filtering;
    other characters are deleted. This deliberately performs no correction.
    """
    s = re.sub(r"[^a-zA-Z\- ']", "", s)
    s = s.replace("- ", " ").replace("--", "").replace(" '", "'").lower()
    return " ".join(s.split())


def _score_row(row: dict) -> dict:
    ok = row.get("status", "ok") == "ok"
    phones_ok=row.get("phoneme_status",row.get("status","ok"))=="ok"
    return {
        "trial_id": row.get("trial_id"),
        "status": row.get("status", "ok"),
        **{key: row[key] for key in _TRACE_KEYS if key in row},
        "word": edit_counts(
            normalize_text(row["reference_text"]).split(),
            normalize_text(row.get("text", "") if ok else "").split(),
        ),
        "phoneme": edit_counts(
            row["reference_phoneme_ids"], row.get("phoneme_ids", []) if phones_ok else []
        ),
    }


def _aggregate(rows: list[dict], per_trial: list[dict]) -> dict:
    totals = {
        kind: {key: sum(row[kind][key] for row in per_trial) for key in _COUNT_KEYS}
        for kind in ("word", "phoneme")
    }
    count = len(rows)
    missing = sum(row.get("status") == "missing" for row in rows)
    failed = sum(row.get("status", "ok") != "ok" for row in rows)
    return {
        **totals,
        "wer": totals["word"]["errors"] / totals["word"]["reference_length"]
        if totals["word"]["reference_length"] else None,
        "per": totals["phoneme"]["errors"] / totals["phoneme"]["reference_length"]
        if totals["phoneme"]["reference_length"] else None,
        "evaluated_count": count,
        "failed_count": failed,
        "successful_count": count - failed,
        "missing_count": missing,
        "skipped_count": 0,
        "prediction_coverage": (count - missing) / count if count else None,
    }


def corpus_metrics(rows: Iterable[dict]) -> dict:
    """Sum word/phone edits before division; non-ok rows are empty failures.

    Rows contain reference_text, reference_phoneme_ids, text, phoneme_ids and
    status (default ok). Empty references contribute insertions, but no
    denominator; a zero total denominator yields None, including an empty run.
    """
    rows = list(rows)
    return _aggregate(rows, [_score_row(row) for row in rows])


def _by_id(rows: Iterable[dict], kind: str) -> dict:
    result = {}
    for row in rows:
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError(f"{kind} requires a nonempty string trial_id")
        if trial_id in result:
            raise ValueError(f"Duplicate {kind} trial_id: {trial_id}")
        result[trial_id] = row
    return result


def _selection(records: dict, manifest: dict, score_split: str) -> list[dict]:
    if score_split not in {"dev_tune", "dev_report", "official_val"}:
        raise ValueError("score_split must be dev_tune, dev_report, or official_val")
    groups = {}
    for name in ("dev_tune", "dev_report"):
        members = manifest.get(name)
        if not isinstance(members, list) or any(not isinstance(member, str) for member in members):
            raise ValueError(f"Split manifest requires a list of IDs for {name}")
        if len(members) != len(set(members)):
            raise ValueError(f"Duplicate IDs in {name}")
        groups[name] = set(members)
    if groups["dev_tune"] & groups["dev_report"]:
        raise ValueError("dev_tune and dev_report must be disjoint")
    partition = groups["dev_tune"] | groups["dev_report"]
    unknown = partition - records.keys()
    if unknown:
        raise ValueError(f"Unknown selected records: {sorted(unknown)}")
    validation = {trial_id for trial_id, row in records.items() if row.get("split") == "val"}
    if partition != validation:
        raise ValueError("Split memberships must partition the available official val records")
    if not manifest.get("manifest_hash"):
        raise ValueError("Split manifest requires manifest_hash")
    selected = partition if score_split == "official_val" else groups[score_split]
    return sorted((records[trial_id] for trial_id in selected), key=_official_order)


def _official_order(record: dict) -> int:
    try:
        order = index(record["official_order_index"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Records require an integer official_order_index") from exc
    if order < 0:
        raise ValueError("official_order_index must be nonnegative")
    return order


def _validate_freeze(freeze: dict | None, expected: dict | None, split_hash: str) -> None:
    if not freeze:
        raise ValueError("A joint freeze manifest is required for reporting")
    studies = freeze.get("studies", {})
    if "session_calibration" not in studies:
        raise ValueError("Freeze studies must include session_calibration")
    if "confidence_lm" not in studies:
        if "paired_dropout" not in studies:
            raise ValueError("Freeze studies must include confidence_lm or paired_dropout")
        if not isinstance(freeze.get("replacement_reason"), str) or not freeze["replacement_reason"].strip():
            raise ValueError("paired_dropout requires a documented replacement_reason")
    if not freeze.get("selection_rules") or not freeze.get("manifest_hash"):
        raise ValueError("Freeze manifest requires selection_rules and manifest_hash")
    if not expected:
        raise ValueError("expected_hashes from the current run are required for reporting")
    if freeze.get("split_hash") != split_hash or expected.get("split_hash") != split_hash:
        raise ValueError("Frozen split hash does not match the current split")
    if set(studies) != set(expected.get("studies", {})):
        raise ValueError("Frozen study hashes do not match the current studies")
    pairs = [("baseline", freeze.get("baseline", {}), expected.get("baseline", {}))]
    pairs.extend(
        (name, hashes, expected["studies"][name]) for name, hashes in studies.items()
    )
    for name, frozen, current in pairs:
        for key in _HASH_KEYS:
            if not isinstance(frozen.get(key), str) or not frozen[key]:
                raise ValueError(f"Freeze requires {name}.{key}")
            if frozen[key] != current.get(key):
                raise ValueError(f"Frozen hash mismatch: {name}.{key}")


def evaluate_predictions(
    predictions: list[dict], records: list[dict], split_manifest: dict,
    reference_loader: Callable[[dict], dict], score_split="dev_tune",
    freeze_manifest=None, expected_hashes=None,
) -> dict:
    """Validate selection and freeze before loading only selected references.

    Records are metadata with trial_id, split='val', official_order_index. The
    loader returns reference_text and phoneme_ids already sliced to true target
    length. Output rows and examples must be derived from scored_rows only.
    Cached predictions outside the selected IDs never enter scored artifacts.
    """
    record_map = _by_id(records, "record")
    prediction_map = _by_id(predictions, "prediction")
    unknown = prediction_map.keys() - record_map.keys()
    if unknown:
        raise ValueError(f"Unknown prediction IDs: {sorted(unknown)}")
    selected = _selection(record_map, split_manifest, score_split)
    if len({_official_order(record) for record in selected}) != len(selected):
        raise ValueError("Selected records have duplicate official_order_index values")
    if score_split != "dev_tune":
        _validate_freeze(freeze_manifest, expected_hashes, split_manifest["manifest_hash"])
    trace = {
        "score_split": score_split,
        "split_manifest_hash": split_manifest["manifest_hash"],
        "freeze_manifest_hash": freeze_manifest.get("manifest_hash")
        if freeze_manifest and score_split != "dev_tune" else None,
    }
    rows = []
    for record in selected:
        trial_id = record["trial_id"]
        prediction = prediction_map.get(trial_id)
        status = prediction.get("status", "ok") if prediction is not None else "missing"
        phoneme_status=prediction.get("phoneme_status",status) if prediction is not None else "missing"
        reference = reference_loader(record)
        if reference.get("reference_text") is None or reference.get("phoneme_ids") is None:
            raise ValueError(f"Missing reference labels for selected trial {trial_id}")
        rows.append({
            **record,
            **trace,
            "reference_text": reference["reference_text"],
            "reference_phoneme_ids": list(reference["phoneme_ids"]),
            "text": prediction.get("text", "") if status == "ok" else "",
            "phoneme_ids": list(prediction.get("phoneme_ids", [])) if phoneme_status == "ok" else [],
            "status": status,
            "phoneme_status":phoneme_status,
        })
    per_trial = [_score_row(row) for row in rows]
    metrics = {
        **_aggregate(rows, per_trial),
        **trace,
    }
    return {"metrics": metrics, "scored_rows": rows, "per_trial": per_trial}


def export_submission(predictions: list[dict], records: list[dict], output_path) -> Path:
    """Write complete test-only predictions in captured traversal order as id,text."""
    record_map = _by_id(records, "record")
    tests = [record for record in record_map.values() if record.get("split") == "test"]
    prediction_map = _by_id(predictions, "prediction")
    if not tests or set(prediction_map) != {record["trial_id"] for record in tests}:
        raise ValueError("Submission predictions must cover every test ID exactly once")
    tests.sort(key=_official_order)
    if [_official_order(record) for record in tests] != list(range(len(tests))):
        raise ValueError("Test official_order_index values must be complete, unique, and sequential")
    for prediction in prediction_map.values():
        if prediction.get("status", "ok") != "ok" or not isinstance(prediction.get("text"), str):
            raise ValueError("Submission cannot contain failed or missing text predictions")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "text"])
        for submission_id, record in enumerate(tests):
            writer.writerow([submission_id, prediction_map[record["trial_id"]]["text"]])
    return output_path
