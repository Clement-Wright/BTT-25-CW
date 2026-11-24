"""Write selection-scoped artifacts without opening reference data."""

import csv
import json
from pathlib import Path

from .evaluation import corpus_metrics, normalize_text


_TRACE = ("score_split", "split_manifest_hash", "freeze_manifest_hash")
_COUNTS = ("substitutions", "deletions", "insertions", "reference_length", "errors")
_BINS = ("0", "1-5", "6-10", "11-20", "21+")


def _unique(rows):
    result = {}
    for row in rows:
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id or trial_id in result:
            raise ValueError("Scored artifacts require unique nonempty trial IDs")
        result[trial_id] = row
    return result


def _validate_result(result):
    metrics = result["metrics"]
    selection = metrics.get("score_split")
    if selection not in {"dev_tune", "dev_report", "official_val"}:
        raise ValueError("Scored artifacts require an explicit score_split")
    if not metrics.get("split_manifest_hash"):
        raise ValueError("Scored artifacts require split_manifest_hash")
    if selection != "dev_tune" and not metrics.get("freeze_manifest_hash"):
        raise ValueError("Reporting artifacts require a recorded freeze manifest hash")
    rows, edits = _unique(result["scored_rows"]), _unique(result["per_trial"])
    if set(rows) != set(edits):
        raise ValueError("Scored rows and per-trial edits must cover identical trial IDs")
    for row in [*rows.values(), *edits.values()]:
        if any(row.get(key) != metrics.get(key) for key in _TRACE):
            raise ValueError("Scored row split/freeze trace differs from metrics")
    return rows, edits


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_value(row, field):
    value = row.get(field)
    if field == "corpus_id" and value in (None, ""):
        value = row.get("corpus")
    return "unknown" if value in (None, "") else str(value)


def _length_bin(row):
    length = len(normalize_text(row["reference_text"]).split())
    return "0" if length == 0 else "1-5" if length <= 5 else "6-10" if length <= 10 else "11-20" if length <= 20 else "21+"


def _subgroups(rows, field, text_available):
    groups = {}
    for row in rows:
        group = _length_bin(row) if field == "length_bin" else _group_value(row, field)
        groups.setdefault(group, []).append(row)
    ordered = [key for key in _BINS if key in groups] if field == "length_bin" else sorted(groups)
    output = []
    for group in ordered:
        metrics = corpus_metrics(groups[group])
        output.append({
            field: group,
            **{key: metrics[key] for key in ("evaluated_count", "failed_count", "missing_count", "prediction_coverage", "per")},
            "wer": metrics["wer"] if text_available else None,
            **{f"{kind}_{key}": metrics[kind][key] for kind in ("word", "phoneme") for key in _COUNTS},
        })
    fields = [field, "evaluated_count", "failed_count", "missing_count", "prediction_coverage", "per", "wer"]
    fields += [f"{kind}_{key}" for kind in ("word", "phoneme") for key in _COUNTS]
    return output, fields


def _rate(value):
    return "undefined (zero reference tokens)" if value is None else f"{value:.2%}"


def _examples(rows, edits, text_available):
    if not rows:
        return []
    kind = "word" if text_available else "phoneme"

    def rank(trial_id):
        counts = edits[trial_id][kind]
        length = counts["reference_length"]
        return (length == 0, counts["errors"] / length if length else counts["errors"], trial_id)

    ordered = sorted(rows, key=rank)
    result = []
    for label, position in (("Best", 0), ("Median", len(ordered) // 2), ("Worst", -1)):
        row = rows[ordered[position]]
        lines = [f"{label} example: `{row['trial_id']}`", f"Status: {row.get('status', 'ok')}"]
        lines.append("Reference: " + json.dumps(row["reference_text"], ensure_ascii=False))
        if text_available:
            lines.append("Prediction: " + json.dumps(row.get("text", ""), ensure_ascii=False))
        else:
            lines.append(f"Reference phones: {row['reference_phoneme_ids']}")
            lines.append(f"Predicted phones: {row.get('phoneme_ids', [])}")
        result.append("\n\n".join(lines))
    return result


def write_evaluation(result, output_dir, metadata=None) -> Path:
    """Save metrics, selection-only rows, subgroups, and deterministic examples.

    No loader is accepted: examples can use only rows already authorized and
    scored by evaluate_predictions. Reference-word bins are fixed at 0, 1–5,
    6–10, 11–20, 21+. Example rank uses WER (PER for phoneme-only runs), with
    trial-ID ties; undefined rates sort last. source_file is omitted on export.
    metadata is JSON-compatible provenance. text_available=False suppresses WER
    rates and word comparisons while retaining raw edit counts for audit.
    """
    rows, edits = _validate_result(result)
    provenance = {**result["metrics"].get("metadata", {}), **(metadata or {})}
    available = provenance.get("text_available", result["metrics"].get("text_available", True))
    if not isinstance(available, bool):
        raise ValueError("text_available must be boolean")
    provenance["text_available"] = available
    metrics = {**result["metrics"], "metadata": provenance, "text_available": available}
    if not available:
        metrics["wer"] = None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "metrics.json", metrics)
    exported = [{key: value for key, value in row.items() if key != "source_file"} for row in result["scored_rows"]]
    for name, values in (("predictions.jsonl", exported), ("per_trial.jsonl", result["per_trial"])):
        (directory / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in values), encoding="utf-8"
        )
    for field, filename in (("session_id", "by_session.csv"), ("corpus_id", "by_corpus.csv"), ("length_bin", "length_bins.csv")):
        grouped, fields = _subgroups(list(rows.values()), field, available)
        _write_csv(directory / filename, grouped, fields)
    lines = [
        "# Evaluation",
        f"Scoring selection: `{metrics['score_split']}`.",
        f"Split manifest: `{metrics['split_manifest_hash']}`. Freeze manifest: `{metrics.get('freeze_manifest_hash') or 'not required'}`.",
        f"Trials: {metrics['evaluated_count']}; failed: {metrics['failed_count']}; missing: {metrics['missing_count']}.",
        f"PER: {_rate(metrics['per'])}",
        f"WER: {_rate(metrics['wer'])}" if available else "WER: not measured. This run produces phonemes only.",
        "Error rates divide summed edit counts by summed reference lengths. Failed and missing predictions remain in the denominators.",
    ]
    if provenance:
        lines += ["## Provenance", "```json\n" + json.dumps(provenance, indent=2, ensure_ascii=False) + "\n```"]
    lines += ["## Examples", "Examples rank by " + ("WER" if available else "PER") + ", with trial ID breaking ties. Undefined rates sort last. Failures remain eligible."]
    lines += _examples(rows, edits, available)
    (directory / "report.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return directory
