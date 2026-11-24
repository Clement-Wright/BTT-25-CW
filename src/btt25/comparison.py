"""Paired corpus error comparisons from already-scored run artifacts."""

import json
from operator import index
from pathlib import Path

import numpy as np

from .evaluation import normalize_text
from .reporting import _group_value, _rate, _validate_result, _write_csv, _write_json


def _load_run(directory):
    directory = Path(directory)
    result = {"metrics": json.loads((directory / "metrics.json").read_text(encoding="utf-8"))}
    for key, filename in (("scored_rows", "predictions.jsonl"), ("per_trial", "per_trial.jsonl")):
        result[key] = [json.loads(line) for line in (directory / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, edits = _validate_result(result)
    for trial_id, row in rows.items():
        expected_lengths = {"word": len(normalize_text(row["reference_text"]).split()), "phoneme": len(row["reference_phoneme_ids"])}
        for kind, length in expected_lengths.items():
            counts = edits[trial_id][kind]
            values = [counts[key] for key in ("substitutions", "deletions", "insertions", "reference_length", "errors")]
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
                raise ValueError("Per-trial edits must be nonnegative integer counts")
            if counts["reference_length"] != length:
                raise ValueError(f"Per-trial reference length mismatch for {trial_id}")
            if counts["errors"] != counts["substitutions"] + counts["deletions"] + counts["insertions"]:
                raise ValueError(f"Per-trial edit total mismatch for {trial_id}")
    resources_path = directory / "resources.json"
    resources = json.loads(resources_path.read_text(encoding="utf-8")) if resources_path.exists() else None
    return result["metrics"], rows, edits, resources


def _bootstrap(errors_a, errors_b, lengths, groups, rng, replicates):
    membership = {}
    for position, group in enumerate(groups):
        membership.setdefault(group, []).append(position)
    members = list(membership.values())
    a, b, n = [np.array([sum(values[positions]) for positions in members], dtype=np.int64) for values in (errors_a, errors_b, lengths)]
    changes = []
    for _ in range(replicates):
        selected = rng.integers(0, len(members), size=len(members))
        denominator = int(n[selected].sum())
        if denominator:
            changes.append(float(b[selected].sum() / denominator - a[selected].sum() / denominator))
    bounds = np.quantile(changes, [0.025, 0.975]).tolist() if changes else [None, None]
    return {"lower": bounds[0], "upper": bounds[1], "units": len(members), "valid_replicates": len(changes)}


def _metric(kind, trial_ids, rows, edits_a, edits_b, seed, replicates):
    errors_a, errors_b, lengths = [
        np.array([source[trial_id][kind][key] for trial_id in trial_ids], dtype=np.int64)
        for source, key in ((edits_a, "errors"), (edits_b, "errors"), (edits_a, "reference_length"))
    ]
    denominator = int(lengths.sum())
    rate_a = float(errors_a.sum() / denominator) if denominator else None
    rate_b = float(errors_b.sum() / denominator) if denominator else None
    sessions = [_group_value(rows[trial_id], "session_id") for trial_id in trial_ids]
    blocks = [(sessions[i], _group_value(rows[trial_id], "block_id")) for i, trial_id in enumerate(trial_ids)]
    rng = np.random.default_rng(seed)
    intervals = {
        name: _bootstrap(errors_a, errors_b, lengths, groups, rng, replicates)
        for name, groups in (("bootstrap", trial_ids), ("session_cluster_bootstrap", sessions), ("session_block_cluster_bootstrap", blocks))
    }
    rescued, harmed = int((errors_b < errors_a).sum()), int((errors_b > errors_a).sum())
    return {
        "measured": True, "errors_a": int(errors_a.sum()), "errors_b": int(errors_b.sum()),
        "reference_length": denominator, "rate_a": rate_a, "rate_b": rate_b,
        "delta_b_minus_a": rate_b - rate_a if denominator else None,
        **intervals, "rescue_count": rescued, "harm_count": harmed,
        "unchanged_count": len(trial_ids) - rescued - harmed,
        "rescue_rate": rescued / len(trial_ids), "harm_rate": harmed / len(trial_ids),
    }


def _comparison_report(result):
    lines = [
        "# Paired comparison",
        f"Scoring selection: `{result['score_split']}`; matched trials: {result['trial_count']}.",
        "Changes are run B minus run A; negative values mean fewer errors. Intervals recompute corpus ratios from paired resamples.",
    ]
    for name in ("per", "wer"):
        metric = result[name]
        if not metric["measured"]:
            lines.append(f"{name.upper()}: not measured. {metric['reason']}")
            continue
        change = metric["delta_b_minus_a"]
        lines.append(f"{name.upper()}: A {_rate(metric['rate_a'])}; B {_rate(metric['rate_b'])}; change " + (f"{change * 100:+.2f} percentage points." if change is not None else "undefined."))
        for key, label in (("bootstrap", "Paired trial"), ("session_cluster_bootstrap", "Session cluster"), ("session_block_cluster_bootstrap", "Session/block cluster")):
            interval = metric[key]
            bounds = "undefined" if interval["lower"] is None else f"[{interval['lower'] * 100:+.2f}, {interval['upper'] * 100:+.2f}] percentage points"
            lines.append(f"{label} 95% interval: {bounds}; {interval['units']} units; {interval['valid_replicates']} valid resamples.")
        lines.append(f"Trials with fewer edits: {metric['rescue_count']} ({metric['rescue_rate']:.2%}); more edits: {metric['harm_count']} ({metric['harm_rate']:.2%}); unchanged: {metric['unchanged_count']}.")
    lines.append("Rescue and harm mean any decrease or increase in a trial's edit count. Intervals describe resampling of this single-participant dataset; they do not establish generalization to other participants.")
    lines.append(f"Seed: {result['seed']}; resamples per interval: {result['replicates']}. Missing session IDs: {result['missing_session_count']}; missing block IDs: {result['missing_block_count']}. Missing group metadata is retained as unknown.")
    for name in ("a", "b"):
        resources = result["runs"][name]["resources"]
        if resources:
            summary = {key: resources[key] for key in ("elapsed_seconds", "peak_tree_rss_bytes", "peak_tree_private_bytes", "minimum_system_available_bytes", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "outcome") if key in resources}
            lines.append(f"Run {name.upper()} resources: " + json.dumps(summary, sort_keys=True))
    return "\n\n".join(lines) + "\n"


def compare_runs(run_a, run_b, output_dir, seed=20261116, replicates=2000) -> dict:
    """Compare matched run directories; negative deltas favor B.

    Bootstrap whole trials in pairs, then whole sessions and session/block
    clusters as sensitivity analyses. Ratios use summed errors and reference
    lengths in each resample. Empty-denominator resamples are excluded and
    counted. Rescue/harm rates use every matched trial, including failures.
    WER is unavailable when either run declares text_available=False.
    """
    replicates = index(replicates)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    metrics_a, rows_a, edits_a, resources_a = _load_run(run_a)
    metrics_b, rows_b, edits_b, resources_b = _load_run(run_b)
    if not rows_a or set(rows_a) != set(rows_b):
        raise ValueError("Paired comparison requires identical nonempty trial IDs")
    for key in ("score_split", "split_manifest_hash", "freeze_manifest_hash"):
        if metrics_a.get(key) != metrics_b.get(key):
            raise ValueError(f"Paired comparison requires matching split/freeze identity: {key}")
    trial_ids = sorted(rows_a)
    for trial_id in trial_ids:
        for field in ("reference_text", "reference_phoneme_ids"):
            if rows_a[trial_id][field] != rows_b[trial_id][field]:
                raise ValueError(f"Paired reference mismatch: {trial_id}.{field}")
        for field in ("session_id", "block_id"):
            if _group_value(rows_a[trial_id], field) != _group_value(rows_b[trial_id], field):
                raise ValueError(f"Paired cluster metadata mismatch: {trial_id}.{field}")
    available = metrics_a.get("text_available", True) and metrics_b.get("text_available", True)
    result = {
        "score_split": metrics_a["score_split"], "split_manifest_hash": metrics_a["split_manifest_hash"],
        "freeze_manifest_hash": metrics_a.get("freeze_manifest_hash"),
        "trial_count": len(trial_ids), "seed": seed, "replicates": replicates,
        "confidence_level": 0.95, "direction": "run_b minus run_a; negative favors run_b",
        "missing_session_count": sum(_group_value(row, "session_id") == "unknown" for row in rows_a.values()),
        "missing_block_count": sum(_group_value(row, "block_id") == "unknown" for row in rows_a.values()),
        "runs": {
            "a": {"name": Path(run_a).name, "metadata": metrics_a.get("metadata", {}), "resources": resources_a},
            "b": {"name": Path(run_b).name, "metadata": metrics_b.get("metadata", {}), "resources": resources_b},
        },
        "per": _metric("phoneme", trial_ids, rows_a, edits_a, edits_b, seed, replicates),
        "wer": _metric("word", trial_ids, rows_a, edits_a, edits_b, seed, replicates)
        if available else {"measured": False, "reason": "At least one run produces phonemes only."},
    }
    paired = []
    for trial_id in trial_ids:
        row = {"trial_id": trial_id, "session_id": _group_value(rows_a[trial_id], "session_id"), "block_id": _group_value(rows_a[trial_id], "block_id")}
        for kind, name in (("phoneme", "per"), ("word", "wer")):
            a, b = edits_a[trial_id][kind], edits_b[trial_id][kind]
            row[f"{name}_reference_length"] = a["reference_length"]
            measured = name == "per" or available
            row.update({f"{name}_errors_a": a["errors"] if measured else None, f"{name}_errors_b": b["errors"] if measured else None, f"{name}_error_change": b["errors"] - a["errors"] if measured else None})
        paired.append(row)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "comparison.json", result)
    _write_csv(directory / "paired_trials.csv", paired, list(paired[0]))
    (directory / "report.md").write_text(_comparison_report(result), encoding="utf-8")
    return result
