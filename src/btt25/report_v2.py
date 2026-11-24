"""Separate v2 freeze and exploratory scoring on the exposed benchmark."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from .assets import sha256
from .comparison import compare_runs
from .data import attach_corpus_metadata, load_split_manifest, object_hash, reference_only, write_json
from .evaluation import _aggregate, _by_id, _score_row, _selection
from .inference import load_config, load_json, read_jsonl
from .reporting import write_evaluation
from .resources import ResourceMonitor


STATUS = "exploratory_after_v1_report_exposure"
V1_FREEZE = "c7b12685c9552344144b0f089a0a3c4d7dc6bec88e6df513e52947e926ba7b56"
REQUIRED = ("predictions.jsonl", "candidates.jsonl", "cache_manifest.json", "run_manifest.json",
            "config.resolved.yaml", "resources.json")


def data(config):
    records = load_json(config["data_manifest"])
    split = load_split_manifest(config["split_manifest"], records)
    if config.get("corpus_metadata"):
        records = attach_corpus_metadata(records, config["corpus_metadata"])
    return records, split


def hash_bundle(directory, names):
    return {name: sha256(directory / name) for name in names}


def verify_run(directory, expected, records, data_hash, split_hash):
    if expected is not None and hash_bundle(directory, REQUIRED) != expected:
        raise ValueError("Frozen scientific run changed")
    manifest = load_json(directory / "run_manifest.json")
    cache = load_json(directory / "cache_manifest.json")
    body = {key: value for key, value in cache.items() if key != "manifest_hash"}
    if object_hash(body) != cache["manifest_hash"]:
        raise ValueError("Candidate cache content hash is invalid")
    resources = load_json(directory / "resources.json")
    if (resources["outcome"] != "complete" or resources.get("stop_reason")
            or max(resources["peak_accounted_host_bytes"], resources.get("peak_cgroup_bytes") or 0)
            > resources["host_ceiling_bytes"]):
        raise ValueError("Decoder resources do not qualify as complete")
    if (cache["data_manifest_sha256"] != data_hash
            or cache["source_cache_manifest"]["split_hash"] != split_hash):
        raise ValueError("Decoder cache belongs to a different data or split manifest")
    rows = read_jsonl(directory / "predictions.jsonl")
    if (manifest["state"] != "complete" or cache["references_loaded"] is not False
            or manifest["saved_trials"] != len(records) or len(rows) != len(records)
            or [r["trial_id"] for r in rows] != [r["trial_id"] for r in records]):
        raise ValueError("Full reference-free validation predictions must match official traversal")
    if any(r["decoder_cache_hash"] != cache["manifest_hash"] for r in rows):
        raise ValueError("Decoder cache stamp differs from predictions")
    return rows


def validate_provenance(tune, decode, split_hash, arm, registration=None):
    provenance = load_json(tune / "provenance.json")["source_sha256"]
    metrics = load_json(tune / "metrics.json")
    config = load_json(tune / "config.resolved.yaml")
    resources = load_json(tune / "resources.json")
    if (metrics["score_split"] != "dev_tune" or metrics["split_hash"] != split_hash
            or config["condition"]["id"] != arm
            or (registration is not None and config["registration"] != load_json(registration))
            or resources["outcome"] != "complete" or resources.get("stop_reason")
            or provenance["predictions"] != sha256(decode / "predictions.jsonl")
            or provenance["candidates"] != sha256(decode / "candidates.jsonl")):
        raise ValueError("Condition lacks paired tuning evidence for its own predictions")
    return metrics


def registration_hashes(args):
    return [sha256(path) for path in (args.registration, *args.extra_registration)]


def evaluator_hashes():
    module = Path(__file__)
    names = (module.name, "data.py", "evaluation.py", "comparison.py", "reporting.py")
    return {name: sha256(module.with_name(name)) for name in names}


def seal(args):
    config = load_config(args.config)
    records, split = data(config)
    validation = _selection(_by_id(records, "metadata"), split, "official_val")
    registration = load_json(args.registration)
    labels = {arm["id"]: args.registration for arm in registration["conditions"]}
    for path in args.extra_registration:
        extra = load_json(path)
        if (extra["reference_access"] != "dev_tune"
                or extra.get("split_manifest_hash", split["manifest_hash"]) != split["manifest_hash"]):
            raise ValueError("Extra condition registration must follow the same reference policy")
        for arm in extra["conditions"]:
            name = arm["id"] if isinstance(arm, dict) else arm
            if name in labels:
                raise ValueError(f"Condition preregistered more than once: {name}")
            labels[name] = path
    if (registration["split_manifest_hash"] != split["manifest_hash"]
            or registration["archived_v1_freeze"] != V1_FREEZE):
        raise ValueError("New freeze must bind the registered split and archived original study")
    conditions = {}
    ranking = []
    data_hash = sha256(config["data_manifest"])
    for spec in args.run:
        name, raw_path = spec.split("=", 1)
        if not re.fullmatch(r"v2-[a-z0-9-]+", name) or name in conditions or name not in labels:
            raise ValueError("Each new condition needs a registered unique ID")
        directory = Path(raw_path).resolve()
        decode = directory / "decode"
        tune = directory / "dev-tune"
        verify_run(decode, None, validation, data_hash, split["manifest_hash"])
        tuning = validate_provenance(tune, decode, split["manifest_hash"], name, labels[name])
        if tuning["trials"] != len(split["dev_tune"]):
            raise ValueError("Freeze comparison requires the full tuning partition")
        decoder = load_json(decode / "cache_manifest.json")["decoder"]
        runtime = load_json(decode / "run_manifest.json")
        ranking.append((tuning["selected_errors"], tuning["oracle"]["50"]["errors"],
                        runtime["p50_trial_seconds"], decoder["beam"], name))
        conditions[name] = {"directory": str(directory), "decode": hash_bundle(decode, REQUIRED),
                            "tune": hash_bundle(tune, ("metrics.json", "config.resolved.yaml", "provenance.json", "resources.json"))}
    if len(conditions) < 2 or args.control not in conditions:
        raise ValueError("Freeze needs at least two completed conditions and an explicit control")
    additional_pairs = (("v2-e2-monophone", "v2-e3-diphone"), ("v2-f1-distilgpt2", "v2-f2-qwen17b"),
                        ("v2-s6-repaired-250", "v2-s8-repaired-aac-250"), ("v2-f2-qwen17b", "v2-f3-qwen4b"),
                        *((f"v2-s{i}-original-{beam}", f"v2-s{i+3}-repaired-{beam}")
                          for i, beam in enumerate((50, 100, 250), 1)))
    pairs = [(args.control, name) for name in sorted(conditions) if name != args.control]
    pairs.extend(pair for pair in additional_pairs if all(name in conditions for name in pair) and pair not in pairs)
    archive = {"data_manifest_sha256": data_hash,
               "corpus_metadata_sha256": sha256(config["corpus_metadata"]) if config.get("corpus_metadata") else None,
               "registrations_sha256": registration_hashes(args), "evaluation_code_sha256": evaluator_hashes(),
               "split_hash": split["manifest_hash"], "evaluation_status": STATUS,
               "v1_freeze": V1_FREEZE, "baseline_condition": args.control,
               "selected_condition": min(ranking)[-1], "selection_rule": registration["selection"],
               "conditions": conditions, "paired_conditions": pairs}
    archive["manifest_hash"] = object_hash(archive)
    args.freeze.parent.mkdir(parents=True, exist_ok=True)
    with args.freeze.open("x", encoding="utf-8") as stream:
        json.dump(archive, stream, indent=2, sort_keys=True)
        stream.write("\n")


def report(args):
    config = load_config(args.config)
    records, split = data(config)
    metadata = _by_id(records, "record")
    validation = _selection(metadata, split, "official_val")
    selected = _selection(metadata, split, args.score_split)
    freeze = load_json(args.freeze)
    digest = freeze.pop("manifest_hash")
    if (digest != object_hash(freeze) or freeze["split_hash"] != split["manifest_hash"]
            or freeze["data_manifest_sha256"] != sha256(config["data_manifest"])
            or freeze["corpus_metadata_sha256"] != (sha256(config["corpus_metadata"])
                if config.get("corpus_metadata") else None)
            or freeze["registrations_sha256"] != registration_hashes(args)
            or freeze["evaluation_code_sha256"] != evaluator_hashes() or freeze["v1_freeze"] != V1_FREEZE
            or freeze["evaluation_status"] != STATUS):
        raise ValueError("Invalid or mismatched exploratory evaluation freeze")
    predictions = {}
    for name, bundle in freeze["conditions"].items():
        directory = Path(bundle["directory"])
        if hash_bundle(directory / "dev-tune", tuple(bundle["tune"])) != bundle["tune"]:
            raise ValueError("Frozen tuning evidence changed")
        predictions[name] = _by_id(verify_run(directory / "decode", bundle["decode"], validation,
                                            freeze["data_manifest_sha256"], split["manifest_hash"]), "prediction")
    # Report labels are loaded only after all condition files have passed checks.
    args.output.mkdir(parents=True, exist_ok=False)
    trace = {"score_split": args.score_split, "split_manifest_hash": split["manifest_hash"],
             "freeze_manifest_hash": digest, "evaluation_status": STATUS}
    write_json(args.output / "config.resolved.yaml", {**trace, "config": config})
    with ResourceMonitor(args.output / "resources.json", max_seconds=600) as monitor:
        labels = {}
        for row in selected:
            monitor.check()
            reference = reference_only(row)
            labels[row["trial_id"]] = {"reference_text": reference["reference_text"],
                                       "reference_phoneme_ids": reference["phoneme_ids"]}
        summary = {}
        by_condition = {}
        for name, by_id in predictions.items():
            rows = []
            for record in selected:
                trial = record["trial_id"]
                pred = by_id[trial]
                rows.append({**record, **trace, **labels[trial], "text": pred["text"],
                             "phoneme_ids": pred["phoneme_ids"], "status": pred["status"],
                             "phoneme_status": pred["phoneme_status"]})
            edits = list(map(_score_row, rows))
            metrics = {**_aggregate(rows, edits), **trace}
            write_evaluation({"metrics": metrics, "scored_rows": rows, "per_trial": edits},
                             args.output / name, {"condition": name, "evaluation_status": STATUS})
            summary[name] = metrics
            by_condition[name] = {row["trial_id"]: row["word"] for row in edits}
        control = freeze["baseline_condition"]
        comparisons = {}
        corpus_changes = {}
        for a_name, name in freeze["paired_conditions"]:
            monitor.check()
            comparison_name = name if a_name == control else f"{name}-minus-{a_name}"
            result = compare_runs(args.output / a_name, args.output / name,
                                  args.output / f"paired-{comparison_name}", seed=20261124)
            comparisons[comparison_name] = {**result["wer"], "control": a_name, "condition": name}
            group = defaultdict(lambda: {"trials": 0, "control_errors": 0, "condition_errors": 0,
                                         "reference_words": 0, "rescued": 0, "harmed": 0})
            for record in selected:
                trial = record["trial_id"]
                a, b = by_condition[a_name][trial], by_condition[name][trial]
                counts = group[record.get("corpus_id") or "unknown"]
                counts["trials"] += 1
                counts["control_errors"] += a["errors"]
                counts["condition_errors"] += b["errors"]
                counts["reference_words"] += a["reference_length"]
                counts["rescued"] += int(b["errors"] < a["errors"])
                counts["harmed"] += int(b["errors"] > a["errors"])
            corpus_changes[comparison_name] = group
        write_json(args.output / "metrics.json", {"selected_condition": freeze["selected_condition"],
                   "conditions": summary, "paired_wer": comparisons, "paired_by_corpus": corpus_changes, **trace})
    (args.output / "report.md").write_text(
        f"# Exploratory v2 comparison\n\nSelection `{args.score_split}`. The v1 report informed this work. "
        "The v2 manifest verifies the exact search artifacts used for the comparison.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_cli = commands.add_parser("seal")
    freeze_cli.add_argument("--run", action="append", required=True)
    freeze_cli.add_argument("--control", required=True)
    report_cli = commands.add_parser("report")
    report_cli.add_argument("--output", type=Path, required=True)
    report_cli.add_argument("--score-split", choices=("dev_report", "official_val"), required=True)
    for name in ("config", "freeze", "registration"):
        for command in (freeze_cli, report_cli):
            command.add_argument("--" + name, type=Path, required=True)
    for command in (freeze_cli, report_cli):
        command.add_argument("--extra-registration", action="append", type=Path, default=[])
    args = parser.parse_args()
    (seal if args.command == "seal" else report)(args)


if __name__ == "__main__":
    main()
