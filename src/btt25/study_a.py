"""Frozen confidence-gated likelihood rescoring of one compact candidate cache.

Tuning reads dev_tune references only. Reporting checks the joint freeze before
label access, then actually scores each policy's selected candidates. Saved
tuning likelihoods are preparation artifacts and never deployment timings.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import median
from time import perf_counter

import numpy as np
import yaml

from .assets import sha256
from .data import attach_corpus_metadata, object_hash, reference_only, write_json
from .evaluation import _by_id, _selection, edit_counts, evaluate_predictions, normalize_text
from .reporting import _group_value, write_evaluation
from .rescoring import DistilGPT2Scorer, _verify_assets, cheap_features
from .resources import ResourceMonitor


LAMBDA_GRID = [0.0, 0.1, 0.3, 1.0]
RANDOM_SEEDS = [20261116, 20261117, 20261118]
_LM_FIELDS = {"lm_log_likelihood", "lm_token_count", "lm_elapsed_seconds"}


def _checked(value):
    value = copy.deepcopy(value) if isinstance(value, dict) else json.loads(Path(value).read_text(encoding="utf-8"))
    if "manifest_hash" in value and value["manifest_hash"] != object_hash({key: item for key, item in value.items() if key != "manifest_hash"}):
        raise ValueError("Manifest content hash mismatch")
    return value


def _config(config):
    result = copy.deepcopy(config) if isinstance(config, dict) else yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if list(result.get("neural_lambda_grid", LAMBDA_GRID)) != LAMBDA_GRID:
        raise ValueError("Experiment A requires the predeclared lambda grid")
    if result.get("call_fraction", 0.5) != 0.5 or list(result.get("random_seeds", RANDOM_SEEDS)) != RANDOM_SEEDS:
        raise ValueError("Experiment A requires fraction 0.5 and the three predeclared seeds")
    return result


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _hashed(value):
    return dict(value, manifest_hash=object_hash(value))


def compact_candidates(candidates):
    """Copy native decoder choices and expose their frozen rank score alias."""
    result = copy.deepcopy(candidates)
    for candidate in result:
        score = candidate.get("score", candidate.get("compact_score", candidate.get("component_scores", {}).get("rank_score")))
        if score is None or not math.isfinite(float(score)) or not isinstance(candidate.get("text"), str):
            raise ValueError("Candidate requires exact text and a finite compact rank score")
        if "rank_score" in candidate.get("component_scores", {}) and float(score) != float(candidate["component_scores"]["rank_score"]):
            raise ValueError("Candidate compact score aliases disagree")
        candidate["score"] = float(score)
    return result


def grouped_gate_split(records, tune_ids, seed=20261116):
    """Deterministically divide whole session/block groups before label access."""
    record_map = _by_id(records, "record")
    if len(tune_ids) != len(set(tune_ids)) or set(tune_ids) - record_map.keys():
        raise ValueError("Gate memberships require unique known tuning IDs")
    groups = {}
    for trial_id in tune_ids:
        row = record_map[trial_id]
        if row.get("session_id") is None or row.get("block_id") is None:
            raise ValueError("Grouped gate tuning requires session_id and block_id metadata")
        groups.setdefault((str(row["session_id"]), str(row["block_id"])), []).append(trial_id)
    if len(groups) < 2:
        raise ValueError("Gate fitting and selection require at least two session/block groups")
    ordered = sorted(groups, key=lambda group: (hashlib.sha256(json.dumps([seed, *group]).encode()).hexdigest(), group))
    fit_groups = set(ordered[: (len(ordered) + 1) // 2])
    return {
        "gate_fit": sorted(trial_id for group, ids in groups.items() if group in fit_groups for trial_id in ids),
        "gate_selection": sorted(trial_id for group, ids in groups.items() if group not in fit_groups for trial_id in ids),
    }


def candidate_length_bin(word_count):
    return "1-5" if word_count <= 5 else "6-10" if word_count <= 10 else "11+"


def length_matched_random(features, gated_ids, seed):
    """Match gated call counts within reference-free compact word-count bins.

    Both policies are eligible only when at least two candidates exist. No
    margin, reference, candidate likelihood, or error enters random selection.
    """
    feature_map = _by_id(features, "feature")
    if len(gated_ids) != len(set(gated_ids)) or set(gated_ids) - feature_map.keys():
        raise ValueError("Random matching requires unique known gated IDs")
    rng, selected = random.Random(seed), []
    for name in ("1-5", "6-10", "11+"):
        eligible = sorted(trial_id for trial_id, row in feature_map.items() if row["candidate_count"] >= 2 and candidate_length_bin(row["word_count"]) == name)
        count = sum(candidate_length_bin(feature_map[trial_id]["word_count"]) == name for trial_id in gated_ids)
        if count > len(eligible):
            raise ValueError("Insufficient eligible candidates for length-matched random calls")
        selected.extend(rng.sample(eligible, count))
    return sorted(selected)


def _context(config, candidate_run, scorer_dir):
    config = _config(config)
    records = json.loads(Path(config["data_manifest"]).read_text(encoding="utf-8"))
    if config.get("corpus_metadata"):
        records = attach_corpus_metadata(records, config["corpus_metadata"])
    record_map = _by_id(records, "record")
    split = _checked(config["split_manifest"])
    _selection(record_map, split, "dev_tune")
    source = Path(candidate_run)
    cache, state = _checked(source / "cache_manifest.json"), _checked(source / "run_manifest.json")
    if state.get("state") != "complete" or cache.get("references_loaded") is not False or state.get("references_loaded") is not False:
        raise ValueError("Experiment A requires a complete reference-free compact candidate cache")
    data_hash = sha256(config["data_manifest"])
    if cache.get("data_manifest_sha256") != data_hash:
        raise ValueError("Candidate data manifest hash changed")
    neural_cache = cache.get("source_cache_manifest", {})
    raw_records = json.loads(Path(config["data_manifest"]).read_text(encoding="utf-8"))
    if (neural_cache.get("checkpoint_hash") != config.get("checkpoint_hash")
            or neural_cache.get("split_hash") != split["manifest_hash"]
            or neural_cache.get("manifest_hash") != object_hash(raw_records)
            or neural_cache.get("references_loaded") is not False
            or neural_cache.get("partition") != "val" or neural_cache.get("dtype") != "float32"):
        raise ValueError("Candidate source encoder checkpoint/split provenance differs")
    packets = _by_id(_rows(source / "candidates.jsonl"), "candidate packet")
    baseline = _by_id(_rows(source / "predictions.jsonl"), "compact prediction")
    val_ids = {trial_id for trial_id, row in record_map.items() if row.get("split") == "val"}
    if set(packets) != val_ids or set(baseline) != val_ids:
        raise ValueError("Frozen candidates and predictions must cover every validation ID exactly once")
    if neural_cache.get("expected_trials") != len(val_ids):
        raise ValueError("Candidate source encoder trial count differs")
    features, choices = {}, {}
    forbidden = {"reference_text", "reference_phoneme_ids", "true_sentence", "oracle_wer"}
    for trial_id in sorted(val_ids):
        packet, prediction = packets[trial_id], baseline[trial_id]
        if forbidden & (packet.keys() | prediction.keys()):
            raise ValueError("Reference fields are forbidden in the compact candidate cache")
        for row in (packet, prediction):
            if row.get("decoder_cache_hash") != cache["manifest_hash"] or row.get("input_id", trial_id) != trial_id or row.get("output_id", trial_id) != trial_id:
                raise ValueError("Candidate/compact cache identity hash mismatch")
        if packet.get("logit_sha256") != prediction.get("logit_sha256"):
            raise ValueError("Candidate/compact logit hash mismatch")
        choices[trial_id] = compact_candidates(packet["candidates"])
        if prediction.get("status") == "ok":
            if not choices[trial_id] or prediction.get("text") != _best(choices[trial_id], 0)["text"]:
                raise ValueError("Compact baseline differs from its frozen candidates")
        features[trial_id] = dict(trial_id=trial_id, **cheap_features(choices[trial_id], prediction.get("input_length", record_map[trial_id].get("input_length", 0))))
    verified = _verify_assets(Path(scorer_dir))
    checkpoint_hash = sha256(config["checkpoint"])
    if config.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("Checkpoint hash changed")
    hashes = {
        "candidates_sha256": sha256(source / "candidates.jsonl"),
        "compact_predictions_sha256": sha256(source / "predictions.jsonl"),
        "decoder_cache_hash": cache["manifest_hash"], "data_manifest_sha256": data_hash,
        "split_hash": split["manifest_hash"], "checkpoint_hash": checkpoint_hash,
        "scorer_asset_manifest_sha256": verified["manifest_sha256"],
        "corpus_metadata_sha256": sha256(config["corpus_metadata"]) if config.get("corpus_metadata") else None,
    }
    return {"config": config, "records": records, "record_map": record_map, "split": split,
            "baseline": baseline, "choices": choices, "features": features, "source_hashes": hashes,
            "source_state": state,
            "ordered_ids": sorted(val_ids, key=lambda trial_id: record_map[trial_id]["official_order_index"])}


def _best(candidates, weight):
    return min(candidates, key=lambda row: (-(row["score"] + weight * row.get("lm_log_likelihood", 0.0)), row["text"]))


def _score(scorer, candidates):
    started = perf_counter()
    try:
        scored = scorer.score_candidates(copy.deepcopy(candidates))
    except (RuntimeError, ValueError) as error:
        return {"candidates": [], "elapsed_seconds": perf_counter() - started, "error": f"{type(error).__name__}: {error}", "candidate_count": len(candidates)}
    elapsed = perf_counter() - started
    if len(scored) != len(candidates):
        raise ValueError("Scorer changed the candidate count")
    for original, row in zip(candidates, scored, strict=True):
        if {key: row.get(key) for key in original} != original or set(row) - set(original) != _LM_FIELDS:
            raise ValueError("Scorer changed frozen candidates or generated new fields")
        if not math.isfinite(float(row["lm_log_likelihood"])) or row["lm_token_count"] < 1 or not math.isfinite(float(row["lm_elapsed_seconds"])) or row["lm_elapsed_seconds"] < 0:
            raise ValueError("Scorer returned invalid candidate likelihoods or timing")
    return {"candidates": scored, "elapsed_seconds": elapsed, "error": None, "candidate_count": len(candidates)}


def _prediction(baseline, scored, weight, called, condition):
    row = copy.deepcopy(baseline)
    row.update(condition=condition, lm_called=called, lm_lambda=weight)
    if called and scored.get("error"):
        # Phone inference remains valid; a word failure contributes deletions.
        row.update(text="", status="lm_error", word_status="lm_error", lm_error=scored["error"],
                   phoneme_status=baseline.get("phoneme_status", baseline.get("status", "ok")))
    elif called:
        selected = _best(scored["candidates"], weight)
        row.update(text=selected["text"], decoder_phoneme_ids=selected.get("phoneme_ids", []),
                   lm_log_likelihood=selected["lm_log_likelihood"], selected_candidate_rank=selected.get("rank"),
                   neural_rerank_score=selected["score"] + weight * selected["lm_log_likelihood"])
    return row


def _word_totals(references, predictions, ids):
    edits = [edit_counts(normalize_text(references[trial_id]["reference_text"]).split(),
                         normalize_text(predictions[trial_id].get("text", "") if predictions[trial_id].get("status") == "ok" else "").split()) for trial_id in ids]
    errors, length = sum(row["errors"] for row in edits), sum(row["reference_length"] for row in edits)
    return {"errors": errors, "reference_length": length, "wer": errors / length if length else None}


def _gated(features, threshold):
    return sorted(trial_id for trial_id, row in features.items() if threshold is not None and row["top_two_margin_per_word"] is not None and row["top_two_margin_per_word"] <= threshold)


def _decisions(context, threshold):
    gated = _gated(context["features"], threshold)
    controls = {str(seed): [] for seed in RANDOM_SEEDS}
    for offset, split in enumerate(("dev_tune", "dev_report")):
        ids=set(context["split"][split])
        rows=[context["features"][trial_id] for trial_id in sorted(ids)]
        selected=[trial_id for trial_id in gated if trial_id in ids]
        for seed in RANDOM_SEEDS:
            controls[str(seed)].extend(length_matched_random(rows, selected, seed + offset*1_000_000_000))
    rows = [dict(context["features"][trial_id], gated=trial_id in gated,
                 random_selected={seed: trial_id in ids for seed, ids in controls.items()}) for trial_id in context["ordered_ids"]]
    return gated, controls, rows


def _timings(calls, latencies=None):
    invoked = [row["elapsed_seconds"] for row in calls]
    return {"lm_calls": len(calls), "lm_candidates": sum(row["candidate_count"] for row in calls),
            "lm_failures": sum(row["error"] is not None for row in calls), "lm_total_seconds": sum(invoked),
            "called_latency_p50_seconds": float(np.quantile(invoked, 0.5)) if invoked else None,
            "called_latency_p95_seconds": float(np.quantile(invoked, 0.95)) if invoked else None,
            "trial_latency_p50_seconds": float(np.quantile(latencies, 0.5)) if latencies else None,
            "trial_latency_p95_seconds": float(np.quantile(latencies, 0.95)) if latencies else None}


def tune(config, candidate_run, output_dir, scorer_dir, device="cuda") -> dict:
    """Fit only lambda and a median-margin gate on grouped dev_tune memberships."""
    context = _context(config, candidate_run, scorer_dir)
    output = Path(output_dir)
    if (output / "selected_config.json").exists():
        raise ValueError("Selected configuration already exists; preserve the frozen tuning result")
    memberships = grouped_gate_split(context["records"], context["split"]["dev_tune"])
    membership_manifest = _hashed(dict(memberships, split_hash=context["split"]["manifest_hash"], source_hashes=context["source_hashes"]))
    output.mkdir(parents=True, exist_ok=True)
    membership_path = output / "gate_memberships.json"
    if membership_path.exists() and _checked(membership_path) != membership_manifest:
        raise ValueError("Existing gate membership/hash state changed")
    write_json(membership_path, membership_manifest)
    baseline_result = evaluate_predictions(list(context["baseline"].values()), context["records"], context["split"], reference_only)
    references = _by_id(baseline_result["scored_rows"], "tune reference")
    scores, calls = {}, []
    with ResourceMonitor(output / "prep_resources.json", max_seconds=context["config"].get("max_seconds", 28200)) as monitor:
        started = perf_counter()
        scorer = DistilGPT2Scorer(scorer_dir, device=device, allow_cpu_fallback=context["config"].get("allow_cpu_fallback", False))
        setup_seconds = perf_counter() - started
        for trial_id in sorted(references):
            monitor.check()
            choices = context["choices"][trial_id]
            if choices:
                scores[trial_id] = _score(scorer, choices)
                calls.append(scores[trial_id])
            else:
                scores[trial_id] = {"candidates": [], "elapsed_seconds": 0.0, "error": "no compact candidates", "candidate_count": 0}
    _write_rows(output / "tuning_scores.jsonl", [dict(trial_id=trial_id, **score) for trial_id, score in scores.items()])
    curves, predictions = {}, {}
    for weight in LAMBDA_GRID:
        predictions[weight] = {trial_id: _prediction(context["baseline"][trial_id], scores[trial_id], weight,
                              weight != 0 and bool(context["choices"][trial_id]), "preparation") for trial_id in references}
        curves[str(weight)] = {name: _word_totals(references, predictions[weight], ids) for name, ids in memberships.items()}
    fit_key = lambda weight: (curves[str(weight)]["gate_fit"]["errors"], weight)
    best_overall_lambda = min(LAMBDA_GRID, key=fit_key)
    selected_lambda = min((weight for weight in LAMBDA_GRID if weight > 0), key=fit_key)
    fit_margins = [context["features"][trial_id]["top_two_margin_per_word"] for trial_id in memberships["gate_fit"]]
    fit_margins = [margin for margin in fit_margins if margin is not None]
    threshold = float(median(fit_margins)) if fit_margins else None
    gated, controls, decisions = _decisions(context, threshold)
    _write_rows(output / "gate_decisions.jsonl", decisions)
    oracle_errors = sum(min(edit_counts(normalize_text(row["reference_text"]).split(), normalize_text(candidate["text"]).split())["errors"]
                            for candidate in (context["choices"][trial_id] or [{"text": ""}])) for trial_id, row in references.items())
    denominator = baseline_result["metrics"]["word"]["reference_length"]
    oracle_wer = oracle_errors / denominator if denominator else None
    selected = {
        "version": 1, "study": "confidence_lm", "configuration": context["config"], "configuration_hash": object_hash(context["config"]),
        "source_hashes": context["source_hashes"], "membership_hash": membership_manifest["manifest_hash"],
        "gate_fit_ids": memberships["gate_fit"], "gate_selection_ids": memberships["gate_selection"],
        "lambda_grid": LAMBDA_GRID, "selected_lambda": selected_lambda,
        "best_overall_lambda": best_overall_lambda, "gate_threshold": threshold,
        "call_fraction_policy": 0.5, "gate_rule": "margin <= gate-fit median; missing margins skip; ties included",
        "random_seeds": RANDOM_SEEDS, "random_eligibility": "at least two frozen candidates", "random_bins": ["1-5", "6-10", "11+"],
        "random_strata": ["dev_tune", "dev_report"],
        "lambda_curves": curves, "selection_rule": "A0 is lambda zero; for A1/A2/A3 minimize gate-fit word edits over positive lambda, ties use the smaller weight; gate-selection is a check only",
        "oracle": {"score_split": "dev_tune", "errors": oracle_errors, "reference_length": denominator, "wer": oracle_wer,
                   "baseline_minus_oracle_wer": baseline_result["metrics"]["wer"] - oracle_wer if denominator else None},
        "preparation": dict(_timings(calls), scorer_setup_seconds=setup_seconds, scorer_metadata=scorer.metadata),
        "full_validation_gated_count": len(gated), "full_validation_random_counts": {seed: len(ids) for seed, ids in controls.items()},
    }
    selection_gated = {trial_id: predictions[selected_lambda][trial_id] if trial_id in gated else context["baseline"][trial_id] for trial_id in references}
    selected["gate_selection_check"] = {
        "A0": _word_totals(references, context["baseline"], memberships["gate_selection"]),
        "A1": _word_totals(references, predictions[selected_lambda], memberships["gate_selection"]),
        "A2": _word_totals(references, selection_gated, memberships["gate_selection"]),
    }
    selected = _hashed(selected)
    write_json(output / "selected_config.json", selected)
    return selected


def _frozen_context(config, candidate_run, scorer_dir, selected_config):
    selected = _checked(selected_config)
    if not selected.get("manifest_hash"):
        raise ValueError("Selected configuration requires a saved content hash")
    context = _context(config, candidate_run, scorer_dir)
    if selected.get("source_hashes") != context["source_hashes"] or selected.get("configuration_hash") != object_hash(context["config"]):
        raise ValueError("Frozen candidate, asset, or configuration hashes changed")
    return context, selected


def study_hashes(config, candidate_run, scorer_dir, selected_config) -> dict:
    """Live confidence_lm entry for the caller's joint freeze/expected manifest."""
    context, selected = _frozen_context(config, candidate_run, scorer_dir, selected_config)
    return {"config_hash": selected["manifest_hash"], "checkpoint_hash": context["source_hashes"]["checkpoint_hash"],
            "assets_hash": object_hash(context["source_hashes"])}


def _effect(baseline, result, ids):
    a, b = _by_id(baseline["per_trial"], "baseline edits"), _by_id(result["per_trial"], "condition edits")
    denominator = sum(a[trial_id]["word"]["reference_length"] for trial_id in ids)
    errors_a = sum(a[trial_id]["word"]["errors"] for trial_id in ids)
    errors_b = sum(b[trial_id]["word"]["errors"] for trial_id in ids)
    rescued = sum(b[trial_id]["word"]["errors"] < a[trial_id]["word"]["errors"] for trial_id in ids)
    harmed = sum(b[trial_id]["word"]["errors"] > a[trial_id]["word"]["errors"] for trial_id in ids)
    return {"trial_count": len(ids), "reference_length": denominator, "errors_a": errors_a, "errors_b": errors_b,
            "delta_b_minus_a": (errors_b - errors_a) / denominator if denominator else None,
            "rescue_count": rescued, "harm_count": harmed, "unchanged_count": len(ids) - rescued - harmed,
            "rescue_rate": rescued / len(ids) if ids else None, "harm_rate": harmed / len(ids) if ids else None}


def _deploy(context, selected, scorer, call_ids, output, name):
    output.mkdir(parents=True, exist_ok=True)
    calls, predictions, candidate_rows, latencies, called_ids = [], [], [], [], []
    started = perf_counter()
    with ResourceMonitor(output / "resources.json", max_seconds=context["config"].get("max_seconds", 28200)) as monitor:
        for trial_id in context["ordered_ids"]:
            monitor.check()
            trial_started = perf_counter()
            called = trial_id in call_ids and bool(context["choices"][trial_id])
            score = _score(scorer, context["choices"][trial_id]) if called else {}
            if called:
                calls.append(score)
                called_ids.append(trial_id)
                candidate_rows.append(dict(trial_id=trial_id, **score))
            row = _prediction(context["baseline"][trial_id], score, selected["selected_lambda"], called, name)
            row["candidate_source_sha256"] = context["source_hashes"]["candidates_sha256"]
            predictions.append(row)
            latencies.append(perf_counter() - trial_started)
    _write_rows(output / "full_predictions.jsonl", predictions)
    _write_rows(output / "scored_candidates.jsonl", candidate_rows)
    stats = dict(_timings(calls, latencies), elapsed_seconds=perf_counter() - started,
                 call_fraction=len(called_ids) / len(context["ordered_ids"]),
                 call_bins={name: sum(candidate_length_bin(context["features"][trial_id]["word_count"]) == name for trial_id in called_ids) for name in ("1-5", "6-10", "11+")},
                 latency_scope="incremental live LM calls and policy loop; shared compact decoding and scorer setup recorded separately")
    write_json(output / "deployment.json", stats)
    return predictions, stats


def report(config, candidate_run, output_dir, scorer_dir, selected_config, freeze_manifest, expected_hashes,
           device="cuda", score_split="dev_report") -> dict:
    """Run fixed A0/A1/A2/three A3 policies, scoring only an approved selection.

    expected_hashes is an artifact registry dict/path with schema_version and
    artifacts. All baseline/A/B hashes are recomputed from current bytes by
    freeze.current_hashes before the shared evaluator permits label access.
    LM failures retain source phones and empty word output, with status=lm_error
    and independent phoneme_status plus explicit deployment failure counts.
    """
    if score_split not in {"dev_report", "official_val"}:
        raise ValueError("Experiment A reporting requires dev_report or official_val")
    context, selected = _frozen_context(config, candidate_run, scorer_dir, selected_config)
    if not freeze_manifest or not expected_hashes:
        raise ValueError("Reporting requires joint freeze and live expected hashes")
    freeze, registry = _checked(freeze_manifest), _checked(expected_hashes)
    if registry.get("schema_version") != 1 or not isinstance(registry.get("artifacts"), dict):
        raise ValueError("expected_hashes must be an artifact registry for independent live verification")
    if freeze.get("registry_hash") != object_hash(registry):
        raise ValueError("Live artifact registry differs from the joint freeze")
    from .freeze import current_hashes
    expected = current_hashes(registry)
    live = {"config_hash": selected["manifest_hash"], "checkpoint_hash": context["source_hashes"]["checkpoint_hash"], "assets_hash": object_hash(context["source_hashes"])}
    if expected.get("studies", {}).get("confidence_lm") != live:
        raise ValueError("Current confidence_lm hashes do not match expected hashes")
    baseline = evaluate_predictions(list(context["baseline"].values()), context["records"], context["split"], reference_only, score_split, freeze, expected)
    allowed_references = {row["trial_id"]: {"reference_text": row["reference_text"], "phoneme_ids": row["reference_phoneme_ids"]} for row in baseline["scored_rows"]}
    gated, controls, decisions = _decisions(context, selected["gate_threshold"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_rows(output / "gate_decisions.jsonl", decisions)
    metadata = {"study": "confidence_lm", "selected_config_hash": selected["manifest_hash"], "candidate_source_sha256": context["source_hashes"]["candidates_sha256"], "text_available": True}
    write_evaluation(baseline, output / "A0", dict(metadata, condition="A0"))
    _write_rows(output / "A0" / "full_predictions.jsonl", [context["baseline"][trial_id] for trial_id in context["ordered_ids"]])
    source_resources = Path(candidate_run) / "resources.json"
    if source_resources.exists():
        write_json(output / "A0" / "resources.json", json.loads(source_resources.read_text(encoding="utf-8")))
    compact_latencies = [row["elapsed_seconds"] for row in context["baseline"].values() if row.get("elapsed_seconds") is not None]
    compact_timing = {
        "recorded_source_run_elapsed_seconds": context["source_state"].get("elapsed_seconds"),
        "sum_trial_decode_seconds": sum(compact_latencies) if compact_latencies else None,
        "scope": "Shared compact decoding. Source run wall time may cover a resumed invocation; saved per-trial durations cover the cached trials.",
    }
    condition_results = {"A0": (baseline, dict(_timings([], compact_latencies), call_fraction=0.0,
                           call_bins={name: 0 for name in ("1-5", "6-10", "11+")}, latency_scope="shared source compact decoder per-trial durations"))}
    with ResourceMonitor(output / "scorer_setup_resources.json", max_seconds=context["config"].get("max_seconds", 28200)):
        setup_started = perf_counter()
        scorer = DistilGPT2Scorer(scorer_dir, device=device, allow_cpu_fallback=context["config"].get("allow_cpu_fallback", False))
        setup_seconds = perf_counter() - setup_started
    policies = {"A1": set(context["ordered_ids"]), "A2": set(gated)}
    policies.update({f"A3_seed{seed}": set(ids) for seed, ids in controls.items()})
    for name, call_ids in policies.items():
        predictions, stats = _deploy(context, selected, scorer, call_ids, output / name, name)
        result = evaluate_predictions(predictions, context["records"], context["split"], lambda row: allowed_references[row["trial_id"]], score_split, freeze, expected)
        write_evaluation(result, output / name, dict(metadata, condition=name, deployment=stats, scorer_metadata=scorer.metadata))
        condition_results[name] = (result, stats)
    conditions = {}
    for name, (result, stats) in condition_results.items():
        ids = [row["trial_id"] for row in result["scored_rows"]]
        executed_policy=policies.get(name,set())
        called = {trial_id for trial_id in executed_policy if context["choices"][trial_id]}
        def call_counts(members):
            selected_calls = called & set(members)
            return {"lm_calls": len(selected_calls), "call_bins": {
                label: sum(candidate_length_bin(context["features"][trial_id]["word_count"]) == label
                           for trial_id in selected_calls) for label in ("1-5", "6-10", "11+")}}
        scored_counts = call_counts(ids)
        corpora = {}
        for row in result["scored_rows"]:
            corpora.setdefault(_group_value(row, "corpus_id"), []).append(row["trial_id"])
        conditions[name] = dict(stats, scored_trial_lm_calls=scored_counts["lm_calls"],
                                scored_call_bins=scored_counts["call_bins"],
                                partition_calls={part: call_counts(context["split"][part]) for part in ("dev_tune", "dev_report")},
                                wer=result["metrics"]["wer"], per=result["metrics"]["per"], word_effect=_effect(baseline, result, ids),
                                corpus_effects={corpus: _effect(baseline, result, members) for corpus, members in sorted(corpora.items())},
                                unusual_random_corpora=[corpus for corpus in sorted(corpora) if "random" in corpus.casefold() or "unusual" in corpus.casefold()])
    summary = {"study": "confidence_lm", "score_split": score_split, "split_hash": context["split"]["manifest_hash"],
               "freeze_manifest_hash": freeze["manifest_hash"], "selected_config_hash": selected["manifest_hash"],
               "selected_lambda": selected["selected_lambda"], "gate_threshold": selected["gate_threshold"],
               "source_hashes": context["source_hashes"], "scorer_setup_seconds": setup_seconds,
               "shared_compact_decode": compact_timing,
               "deployed_predictions_sha256":{name:sha256(output/name/"full_predictions.jsonl") for name in conditions},
               "scorer_metadata": scorer.metadata, "conditions": conditions,
               "interpretation": "All fixed conditions retained, including harmful comparisons. Corpus diagnostics use only the approved scored selection. No population generalization claim."}
    write_json(output / "study_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["tune", "report", "hashes"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--scorer-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selected-config")
    parser.add_argument("--freeze-manifest")
    parser.add_argument("--expected-hashes", "--artifact-registry", dest="expected_hashes")
    parser.add_argument("--score-split", choices=["dev_report", "official_val"], default="dev_report")
    args = parser.parse_args(argv)
    if args.phase == "tune":
        if not args.output_dir:
            parser.error("tune requires --output-dir")
        result = tune(args.config, args.candidate_run, args.output_dir, args.scorer_dir, args.device)
        result = {"selected_config": str(Path(args.output_dir) / "selected_config.json"), "selected_lambda": result["selected_lambda"], "gate_threshold": result["gate_threshold"]}
    elif args.phase == "hashes":
        if not args.selected_config:
            parser.error("hashes requires --selected-config")
        result = study_hashes(args.config, args.candidate_run, args.scorer_dir, args.selected_config)
    else:
        if not all((args.output_dir, args.selected_config, args.freeze_manifest, args.expected_hashes)):
            parser.error("report requires output, selected config, joint freeze, and expected hashes")
        result = report(args.config, args.candidate_run, args.output_dir, args.scorer_dir, args.selected_config,
                        args.freeze_manifest, args.expected_hashes, args.device, args.score_split)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
