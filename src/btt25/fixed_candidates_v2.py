"""Compare two pinned neural priors over a frozen Flashlight candidate set."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .assets import sha256
from .benchmark import timing_summary
from .data import object_hash, reference_only, write_json
from .evaluation import _by_id, _selection, edit_counts, normalize_text
from .inference import load_config, load_json
from .qwen_scorer import QwenScorer, cluster_guard
from .rescoring import DistilGPT2Scorer
from .resources import BudgetExceeded, ResourceMonitor
from .report_v2 import data, validate_provenance, verify_run
from .search_diagnostic import diagnostics, rows


def choose_source(paths, split, search_registrations):
    definitions = {}
    for registration in search_registrations:
        search_config = load_json(registration)
        if search_config["split_manifest_hash"] != split["manifest_hash"]:
            raise ValueError("Source search registration uses a different tuning split")
        for arm in search_config["conditions"]:
            if arm["id"] in definitions:
                raise ValueError("Duplicate source search condition registration")
            definitions[arm["id"]] = registration
    candidates = []
    for directory in paths:
        tune = directory / "dev-tune/metrics.json"
        decode = directory / "decode/run_manifest.json"
        if not tune.is_file() or not decode.is_file():
            continue
        b = load_json(decode)
        if b["state"] != "complete":
            continue
        name = load_json(directory / "dev-tune/config.resolved.yaml")["condition"]["id"]
        search_registration = definitions.get(name)
        if search_registration is None:
            raise ValueError(f"Unregistered source condition: {name}")
        a = validate_provenance(directory / "dev-tune", directory / "decode", split["manifest_hash"], name,
                                search_registration)
        stamp = load_json(directory / "dev-tune/provenance.json")["source_sha256"]
        if stamp["registration"] != sha256(search_registration) or a["trials"] != len(split["dev_tune"]):
            raise ValueError(f"Unregistered or partial source condition: {name}")
        meta = load_json(directory / "decode/cache_manifest.json")
        key = (a["selected_errors"], a["oracle"]["50"]["errors"], b["p50_trial_seconds"],
               meta["decoder"]["beam"], name)
        candidates.append((key, directory))
    if not candidates:
        raise ValueError("No completed full validation candidate set with paired tuning scores")
    return min(candidates)[1]


def set_weights(packets, reference, weights):
    edits = Counter({w: 0 for w in weights})
    for trial, ref in reference.items():
        choices = packets[trial]
        errors = [edit_counts(ref, normalize_text(c["text"]).split())["errors"] for c in choices]
        for weight in weights:
            selected = max(range(len(choices)), key=lambda i: rank(choices[i], weight), default=None)
            edits[weight] += len(ref) if selected is None else errors[selected]
    return min(weights, key=lambda w: (edits[w], w)), dict(edits)


def rank(choice, weight):
    return choice["component_scores"]["acoustic"] + weight * choice["neural_logprob"]


def save_condition(where, name, prediction_rows, packets, source, source_cache, registration, metadata,
                   selected_weight, grid, records, split, resource_path):
    out = where / name / "decode"
    out.mkdir(parents=True, exist_ok=False)
    cache = {key: value for key, value in source_cache.items() if key != "manifest_hash"}
    cache.update(source_candidate_cache_hash=source_cache["manifest_hash"], fixed_neural_model=metadata,
                 hypothesis_limit=registration["candidate_limit"], neural_weight=selected_weight)
    cache["manifest_hash"] = object_hash(cache)
    predictions, ranked_packets = [], []
    for original in prediction_rows:
        trial = original["trial_id"]
        ranked = sorted(packets[trial], key=lambda c: rank(c, selected_weight), reverse=True)
        for order, row in enumerate(ranked, 1):
            row["rank"] = order
            row["component_scores"]["neural_contribution"] = selected_weight * row["neural_logprob"]
            row["component_scores"]["final_rank_score"] = rank(row, selected_weight)
        predictions.append({**original, "text": ranked[0]["text"] if ranked else "",
                            "status": "ok" if ranked else "decoder_error", "decoder_cache_hash": cache["manifest_hash"]})
        ranked_packets.append({"trial_id": trial, "logit_sha256": original["logit_sha256"],
                               "decoder_cache_hash": cache["manifest_hash"], "candidates": ranked})
    for filename, data_rows in (("predictions.jsonl", predictions), ("candidates.jsonl", ranked_packets)):
        with (out / filename).open("w", encoding="utf-8") as stream:
            stream.writelines(json.dumps(r, allow_nan=False) + "\n" for r in data_rows)
    for filename, data_obj in (("cache_manifest.json", cache), ("config.resolved.yaml", registration),
        ("run_manifest.json", {"state": "complete", "saved_trials": len(predictions), "references_loaded": False,
                              "p50_trial_seconds": metadata["timing"]["latency_seconds"]["p50"]})):
        write_json(out / filename, data_obj)
    (out / "resources.json").write_bytes(resource_path.read_bytes())
    tune = out.parent / "dev-tune"
    tune.mkdir()
    lexicon = next((p for p in (source / "lexicon/searchable_lexicon.txt", source.parent / "lexicon/searchable_lexicon.txt")
                    if p.is_file()), None)
    vocabulary = {line.split()[0] for line in lexicon.read_text(encoding="utf-8").splitlines() if line.strip()} if lexicon else None
    metrics, examples = diagnostics(records, split, predictions, ranked_packets, [10, 25, 50], vocabulary,
                                    cache_hash=cache["manifest_hash"])
    metrics.update(chosen_weight=selected_weight, tuning_weight_grid=grid, model=metadata)
    write_json(tune / "metrics.json", metrics)
    write_json(tune / "config.resolved.yaml", {"condition": {"id": name}, "registration": registration})
    write_json(tune / "provenance.json", {"source_sha256": {key: sha256(out / f"{key}.jsonl")
               for key in ("predictions", "candidates")}})
    (tune / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in examples), encoding="utf-8")
    note = f"# Fixed-candidate rescoring\n\n{metrics['selected_errors']} / {metrics['reference_words']} word errors " \
           f"on `dev_tune` ({metrics['selected_wer']:.2%}). Chosen weight {selected_weight}.\n"
    for directory in (out, tune):
        (directory / "report.md").write_text(note, encoding="utf-8")
    return metrics


def run(args):
    cluster_guard()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Require an allocated BF16-capable CUDA device")
    config, registration = load_config(args.config), load_json(args.registration)
    if registration["reference_access"] != "dev_tune":
        raise ValueError("The study uses only the tuning allowlist")
    records, split = data(config)
    registrations = [args.registration.with_name("search_v2.json"), *args.search_registration]
    source = choose_source(args.source, split, registrations)
    names = registration["conditions"]
    if "v2-f3-qwen4b" in names:
        if args.prior is None or names != ["v2-f3-qwen4b"]:
            raise ValueError("The size comparison requires the exact original completed scorer receipt")
        prior = load_json(args.prior / "config.resolved.yaml")
        prior_state = load_json(args.prior / "run_manifest.json")
        original_registration = prior["registration"]
        if (prior_state["state"] != "complete" or str(source) != prior["source"]
                or sha256(source / "decode/candidates.jsonl") != prior["candidate_sha256"]
                or any(registration[k] != original_registration[k] for k in ("candidate_limit", "batch_size", "weights"))):
            raise ValueError("The Qwen size comparison must reuse the same scored candidate contract")
    source_predictions = verify_run(source / "decode", None, _selection(_by_id(records, "record"), split, "official_val"),
                                    sha256(config["data_manifest"]), split["manifest_hash"])
    source_meta = load_json(source / "decode/cache_manifest.json")
    source_packets = rows(source / "decode/candidates.jsonl")
    packet_map = _by_id(source_packets, "candidate packet")
    fixed = {}
    cap = registration["candidate_limit"]
    for row in source_predictions:
        trial = row["trial_id"]
        packet = packet_map[trial]
        if (packet["logit_sha256"] != row["logit_sha256"]
                or packet["decoder_cache_hash"] != source_meta["manifest_hash"]):
            raise ValueError("Fixed-candidate identity mismatch")
        choices = packet["candidates"][:cap]
        if any(not isfinite(c["component_scores"]["acoustic"]) for c in choices):
            raise ValueError("Nonfinite source acoustic score")
        fixed[trial] = choices
    eligible = sorted(_selection(_by_id(records, "record"), split, "dev_tune"), key=lambda r: r["input_length"])
    pilot = [eligible[i] for i in np.linspace(0, len(eligible) - 1, 24, dtype=int)]
    indexed = _by_id(records, "record")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    provenance = {"registration": registration, "source": str(source),
                  "candidate_sha256": sha256(source / "decode/candidates.jsonl"),
                  "registration_sha256": sha256(args.registration), "search_registrations_sha256": [sha256(p) for p in registrations],
                  "code_sha256": sha256(__file__),
                  "prior_scorer_receipt_sha256": sha256(args.prior / "run_manifest.json") if args.prior else None}
    write_json(out / "config.resolved.yaml", provenance)
    measured = {name: [] for name in names}
    scored = {name: {} for name in names}
    meta = {}
    result = {"state": "running", "pilot_passes": 0}
    try:
        with ResourceMonitor(out / "resources.json", max_seconds=args.max_seconds) as monitor, ExitStack() as stack:
            timing = stack.enter_context((out / "timing.jsonl").open("w", encoding="utf-8"))
            saved = stack.enter_context((out / "predictions.jsonl").open("w", encoding="utf-8"))
            start = perf_counter()
            scorers = {}
            for name in names:
                if name == "v2-f1-distilgpt2":
                    scorers[name] = DistilGPT2Scorer(args.distil, device="cuda:0", dtype=torch.bfloat16)
                elif name == "v2-f2-qwen17b":
                    scorers[name] = QwenScorer(args.qwen)
                elif name == "v2-f3-qwen4b":
                    scorers[name] = QwenScorer(args.qwen, size="4b")
                else:
                    raise ValueError(f"Unknown fixed-candidate scorer condition: {name}")
            meta = {name: scorer.metadata for name, scorer in scorers.items()}
            if any(meta[name]["revision"] != registration["models"][name] for name in names):
                raise ValueError("Mismatched model revision")
            result["cold_model_setup_s"] = perf_counter() - start
            batch = registration["batch_size"]

            def operate(record, name, stage):
                monitor.check()
                trial = record["trial_id"]
                began = perf_counter()
                choices = fixed[trial]
                texts = [c["text"] for c in choices]
                scorer = scorers[name]
                if isinstance(scorer, DistilGPT2Scorer):
                    scores = [(x["log_likelihood"], x["token_count"]) for x in scorer.score_many(texts, batch_size=batch)]
                else:
                    scores = scorer.score(texts, batch_size=batch)
                torch.cuda.synchronize()
                row = {"trial_id": trial, "condition": name, "stage": stage, "latency_s": perf_counter() - began,
                       "status": "ok" if choices else "decoder_error", "input_seconds": record["input_length"] * .02,
                       "candidates": len(choices), "tokens": sum(n for _, n in scores), "stages": {}}
                timing.write(json.dumps(row) + "\n")
                timing.flush()
                measured[name].append(row)
                if stage == "validation":
                    packet = [{**c, "neural_logprob": value, "neural_tokens": count,
                               "component_scores": dict(c["component_scores"])}
                              for c, (value, count) in zip(choices, scores, strict=True)]
                    scored[name][trial] = packet
                    saved.write(json.dumps({**row, "scored_candidates": packet}, allow_nan=False) + "\n")
                    saved.flush()

            warm = perf_counter()
            while result["pilot_passes"] < 3 or perf_counter() - warm < registration["pilot_seconds"]:
                order = names if result["pilot_passes"] % 2 == 0 else list(reversed(names))
                for record in pilot:
                    for name in order:
                        operate(record, name, "pilot")
                result["pilot_passes"] += 1
            forecast = 2 * len(source_predictions) * sum(
                np.quantile([r["latency_s"] for r in measured[name]], .95) for name in names) + 120
            result.update(pilot_seconds=perf_counter() - warm, forecast_seconds=float(forecast))
            write_json(out / "admission.json", result)
            if perf_counter() - start + forecast > args.max_seconds:
                raise BudgetExceeded("Matched fixed-candidate pilot forecast exceeds the job budget")
            for row in source_predictions:
                for name in names:
                    operate(indexed[row["trial_id"]], name, "validation")
            monitor.check()
        if load_json(out / "resources.json")["outcome"] != "complete":
            raise BudgetExceeded("Fixed-candidate scoring exceeded the resource envelope at the final sample")
        scorers.clear()
        torch.cuda.empty_cache()
        with ResourceMonitor(out / "tuning-resources.json", max_seconds=args.max_seconds - (perf_counter() - start)) as monitor:
            reference = {}
            for r in eligible:
                monitor.check()
                reference[r["trial_id"]] = normalize_text(reference_only(r)["reference_text"]).split()
            result["conditions"] = {}
            for name in names:
                monitor.check()
                weight, grid = set_weights(scored[name], reference, registration["weights"])
                values = [row for row in measured[name] if row["stage"] == "validation"]
                meta[name].update(timing=timing_summary(values), candidates=sum(r["candidates"] for r in values),
                                  tokens=sum(r["tokens"] for r in values))
                result["conditions"][name] = save_condition(out, name, source_predictions, scored[name], source, source_meta,
                    registration, meta[name], weight, grid, records, split, out / "resources.json")
            monitor.check()
        if load_json(out / "tuning-resources.json")["outcome"] != "complete":
            raise BudgetExceeded("Tuning fold exceeded its allocated resources")
        for name in names:
            (out / name / "dev-tune/resources.json").write_bytes((out / "tuning-resources.json").read_bytes())
        result["state"] = "complete"
    except BaseException as error:
        result.update(state="budget_stopped" if isinstance(error, (BudgetExceeded, KeyboardInterrupt)) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(out / "metrics.json", result)
        write_json(out / "run_manifest.json", result)
        (out / "report.md").write_text(f"# Fixed-candidate study\n\nState `{result['state']}`. "
            "Timing includes candidate tokenization and synchronized teacher-forced scoring.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "registration", "distil", "qwen", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--search-registration", type=Path, action="append", default=[])
    parser.add_argument("--prior", type=Path, help="Completed matched smaller-model study for the 4B condition")
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=2600)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
