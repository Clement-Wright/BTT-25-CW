"""Time complete T15 word decoding with or without explicit CTC alignment summation."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .assets import sha256
from .benchmark import feature_encoder, native_decoder
from .data import load_split_manifest, object_hash, write_json
from .evaluation import _by_id, _selection
from .export_v2_test import verify_native_module
from .inference import load_config, load_json
from .resources import ResourceMonitor
from .strong_rate_decode import load_distil_scorer, score_candidates
from .v4_summary import distribution


def benchmark(args):
    cfg = load_config(args.config)
    freeze = load_json(args.freeze)
    digest = freeze.pop("manifest_hash")
    if object_hash(freeze) != digest or digest != "db4bae4fbcd3783cbdd55b50bfb5569002db7eeef5ded7c51db54cc867fef9b8":
        raise ValueError("Unexpected v2 freeze")
    binding = freeze["conditions"][freeze["selected_condition"]]
    cache_file = Path(binding["directory"])/"decode/cache_manifest.json"
    if sha256(cache_file) != binding["decode"]["cache_manifest.json"]:
        raise ValueError("Bad decoder identity")
    cache = load_json(cache_file)
    native = cache["decoder"]
    args.kenlm_model = cache["kenlm_model_path"]
    if sha256(args.kenlm_model) != cache["kenlm_model_sha256"] or sha256(args.lexicon) != cache["lexicon_sha256"]:
        raise ValueError("Bad lexical assets")
    args.nbest, args.beam_threshold, args.lm_weight = native["nbest"], native["beam_threshold"], native["lm_weight"]
    args.start_timeout, args.request_timeout = 120., 30.
    settings = {"model": "distilgpt2", "path": str(args.distil), "revision": cache["fixed_neural_model"]["revision"],
                "batch_size": 8, "candidate_limit": cache["hypothesis_limit"], "weight": cache["neural_weight"]}
    records = load_json(cfg["data_manifest"])
    split = load_split_manifest(cfg["split_manifest"], records)
    tuning = _selection(_by_id(records, "trial"), split, "dev_tune")
    ordered = sorted(tuning, key=lambda r: (r["input_length"], r["trial_id"]))
    cohort = [ordered[i] for i in np.linspace(0,len(ordered)-1,24,dtype=int)]
    prior = {"native_distil": {}, "alignment_distil": {}}
    with args.expected.open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            if r["model"] == "distilgpt2" and r["rule"] in ("native", "alignment_sum"):
                prior["native_distil" if r["rule"] == "native" else "alignment_distil"][r["trial_id"]] = r["text"]
    if any(set(rows) != {r["trial_id"] for r in tuning} for rows in prior.values()):
        raise ValueError("Could not verify full reference prediction cohort")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    state = {"status": "started", "trials": len(cohort), "split": "dev_tune",
             "passes": 3, "latency_scope": "HDF5 features to selected final text"}
    write_json(out/"config.resolved.yaml", {**state, "freeze": digest, "scorer": settings,
                                           "trials": [r["trial_id"] for r in cohort]})
    results = defaultdict(list)
    try:
        with ResourceMonitor(out/"resources.json", max_seconds=1800) as monitor:
            t0 = perf_counter()
            native_digest = verify_native_module(args.decoder_python, args.decoder_module_path)
            encoder = feature_encoder(cfg)
            scorer = load_distil_scorer(settings)
            if scorer.metadata["asset_manifest_sha256"] != cache["fixed_neural_model"]["asset_manifest_sha256"]:
                raise ValueError("Unsealed language model")
            write_json(out/"provenance.json", {"encoder": cfg["checkpoint_hash"], "native": native_digest,
                "scorer_manifest": scorer.metadata["asset_manifest_sha256"], "benchmark": sha256(__file__)})
            with native_decoder(args, native["beam"], out/"decoder.log") as (decode, setup), (
                    out/"predictions.jsonl").open("x", buffering=1) as stream:
                state["cold_load_s"] = perf_counter()-t0
                state["native_setup_s"] = setup["native_setup_s"]
                for sweep in range(3):
                    cohort_order = list(cohort)
                    np.random.default_rng(40+sweep).shuffle(cohort_order)
                    for i, row in enumerate(([cohort_order[0]] if sweep == 0 else [])+cohort_order):
                        warmup = sweep == 0 and i == 0
                        modes = ["native_distil", "alignment_distil"]
                        if (sweep+i)%2:
                            modes.reverse()
                        for mode in modes:
                            monitor.check()
                            state["active_request"] = {"trial_id": row["trial_id"], "mode": mode,
                                                       "pass": sweep, "warmup": warmup}
                            torch.cuda.synchronize()
                            start = perf_counter()
                            logits, _ = encoder(row)
                            choices = decode(row["trial_id"], logits)["candidates"]
                            if not choices:
                                text = ""
                            elif mode == "native_distil":
                                subset = choices[:settings["candidate_limit"]]
                                values = scorer.score_many([c["text"] for c in subset], batch_size=settings["batch_size"])
                                text = max(zip(subset,values,strict=True), key=lambda cv:
                                    cv[0]["component_scores"]["acoustic"] + settings["weight"]*cv[1]["log_likelihood"]
                                    )[0]["text"]
                            else:
                                text = score_candidates(choices, logits, scorer, settings)["aligned_text"]
                            torch.cuda.synchronize()
                            elapsed = perf_counter()-start
                            datum = {"trial_id": row["trial_id"], "mode": mode, "pass": sweep, "warmup": warmup,
                                     "text": text, "elapsed_s": elapsed, "raw_frames": row["input_length"],
                                     "rtf": elapsed/(row["input_length"]*.020),
                                     "matches_archived": text == prior[mode][row["trial_id"]]}
                            stream.write(json.dumps(datum)+"\n")
                            if not warmup:
                                results[mode].append(datum)
                            del state["active_request"]
                monitor.sample()
                monitor.check()
        if load_json(out/"resources.json")["outcome"] != "complete":
            raise RuntimeError("Final resource check failed")
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="incomplete", error=repr(exc))
        if "active_request" in state:
            write_json(out/"failed_request.json", {**state["active_request"], "error": repr(exc),
                       "status": "timeout" if isinstance(exc, TimeoutError) else "failed"})
        raise
    finally:
        state["timing"] = {k: {"requests": len(v), "elapsed_s": distribution([x["elapsed_s"] for x in v]),
            "rtf": distribution([x["rtf"] for x in v]), "empty": sum(not x["text"] for x in v),
            "replay_differences": sum(not x["matches_archived"] for x in v),
            "pooled_rtf": sum(x["elapsed_s"] for x in v)/sum(x["raw_frames"]*.020 for x in v)}
            for k,v in results.items()}
        for k, rows in results.items():
            durations = np.array([r["raw_frames"]*.020 for r in rows])
            cuts = np.percentile(durations,[25,50,75])
            groups = np.searchsorted(cuts,durations,side="right")
            latency = np.array([r["elapsed_s"] for r in rows])
            state["timing"][k].update(requests_per_second=len(rows)/sum(latency),
                duration_edges_s=cuts.tolist(), latency_by_duration_quartile=[
                    distribution(latency[groups == i]) if np.any(groups == i) else None for i in range(4)])
        write_json(out/"metrics.json", state)
        (out/"report.md").write_text(f"# Word decoder timing\n\n{state['status']}. Shared resident model, "
                                    "alternating complete requests and three warm passes.\n")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for key in ("output","config","freeze","lexicon","decoder-python","decoder-module-path","distil","expected"):
        p.add_argument("--"+key, required=True, type=Path)
    benchmark(p.parse_args())
