"""Apply the frozen v2 word decoder to all three primary trained rate encoders."""
from argparse import ArgumentParser, Namespace
from collections import Counter
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from .assets import sha256
from .benchmark import native_decoder, cached_encoder
from .data import load_split_manifest, object_hash, reference_only, write_json
from .evaluation import _by_id, _selection, edit_counts, normalize_text
from .export_v2_test import verify_native_module
from .inference import load_config, load_json, read_jsonl
from .resources import ResourceMonitor, BudgetExceeded
from .qwen_scorer import cluster_guard
from .rescoring import DistilGPT2Scorer


def load_distil_scorer(settings):
    cluster_guard()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This study requires an allocated BF16 CUDA GPU")
    scorer = DistilGPT2Scorer(Path(settings["path"]), device="cuda:0", dtype=torch.bfloat16)
    if scorer.metadata["revision"] != settings["revision"]:
        raise ValueError("Unexpected DistilGPT2 revision")
    return scorer


def score_candidates(choices, logits, scorer, settings):
    """Share one teacher-forced score per candidate between path and alignment-sum decisions."""
    subset = choices[:settings["candidate_limit"]]
    if not subset:
        return {"text": "", "aligned_text": "", "scores": []}
    paths = [tuple(c["phoneme_ids"]) for c in subset]
    unique = list(dict.fromkeys(paths))
    if any(not 0 < p < logits.shape[1] for seq in unique for p in seq):
        raise ValueError("Native candidate path contains a blank or unknown phone")
    logp = torch.from_numpy(logits).double().log_softmax(-1).unsqueeze(1)
    lengths = torch.tensor([len(s) for s in unique], dtype=torch.long)
    targets = torch.tensor([p for seq in unique for p in seq], dtype=torch.long)
    losses = torch.nn.functional.ctc_loss(
        logp.expand(-1, len(unique), -1).contiguous(), targets, torch.full_like(lengths, len(logits)),
        lengths, reduction="none", zero_infinity=False)
    if not torch.isfinite(losses).all():
        raise ValueError("Native candidate has no finite CTC alignment")
    mass = dict(zip(unique, (-losses).tolist()))
    values = scorer.score_many([c["text"] for c in subset], batch_size=settings["batch_size"])
    torch.cuda.synchronize(scorer.device)
    scores = []
    for i, (c, seq, val) in enumerate(zip(subset, paths, values, strict=True), 1):
        lm = float(val["log_likelihood"])
        weight = float(settings["weight"])
        acoustic = c["component_scores"]["acoustic"]
        scores.append({"source_rank": i, "text": c["text"], "phoneme_ids": list(seq),
                       "token_count": val["token_count"], "neural_log_likelihood": lm,
                       "path_acoustic": acoustic, "ctc_log_mass": mass[seq],
                       "kenlm_contribution": c["component_scores"]["kenlm_contribution"],
                       "native_score": c["component_scores"]["rank_score"],
                       "path_fused": acoustic + weight * lm, "alignment_fused": mass[seq] + weight * lm})
    legacy = sorted(scores, key=lambda r: (-r["path_fused"], r["source_rank"]))
    for rank, row in enumerate(legacy, 1):
        row["path_rank"] = rank
    aligned = sorted(scores, key=lambda r: (-r["alignment_fused"], r["source_rank"]))
    for rank, row in enumerate(aligned, 1):
        row["aligned_rank"] = rank
    return {"text": legacy[0]["text"], "aligned_text": aligned[0]["text"], "scores": aligned}


def check_original_control(args, cfg, records, split, tune, scorer, ranking, out):
    project = args.freeze.resolve().parents[3]
    candidate_path = project/"v2/runs/v2-search-attempt-12523602/v2-s6-repaired-250/decode/candidates.jsonl"
    archive = project/"v4-rate-20261127/runs/phone-priors-12625459/study/predictions.jsonl"
    ordered = sorted(tune, key=lambda r:(r["input_length"],r["trial_id"]))
    subset = [ordered[i] for i in np.linspace(0,len(ordered)-1,12,dtype=int)]
    ids = {r["trial_id"] for r in subset}
    expected = {r["trial_id"]:r["text"] for r in read_jsonl(archive) if r["trial_id"] in ids
                and r["model"] == "distilgpt2" and r["rule"] == "alignment_sum"}
    selected = {r["trial_id"]:r for r in read_jsonl(candidate_path) if r["trial_id"] in ids}
    encode = cached_encoder(Path("runs/baseline-val-a100"),cfg,records,split)
    replay = []
    for row in subset:
        trial = row["trial_id"]
        logits,_ = encode(row)
        result = score_candidates(selected[trial]["candidates"],logits,scorer,ranking)
        replay.append({"trial_id":trial,"expected":expected[trial],"observed":result["aligned_text"],
                       "match":expected[trial] == result["aligned_text"]})
    write_json(out/"original_gru_parity.json", {"trials":len(replay), "predictor":sha256(archive),
               "source_candidates":sha256(candidate_path), "scope":"12 length-stratified dev_tune trials",
               "rows": replay, "all_match": all(r["match"] for r in replay)})
    if not all(r["match"] for r in replay):
        raise ValueError("Original-GRU alignment scorer parity failed")


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    freeze = load_json(args.freeze)
    freeze_id = freeze.pop("manifest_hash")
    expected = "db4bae4fbcd3783cbdd55b50bfb5569002db7eeef5ded7c51db54cc867fef9b8"
    if object_hash(freeze) != freeze_id or freeze_id != expected:
        raise ValueError("Unexpected original freeze")
    condition = freeze["selected_condition"]
    if condition != "v2-f1-distilgpt2":
        raise ValueError("Unexpected v2 decoder selection")
    binding = freeze["conditions"][condition]
    source = Path(binding["directory"]) / "decode"
    if sha256(source / "cache_manifest.json") != binding["decode"]["cache_manifest.json"]:
        raise ValueError("Original v2 decoder settings changed")
    cache = load_json(source / "cache_manifest.json")
    n = cache["decoder"]
    settings = Namespace(decoder_python=args.decoder_python, decoder_module_path=args.decoder_module_path,
                         kenlm_model=cache["kenlm_model_path"], lexicon=args.lexicon, nbest=n["nbest"],
                         beam_threshold=n["beam_threshold"], lm_weight=n["lm_weight"],
                         start_timeout=120, request_timeout=30)
    if sha256(args.lexicon) != cache["lexicon_sha256"] or sha256(settings.kenlm_model) != cache["kenlm_model_sha256"]:
        raise ValueError("Decoder asset mismatch")
    ranking = {"model": "distilgpt2", "path": str(args.distil), "revision": cache["fixed_neural_model"]["revision"],
               "batch_size": 8, "candidate_limit": cache["hypothesis_limit"], "weight": cache["neural_weight"]}
    records = load_json(cfg["data_manifest"])
    split = load_split_manifest(cfg["split_manifest"], records)
    tune = _selection(_by_id(records, "trial"), split, "dev_tune")
    ids = [r["trial_id"] for r in tune]
    roots, entries, provenance = {}, {}, {"code": sha256(__file__)}
    for source_dir in args.sources:
        name = source_dir.name
        if name in roots:
            raise ValueError("Duplicate input condition")
        roots[name] = source_dir
        m = load_json(source_dir/"metrics.json")
        entries[name] = _by_id(read_jsonl(source_dir/"predictions.jsonl"), "emissions")
        if m["status"] != "complete" or m["updates"] != 2048 or set(entries[name]) != set(ids):
            raise ValueError(f"Incomplete trained emissions: {name}")
        provenance[name] = {"predictions_sha256": sha256(source_dir/"predictions.jsonl"),
                            "delta_sha256": m["checkpoint_sha256"]}
    if set(roots) != {"gru80", "gru40", "two_rate"}:
        raise ValueError("Expected exactly the three primary clocks")
    write_json(out/"config.resolved.yaml", {"split": "dev_tune", "split_hash": split["manifest_hash"],
        "v2_freeze": freeze_id, "scorer": ranking, "search": n, "max_seconds": 5400, "pilot_s": 900,
        "aligned_text": "Secondary dev-informed comparison. Exact CTC mass of one retained pronunciation plus the same frozen neural weight. No KenLM in final score."})
    totals = {k: Counter() for k in roots}
    status = {"status": "started"}
    try:
        with ResourceMonitor(out/"resources.json", max_seconds=5400) as monitor:
            provenance["native_binding"] = verify_native_module(args.decoder_python, args.decoder_module_path)
            scorer = load_distil_scorer(ranking)
            meta = scorer.metadata
            if meta["asset_manifest_sha256"] != cache["fixed_neural_model"]["asset_manifest_sha256"]:
                raise ValueError("Scorer asset mismatch")
            provenance["scorer_assets"] = meta["asset_manifest_sha256"]
            write_json(out/"provenance.json", provenance)
            check_original_control(args,cfg,records,split,tune,scorer,ranking,out)
            with native_decoder(settings, n["beam"], out/"decoder.log") as (decode, setup):
                status["decoder_setup"] = setup
                def predict(name, row):
                    trial = row["trial_id"]
                    data = entries[name][trial]
                    logits = np.load(roots[name]/data["logits"], allow_pickle=False)
                    if logits.shape != (data["emission_frames"], 41):
                        raise ValueError(f"Invalid logit shape on {trial}")
                    c = decode(trial, logits)["candidates"]
                    return c, score_candidates(c, logits, scorer, ranking)
                ordered = sorted(tune, key=lambda r: (r["input_length"], r["trial_id"]))
                pilot = [ordered[i] for i in np.linspace(0, len(ordered)-1, 24, dtype=int)]
                start = monotonic()
                times = []
                with (out/"pilot_predictions.jsonl").open("x", buffering=1) as stream:
                    while monotonic()-start < 900:
                        for row in pilot:
                            for name in roots:
                                monitor.check()
                                t0 = monotonic()
                                choices, scored = predict(name, row)
                                elapsed = monotonic()-t0
                                times.append(elapsed)
                                stream.write(json.dumps({"name": name, "trial_id": row["trial_id"],
                                    "text": scored["text"], "aligned_text": scored["aligned_text"],
                                    "candidates": len(choices), "elapsed_s": elapsed})+"\n")
                estimate = 1.35 * len(roots)*len(tune)*float(np.mean(times)) + 90
                available = 5400 - (monotonic()-monitor.start)
                write_json(out/"admission.json", {"pilot_seconds": monotonic()-start, "requests": len(times),
                    "forecast_s": estimate, "remaining_s": available, "admitted": estimate <= available})
                if estimate > available:
                    raise BudgetExceeded("Measured pilot does not fit remaining budget")
                with (out/"predictions.jsonl").open("x", buffering=1) as stream:
                    for row in tune:
                        results = {}
                        for name in roots:
                            monitor.check()
                            choices, scored = predict(name, row)
                            results[name] = {**scored, "first": choices[0]["text"] if choices else "",
                                             "candidates": [c["text"] for c in choices]}
                        ref = normalize_text(reference_only(row)["reference_text"]).split()
                        for name, pred in results.items():
                            errors = {}
                            for k in ("text", "aligned_text", "first"):
                                errors[k] = edit_counts(ref, normalize_text(pred[k]).split())["errors"]
                            aligned_counts = edit_counts(ref, normalize_text(pred["aligned_text"]).split())
                            pred["aligned_counts"] = aligned_counts
                            errors["oracle"] = min((edit_counts(ref, normalize_text(t).split())["errors"]
                                                    for t in pred["candidates"]), default=len(ref))
                            pred["edits"] = errors
                            totals[name].update(trials=1, words=len(ref), empty=int(not pred["text"]),
                                aligned_substitutions=aligned_counts["substitutions"], aligned_insertions=aligned_counts["insertions"],
                                aligned_deletions=aligned_counts["deletions"], **errors)
                        stream.write(json.dumps({"trial_id": row["trial_id"], "results": results})+"\n")
                monitor.check()
        if load_json(out/"resources.json")["outcome"] != "complete":
            raise BudgetExceeded("Final monitor sample reached limit")
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="incomplete", error=repr(exc))
        raise
    finally:
        write_json(out/"metrics.json", {**status, "counts": totals})
        (out/"report.md").write_text(f"# Strong word decoder with trained rate encoders\n\n"
           f"Status {status['status']}. All neural-LM settings came from the v2 freeze.\n")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for key in ("output", "config", "freeze", "lexicon", "decoder-python", "decoder-module-path", "distil"):
        p.add_argument("--"+key, required=True, type=Path)
    p.add_argument("sources", nargs="+", type=Path)
    run(p.parse_args())
