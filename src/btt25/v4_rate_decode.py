"""Pool or temper trained 40 ms emissions, decode blind, then score frozen dev_tune."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from btt25.assets import sha256
from btt25.benchmark import native_decoder
from btt25.comparison import _bootstrap
from btt25.ctc import greedy_ctc
from btt25.data import attach_corpus_metadata, load_split_manifest, read_trial, reference_only, write_json
from btt25.evaluation import _by_id, _selection, edit_counts, normalize_text
from btt25.inference import load_config, load_json, read_jsonl
from btt25.model import gaussian_kernel, load_encoder
from btt25.resources import BudgetExceeded, ResourceMonitor
from btt25.v4_rate import RateEncoder, geometry


METHODS = ("native", "temperature2", "pair_average", "merge_agree")


def transform(x, method):
    """Pairs start at the first emitted frame. Keep every unpaired last frame."""
    if method not in METHODS or x.ndim != 2 or x.shape[1] != 41 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid rate method or T15 emissions")
    if method == "native":
        return x
    if method == "temperature2":
        return x * np.float32(.5)
    count = len(x) // 2
    if not count:
        return x
    # Log of arithmetic mean of normalized posteriors, robust to large logits.
    mx = x.max(-1, keepdims=True)
    z = (x - mx) - np.log(np.exp(x - mx).sum(-1, keepdims=True))
    left, right = z[:2*count:2], z[1:2*count:2]
    pooled = np.logaddexp(left, right) - np.float32(np.log(2))
    merge = np.ones(count, dtype=bool) if method == "pair_average" else left.argmax(-1) == right.argmax(-1)
    output = z.copy()
    output[:2*count:2][merge] = pooled[merge]
    keep = np.ones(len(x), dtype=bool)
    keep[1:2*count:2] = ~merge
    return output[keep]


def emissions(args, name, root, val, tuning, cfg, monitor):
    ckpt = root / "checkpoints/last.pt"
    prior = load_json(root / "metrics.json")
    if prior["status"] != "complete" or sha256(ckpt) != prior["checkpoint_sha256"]:
        raise ValueError(f"Need completed and verified {name} checkpoint")
    official, upstream = load_encoder(cfg["checkpoint"], cfg["model_args"], "cpu", cfg["checkpoint_hash"])
    if official.contract_hash != cfg["contract_hash"]:
        raise ValueError("Official feature contract changed")
    settings = upstream["dataset"]["data_transforms"]
    model = RateEncoder(official, name, gaussian_kernel(settings["smooth_kernel_std"], settings["smooth_kernel_size"]))
    delta = torch.load(ckpt, map_location="cpu", weights_only=True)
    if (delta["condition"] != name or delta["source_sha256"] != cfg["checkpoint_hash"] or
          delta["step"] != prior["updates"] or model.contract != prior["contract"]):
        raise ValueError("Trained model identity mismatch")
    model.load_delta(delta["trainable"])
    del delta, official
    model.cuda().eval()
    cache = args.output / name / "additional_logits"
    cache.mkdir(parents=True)
    paths, digest, extra = {}, hashlib.sha256(), 0
    cached = {r["trial_id"] for r in tuning}
    with torch.inference_mode():
        for row in val:
            monitor.check()
            trial, pos = row["trial_id"], row["official_order_index"]
            path = root / f"logits/{pos:05d}.npy" if trial in cached else cache / f"{pos:05d}.npy"
            if trial not in cached:
                tensor = torch.from_numpy(read_trial(row, include_labels=False)["features"]).cuda().unsqueeze(0)
                x = model(tensor, row["day_index"])[0].cpu().numpy()
                np.save(path, x, allow_pickle=False)
                extra += 1
            else:
                x = np.load(path, allow_pickle=False)
            if x.shape != (geometry(name, row["input_length"]), 41) or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError(f"Invalid emissions on {trial}")
            digest.update(trial.encode() + b"\0" + hashlib.sha256(x.tobytes()).digest())
            paths[trial] = path
    write_json(args.output / name / "source.json", {"checkpoint_sha256": prior["checkpoint_sha256"],
               "registered_updates": prior["updates"], "official_source_sha256": cfg["checkpoint_hash"],
               "cached_tuning_trials": len(cached), "additional_blind_trials": extra,
               "ordered_trial_and_emissions_sha256": digest.hexdigest()})
    return paths


def decode_one(row, paths, method, decode):
    trial = row["trial_id"]
    begin = monotonic()
    x = np.load(paths[trial], allow_pickle=False)
    z = transform(x, method)
    phone_ids = greedy_ctc(z.argmax(-1).tolist())
    response = decode(trial, z)
    candidates = response["candidates"]
    text = candidates[0]["text"] if candidates else ""
    return {"trial_id": trial, "text": text, "phones": phone_ids, "status": "ok" if text else "empty",
            "original_frames": len(x), "decoder_frames": len(z), "source_bins": row["input_length"],
            "cached_logit_to_text_s": monotonic() - begin, "search_s": response["search_s"]}


def score(rows, allow, refs, control, original, method, *, lexical_disagreements=None):
    totals, sessions, corpora = Counter(), defaultdict(Counter), defaultdict(Counter)
    errors = {"word": [], "phone": []}
    lengths = {"word": [], "phone": []}
    per_trial = []
    for row in allow:
        trial = row["trial_id"]
        pred, ref = rows[trial], refs[trial]
        if method == "native":
            if original[trial]["phones"] != pred["phones"]:
                raise ValueError(f"Source phones differ on {trial}")
            if original[trial]["text"] != pred["text"]:
                if lexical_disagreements is None:
                    raise ValueError(f"Native did not replay the original trained arm on {trial}")
                lexical_disagreements.append({"trial_id":trial, "trained_text":original[trial]["text"],
                                               "replayed_text":pred["text"]})
        if method in {"temperature2", "merge_agree"} and pred["phones"] != control[trial]["phones"]:
            raise ValueError(f"Greedy path invariant failed on {trial}")
        w = edit_counts(normalize_text(ref["reference_text"]).split(), normalize_text(pred["text"]).split())
        p = edit_counts(ref["phoneme_ids"], pred["phones"])
        cnt = Counter(trials=1, empty=int(not pred["text"]), frames=pred["decoder_frames"],
                      original_frames=pred["original_frames"])
        for kind, val in (("word", w), ("phone", p)):
            errors[kind].append(val["errors"])
            lengths[kind].append(val["reference_length"])
            for key, v in val.items():
                cnt[kind+"_"+key] = v
        totals.update(cnt)
        sessions[row["session_id"]].update(cnt)
        corpora[row.get("corpus_id") or "unclassified"].update(cnt)
        per_trial.append({"trial_id": trial, "word": w, "phone": p, "counts": cnt})
    return ({"counts": totals, "sessions": sessions, "corpora": corpora, "errors": errors,
             "lengths": lengths, "wer": totals["word_errors"]/totals["word_reference_length"],
             "per": totals["phone_errors"]/totals["phone_reference_length"]}, per_trial)


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Use an allocated Gilbreth GPU for the additional blind inference")
    if not 900 <= args.pilot_seconds <= 1800 or not args.pilot_seconds < args.max_seconds <= 28200:
        raise ValueError("Invalid sustained pilot or job wall budget")
    cfg = load_config(args.config)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = load_json(cfg["data_manifest"])
    if args.corpus_metadata:
        manifest = attach_corpus_metadata(manifest, args.corpus_metadata)
    split = load_split_manifest(cfg["split_manifest"], manifest)
    tuning = _selection(_by_id(manifest, "trial"), split, "dev_tune")
    val = sorted((r for r in manifest if r["split"] == "val"), key=lambda r: r["official_order_index"])
    if [r["official_order_index"] for r in val] != list(range(len(val))):
        raise ValueError("Official validation order is incomplete")
    roots = dict(gru40=args.gru40, two_rate=args.two_rate)
    ordered = sorted(tuning, key=lambda r: r["input_length"])
    pilot = [ordered[i] for i in np.linspace(0, len(ordered)-1, 24, dtype=int)]
    ledger = {"status":"started", "conditions":{}, "score_split":"dev_tune"}
    hashes = {str(p): sha256(p) for p in (args.config, args.lexicon, args.kenlm_model, Path(__file__))}
    identity = {"hashes": hashes, "split_hash": split["manifest_hash"], "official_val_trials":len(val),
                "dev_tune_trials":len(tuning), "methods":list(METHODS), "decode":{"beam":50,"nbest":10,
                "beam_threshold":25,"lm_weight":2}, "pilot_seconds":args.pilot_seconds,
                "max_seconds":args.max_seconds, "scope":"cached logits to final text"}
    write_json(args.output / "config.resolved.yaml", identity)
    write_json(args.output / "provenance.json", hashes)
    start = monotonic()
    try:
        with ResourceMonitor(args.output / "resources.json", max_seconds=args.max_seconds) as monitor:
            sources = {name: emissions(args, name, root, val, tuning, cfg, monitor) for name,root in roots.items()}
            conditions = [(name,m) for name in roots for m in METHODS]
            with native_decoder(args,50,args.output/"decoder.log") as (decode, info):
                ledger["decoder"] = info
                count, begin = 0, monotonic()
                with (args.output/"pilot.jsonl").open("w",encoding="utf-8",buffering=1) as log:
                    while monotonic()-begin < args.pilot_seconds:
                        name, method = conditions[count % len(conditions)]
                        row = pilot[(count//len(conditions))%len(pilot)]
                        monitor.check()
                        pred = decode_one(row,sources[name],method,decode)
                        log.write(json.dumps({"encoder":name,"method":method,**pred})+"\n")
                        count += 1
                pilot_s = monotonic()-begin
                need = 1.30 * pilot_s/count * len(conditions)*len(val) + 180
                remaining = args.max_seconds - (monotonic()-start)
                write_json(args.output/"admission.json", {"pilot_seconds":pilot_s,"requests":count,
                           "forecast_with_margin_s":need,"remaining_s":remaining,"admitted":need<=remaining})
                if need > remaining:
                    raise BudgetExceeded("Pilot forecast exceeds remaining compute budget")
                with ExitStack() as stack:
                    files = {}
                    for name,method in conditions:
                        directory = args.output/name/method
                        directory.mkdir()
                        files[name,method] = stack.enter_context((directory/"predictions.jsonl").open("w",encoding="utf-8",buffering=1))
                    # Rotate order between trials for timing; maintain identical official order in each stream.
                    for i,row in enumerate(val):
                        for offset in range(len(conditions)):
                            name,method = conditions[(i+offset)%len(conditions)]
                            monitor.check()
                            pred = decode_one(row,sources[name],method,decode)
                            files[name,method].write(json.dumps(pred)+"\n")
            # Label reads are deferred until every blind output has been saved.
            refs = {r["trial_id"]: reference_only(r) for r in tuning}
            membership = [(r["session_id"],r["block_id"]) for r in tuning]
            for name in roots:
                original = _by_id(read_jsonl(roots[name]/"predictions.jsonl"),name)
                for method in METHODS:
                    directory = args.output/name/method
                    records = read_jsonl(directory/"predictions.jsonl")
                    rows = _by_id(records,method)
                    if list(rows) != [r["trial_id"] for r in val]:
                        raise ValueError("Wrong blind coverage or order")
                    if method == "native":
                        control = rows
                    metrics, details = score(rows,tuning,refs,control,original,method)
                    latency = np.array([r["cached_logit_to_text_s"] for r in records])
                    metrics.update(blind_trials=len(records), median_cached_s=float(np.median(latency)),
                                   p95_cached_s=float(np.quantile(latency,.95)),
                                   mean_search_s=float(np.mean([r["search_s"] for r in records])),
                                   validation_input_frames=sum(r["original_frames"] for r in records),
                                   validation_output_frames=sum(r["decoder_frames"] for r in records),
                                   prediction_sha256=sha256(directory/"predictions.jsonl"))
                    err, lengths = metrics.pop("errors"), metrics.pop("lengths")
                    if method=="native":
                        baseline = err
                    else:
                        metrics["paired_vs_native"] = {k:_bootstrap(np.array(baseline[k]),np.array(err[k]),
                            np.array(lengths[k]),membership,np.random.default_rng(20261127),2000) for k in err}
                    ledger["conditions"][f"{name}/{method}"] = metrics
                    write_json(directory/"metrics.json",metrics)
                    write_json(directory/"tuning_edits.json",details)
                    write_json(directory/"config.resolved.yaml",{**identity,"encoder":name,"method":method})
                    write_json(directory/"provenance.json",load_json(args.output/name/"source.json"))
                    (directory/"report.md").write_text(f"# {name}, {method}\n\n{metrics['wer']:.2%} tuning WER "
                        f"and {metrics['per']:.2%} PER. {len(records)} blind predictions. "
                        f"Shared allocation telemetry is in ../../resources.json.\n")
            monitor.check()
        if load_json(args.output/"resources.json")["outcome"] != "complete":
            raise BudgetExceeded("Budget at final sample")
        ledger["status"] = "complete"
    except BaseException as exc:
        ledger.update(status="budget_stopped" if isinstance(exc,BudgetExceeded) else "failed",error=repr(exc))
        raise
    finally:
        write_json(args.output/"metrics.json",ledger)
        (args.output/"report.md").write_text(f"# Rate-aware decoding\n\nStatus {ledger['status']}. "
            "All word and phone edit counts apply to dev_tune only. Latency starts at cached-logit file read.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("config","lexicon","kenlm-model","decoder-python","decoder-module-path","gru40","two-rate","output"):
        p.add_argument("--"+arg,type=Path,required=True)
    p.add_argument("--corpus-metadata",type=Path)
    p.add_argument("--pilot-seconds",type=int,default=900)
    p.add_argument("--max-seconds",type=int,default=6000)
    p.set_defaults(nbest=10,lm_weight=2.,beam_threshold=25.,start_timeout=120.,request_timeout=30.)
    run(p.parse_args())
