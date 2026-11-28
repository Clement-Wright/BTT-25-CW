"""Reconcile saved blind decoder outputs with independently obtained native controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from btt25.assets import sha256
from btt25.comparison import _bootstrap
from btt25.ctc import greedy_ctc
from btt25.data import attach_corpus_metadata, load_split_manifest, reference_only, write_json
from btt25.evaluation import _by_id, _selection, normalize_text, edit_counts
from btt25.inference import load_config, load_json, read_jsonl
from btt25.resources import ResourceMonitor
if __package__:
    from .v4_rate_decode import METHODS, score, transform
else:
    from v4_rate_decode import METHODS, score, transform


def audit(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Score authorized trial data within the cluster allocation")
    args.output.mkdir(parents=True,exist_ok=False)
    initial = load_json(args.blind/"metrics.json")
    if initial["status"]!="failed" or "Native did not replay the original trained arm" not in initial.get("error",""):
        raise ValueError("Only lexical replay discrepancies admit this recovery")
    cfg = load_config(args.config)
    records = attach_corpus_metadata(load_json(cfg["data_manifest"]),args.corpus_metadata)
    split = load_split_manifest(cfg["split_manifest"],records)
    allow = _selection(_by_id(records,"trial"),split,"dev_tune")
    val = sorted((r for r in records if r["split"]=="val"),key=lambda r:r["official_order_index"])
    ids = [r["trial_id"] for r in val]
    tune_ids = {r["trial_id"] for r in allow}
    prereg = load_json(args.blind/"config.resolved.yaml")
    if prereg["split_hash"]!=split["manifest_hash"] or prereg["methods"]!=list(METHODS):
        raise ValueError("Original decoder configuration identity mismatch")
    write_json(args.output/"config.resolved.yaml",prereg)
    provenance = {"original_config_sha256":sha256(args.blind/"config.resolved.yaml"),
                  "blind_predictions_job":args.blind.parent.name,"audit_code_sha256":sha256(Path(__file__)),
                  "score_code_sha256":sha256(Path(__file__).with_name("v4_rate_decode.py"))}
    write_json(args.output/"provenance.json",provenance)
    out = {"status":"started","score_split":"dev_tune","conditions":{},"native_archive":{},
           "blind_job_status":"failed at lexical replay assertion after saving predictions"}
    try:
        with ResourceMonitor(args.output/"resources.json",max_seconds=850) as monitor:
            indexed = {}
            originals = {}
            for name,source in (("gru40",args.gru40),("two_rate",args.two_rate)):
                source_cfg=load_json(source/"config.resolved.yaml")
                if source_cfg["decode"]!=prereg["decode"] or source_cfg["split_hash"]!=split["manifest_hash"]:
                    raise ValueError("Trained search weights or split differ from the replay")
                input_hash = {Path(k).name:v for k,v in prereg["hashes"].items()}
                for key in (args.config.name,"lexicon.txt","train-3gram.binary"):
                    if input_hash[key]!=source_cfg["hashes"][key]:
                        raise ValueError("Asset hash differs from the trained control: "+key)
                checkpoint=load_json(source/"metrics.json")
                receipt=load_json(args.blind/name/"source.json")
                if checkpoint["status"]!="complete" or checkpoint["checkpoint_sha256"]!=receipt["checkpoint_sha256"]:
                    raise ValueError("Checkpoint provenance mismatch")
                originals[name] = _by_id(read_jsonl(source/"predictions.jsonl"),name)
                if set(originals[name])!=tune_ids:
                    raise ValueError("Original training predictions have different tuning coverage")
                for method in METHODS:
                    parent=args.blind/name/method
                    rows=read_jsonl(parent/"predictions.jsonl")
                    if [r["trial_id"] for r in rows]!=ids:
                        raise ValueError("Incomplete or reordered blind stream")
                    indexed[name,method]=_by_id(rows,method)
                digest=hashlib.sha256()
                for row in val:
                    monitor.check()
                    trial=row["trial_id"]
                    d=source/"logits" if trial in tune_ids else args.blind/name/"additional_logits"
                    x=np.load(d/f"{row['official_order_index']:05d}.npy",allow_pickle=False)
                    digest.update(trial.encode()+b"\0"+hashlib.sha256(x.tobytes()).digest())
                    for method in METHODS:
                        pred=indexed[name,method][trial]
                        z=transform(x,method)
                        if pred["phones"]!=greedy_ctc(z.argmax(-1).tolist()) or len(z)!=pred["decoder_frames"]:
                            raise ValueError("Saved prediction disagrees with exact source emissions")
                if digest.hexdigest()!=receipt["ordered_trial_and_emissions_sha256"]:
                    raise ValueError("Blind emissions hash mismatch")
            refs = {r["trial_id"]:reference_only(r) for r in allow}
            assignment=[(r["session_id"],r["block_id"]) for r in allow]
            for name in ("gru40","two_rate"):
                control=indexed[name,"native"]
                original=originals[name]
                changes=[]
                for method in METHODS:
                    monitor.check()
                    directory=args.output/name/method
                    directory.mkdir(parents=True)
                    parent=args.blind/name/method
                    os.link(parent/"predictions.jsonl",directory/"predictions.jsonl")
                    rows=indexed[name,method]
                    metrics,details=score(rows,allow,refs,control,original,method,lexical_disagreements=changes)
                    latency=[p["cached_logit_to_text_s"] for p in rows.values()]
                    metrics.update(blind_trials=len(rows),prediction_sha256=sha256(directory/"predictions.jsonl"),
                         median_cached_s=float(np.median(latency)),p95_cached_s=float(np.quantile(latency,.95)),
                         mean_search_s=float(np.mean([p["search_s"] for p in rows.values()])),
                         validation_input_frames=sum(p["original_frames"] for p in rows.values()),
                         validation_output_frames=sum(p["decoder_frames"] for p in rows.values()))
                    errors,lengths=metrics.pop("errors"),metrics.pop("lengths")
                    if method=="native":
                        baseline=errors
                    else:
                        metrics["paired_vs_native"]={k:_bootstrap(np.array(baseline[k]),np.array(errors[k]),
                             np.array(lengths[k]),assignment,np.random.default_rng(20261127),2000) for k in errors}
                    out["conditions"][f"{name}/{method}"]=metrics
                    write_json(directory/"metrics.json",metrics)
                    write_json(directory/"tuning_edits.json",details)
                    write_json(directory/"config.resolved.yaml",{**prereg,"encoder":name,"method":method})
                    write_json(directory/"provenance.json",{**provenance,"source":load_json(args.blind/name/"source.json")})
                    write_json(directory/"resources.json",{"scope":"shared allocation",
                               "decode_receipt_sha256":sha256(args.blind/"resources.json")})
                    (directory/"report.md").write_text(f"# {name}, {method}\n\nTuning WER {metrics['wer']:.2%}, "
                        f"PER {metrics['per']:.2%}. The search control comes from the same blind decode job.\n")
                for change in changes:
                    words=normalize_text(refs[change["trial_id"]]["reference_text"]).split()
                    for k in ("trained_text","replayed_text"):
                        change[k+"_edits"]=edit_counts(words,normalize_text(change[k]).split())["errors"]
                original_errors=sum(c["counts"]["word_errors"] for c in original.values())
                replay_errors=out["conditions"][name+"/native"]["counts"]["word_errors"]
                if sum(c["replayed_text_edits"]-c["trained_text_edits"] for c in changes)!=replay_errors-original_errors:
                    raise ValueError("Recorded text disagreements do not reconcile total word edits")
                write_json(args.output/name/"replay_disagreements.json",changes)
                out["native_archive"][name]={"trial_count":len(allow),"different_texts":len(changes),
                    "different_greedy_phones":0,"archive_word_errors":original_errors,"replay_word_errors":replay_errors,
                    "archive_prediction_sha256":sha256((args.gru40 if name=="gru40" else args.two_rate)/"predictions.jsonl")}
            monitor.check()
        if load_json(args.output/"resources.json")["outcome"]!="complete":
            raise RuntimeError("Final telemetry exceeded budget")
        out["status"]="complete"
    except BaseException as exc:
        out.update(status="incomplete",error=repr(exc))
        raise
    finally:
        write_json(args.output/"metrics.json",out)
        (args.output/"report.md").write_text(f"# Rate-aware decoder replay audit\n\nStatus {out['status']}. "
            "The native controls were decoded in the same process as the three transformations. "
            "Archived and fresh native word totals are recorded separately.\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config","blind","corpus-metadata","gru40","two-rate","output"):
        p.add_argument("--"+name,type=Path,required=True)
    audit(p.parse_args())
