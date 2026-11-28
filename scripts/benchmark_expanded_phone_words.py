"""Time the chosen expanded word rule end to end, paired with the original-list control."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from btt25.assets import sha256
from btt25.benchmark import feature_encoder, native_decoder
from btt25.data import load_split_manifest, write_json
from btt25.evaluation import _by_id, _selection
from btt25.inference import load_config, load_json, read_jsonl
from btt25.phone_edit_candidates import Neighbors, Trigram
from btt25.phone_robustness import path_scores
from btt25.phone_word_fusion import assemble, select
from btt25.qwen_scorer import cluster_guard
from btt25.rescoring import DistilGPT2Scorer
from btt25.resources import ResourceMonitor


def distribution(values):
    return dict(zip(("p50","p90","p95","p99","max"),np.percentile(values,[50,90,95,99,100]).tolist()))


def run(args):
    cluster_guard()
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The matched timing campaign requires an allocated BF16 A100")
    cfg=load_config(args.config)
    records=load_json(cfg["data_manifest"])
    split=load_split_manifest(cfg["split_manifest"],records)
    tune=_selection(_by_id(records,"trial"),split,"dev_tune")
    sorted_ids=sorted(tune,key=lambda r:(r["input_length"],r["trial_id"]))
    cohort=[sorted_ids[i] for i in np.linspace(0,len(tune)-1,24,dtype=int)]
    reference=load_json(args.study/"metrics.json")
    if reference["status"]!="complete":
        raise ValueError("The candidate comparison has not completed")
    totals=reference["overall"]
    winner=min(("expanded_alignment","expanded_path"),key=lambda k:totals[k]["errors"])
    if totals[winner]["errors"]>=totals["original_alignment"]["errors"]:
        raise ValueError("Expanded decision did not improve the registered alignment control")
    rules=("original_alignment",winner)
    historical={r["trial_id"]:r["results"] for r in read_jsonl(args.study/"predictions.jsonl")}
    if set(historical)!={r["trial_id"] for r in tune}:
        raise ValueError("Expected a complete score cohort")
    reg=load_json(args.registration); edits=load_json(args.proposal_config)
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"config.resolved.yaml",{"rules":rules,"cohort":[r["trial_id"] for r in cohort],
               "repetitions":3,"seed":40,"candidate_registration":reg,"proposal_registration":edits,
               "clock":"request starts before opening HDF5 and ends after completed word selection",
               "scorer_batch":8,"beam":250,"nbest":50,"lm_weight":2.})
    files={"code":__file__,"study":args.study/"metrics.json","word_search":args.kenlm_model,"lexicon":args.lexicon,
           "arpa":args.arpa,"unigrams":args.unigrams,"config":args.config,"registration":args.registration}
    write_json(args.output/"provenance.json",{"hashes":{k:sha256(p) for k,p in files.items()},
               "split_hash":split["manifest_hash"],"encoder":cfg["checkpoint_hash"]})
    evidence=defaultdict(list)
    with ResourceMonitor(args.output/"resources.json",max_seconds=900) as monitor, \
         (args.output/"predictions.jsonl").open("x",encoding="utf-8",buffering=1) as stream:
        cold=perf_counter()
        encoder=feature_encoder(cfg)
        scorer=DistilGPT2Scorer(args.distil,device="cuda:0",dtype=torch.bfloat16)
        if (scorer.metadata["revision"]!=reg["model_revision"] or scorer.metadata["asset_manifest_sha256"]
                !=load_json(args.study/"provenance.json")["scorer"]):
            raise ValueError("Scorer hash mismatch")
        lex=Neighbors(args.lexicon,load_json(args.unigrams)["counts"],edits)
        lm=Trigram(args.arpa)
        with native_decoder(args,250,args.output/"decoder.log") as (search,setup):
            torch.cuda.synchronize()
            loading_s=perf_counter()-cold
            for sweep in range(3):
                order=list(cohort); np.random.default_rng(40+sweep).shuffle(order)
                for i,row in enumerate(([order[0]] if not sweep else [])+order):
                    warmup=sweep==0 and i==0
                    for name in (rules if (i+sweep)%2==0 else rules[::-1]):
                        monitor.check()
                        torch.cuda.synchronize()
                        start=perf_counter()
                        logits,_=encoder(row)
                        native=search(row["trial_id"],logits)["candidates"]
                        if name==winner:
                            _,pool,_=assemble(native,logits,lex,lm,edits,reg)
                        else:
                            top=native[:reg["native_limit"]]
                            sums=path_scores(logits,[c["phoneme_ids"] for c in top]) if top else []
                            pool={c["text"]:(c["component_scores"]["acoustic"],float(s)) for c,s in zip(top,sums)}
                        texts=list(pool)
                        scores={t:v["log_likelihood"] for t,v in zip(texts,scorer.score_many(texts,batch_size=reg["batch_size"]),strict=True)}
                        text=select(pool,scores,reg["weight"],name.endswith("alignment"))
                        torch.cuda.synchronize()
                        elapsed=perf_counter()-start
                        entry={"trial_id":row["trial_id"],"method":name,"pass":sweep,"warmup":warmup,
                               "status":"ok" if text else "empty","text":text,"elapsed_s":elapsed,
                               "raw_frames":row["input_length"],"rtf":elapsed/(row["input_length"]*.020),
                               "candidates":len(pool),"matches_archived":text==historical[row["trial_id"]][name]["text"]}
                        stream.write(json.dumps(entry)+"\n")
                        if not warmup:
                            evidence[name].append(entry)
            monitor.sample()
            monitor.check()
    result={}
    for name,rows in evidence.items():
        raw=np.array([r["raw_frames"]*.020 for r in rows])
        edges=np.percentile(raw,[25,50,75])
        sec=np.array([r["elapsed_s"] for r in rows])
        quartile=np.searchsorted(edges,raw,side="right")
        result[name]={"requests":len(rows),"p":distribution(sec),"rtf":distribution([r["rtf"] for r in rows]),
                      "pooled_rtf":float(sec.sum()/raw.sum()),"requests_per_second":float(len(sec)/sec.sum()),
                      "length_edges_s":edges.tolist(),"length_quartiles":[distribution(sec[quartile==i]) for i in range(4)],
                      "empty":sum(not r["text"] for r in rows),"replay_differences":sum(not r["matches_archived"] for r in rows)}
    write_json(args.output/"metrics.json",{"status":"complete","timing":result,"cold_s":loading_s,
               "decoder_startup":setup,"gpu":torch.cuda.get_device_name(),"precision":"BF16 LM, FP32 encoder, FP64 CTC",
               "scope":"cold model setup separately, warm offline HDF5-to-final-text requests"})
    (args.output/"report.md").write_text(
        "# Expanded word-list deployment timing\n\n" + "\n".join(
            f"{key} completed {v['requests']} warm requests with p95 {v['p']['p95']:.3f} s and pooled RTF {v['pooled_rtf']:.4f}."
            for key,v in result.items())+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("registration","config","study","distil","proposal-config","lexicon","arpa","unigrams",
                "decoder-python","decoder-module-path","kenlm-model","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.set_defaults(nbest=50,beam_threshold=25.,lm_weight=2.,start_timeout=120.,request_timeout=30.)
    run(p.parse_args())
