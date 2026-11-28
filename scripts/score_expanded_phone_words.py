"""Apply a fixed alignment-aware DistilGPT2 decision to existing and expanded word lists."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from btt25.assets import sha256
from btt25.benchmark import cached_encoder
from btt25.ctc import ctc_minimum_frames, greedy_ctc
from btt25.data import attach_corpus_metadata, load_split_manifest, reference_only, write_json
from btt25.evaluation import _by_id, _selection, edit_counts, normalize_text
from btt25.inference import load_config, load_json, read_jsonl
from btt25.phone_edit_candidates import Neighbors, Trigram, proposals
from btt25.phone_word_fusion import assemble, select
from btt25.qwen_scorer import cluster_guard
from btt25.rescoring import DistilGPT2Scorer
from btt25.resources import ResourceMonitor


def packets(path, ids, only_normal=False):
    selected={}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            r=json.loads(line)
            trial=r["trial_id"]
            if trial in ids and (not only_normal or r["input"] == "neural"):
                if trial in selected:
                    raise ValueError("Duplicate trial packet")
                selected[trial]=r
    if selected.keys()!=ids:
        raise ValueError("Candidate coverage differs")
    return selected


def run(args):
    cluster_guard()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Require Gilbreth BF16 CUDA")
    cfg = load_config(args.config)
    records = load_json(cfg["data_manifest"])
    split = load_split_manifest(cfg["split_manifest"],records)
    selected = _selection(_by_id(attach_corpus_metadata(records,cfg["corpus_metadata"]),"trial"),split,"dev_tune")
    ids={r["trial_id"] for r in selected}
    args.output.mkdir(parents=True,exist_ok=False)
    reg=load_json(args.registration)
    write_json(args.output/"config.resolved.yaml",reg)
    digests={key:sha256(getattr(args,key)) for key in
             ("registration","proposals","source","old_scores","proposal_config","arpa","lexicon","unigrams")}
    digests["experiment"]=sha256(__file__)
    digests["emission_manifest"]=sha256(args.cache/"cache_manifest.json")
    provenance={"hashes":digests,"split_hash":split["manifest_hash"]}
    write_json(args.output/"provenance.json",provenance)
    encode=cached_encoder(args.cache,cfg,records,split)
    overall,corpus,sessions=defaultdict(Counter),defaultdict(lambda:defaultdict(Counter)),defaultdict(lambda:defaultdict(Counter))
    paired=defaultdict(list)
    totals=Counter()
    torch.set_num_threads(1)
    with ResourceMonitor(args.output/"resources.json",max_seconds=670) as monitor, \
         (args.output/"predictions.jsonl").open("x",encoding="utf-8",buffering=1) as stream, \
         (args.output/"candidate-scores.jsonl").open("x",encoding="utf-8",buffering=1) as saved:
        old=packets(args.old_scores,ids)
        frozen=packets(args.source,ids,True)
        extra=packets(args.proposals,ids,True)
        cache_manifest=load_json(args.old_scores.parent/"cache_manifest.json")
        scorer=DistilGPT2Scorer(args.distil,device="cuda:0",dtype=torch.bfloat16)
        if (scorer.metadata["revision"]!=reg["model_revision"] or scorer.metadata["asset_manifest_sha256"]
                !=cache_manifest["fixed_neural_model"]["asset_manifest_sha256"]):
            raise ValueError("Model differs from the archived scorer")
        provenance["scorer"]=scorer.metadata["asset_manifest_sha256"]
        write_json(args.output/"provenance.json",provenance)
        cfg_edit=load_json(args.proposal_config)
        unigrams=load_json(args.unigrams)
        if unigrams["source_split"]!="train":
            raise ValueError("Unexpected unigram source")
        if sha256(args.arpa)!=load_json(args.arpa.with_name("manifest.json"))["arpa_sha256"]:
            raise ValueError("ARPA differs from train-only model manifest")
        lm=Trigram(args.arpa)
        index=Neighbors(args.lexicon,unigrams["counts"],cfg_edit)
        for record in selected:
            monitor.check()
            trial=record["trial_id"]
            source=frozen[trial]["candidates"]
            previously={c["text"]:c for c in old[trial]["candidates"]}
            top=[c["text"] for c in source[:25]]
            if set(top)!=previously.keys():
                raise ValueError("Top 25 candidates differ from verified LM archive")
            values,_=encode(record)
            orig,combined,generated=assemble(source,values,index,lm,cfg_edit,reg,archived=True)
            if [t for t,_ in generated]!=extra[trial]["proposed_texts"]:
                raise ValueError("Expanded candidate replay differs from completed proposal study")
            texts=list(combined)
            scores={t:previously[t]["neural_logprob"] for t in top}
            token_counts={t:previously[t]["neural_tokens"] for t in top}
            missing=[t for t in texts if t not in scores]
            start=perf_counter()
            for t,item in zip(missing,scorer.score_many(missing,batch_size=reg["batch_size"]),strict=True):
                scores[t]=item["log_likelihood"]
                token_counts[t]=item["token_count"]
            elapsed=perf_counter()-start
            totals.update(reused_likelihoods=len(top),new_likelihoods=len(missing),
                          new_tokens=sum(token_counts[t] for t in missing))
            reference=normalize_text(reference_only(record)["reference_text"]).split()
            choices={}
            for k in reg["cells"]:
                pool=combined if k.startswith("expanded") else orig
                winner=select(pool,scores,reg["weight"],k.endswith("alignment"))
                edits=edit_counts(reference,normalize_text(winner).split())
                oracle=min((edit_counts(reference,normalize_text(t).split())["errors"] for t in pool),default=len(reference))
                counts={**edits,"oracle_edits":oracle,"trials":1,"empty":int(not winner),"candidates":len(pool)}
                for summary in [overall[k],corpus[record.get("corpus_id") or "unknown"][k],sessions[record["session_id"]][k]]:
                    summary.update(counts)
                paired[k].append(edits["errors"])
                choices[k]={"text":winner,"counts":counts}
            stream.write(json.dumps({"trial_id":trial,"results":choices,"new_lm_calls_seconds":elapsed,
                                     "new_likelihoods":len(missing),"reused":len(top)})+"\n")
            saved.write(json.dumps({"trial_id":trial,"scores":[{"text":t,"neural_logprob":scores[t],
                       "token_count":token_counts[t],"native_acoustic":orig.get(t),"union_acoustic":combined[t]} for t in texts]})+"\n")
    if (overall["original_alignment"]["errors"]!=reg["control_edits"]
            or overall["original_path"]["errors"]!=reg["control_path_edits"]
            or overall["original_alignment"]["reference_length"]!=reg["control_denominator"]):
        raise ValueError("Top-25 ranking does not reproduce the prior result")
    ref_lengths=[r["results"]["original_alignment"]["counts"]["reference_length"] for r in read_jsonl(args.output/"predictions.jsonl")]
    resample=np.random.default_rng(20261127).integers(len(selected),size=(2000,len(selected)))
    for name,errors in paired.items():
        control="original_alignment" if name.endswith("alignment") else "original_path"
        diff=np.array(errors)-np.array(paired[control])
        overall[name].update(improved_trials=int((diff<0).sum()),harmed_trials=int((diff>0).sum()))
        overall[name]["paired95_vs_"+control]=np.quantile(diff[resample].sum(1)/np.array(ref_lengths)[resample].sum(1),[.025,.975]).tolist()
    write_json(args.output/"metrics.json",{"status":"complete","scope":"dev_tune","overall":overall,"corpus":corpus,
                                           "sessions":sessions,"score_work":totals})
    lines=["# Exact CTC and DistilGPT2 on expanded lists","","| Condition | Word errors | Oracle | WER |","|---|---:|---:|---:|"]
    for name,c in overall.items():
        lines.append(f"| {name} | {c['errors']} | {c['oracle_edits']} | {c['errors']/c['reference_length']:.2%} |")
    (args.output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("registration","config","cache","source","proposals","proposal-config","arpa","lexicon","unigrams","distil","old-scores","output"):
        p.add_argument("--"+k,type=Path,required=True)
    run(p.parse_args())
