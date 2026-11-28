"""Decode released T15 features with the fixed expanded-list GRU/DistilGPT2 system."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter

import torch

from btt25.assets import sha256
from btt25.benchmark import SERVER, feature_encoder, native_decoder
from btt25.data import read_trial, write_json
from btt25.evaluation import export_submission
from btt25.inference import load_config, load_json
from btt25.phone_edit_candidates import Neighbors, Trigram
from btt25.phone_word_fusion import assemble, select
from btt25.qwen_scorer import cluster_guard
from btt25.rescoring import DistilGPT2Scorer
from btt25.resources import ResourceMonitor


def predict(record, encoder, search, scorer, lex, lm, edits, reg):
    logits,_ = encoder(record)
    native = search(record["trial_id"],logits)["candidates"]
    _,pool,_ = assemble(native,logits,lex,lm,edits,reg)
    texts = list(pool)
    vals = scorer.score_many(texts,batch_size=reg["batch_size"])
    scores = {text:v["log_likelihood"] for text,v in zip(texts,vals,strict=True)}
    text = select(pool,scores,reg["weight"],aligned=True)
    return text,len(pool),sum(v["token_count"] for v in vals)


def run(args):
    cluster_guard()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A Gilbreth BF16 GPU allocation is required")
    trained_two_rate = getattr(args, "trained_two_rate", None)
    trained_hash = getattr(args, "trained_two_rate_sha256", None)
    merge_agree = getattr(args, "merge_agree", False)
    if trained_hash is not None and not trained_two_rate:
        raise ValueError("Provide the trained checkpoint directory for its expected hash")
    if merge_agree and not trained_two_rate:
        raise ValueError("Agreement-pair merging is registered for the trained two-rate encoder")
    cfg=load_config(args.config)
    data=load_json(cfg["data_manifest"])
    rows=sorted((r for r in data if r["split"]==args.partition),key=lambda r:r["official_order_index"])
    if not rows or [r["official_order_index"] for r in rows]!=list(range(len(rows))):
        raise ValueError("Missing or noncontiguous official trial order")
    if len({r["trial_id"] for r in rows})!=len(rows):
        raise ValueError("Duplicate trial IDs")
    reg=load_json(args.registration)
    edits=load_json(args.proposal_config)
    if (reg["native_limit"],reg["proposal_limit"],reg["weight"])!=(25,25,1.25):
        raise ValueError("Expected the measured 25+25 alignment selection rule")
    lm_info=load_json(args.train_lm_manifest)
    if (lm_info["source_split"]!="official train only" or sha256(args.arpa)!=lm_info["arpa_sha256"]
            or sha256(args.kenlm_model)!=lm_info["binary_sha256"]
            or sha256(args.training_text)!=lm_info["text_sha256"]):
        raise ValueError("Training text and ARPA/binary identities disagree")
    counts=Counter(args.training_text.read_text(encoding="utf-8").split())
    if len(counts)!=lm_info["vocabulary_words"] or sum(counts.values())!=lm_info["word_tokens"]:
        raise ValueError("Training counts differ from manifest")
    if args.verify_counts and counts!=load_json(args.verify_counts)["counts"]:
        raise ValueError("Reconstructed word counts differ from the development run")
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"config.resolved.yaml",{"partition":args.partition,"trials":len(rows),"stage":"features to final text",
               "references_loaded":False,"encoder":cfg["checkpoint_hash"],
               "frontend":"trained_two_rate" if trained_two_rate else "official",
               "merge_agree":merge_agree,"beam":args.beam,"nbest":args.nbest,
               "lm_weight":args.lm_weight,"registration":reg,"proposals":edits,"max_seconds":args.max_seconds})
    hashes={k:sha256(v) for k,v in {"features_manifest":cfg["data_manifest"],"config":args.config,
            "registration":args.registration,"proposal_rules":args.proposal_config,"lexicon":args.lexicon,
            "arpa":args.arpa,"binary":args.kenlm_model,"train_text":args.training_text,"executable":__file__,
            "decoder_server":SERVER,"decoder_worker":SERVER.with_name("decode_worker.py")}.items()}
    predictions=[]
    outcome={"status":"started","partition":args.partition,"references_loaded":False,
             "expected_trials":len(rows),"selected_rule":"expanded_alignment"}
    with (out/"predictions.jsonl").open("x",encoding="utf-8",buffering=1) as stream:
        try:
            with ResourceMonitor(out/"resources.json",max_seconds=args.max_seconds) as monitor:
                if trained_two_rate:
                    from btt25.two_rate_expanded import DELTA, load_trained
                    from btt25.v4_rate_decode import transform
                    expected_sha256=DELTA if trained_hash is None else trained_hash
                    net=load_trained(cfg,trained_two_rate,expected_sha256)
                    hashes["two_rate_delta"]=expected_sha256
                    method="merge_agree" if merge_agree else "native"

                    @torch.inference_mode()
                    def encoder(row):
                        features=read_trial(row,include_labels=False)["features"]
                        logits=net(torch.from_numpy(features).cuda()[None],row["day_index"])[0].float().cpu().numpy()
                        return transform(logits,method), {}
                else:
                    encoder=feature_encoder(cfg)
                scorer=DistilGPT2Scorer(args.distil,device="cuda:0",dtype=torch.bfloat16)
                if scorer.metadata["revision"]!=reg["model_revision"]:
                    raise ValueError("Frozen model revision mismatch")
                write_json(out/"provenance.json",{"hashes":hashes,"scorer_assets":scorer.metadata["asset_manifest_sha256"],
                           "encoder":cfg["checkpoint_hash"],"contract":cfg["contract_hash"],
                           "training_ids":lm_info["training_id_hash"]})
                lex=Neighbors(args.lexicon,counts,edits)
                lm=Trigram(args.arpa)
                with native_decoder(args,args.beam,out/"decoder.log") as (decode,service):
                    for r in rows:
                        monitor.check()
                        torch.cuda.synchronize()
                        started=perf_counter()
                        text,choices,tokens=predict(r,encoder,decode,scorer,lex,lm,edits,reg)
                        torch.cuda.synchronize()
                        item={"trial_id":r["trial_id"],"official_order_index":r["official_order_index"],
                              "text":text,"status":"ok","empty":not bool(text),"candidates":choices,
                              "lm_tokens":tokens,"elapsed_s":perf_counter()-started,"raw_frames":r["input_length"]}
                        predictions.append(item)
                        stream.write(json.dumps(item,allow_nan=False)+"\n")
                    monitor.check()
            if load_json(out/"resources.json")["outcome"]!="complete":
                raise RuntimeError("Resource stop at shutdown")
            if args.partition=="test":
                submission=export_submission(predictions,data,out/"submission.csv")
                with submission.open(newline="",encoding="utf-8") as s:
                    table=list(csv.DictReader(s))
                if ([row["id"] for row in table]!=[str(i) for i in range(len(rows))]
                        or [row["text"] for row in table]!=[r["text"] for r in predictions]):
                    raise ValueError("Round-trip submission verification failed")
                outcome["submission_sha256"]=sha256(submission)
            outcome["status"]="complete"
        except BaseException as error:
            outcome.update(status="incomplete",error=repr(error))
            raise
        finally:
            outcome.update(saved_predictions=len(predictions),empty=sum(r["empty"] for r in predictions))
            write_json(out/"metrics.json",outcome)
            (out/"report.md").write_text(f"# Reference-free {args.partition} decoding\n\n"
                f"{outcome['status']}. Saved {len(predictions)} of {len(rows)} predicted texts. "
                f"Empty predictions: {outcome['empty']}. No reference-based score was computed.\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("registration","config","distil","proposal-config","lexicon","arpa","training-text",
                "train-lm-manifest","decoder-python","decoder-module-path","kenlm-model","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--verify-counts",type=Path)
    p.add_argument("--trained-two-rate",type=Path,help="Directory containing the verified 2048-update study metrics and checkpoint")
    p.add_argument("--trained-two-rate-sha256",help="Expected SHA-256 for an independently trained compatible delta; default uses the measured released-run digest")
    p.add_argument("--merge-agree",action="store_true",help="Average posteriors for neighboring pairs with equal argmax")
    p.add_argument("--partition",choices=["test","val"],default="test")
    p.add_argument("--max-seconds",type=int,default=1350)
    p.set_defaults(beam=250,nbest=50,beam_threshold=25.,lm_weight=2.,start_timeout=120.,request_timeout=30.)
    run(p.parse_args())
