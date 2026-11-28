"""Apply the fixed expanded word decoder before and after merging agreeing 40 ms pairs."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256 as data_hash
import json
from pathlib import Path
from time import perf_counter

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
from btt25.phone_edit_candidates import Neighbors, Trigram
from btt25.phone_word_fusion import assemble, select
from btt25.qwen_scorer import cluster_guard
from btt25.rescoring import DistilGPT2Scorer
from btt25.resources import ResourceMonitor
from btt25.v4_rate import RateEncoder, geometry
from btt25.v4_rate_decode import transform
from btt25.v4_summary import distribution


METHODS = ("native", "merge_agree")
DELTA = "cb1c7b359896433a96dd05509e4ef063fad1b2d47a116b60df788a4f33aaa4c7"


def load_trained(config, root, expected_sha256=DELTA):
    if len(expected_sha256)!=64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("Expected a lowercase SHA-256 digest for the trained two-rate delta")
    result = load_json(root / "metrics.json")
    path = root / "checkpoints/last.pt"
    if (result["status"] != "complete" or result["updates"] != 2048 or
            result["checkpoint_sha256"] != expected_sha256 or sha256(path) != expected_sha256):
        raise ValueError("Unverified two-rate checkpoint")
    official, upstream = load_encoder(config["checkpoint"], config["model_args"], "cpu", config["checkpoint_hash"])
    if official.contract_hash != config["contract_hash"]:
        raise ValueError("Official GRU parent contract mismatch")
    prep = upstream["dataset"]["data_transforms"]
    module = RateEncoder(official, "two_rate", gaussian_kernel(prep["smooth_kernel_std"], prep["smooth_kernel_size"]))
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (module.contract != result["contract"] or state["condition"] != "two_rate" or
            state["source_sha256"] != config["checkpoint_hash"] or state["step"] != result["updates"]):
        raise ValueError("Wrong-trained-encoder architecture or parent")
    module.load_delta(state["trainable"])
    return module.eval().cuda()


def rank(pool, scorer, reg, shared):
    missing = [t for t in pool if t not in shared]
    scored = scorer.score_many(missing, batch_size=reg["batch_size"])
    for text, row in zip(missing, scored, strict=True):
        shared[text] = (row["log_likelihood"], row["token_count"])
    values = {t: shared[t][0] for t in pool}
    return select(pool, values, reg["weight"], aligned=True), len(missing)


def admission(args, trials):
    pilot = load_json(args.strong / "admission.json")
    benchmark = load_json(args.prior_timing / "metrics.json")
    if pilot["pilot_seconds"] < 900 or not pilot["admitted"] or benchmark["status"] != "complete":
        raise ValueError("Missing sustained pilot or measured word decoder benchmark")
    p95 = benchmark["timing"]["expanded_alignment"]["p"]["p95"]
    cold = benchmark["cold_s"]
    requests = 2 * trials + 2 * 24 * 3 + 2
    # Doubling source emissions can double search work. This forecast also charges
    # the original arm full cost even though its native lists and scores are saved.
    seconds = 2 * cold + 1.35 * requests * max(2 * p95, pilot["pilot_seconds"] / pilot["requests"]) + 90
    result = {"strong_pilot_seconds": pilot["pilot_seconds"], "strong_pilot_requests": pilot["requests"],
              "measured_expanded_p95_s": p95, "requests": requests, "safety_margin": 1.35,
              "forecast_s": seconds, "app_budget_s": args.max_seconds, "admitted": seconds < args.max_seconds}
    write_json(args.output / "admission.json", result)
    if not result["admitted"]:
        raise ValueError("Measured forecast exceeds the application budget")


def run(args):
    cluster_guard()
    if "A100" not in torch.cuda.get_device_name() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Expected A100 BF16 CUDA")
    torch.set_num_threads(4)
    config = load_config(args.config)
    records = load_json(config["data_manifest"])
    split = load_split_manifest(config["split_manifest"], records)
    tune = _selection(_by_id(attach_corpus_metadata(records, config["corpus_metadata"]), "trial"), split, "dev_tune")
    ids = {r["trial_id"] for r in tune}
    cache = _by_id(read_jsonl(args.trained / "predictions.jsonl"), "two-rate emissions")
    old = _by_id(read_jsonl(args.strong / "predictions.jsonl"), "two-rate native shortlist")
    if set(cache) != ids or set(old) != ids:
        raise ValueError("Trained and language-model caches must cover exactly dev_tune")
    reg = load_json(args.registration)
    edits = load_json(args.proposal_config)
    if (reg["native_limit"], reg["proposal_limit"], reg["weight"]) != (25, 25, 1.25):
        raise ValueError("Changed fixed word rule")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "logits/merge_agree").mkdir(parents=True)
    admission(args, len(tune))
    cfg = {"checkpoint": DELTA, "condition": "two_rate", "methods": METHODS, "score_split": "dev_tune",
           "split_hash": split["manifest_hash"], "word_rule": reg, "proposal_rules": edits,
           "beam": 250, "nbest": args.nbest, "cache_note": "Retain raw trained source and save every transformed array",
           "timing": "Three alternating passes over 24 length-stratified trials from HDF5; fresh encoder, candidates and LM scores on every request"}
    write_json(out / "config.resolved.yaml", cfg)
    hashes = {name:sha256(path) for name,path in {
        "source": __file__, "registration": args.registration, "proposal_config": args.proposal_config,
        "native_lm_predictions": args.strong / "predictions.jsonl", "trained_predictions": args.trained / "predictions.jsonl",
        "prior_timing": args.prior_timing / "metrics.json", "strong_pilot": args.strong / "admission.json",
        "arpa": args.arpa, "lm": args.kenlm_model, "lexicon": args.lexicon, "counts": args.unigrams}.items()}
    provenance = {"hashes": hashes, "encoder_delta": DELTA, "parent": config["checkpoint_hash"]}
    write_json(out / "provenance.json", provenance)
    totals, corpora, sessions = defaultdict(Counter), defaultdict(lambda:defaultdict(Counter)), defaultdict(lambda:defaultdict(Counter))
    paired, denominator, groups, expected = defaultdict(list), [], [], {}
    frame_sums, scope_times = Counter(), defaultdict(list)
    timing, statuses = defaultdict(list), {"status": "started", "trials_written":0}
    sequence_hash = {method:data_hash() for method in METHODS}
    started = perf_counter()
    try:
        with ExitStack() as stack:
            monitor = stack.enter_context(ResourceMonitor(out / "resources.json", max_seconds=args.max_seconds))
            net = load_trained(config, args.trained)
            scorer = DistilGPT2Scorer(args.distil, device="cuda:0", dtype=torch.bfloat16)
            if (scorer.metadata["revision"] != reg["model_revision"] or
                    scorer.metadata["asset_manifest_sha256"] != load_json(args.strong / "provenance.json")["scorer_assets"]):
                raise ValueError("Teacher-forced LM identity differs")
            lm=Trigram(args.arpa)
            unigrams=load_json(args.unigrams)
            if unigrams["source_split"]!="train":
                raise ValueError("Unexpected unigram training split")
            lex=Neighbors(args.lexicon,unigrams["counts"],edits)
            decode,setup=stack.enter_context(native_decoder(args,250,out/"decoder.log"))
            files=[stack.enter_context((out/name).open("x",encoding="utf-8",buffering=1))
                   for name in ("predictions.jsonl","candidate-scores.jsonl","timing.jsonl")]
            torch.cuda.synchronize()
            statuses["cold_models_lexicon_and_worker_s"]=perf_counter()-started
            for record in tune:
                monitor.check()
                trial=record["trial_id"]
                source=args.trained / cache[trial]["logits"]
                x=np.load(source,allow_pickle=False)
                if (x.shape!=(geometry("two_rate",record["input_length"]),41) or x.dtype!=np.float32
                        or not np.isfinite(x).all()):
                    raise ValueError(f"Invalid trained emissions on {trial}")
                cached=sorted(old[trial]["results"]["two_rate"]["scores"],key=lambda c:c["source_rank"])
                shared={c["text"]:(c["neural_log_likelihood"],c["token_count"]) for c in cached}
                stored={}; pools={}
                for method in METHODS:
                    t0=perf_counter()
                    z=transform(x,method)
                    path=source
                    if method=="merge_agree":
                        path=out/"logits/merge_agree"/f"{record['official_order_index']:05d}.npy"
                        np.save(path,z,allow_pickle=False)
                    sequence_hash[method].update(trial.encode()+b"\0"+data_hash(z.tobytes()).digest())
                    if method=="native":
                        choices=cached
                    else:
                        choices=decode(trial,z)["candidates"]
                    _,pool,_=assemble(choices,z,lex,lm,edits,reg,archived=method=="native")
                    text,new=rank(pool,scorer,reg,shared)
                    pools[method]=list(pool)
                    stored[method]={"text":text,"status":"ok" if text else "empty","candidates":len(pool),
                                    "phone_ids":greedy_ctc(z.argmax(1).tolist()),"frames":len(z),
                                    "emissions":str(path),"new_likelihoods":new,"seconds":perf_counter()-t0}
                    files[1].write(json.dumps({"trial_id":trial,"method":method,"scores":[
                        {"text":t,"ctc_log_mass":ac[1],"neural_log_likelihood":shared[t][0],"token_count":shared[t][1]}
                        for t,ac in pool.items()]},allow_nan=False)+"\n")
                reference=reference_only(record)
                words=normalize_text(reference["reference_text"]).split()
                denominator.append(len(words)); groups.append((record["session_id"],record["block_id"]))
                for method,v in stored.items():
                    word=edit_counts(words,normalize_text(v["text"]).split())
                    phone=edit_counts(reference["phoneme_ids"],v["phone_ids"])
                    oracle=min((edit_counts(words,normalize_text(t).split())["errors"] for t in pools[method]),default=len(words))
                    counts={"trials":1,"word_errors":word["errors"],"words":len(words),"phone_errors":phone["errors"],
                            "phones":len(reference["phoneme_ids"]),"oracle":oracle,"empty":int(not v["text"]),
                            "candidates":v["candidates"],"new_likelihoods":v["new_likelihoods"],"frames":v["frames"],
                            **{k:word[k] for k in ("insertions","deletions","substitutions")}}
                    for target in (totals[method],corpora[record.get("corpus_id") or "unknown"][method],sessions[record["session_id"]][method]):
                        target.update(counts)
                    v["counts"]=counts
                    paired[method].append(word["errors"])
                if stored["native"]["phone_ids"]!=stored["merge_agree"]["phone_ids"]:
                    raise ValueError("Greedy phone sequence changed under agreement-only merging")
                expected[trial]={m:v["text"] for m,v in stored.items()}
                files[0].write(json.dumps({"trial_id":trial,"results":stored},allow_nan=False)+"\n")
                statuses["trials_written"]+=1
            statuses["cached_evaluation_s"]=perf_counter()-started-statuses["cold_models_lexicon_and_worker_s"]
            ordered=sorted(tune,key=lambda r:(r["input_length"],r["trial_id"]))
            cohort=[ordered[i] for i in np.linspace(0,len(ordered)-1,24,dtype=int)]
            with torch.inference_mode():
                for sweep in range(3):
                    order=list(cohort)
                    np.random.default_rng(40+sweep).shuffle(order)
                    for i,record in enumerate(([order[0]] if not sweep else [])+order):
                        for method in (METHODS if (sweep+i)%2==0 else METHODS[::-1]):
                            monitor.check()
                            torch.cuda.synchronize()
                            t0=perf_counter()
                            features=read_trial(record,include_labels=False)["features"]
                            logits=net(torch.from_numpy(features).cuda()[None],record["day_index"])[0].float().cpu().numpy()
                            z=transform(logits,method)
                            choices=decode(record["trial_id"],z)["candidates"]
                            _,pool,_=assemble(choices,z,lex,lm,edits,reg)
                            text,_=rank(pool,scorer,reg,{})
                            torch.cuda.synchronize()
                            sec=perf_counter()-t0
                            entry={"method":method,"trial_id":record["trial_id"],"pass":sweep,"warmup":sweep==0 and i==0,
                                   "text":text,"elapsed_s":sec,"frames":len(z),"raw_frames":record["input_length"],
                                   "rtf":sec/(record["input_length"]*.020),"match":text==expected[record["trial_id"]][method]}
                            files[2].write(json.dumps(entry)+"\n")
                            if not entry["warmup"]:
                                timing[method].append(entry)
            monitor.sample(); monitor.check()
        if load_json(out/"resources.json")["outcome"]!="complete":
            raise RuntimeError("Resource cap reached")
        statuses["status"]="complete"
    except BaseException as error:
        statuses.update(status="incomplete",error=repr(error))
        raise
    finally:
        result={**statuses,"counts":totals,"by_corpus":corpora,"by_session":sessions,
                "emission_sha256":{m:h.hexdigest() for m,h in sequence_hash.items()}}
        if statuses["status"]=="complete":
            a,b=np.array(paired["native"]),np.array(paired["merge_agree"])
            result["paired"]={"helped":int((b<a).sum()),"harmed":int((b>a).sum()),
                              **_bootstrap(a,b,np.array(denominator),groups,np.random.default_rng(20261127),2000)}
            result["timing"]={}
            for m,rows in timing.items():
                secs=np.array([r["elapsed_s"] for r in rows]); raw=np.array([r["raw_frames"]*.020 for r in rows])
                edges=np.percentile(raw,[25,50,75]); quartile=np.searchsorted(edges,raw,side="right")
                result["timing"][m]={"requests":len(rows),"seconds":distribution(secs),"rtf":distribution(secs/raw),
                                     "pooled_rtf":float(secs.sum()/raw.sum()),"requests_per_s":float(len(secs)/secs.sum()),
                                     "empty":sum(not r["text"] for r in rows),"changed":sum(not r["match"] for r in rows),
                                     "length_edges_s":edges.tolist(),"length_quartiles":[distribution(secs[quartile==q]) for q in range(4)]}
        write_json(out/"metrics.json",result)
        (out/"report.md").write_text("# Trained two-rate decoder\n\n"+statuses["status"]+".\n\n"+
            "\n".join(f"{m}: {v['word_errors']}/{v['words']} word edits; {v['phone_errors']}/{v['phones']} phone edits."
                      for m,v in totals.items())+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("config","trained","strong","prior-timing","registration","proposal-config","kenlm-model","arpa",
                "lexicon","unigrams","distil","decoder-python","decoder-module-path","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--max-seconds",type=int,default=2550)
    p.set_defaults(nbest=50,lm_weight=2.,beam_threshold=25.,start_timeout=120.,request_timeout=30.)
    run(p.parse_args())
