"""Apply fixed word-decision rules to previously measured LM scores inside Slurm."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
import os
from pathlib import Path

import numpy as np

from btt25.assets import sha256
from btt25.data import attach_corpus_metadata, load_split_manifest, reference_only, write_json
from btt25.evaluation import _by_id, _selection, edit_counts, normalize_text
from btt25.inference import load_config, load_json
from btt25.resources import ResourceMonitor
from btt25.phone_robustness import edit_distances, mbr


def keep(path, allowed, only_neural=False):
    index = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            trial = row["trial_id"]
            if trial not in allowed or (only_neural and row["input"] != "neural"):
                continue
            if trial in index:
                raise ValueError("Duplicate eligible candidate packet")
            index[trial] = row["candidates"]
    if index.keys() != allowed:
        raise ValueError("Existing cache misses trial IDs")
    return index


def explicit_compact(path, records, references, output, monitor, sampling):
    """Replace all merged native scores with one CTC sum and one explicit lexical prior."""
    rules = ("native", "alignment_sum", "alignment_sum_mbr")
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    edits_by_trial = defaultdict(dict)
    indexed = {r["trial_id"]: r for r in records}
    with path.open(encoding="utf-8") as f, (output/"explicit-compact-predictions.jsonl").open("x",encoding="utf-8",buffering=1) as out:
        for line in f:
            monitor.check()
            packet = json.loads(line)
            trial,kind,candidates = packet["trial_id"],packet["input"],packet["candidates"]
            if trial not in indexed or trial in edits_by_trial[kind]:
                raise ValueError("Unexpected compact comparison trial")
            choices = [""] * len(rules)
            if candidates:
                texts = [c["text"] for c in candidates]
                ac_lm = np.array([c["ctc_log_mass"] + c["kenlm_contribution"] for c in candidates])
                distances = edit_distances([normalize_text(t).split() for t in texts])
                choices = [texts[0], texts[int(ac_lm.argmax())], texts[mbr(ac_lm, distances, 2.)]]
            record = indexed[trial]
            errors = []
            for rule,text in zip(rules,choices):
                edits = edit_counts(references[trial], normalize_text(text).split())
                errors.append(edits["errors"])
                for area in ("overall", "corpus/" + (record.get("corpus_id") or "unknown"), "session/" + record["session_id"]):
                    groups[kind][area][rule].update({**edits, "trials":1, "empty":int(not text)})
                out.write(json.dumps({"trial_id":trial,"input":kind,"rule":rule,"text":text,"edits":edits})+"\n")
            edits_by_trial[kind][trial] = errors
    ids = list(indexed)
    lengths = np.array([len(references[trial]) for trial in ids])
    for kind, by_id in edits_by_trial.items():
        if by_id.keys() != indexed.keys():
            raise ValueError("Incomplete compact comparison trial set")
        errors = np.array([by_id[trial] for trial in ids])
        for j,rule in enumerate(rules):
            delta = errors[:,j] - errors[:,0]
            summary = groups[kind]["overall"][rule]
            summary["paired_95_vs_own_native"] = np.quantile(delta[sampling].sum(1) / lengths[sampling].sum(1),
                                                             [.025,.975]).tolist()
            summary.update(improved_trials=int((delta < 0).sum()), harmed_trials=int((delta > 0).sum()),
                           delta_edits_vs_own_native=int(delta.sum()))
    write_json(output/"explicit-compact-metrics.json", groups)


def compare(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Score-cache experiment requires a Gilbreth allocation")
    args.output.mkdir(parents=True,exist_ok=False)
    config = load_config(args.config)
    records = load_json(config["data_manifest"])
    split = load_split_manifest(config["split_manifest"],records)
    selected = _selection(_by_id(attach_corpus_metadata(records,config["corpus_metadata"]),"trial"),split,"dev_tune")
    ids = {r["trial_id"] for r in selected}
    sources = dict(args.score)
    if len(sources) != len(args.score):
        raise ValueError("Model names must be unique")
    reference = {r["trial_id"]: normalize_text(reference_only(r)["reference_text"]).split() for r in selected}
    write_json(args.output/"config.resolved.yaml",{"conditions":list(sources),"neural_weight":1.25,
               "risk_temperature":2,"candidate_limit":25,"split":"dev_tune","inputs":"saved likelihoods only",
               "tie_break":"original search rank","bootstrap_unit":"paired trial, seed 20261127, 2000 draws",
               "compact_exact":"stored phone CTC log-mass + explicit weighted KenLM natural log likelihood, zero insertion bonus"})
    write_json(args.output/"provenance.json",{"files":{k:sha256(v) for k,v in
               {**sources,"acoustics":args.acoustics,"source":__file__,"config":args.config}.items()},
               "split_hash":split["manifest_hash"]})
    overall, corpus, sessions = {}, {}, {}
    rng=np.random.default_rng(20261127)
    sampling=rng.integers(0,len(selected),(2000,len(selected)))
    lengths=np.array([len(reference[r["trial_id"]]) for r in selected])
    prior_controls={"distilgpt2":802,"qwen36":840,"qwen38":846}
    with ExitStack() as stack:
        monitor=stack.enter_context(ResourceMonitor(args.output/"resources.json",max_seconds=480))
        stream=stack.enter_context((args.output/"predictions.jsonl").open("x",encoding="utf-8",buffering=1))
        explicit_compact(args.acoustics, selected, reference, args.output, monitor, sampling)
        alignments=keep(args.acoustics,ids,True)
        for model,path in sources.items():
            scored=keep(path,ids)
            outcomes=defaultdict(list)
            summary=defaultdict(Counter)
            groups=defaultdict(lambda:defaultdict(Counter))
            days=defaultdict(lambda:defaultdict(Counter))
            for record in selected:
                monitor.check()
                trial=record["trial_id"]
                candidates=alignments[trial][:25]
                provided={c["text"]:c for c in scored[trial]}
                if len(provided)!=len(scored[trial]) or set(provided)!={c["text"] for c in candidates}:
                    raise ValueError(f"Candidate identities differ on {model} / {trial}")
                rules=("native","native_mbr","alignment_sum","alignment_sum_mbr")
                choices=[""]*4
                if candidates:
                    for c in candidates:
                        s=provided[c["text"]]
                        if s["phoneme_ids"]!=c["phones"] or abs(s["component_scores"]["acoustic"]-c["path_acoustic"])>1e-5:
                            raise ValueError("Mismatched native paths or acoustics")
                    prior=1.25*np.array([provided[c["text"]]["neural_logprob"] for c in candidates])
                    native=prior+np.array([c["path_acoustic"] for c in candidates])
                    exact=prior+np.array([c["ctc_log_mass"] for c in candidates])
                    texts=[c["text"] for c in candidates]
                    dist=edit_distances([normalize_text(t).split() for t in texts])
                    choices=[texts[i] for i in (native.argmax(),mbr(native,dist,2.),exact.argmax(),mbr(exact,dist,2.))]
                for rule,text in zip(rules,choices):
                    edits=edit_counts(reference[trial],normalize_text(text).split())
                    counts={**edits,"trials":1,"empty":int(not text)}
                    summary[rule].update(counts)
                    groups[record.get("corpus_id") or "unknown"][rule].update(counts)
                    days[record["session_id"]][rule].update(counts)
                    outcomes[rule].append(edits["errors"])
                    stream.write(json.dumps({"model":model,"rule":rule,"trial_id":trial,"text":text,"counts":edits,
                                             "status":"ok" if text else "empty"})+"\n")
            if summary["native"]["errors"]!=prior_controls[model]:
                raise ValueError(f"{model} top-1 differs from the archived study")
            for rule,errors in outcomes.items():
                delta=np.array(errors)-np.array(outcomes["native"])
                draws=delta[sampling].sum(1)/lengths[sampling].sum(1)
                summary[rule].update({"improved_trials":int((delta<0).sum()),"harmed_trials":int((delta>0).sum())})
                summary[rule]["paired_95_vs_own_native"]=np.quantile(draws,[.025,.975]).tolist()
            overall[model],corpus[model],sessions[model]=summary,groups,days
    write_json(args.output/"metrics.json",{"status":"complete","overall":overall,"by_session":sessions,
                                          "by_corpus":corpus,"split":"dev_tune","trials":len(selected)})
    lines=["# Reused language-model scores","","| Prior | Selection | Edits | WER |","|---|---|---:|---:|"]
    for model,rules in overall.items():
        for rule,v in rules.items():
            lines.append(f"| {model} | {rule} | {v['errors']} | {v['errors']/v['reference_length']:.2%} |")
    (args.output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--acoustics",type=Path,required=True)
    p.add_argument("--score",type=lambda s:(s.split("=",1)[0],Path(s.split("=",1)[1])),action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    compare(p.parse_args())
