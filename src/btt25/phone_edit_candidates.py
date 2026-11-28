"""Propose words within one phone edit of the GRU's SIL-delimited output."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from math import log
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from btt25.assets import sha256
from btt25.benchmark import cached_encoder
from btt25.ctc import ctc_minimum_frames, greedy_ctc
from btt25.data import TOKENS, attach_corpus_metadata, load_split_manifest, reference_only, write_json
from btt25.evaluation import _by_id, _selection, edit_counts, normalize_text
from btt25.inference import load_config, load_json
from btt25.phone_robustness import path_scores
from btt25.reference_phone_diagnostic import reference_emissions
from btt25.resources import ResourceMonitor


class Trigram:
    """ARPA trigram lookup. Store probabilities and backoff weights in natural logs."""
    def __init__(self, path):
        self.prob, self.backoff = {}, {}
        order = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("\\"):
                order = int(line[1]) if line.endswith("-grams:") else None
            elif order and line.strip():
                row = line.split()
                key = tuple(row[1:1 + order])
                self.prob[key] = float(row[0]) * log(10)
                self.backoff[key] = float(row[-1]) * log(10) if len(row) == order + 2 else 0.
        if ("<unk>",) not in self.prob or ("</s>",) not in self.prob:
            raise ValueError("Missing ARPA sentinels")

    @lru_cache(maxsize=100_000)
    def step(self, context, word):
        if (word,) not in self.prob:
            word = "<unk>"
        key = context + (word,)
        score = 0.
        while key not in self.prob:
            score += self.backoff.get(key[:-1], 0.)
            key = key[1:]
        return (context + (word,))[-2:], score + self.prob[key]

    def sentence(self, words):
        context, score = ("<s>",), 0.
        for word in [*words, "</s>"]:
            context, contribution = self.step(context, word)
            score += contribution
        return score


class Neighbors:
    def __init__(self, path, counts, config):
        ids = {name: i for i, name in enumerate(TOKENS)}
        self.lex, self.deleted = defaultdict(set), defaultdict(set)
        self.pairs = {frozenset((ids[a], ids[b])) for a,b in
                      [("P","B"),("T","D"),("K","G"),("S","Z"),("SH","ZH"),("F","V"),("TH","DH"),("CH","JH")]}
        self.counts, self.config = counts, config
        for row in path.read_text(encoding="utf-8").splitlines():
            if not row.strip():
                continue
            word,*phones = row.split()
            self.lex[tuple(ids[p] for p in phones)].add(word)
        for seq in self.lex:
            for i in range(len(seq)):
                self.deleted[seq[:i] + seq[i+1:]].add(seq)

    @lru_cache(maxsize=8192)
    def lookup(self, segment):
        seqs = {segment} | self.deleted.get(segment, set())
        for i in range(len(segment)):
            shortened = segment[:i] + segment[i+1:]
            seqs.add(shortened)
            seqs.update(self.deleted.get(shortened, ()))
        options = []
        for seq in seqs:
            if len(seq) == len(segment):
                mismatches = [(a,b) for a,b in zip(seq,segment) if a!=b]
                if len(mismatches) > 1:
                    continue
                distance = 0. if not mismatches else (
                    self.config["voicing_substitution_cost"] if frozenset(mismatches[0]) in self.pairs
                    else self.config["other_edit_cost"])
            else:
                # Unequal lengths enter through exactly one dictionary deletion.
                distance = self.config["other_edit_cost"]
            for word in self.lex.get(seq, ()):
                options.append((distance, -self.counts.get(word, 0), word, seq))
        options.sort()
        return options[:self.config["word_alternatives"]]


def proposals(phones, index, lm, config):
    # Keep all SILs in the CTC string, including leading and trailing boundaries.
    chunks = []
    for phone in phones:
        if phone == config["sil_id"]:
            chunks.append(None)
        elif not chunks or chunks[-1] is None:
            chunks.append([phone])
        else:
            chunks[-1].append(phone)
    beam = [(0., ("<s>",), (), ())]
    for chunk in chunks:
        if chunk is None:
            beam = [(score,ctx,words,path+(config["sil_id"],)) for score,ctx,words,path in beam]
            continue
        expanded = {}
        for cost, _, word, word_path in index.lookup(tuple(chunk)):
            for score,ctx,words,path in beam:
                next_ctx, lp = lm.step(ctx, word)
                item = (score+lp-config["proposal_edit_penalty_nats"]*cost, next_ctx, words+(word,), path+word_path)
                if item[2] not in expanded or item[0] > expanded[item[2]][0]:
                    expanded[item[2]] = item
        if not expanded:
            return []
        beam = sorted(expanded.values(),key=lambda x:(-x[0],x[2],x[3]))[:config["beam_width"]]
    beam.sort(key=lambda x:(-(x[0]+lm.step(x[1],"</s>")[1]), x[2],x[3]))
    return [(" ".join(words), path) for _,_,words,path in beam[:config["new_sentence_limit"]] if words]


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    cfg = load_json(args.registration)
    config = load_config(args.config)
    records = load_json(config["data_manifest"])
    split = load_split_manifest(config["split_manifest"],records)
    selected = _selection(_by_id(attach_corpus_metadata(records,config["corpus_metadata"]),"trial"),split,"dev_tune")
    trials = {r["trial_id"]:r for r in selected}
    encode = cached_encoder(args.cache,config,records,split)
    write_json(args.output/"config.resolved.yaml", cfg)
    provenance = {key:sha256(path) for key,path in {"registration":args.registration,"source":__file__,
                  "native":args.candidates,"lexicon":args.lexicon,"arpa":args.arpa,"unigrams":args.unigrams}.items()}
    lm_manifest = load_json(args.arpa.with_name("manifest.json"))
    if lm_manifest["source_split"] != "official train only" or lm_manifest["arpa_sha256"] != provenance["arpa"]:
        raise ValueError("Expected verified training-only LM")
    unigrams = load_json(args.unigrams)
    if unigrams["source_split"] != "train":
        raise ValueError("Unexpected word-count provenance")
    write_json(args.output/"provenance.json",{"files":provenance,"split_hash":split["manifest_hash"],
                                              "training_id_hash":lm_manifest["training_id_hash"]})
    summary = defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    per_id = defaultdict(dict)
    torch.set_num_threads(1)
    with ResourceMonitor(args.output/"resources.json",max_seconds=700) as monitor, \
         (args.output/"predictions.jsonl").open("x",encoding="utf-8",buffering=1) as stream, \
         args.candidates.open(encoding="utf-8") as packets:
        lm = Trigram(args.arpa)
        index = Neighbors(args.lexicon,unigrams["counts"],cfg)
        for line in packets:
            packet = json.loads(line)
            kind,trial = packet["input"],packet["trial_id"]
            if kind not in cfg["inputs"]:
                continue
            monitor.check()
            if trial not in trials or trial in per_id[kind]:
                raise ValueError("Unexpected or duplicate candidate ID")
            record = trials[trial]
            label = reference_only(record)
            reference = normalize_text(label["reference_text"]).split()
            logits,_ = encode(record)
            if kind == "reference_spikes":
                logits = reference_emissions(label["phoneme_ids"],len(logits),"spike")
            phones = greedy_ctc(logits.argmax(1).tolist())
            start = perf_counter()
            generated = proposals(phones,index,lm,cfg)
            feasible = [(text,path) for text,path in generated if ctc_minimum_frames(path) <= len(logits)]
            weighted = cfg["lm_weight_nats"]
            native = {c["text"]: c["ctc_log_mass"] + weighted * lm.sentence(c["text"].split())
                      for c in packet["candidates"]}
            arpa_diffs = [abs(weighted*lm.sentence(c["text"].split())-c["kenlm_contribution"]) for c in packet["candidates"]]
            combined = dict(native)
            if feasible:
                scores = path_scores(logits,[path for _,path in feasible])
                for (text,path),score in zip(feasible,scores):
                    value = float(score) + weighted*lm.sentence(text.split())
                    combined[text] = max(combined.get(text,-np.inf),value)
            elapsed = perf_counter()-start
            values, edits_by_rule = {}, {}
            for name, pool in [("native_list",native),("union",combined)]:
                best = max(pool,key=pool.get,default="")
                edits = edit_counts(reference,normalize_text(best).split())
                oracle = min((edit_counts(reference,normalize_text(text).split())["errors"] for text in pool),default=len(reference))
                counts = {**edits,"oracle_edits":oracle,"trials":1,"empty":int(not best),"candidates":len(pool),
                          "trials_with_new_text":int(bool(set(combined)-set(native))),"infeasible_paths":len(generated)-len(feasible)}
                for area in ["overall","corpus/"+(record.get("corpus_id") or "unknown"),"session/"+record["session_id"]]:
                    summary[kind][area][name].update(counts)
                edits_by_rule[name] = (edits["errors"],oracle)
                values[name] = {"text":best, "edits":edits,"oracle":oracle,"candidate_count":len(pool)}
            per_id[kind][trial] = (len(reference),edits_by_rule)
            stream.write(json.dumps({"trial_id":trial,"input":kind,"predictions":values,
                                     "proposed_texts":[t for t,p in feasible],
                                     "seconds":elapsed,"max_abs_native_lm_disagreement_nats":max(arpa_diffs,default=0.)})+"\n")
    for kind in cfg["inputs"]:
        if per_id[kind].keys() != trials.keys():
            raise ValueError("Incomplete trial coverage")
        dist = np.random.default_rng(20261127).integers(len(trials),size=(2000,len(trials)))
        rows = [per_id[kind][t] for t in trials]
        lengths = np.array([r[0] for r in rows])
        for j,metric in enumerate(("selected","oracle")):
            delta = np.array([r[1]["union"][j]-r[1]["native_list"][j] for r in rows])
            summary[kind]["overall"]["union"][metric+"_paired95"] = np.quantile(
                delta[dist].sum(1)/lengths[dist].sum(1),[.025,.975]).tolist()
    write_json(args.output/"metrics.json",{"status":"complete","split":"dev_tune","results":summary})
    lines=["# One-edit word proposals","","| Input | Candidate list | Word edits | Oracle edits | Reference words |",
           "|---|---|---:|---:|---:|"]
    for kind, areas in summary.items():
        for name, result in areas["overall"].items():
            lines.append(f"| {kind} | {name} | {result['errors']} | {result['oracle_edits']} | {result['reference_length']} |")
    (args.output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("config","cache","registration","arpa","candidates","unigrams","lexicon","output"):
        p.add_argument("--"+k,type=Path,required=True)
    run(p.parse_args())
