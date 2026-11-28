"""Paired dev_tune experiment on phone uncertainty, alignment mass and word decisions."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from btt25.assets import sha256
from btt25.benchmark import SERVER, cached_encoder, native_decoder
from btt25.ctc import greedy_ctc
from btt25.data import attach_corpus_metadata, load_split_manifest, reference_only, write_json
from btt25.evaluation import _by_id, _selection, edit_counts, normalize_text
from btt25.inference import load_config, load_json, read_jsonl
from btt25.reference_phone_diagnostic import reference_emissions
from btt25.resources import ResourceMonitor


INPUTS = ("neural", "neural_hard_frames", "neural_spikes", "reference_spikes")
RULES = ("native", "native_mbr", "alignment_sum", "alignment_sum_mbr")


def path_scores(logits, paths):
    """Log mass of all CTC alignments to each candidate's single retained phone string."""
    unique = list(dict.fromkeys(tuple(p) for p in paths))
    if any(not 0 < p < logits.shape[1] for seq in unique for p in seq):
        raise ValueError("Invalid nonblank phone in native candidate")
    logp = torch.from_numpy(logits).double().log_softmax(-1).unsqueeze(1)
    lengths = torch.tensor([len(seq) for seq in unique], dtype=torch.long)
    targets = torch.tensor([p for seq in unique for p in seq], dtype=torch.long)
    losses = torch.nn.functional.ctc_loss(
        logp.expand(-1, len(unique), -1).contiguous(), targets,
        torch.full_like(lengths, logits.shape[0]), lengths, reduction="none", zero_infinity=False,
    )
    if not torch.isfinite(losses).all():
        raise ValueError("Native phone path has no finite CTC alignment")
    scores = dict(zip(unique, (-losses).tolist()))
    return np.array([scores[tuple(p)] for p in paths])


def edit_distances(words):
    """Myers bit-vector edit counts over word tokens. No limit on sentence length."""
    distances = np.zeros((len(words), len(words)), dtype=np.int32)
    for i, pattern in enumerate(words):
        length = len(pattern)
        if not length:
            distances[i, :i] = distances[:i, i] = [len(seq) for seq in words[:i]]
            continue
        masks = {}
        for k, token in enumerate(pattern):
            masks[token] = masks.get(token, 0) | (1 << k)
        for j, seq in enumerate(words[:i]):
            pos, neg, cost, last_bit = -1, 0, length, 1 << (length - 1)
            for token in seq:
                equal = masks.get(token, 0)
                vertical = equal | neg
                horizontal = (((equal & pos) + pos) ^ pos) | equal
                hp, hn = neg | ~(horizontal | pos), pos & horizontal
                cost += bool(hp & last_bit) - bool(hn & last_bit)
                hp, hn = (hp << 1) | 1, hn << 1
                pos, neg = hn | ~(vertical | hp), hp & vertical
            distances[i, j] = distances[j, i] = cost
    return distances


def mbr(scores, distances, temperature):
    weights = np.exp((scores - scores.max()) / temperature)
    # The symmetric loss is the count of word edits, matching the corpus WER numerator.
    risks = distances @ (weights / weights.sum())
    return int(np.lexsort((np.arange(len(scores)), -scores, risks))[0])


def select(candidates, logits, temperature):
    if not candidates:
        return {k: "" for k in RULES}, []
    words = [normalize_text(c["text"]).split() for c in candidates]
    dist = edit_distances(words)
    native = np.array([c["component_scores"]["rank_score"] for c in candidates])
    acoustic = np.array([c["component_scores"]["acoustic"] for c in candidates])
    mass = path_scores(logits, [c["phoneme_ids"] for c in candidates])
    aligned = mass + native - acoustic
    ranks = (0, mbr(native, dist, temperature), int(aligned.argmax()), mbr(aligned, dist, temperature))
    choices = {key: candidates[i]["text"] for key, i in zip(RULES, ranks)}
    evidence = [{"text": c["text"], "phones": c["phoneme_ids"], "native": float(n),
                 "path_acoustic": float(a), "ctc_log_mass": float(m),
                 "kenlm_contribution": c["component_scores"]["kenlm_contribution"]}
                for c, n, a, m in zip(candidates, native, acoustic, mass)]
    return choices, evidence


def summarize(rows):
    totals = {}
    for field in ("overall", "by_session", "by_corpus"):
        groups = defaultdict(lambda: defaultdict(Counter))
        for r in rows:
            group = {"overall": "all", "by_session": r["session"], "by_corpus": r["corpus"]}[field]
            for rule, text in r["predictions"].items():
                counts = groups[group][rule]
                edits = r["edits"][rule]
                counts.update(edits)
                counts.update(trials=1, empty=int(not text), candidates=r["candidate_count"],
                              oracle_errors=r["oracle"], extended_frames=int(r["extended_frames"] > 0),
                              improved_trials=int(edits["errors"] < r["edits"]["native"]["errors"]),
                              harmed_trials=int(edits["errors"] > r["edits"]["native"]["errors"]))
        totals[field] = groups["all"] if field == "overall" else groups
    return totals


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    records = load_json(config["data_manifest"])
    split = load_split_manifest(config["split_manifest"], records)
    selected = _selection(_by_id(attach_corpus_metadata(records, config["corpus_metadata"]), "trial"),
                          split, "dev_tune")
    encode = cached_encoder(args.cache, config, records, split)
    saved = _by_id(read_jsonl(args.control), "saved text")
    control_candidates = args.control.with_name("candidates.jsonl")
    archived = _by_id(read_jsonl(control_candidates), "saved candidate packet")
    torch.set_num_threads(1)
    write_json(args.output / "config.resolved.yaml", {**{k: str(v) for k, v in vars(args).items()},
               "inputs": INPUTS, "rules": RULES, "split": "dev_tune", "reference_phones": "diagnostic only",
               "search": "native CTC beam, training-transcript KenLM 3-gram, no insertion bonus",
               "scope": "one pass, cached GRU logits; CPU decoder and CPU float64 CTC rescoring"})
    files = {"source": __file__, "config": args.config, "splits": config["split_manifest"],
             "data_manifest": config["data_manifest"], "lexicon": args.lexicon, "kenlm": args.kenlm_model,
             "control": args.control, "control_candidates": control_candidates,
             "cache": args.cache / "cache_manifest.json", "server": SERVER,
             "worker": SERVER.with_name("decode_worker.py")}
    write_json(args.output / "provenance.json", {"files": {k: sha256(v) for k, v in files.items()},
               "split_hash": split["manifest_hash"], "torch": torch.__version__, "numpy": np.__version__})
    by_input = defaultdict(list)
    with ExitStack() as stack:
        monitor = stack.enter_context(ResourceMonitor(args.output / "resources.json", max_seconds=1800))
        decode, backend = stack.enter_context(native_decoder(args, args.beam, args.output / "decoder.stderr.log"))
        streams = [stack.enter_context((args.output / n).open("x", encoding="utf-8", buffering=1))
                   for n in ("predictions.jsonl", "candidates.jsonl")]
        for record in selected:
            monitor.check()
            trial = record["trial_id"]
            logits, _ = encode(record)
            reference = reference_only(record)
            ref_words = normalize_text(reference["reference_text"]).split()
            target = reference["phoneme_ids"]
            frames = logits.argmax(axis=1)
            phones = greedy_ctc(frames.tolist())
            hard = np.full_like(logits, -100)
            hard[np.arange(len(logits)), frames] = 0
            inputs = [logits, hard, reference_emissions(phones, len(logits), "spike") if phones else hard,
                      reference_emissions(target, len(logits), "spike")]
            for kind, values in zip(INPUTS, inputs):
                start = perf_counter()
                candidates = decode(trial, values)["candidates"]
                search_s = perf_counter() - start
                replay = None
                if kind == "neural":
                    packet = archived[trial]
                    if packet["logit_sha256"] != saved[trial]["logit_sha256"]:
                        raise ValueError("Archived control files disagree on their emission hash")
                    actual = candidates[0]["text"] if candidates else ""
                    expected = saved[trial]["text"] or ""
                    replay = {"text": actual, "matches_archived": actual == expected,
                              "candidate_set_agrees": {c["text"] for c in candidates} ==
                                                       {c["text"] for c in packet["candidates"]}}
                    candidates = packet["candidates"]
                start = perf_counter()
                choices, evidence = select(candidates, values, args.temperature)
                rank_s = perf_counter() - start
                if kind == "neural" and choices["native"] != (saved[trial]["text"] or ""):
                    raise ValueError(f"Archived candidates disagree with frozen prediction on {trial}")
                edits = {rule: edit_counts(ref_words, normalize_text(text).split()) for rule, text in choices.items()}
                all_edits = edit_distances([ref_words] + [normalize_text(c["text"]).split() for c in candidates])[0, 1:] if candidates else []
                row = {"trial_id": trial, "input": kind, "session": record["session_id"],
                       "corpus": record.get("corpus_id") or "unknown", "predictions": choices, "edits": edits,
                       "status": "ok" if candidates else "empty", "candidate_count": len(candidates),
                       "oracle": int(min(all_edits, default=len(ref_words))), "extended_frames": len(values)-len(logits),
                       "reference_phones": len(target), "phone_errors": edit_counts(target, greedy_ctc(values.argmax(1).tolist()))["errors"],
                       "search_seconds": search_s, "all_four_selections_seconds": rank_s,
                       "native_replay": replay}
                by_input[kind].append(row)
                streams[0].write(json.dumps(row, allow_nan=False) + "\n")
                streams[1].write(json.dumps({"trial_id": trial, "input": kind, "candidates": evidence}, allow_nan=False) + "\n")
    summary = {k: summarize(rows) for k, rows in by_input.items()}
    rng = np.random.default_rng(20261127)
    sampling = rng.integers(0, len(selected), (2000, len(selected)))
    paired = {}
    reference_words = np.array([r["edits"]["native"]["reference_length"] for r in by_input["neural"]])
    for kind, rows in by_input.items():
        for rule in RULES:
            delta = np.array([r["edits"][rule]["errors"] - n["edits"]["native"]["errors"]
                              for r, n in zip(rows, by_input["neural"])])
            draws = delta[sampling].sum(1) / reference_words[sampling].sum(1)
            paired[f"{kind}/{rule}"] = {"delta_errors_vs_normal_native": int(delta.sum()),
                                         "paired_trial_bootstrap_95": np.quantile(draws,[.025,.975]).tolist()}
        summary[kind]["phones"] = {"errors": sum(r["phone_errors"] for r in rows),
                                    "reference_length": sum(r["reference_phones"] for r in rows)}
    write_json(args.output / "metrics.json", {"status": "complete", "score_split": "dev_tune",
               "trials": len(selected), "results": summary, "paired": paired, "decoder": backend,
               "normal_replay": {"top_text_disagreements": sum(not r["native_replay"]["matches_archived"]
                                    for r in by_input["neural"]),
                                 "candidate_set_disagreements": sum(not r["native_replay"]["candidate_set_agrees"]
                                    for r in by_input["neural"])}})
    lines = ["# Phone evidence and word decisions", "", "| Input | Selection | Word errors | Words | WER |",
             "| --- | --- | ---: | ---: | ---: |"]
    for kind in INPUTS:
        for rule, count in summary[kind]["overall"].items():
            lines.append(f"| {kind} | {rule} | {count['errors']} | {count['reference_length']} | {count['errors']/count['reference_length']:.2%} |")
    lines += ["", "The reference-spike input uses held-out tuning labels and constructed timing. Its WER is diagnostic."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "cache", "control", "lexicon", "kenlm-model", "decoder-python", "decoder-module-path", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--beam", type=int, default=250)
    p.add_argument("--nbest", type=int, default=50)
    p.add_argument("--beam-threshold", type=float, default=25.)
    p.add_argument("--lm-weight", type=float, default=2.)
    p.add_argument("--temperature", type=float, default=2.)
    p.add_argument("--start-timeout", type=float, default=90.)
    p.add_argument("--request-timeout", type=float, default=30.)
    run(p.parse_args())
