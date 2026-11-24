"""Compute tuning-only candidate coverage from saved, reference-free output."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .data import attach_corpus_metadata, load_split_manifest, reference_only, write_json
from .evaluation import _by_id, _selection, edit_counts, normalize_text
from .inference import sha256
from .resources import ResourceMonitor


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def words(text):
    return tuple(normalize_text(text).split())


def diagnostics(records, split, predictions, packets, ks, lexicon=None, lm_vocab=None, cache_hash=None):
    record_map = _by_id(records, "record")
    selected = _selection(record_map, split, "dev_tune")
    allowed = {r["trial_id"] for r in records if r["split"] == "val"}
    prediction_map = _by_id(predictions, "prediction")
    packet_map = _by_id(packets, "candidate packet")
    if (prediction_map.keys() | packet_map.keys()) - allowed:
        raise ValueError("Candidate diagnostic inputs must be validation predictions")
    totals = Counter()
    per_session, per_corpus = {}, {}
    counts = Counter()
    examples = []
    for record in selected:
        trial = record["trial_id"]
        prediction = prediction_map.get(trial, {})
        packet = packet_map.get(trial, {})
        if cache_hash and (any(row and row.get("decoder_cache_hash") != cache_hash for row in (prediction, packet))
                           or (prediction and packet and prediction.get("logit_sha256") != packet.get("logit_sha256"))):
            raise ValueError("Predictions and candidates must belong to the same logit and decoder cache")
        ok = prediction.get("status") == "ok"
        hypotheses = list(dict.fromkeys(words(row["text"]) for row in
                                       packet.get("candidates", []))) if ok else []
        selected_words = words(prediction["text"]) if ok else ()
        if ok and (not hypotheses or selected_words != hypotheses[0]):
            raise ValueError("Selected prediction does not match saved first candidate")
        ref = words(reference_only(record)["reference_text"])
        selected_edits = edit_counts(ref, selected_words)
        edits = [edit_counts(ref, hyp)["errors"] for hyp in hypotheses[:max(ks)]]
        row = {"reference_words": len(ref), "trials": 1, "successful": int(ok),
               "selected_errors": selected_edits["errors"]}
        for field in ("insertions", "deletions", "substitutions"):
            row[field] = selected_edits[field]
        for k in ks:
            row[f"oracle_{k}_errors"] = min(edits[:k], default=len(ref))
            row[f"exact_at_{k}"] = int(0 in edits[:k])
        for name, vocab in (("lexicon", lexicon), ("lm", lm_vocab)):
            if vocab is not None:
                row[f"{name}_oov_tokens"] = sum(word not in vocab for word in ref)
                row[f"{name}_oov_trials"] = int(any(word not in vocab for word in ref))
        counts[len(hypotheses)] += 1
        totals.update(row)
        per_session.setdefault(record["session_id"], Counter()).update(row)
        per_corpus.setdefault(record.get("corpus_id") or "unclassified", Counter()).update(row)
        examples.append({"trial_id": trial, "text": " ".join(selected_words),
                         "status": prediction.get("status", "missing"), "unique_candidates": len(hypotheses)})

    def summarize(counts):
        denominator = counts["reference_words"]
        return {**counts, "selected_wer": counts["selected_errors"] / denominator if denominator else None,
                "oracle": {str(k): {"errors": counts[f"oracle_{k}_errors"],
                                    "wer": counts[f"oracle_{k}_errors"] / denominator if denominator else None,
                                    "exact_trials": counts[f"exact_at_{k}"],
                                    "ranking_gap": (counts["selected_errors"] - counts[f"oracle_{k}_errors"])
                                    / denominator if denominator else None} for k in ks}}

    return {**summarize(totals), "score_split": "dev_tune", "split_hash": split["manifest_hash"],
            "candidate_count_histogram": dict(sorted(counts.items())),
            "by_session": {key: summarize(value) for key, value in per_session.items()},
            "by_corpus": {key: summarize(value) for key, value in per_corpus.items()}}, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("registration", "data-manifest", "split-manifest", "predictions", "candidates", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--corpus-metadata", type=Path)
    parser.add_argument("--lexicon", type=Path, help="Effective reachable lexicon, one word per line")
    parser.add_argument("--lm-vocab", type=Path, help="Training vocabulary text or JSON with a counts key")
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    condition, = [arm for arm in registration["conditions"] if arm["id"] == args.condition]
    records = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    if args.corpus_metadata:
        records = attach_corpus_metadata(records, args.corpus_metadata)
    split = load_split_manifest(args.split_manifest, records)
    if split["manifest_hash"] != registration["split_manifest_hash"] or registration["reference_access"] != "dev_tune":
        raise ValueError("Registration must specify the unchanged tuning split")
    lexicon = {line.split()[0] for line in args.lexicon.read_text(encoding="utf-8").splitlines()
               if line.strip()} if args.lexicon else None
    lm_vocab = None
    if args.lm_vocab:
        vocabulary = args.lm_vocab.read_text(encoding="utf-8")
        lm_vocab = set(json.loads(vocabulary)["counts"] if args.lm_vocab.suffix == ".json"
                       else vocabulary.splitlines())
    cache_hash = None
    if condition["id"] != "v2-s0-archived":
        if not args.candidate_cache:
            raise ValueError("New search conditions require a bound decoder cache")
        cache = json.loads(args.candidate_cache.read_text(encoding="utf-8"))
        settings = cache["decoder"]
        expect = {"beam": condition["beam"], "nbest": condition["nbest"], "variant": condition["backend"],
                  "lm_weight": registration["fixed"]["lm_weight_natural_log"],
                  "beam_threshold": registration["fixed"]["beam_threshold"]}
        if any(settings.get(key) != value for key, value in expect.items()):
            raise ValueError("Decoder cache settings disagree with the registered search condition")
        cache_hash = cache["manifest_hash"]
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {key: sha256(path) for key in
              ("registration", "data_manifest", "split_manifest", "predictions", "candidates",
               "corpus_metadata", "lexicon", "lm_vocab", "candidate_cache") if (path := getattr(args, key))}
    write_json(args.output / "config.resolved.yaml", {"condition": condition, "registration": registration})
    write_json(args.output / "provenance.json", {"source_sha256": hashes})
    with ResourceMonitor(args.output / "resources.json", max_seconds=300) as monitor:
        metrics, selected = diagnostics(records, split, rows(args.predictions), rows(args.candidates),
                                        registration["oracle_k"], lexicon, lm_vocab, cache_hash)
        monitor.check()
        write_json(args.output / "metrics.json", metrics)
        with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps(row) + "\n")
    lines = ["# Candidate coverage", "", f"Condition `{condition['id']}`, scored on `dev_tune`.",
             "", f"Selected WER {metrics['selected_wer']:.2%} "
             f"({metrics['selected_errors']}/{metrics['reference_words']})."]
    lines.extend(f"Oracle@{k} WER {value['wer']:.2%} ({value['errors']}/{metrics['reference_words']})."
                 for k, value in metrics["oracle"].items())
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
