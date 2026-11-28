"""Summarize saved tuning predictions and align errors to individual phones."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .assets import sha256
from .comparison import _bootstrap
from .data import TOKENS, attach_corpus_metadata, load_split_manifest, reference_only, write_json
from .evaluation import _by_id, _selection, edit_counts, normalize_text
from .inference import load_config, load_json, read_jsonl


def align(reference, hypothesis):
    """Backtrack with the corpus evaluator's diagonal, deletion, insertion tie order."""
    width = len(hypothesis) + 1
    cost, trace = [list(range(width))], [[2] * width]
    for i, target in enumerate(reference, 1):
        previous, current, moves = cost[-1], [i], [1]
        for j, actual in enumerate(hypothesis, 1):
            choices = (previous[j - 1] + (target != actual), previous[j] + 1, current[-1] + 1)
            best = min(choices)
            moves.append(choices.index(best))
            current.append(best)
        cost.append(current)
        trace.append(moves)
    i, j = len(reference), len(hypothesis)
    while i or j:
        move = trace[i][j]
        if move == 0:
            i, j = i - 1, j - 1
            yield reference[i], hypothesis[j]
        elif move == 1:
            i -= 1
            yield reference[i], None
        else:
            j -= 1
            yield None, hypothesis[j]


def summarize(config, paths, output, corpus_metadata=None):
    config = load_config(config)
    records = load_json(config["data_manifest"])
    if corpus_metadata is not None:
        records = attach_corpus_metadata(records, corpus_metadata)
    split = load_split_manifest(config["split_manifest"], records)
    selected = _selection(_by_id(records, "trial"), split, "dev_tune")
    ids = [r["trial_id"] for r in selected]
    membership = [(r["session_id"], r["block_id"]) for r in selected]
    refs = [reference_only(r) for r in selected]
    results = {}
    metrics = ("word_errors", "phone_errors", "words", "phones")
    denominator_words = np.array([len(normalize_text(r["reference_text"]).split()) for r in refs])
    denominator_phones = np.array([len(r["phoneme_ids"]) for r in refs])
    for path in paths:
        name = path.parent.name
        if name in results:
            raise ValueError(f"Duplicate condition name {name}")
        hypotheses = _by_id(read_jsonl(path), "predictions")
        if set(hypotheses) != set(ids):
            raise ValueError(f"{name} has incomplete or unrelated trial IDs")
        all_counts = Counter()
        sessions, corpora, edits = defaultdict(Counter), defaultdict(Counter), defaultdict(list)
        phone_edits, changed = defaultdict(Counter), Counter()
        for r, reference, trial in zip(selected, refs, ids, strict=True):
            hypothesis = hypotheses[trial]
            counts = Counter(hypothesis["counts"])
            phone_errors = 0
            for target, predicted in align(reference["phoneme_ids"], hypothesis["phones"]):
                if target is not None:
                    phone_edits[TOKENS[target]]["reference"] += 1
                    category = "correct" if target == predicted else "deletion" if predicted is None else "substitution"
                    phone_edits[TOKENS[target]][category] += 1
                    if category == "substitution":
                        changed[(TOKENS[target], TOKENS[predicted])] += 1
                else:
                    phone_edits[TOKENS[predicted]]["insertion"] += 1
                phone_errors += target != predicted
            word = edit_counts(normalize_text(reference["reference_text"]).split(),
                               normalize_text(hypothesis["text"]).split())
            if ((counts["phone_errors"], counts["phones"]) != (phone_errors, len(reference["phoneme_ids"]))
                    or (counts["word_errors"], counts["words"]) != (word["errors"], word["reference_length"])):
                raise ValueError(f"Stored counts do not agree for {trial}")
            for key in ("substitutions", "deletions", "insertions"):
                counts[f"word_{key}"] = word[key]
            all_counts.update(counts)
            sessions[r["session_id"]].update(counts)
            corpora[r.get("corpus_id") or "unclassified"].update(counts)
            for key in metrics:
                edits[key].append(counts[key])
        condition = {"counts": all_counts, "by_session": sessions, "by_corpus": corpora, "by_phone": phone_edits,
                     "substitutions": [{"reference": a, "prediction": b, "count": n}
                                       for (a, b), n in changed.most_common()],
                     "per": all_counts["phone_errors"] / all_counts["phones"],
                     "wer": all_counts["word_errors"] / all_counts["words"],
                     "paired_vs_first": {}, "input_sha256": sha256(path)}
        if not results:
            control = {k: np.asarray(v, dtype=np.int64) for k, v in edits.items()}
        else:
            for kind, key, lengths in (("wer", "word_errors", denominator_words),
                                       ("per", "phone_errors", denominator_phones)):
                counts = np.asarray(edits[key], dtype=np.int64)
                difference = counts - control[key]
                condition["paired_vs_first"][kind] = {"delta": float(difference.sum() / lengths.sum()),
                    "better_trials": int((difference < 0).sum()), "worse_trials": int((difference > 0).sum()),
                    "session_block_bootstrap": _bootstrap(control[key], counts, lengths, membership,
                                                           np.random.default_rng(20261127), 2000)}
        results[name] = condition
    write_json(output, {"score_split": "dev_tune", "split_hash": split["manifest_hash"], "conditions": results})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--corpus-metadata", type=Path)
    cli.add_argument("predictions", nargs="+", type=Path)
    args = cli.parse_args()
    summarize(args.config, args.predictions, args.output, args.corpus_metadata)
