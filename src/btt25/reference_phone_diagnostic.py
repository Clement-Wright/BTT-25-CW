"""Compare saved neural predictions with a decoder fed the tuning reference phones."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .assets import sha256
from .benchmark import SERVER, native_decoder
from .ctc import ctc_minimum_frames, greedy_ctc
from .data import TOKENS, attach_corpus_metadata, load_split_manifest, reference_only, write_json
from .evaluation import _by_id, _selection, edit_counts, normalize_text
from .inference import load_json, read_jsonl
from .resources import ResourceMonitor


def reference_emissions(target, frames, alignment="dwell"):
    """Create a verified CTC path with either sustained or one-frame phones."""
    if not target or any(not 0 < phone < len(TOKENS) for phone in target):
        raise ValueError("Expected an unpadded nonempty phone target")
    frames = max(frames, 2 * len(target) + 1)
    if alignment == "spike":
        # Flashlight returns to the root as soon as a word is emitted.
        # Keeping extra frames blank avoids demanding repeated terminal phones.
        blanks, remainder = divmod(frames - len(target), len(target) + 1)
        path = [0] * (blanks + (remainder > 0))
        for i, phone in enumerate(target, 1):
            path.append(phone)
            path.extend([0] * (blanks + (i < remainder)))
    elif alignment == "dwell":
        dwell, remainder = divmod(frames - len(target) - 1, len(target))
        path = [0]
        for i, phone in enumerate(target):
            path.extend([phone] * (dwell + (i < remainder)))
            path.append(0)
    else:
        raise ValueError(f"Unsupported alignment policy {alignment}")
    if len(path) != frames or greedy_ctc(path) != target:
        raise ValueError("Invalid constructed CTC path")
    logits = np.full((frames, len(TOKENS)), -100., dtype=np.float32)
    logits[np.arange(frames), path] = 0.
    return logits


def exact_lexicon(path):
    by_phone = defaultdict(set)
    token_ids = {token: i for i, token in enumerate(TOKENS)}
    for row in path.read_text(encoding="utf-8").splitlines():
        if row.strip():
            word, *phones = row.split()
            by_phone[tuple(token_ids[phone] for phone in phones)].add(word)
    return by_phone


def exact_word_choices(phones, lexicon):
    """Use the released SIL targets as boundaries and preserve homophones."""
    segments, current = [], []
    for phone in [*phones, 40]:
        if phone == 40:
            if current:
                segments.append(lexicon.get(tuple(current), set()))
                current.clear()
        else:
            current.append(phone)
    return segments


def diagnostic(args):
    records = load_json(args.data_manifest)
    if args.corpus_metadata:
        records = attach_corpus_metadata(records, args.corpus_metadata)
    split = load_split_manifest(args.split_manifest, records)
    selected = _selection(_by_id(records, "trial"), split, "dev_tune")
    neural = _by_id(read_jsonl(args.neural_predictions), "neural prediction")
    saved = _by_id(read_jsonl(args.word_predictions), "word prediction")
    for row in selected:
        trial = row["trial_id"]
        if neural[trial]["logit_sha256"] != saved[trial]["logit_sha256"]:
            raise ValueError(f"Saved predictions used different emissions for {trial}")

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.resolved.yaml", {**{k: str(v) if isinstance(v, Path) else v
                                                        for k, v in vars(args).items()},
                                                     "scope": "dev_tune only"})
    inputs = ("data_manifest", "split_manifest", "neural_predictions", "word_predictions",
              "lexicon", "kenlm_model", "corpus_metadata", "training_unigrams")
    write_json(args.output / "provenance.json", {
        k: sha256(getattr(args, k)) for k in inputs if getattr(args, k)
    } | {"diagnostic": sha256(__file__), "decoder_server": sha256(SERVER),
         "decoder_worker": sha256(SERVER.with_name("decode_worker.py"))})
    totals, groups, sessions = Counter(), {}, {}
    lexicon = exact_lexicon(args.lexicon)
    prior = load_json(args.training_unigrams) if args.training_unigrams else None
    if prior and prior.get("source_split") != "train":
        raise ValueError("Homophone counts must come from the official training split")
    unigrams = prior["counts"] if prior else {}

    def select_words(segments):
        return " ".join(min(options, key=lambda word: (-unigrams.get(word, 0), word))
                        for options in segments if options)

    started = perf_counter()
    with ResourceMonitor(args.output / "resources.json", max_seconds=600) as monitor, \
            native_decoder(args, args.beam, args.output / "decoder.stderr.log") as (decode, service), \
            (args.output / "predictions.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
        for record in selected:
            monitor.check()
            trial = record["trial_id"]
            label = reference_only(record)
            target = label["phoneme_ids"]
            reference = normalize_text(label["reference_text"]).split()
            actual = neural[trial]
            logits = reference_emissions(target, actual["output_length"], args.alignment)
            candidates = decode(trial, logits)["candidates"]
            best = candidates[0] if candidates else {"text": "", "phoneme_ids": []}
            choices = exact_word_choices(target, lexicon)
            neural_choices = exact_word_choices(actual["phoneme_ids"], lexicon)
            exact_text = select_words(choices)
            neural_lexicon_text = select_words(neural_choices)
            matched_words = len(reference) == len(choices)
            counts = Counter(trials=1, reference_words=len(reference), reference_phones=len(target),
                             length_extended=int(len(logits) > actual["output_length"]),
                             infeasible_at_baseline_length=int(ctc_minimum_frames(target) > actual["output_length"]),
                             saved_phone_errors=edit_counts(target, actual["phoneme_ids"])["errors"],
                             decoder_oracle_errors=min((edit_counts(reference, normalize_text(c["text"]).split())["errors"]
                                                        for c in candidates), default=len(reference)),
                             empty_reference_phone_outputs=int(not best["text"]),
                             exact_input_phone_path=int(best["phoneme_ids"] == target),
                             exact_segments=len(choices),
                             empty_exact_segments=sum(not options for options in choices),
                             neural_exact_segments=len(neural_choices),
                             neural_empty_exact_segments=sum(not options for options in neural_choices),
                             homophone_segments=sum(len(options) > 1 for options in choices),
                             matched_segment_words=len(reference) if matched_words else 0,
                             reference_word_absent_from_aligned_segment=sum(
                                 word not in options for word, options in zip(reference, choices)
                             ) if matched_words else 0)
            for name, text in (("saved", saved[trial]["text"] or ""), ("reference_phone", best["text"]),
                               ("exact_lexicon", exact_text), ("neural_exact_lexicon", neural_lexicon_text)):
                for key, value in edit_counts(reference, normalize_text(text).split()).items():
                    if key != "reference_length":
                        counts[f"{name}_{key}"] = value
            totals.update(counts)
            groups.setdefault(record.get("corpus_id") or "unclassified", Counter()).update(counts)
            sessions.setdefault(record["session_id"], Counter()).update(counts)
            stream.write(json.dumps({"trial_id": trial, "status": "ok" if best["text"] else "empty",
                                     "text": best["text"], "decoded_phone_ids": best["phoneme_ids"],
                                     "exact_lexicon_text": exact_text, "neural_exact_lexicon_text": neural_lexicon_text,
                                     "diagnostic_counts": counts}) + "\n")
    metrics = {"status": "complete", "split_hash": split["manifest_hash"], "score_split": "dev_tune",
               "overall": totals, "by_session": sessions, "by_corpus": groups, "backend": service,
               "elapsed_seconds": perf_counter() - started}
    write_json(args.output / "metrics.json", metrics)
    (args.output / "report.md").write_text(
        f"# Reference-phone diagnostic\n\n{service['mode']} on {len(selected)} tuning trials.\n\n"
        f"Saved-encoder WER {totals['saved_errors']/totals['reference_words']:.2%}. "
        f"Native reference-phone injection WER {totals['reference_phone_errors']/totals['reference_words']:.2%}. "
        f"The native decoder kept the injected phone string on {totals['exact_input_phone_path']} trials. "
        f"Exact lexicon WER {totals['exact_lexicon_errors']/totals['reference_words']:.2%} with the reference phones "
        f"and {totals['neural_exact_lexicon_errors']/totals['reference_words']:.2%} with the saved GRU phones.\n\n"
        "Reference phone timing is constructed, and the sharp synthetic logits change confidence. "
        "These paired scores diagnose the decoder and are ineligible as end-to-end results.\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-manifest", "split-manifest", "neural-predictions", "word-predictions",
                 "lexicon", "decoder-python", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("kenlm-model", "corpus-metadata", "decoder-module-path", "training-unigrams"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--beam", type=int, default=50)
    parser.add_argument("--nbest", type=int, default=10)
    parser.add_argument("--beam-threshold", type=float, default=25.)
    parser.add_argument("--lm-weight", type=float, default=1.)
    parser.add_argument("--alignment", choices=("dwell", "spike"), default="dwell")
    parser.add_argument("--start-timeout", type=float, default=90.)
    parser.add_argument("--request-timeout", type=float, default=30.)
    args = parser.parse_args()
    diagnostic(args)


if __name__ == "__main__":
    main()
