"""Compare completed, fixed-candidate rerankers on the T15 tuning IDs."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .assets import sha256
from .comparison import _bootstrap
from .data import attach_corpus_metadata, load_split_manifest, reference_only, write_json
from .evaluation import _by_id, _selection, edit_counts, normalize_text
from .inference import load_config, load_json, read_jsonl


def compare(config_path, output, predictions, pilot_count=None, corpus_metadata=None):
    config = load_config(config_path)
    records = load_json(config["data_manifest"])
    if corpus_metadata is not None:
        records = attach_corpus_metadata(records, corpus_metadata)
    split = load_split_manifest(config["split_manifest"], records)
    eligible = _selection(_by_id(records, "trial"), split, "dev_tune")
    if pilot_count is not None:
        if not 1 <= pilot_count <= len(eligible):
            raise ValueError("Pilot count must fit the tuning split")
        ordered = sorted(eligible, key=lambda row: row["input_length"])
        eligible = [ordered[i] for i in np.linspace(0, len(ordered) - 1, pilot_count, dtype=int)]
    ids = [row["trial_id"] for row in eligible]
    blocks = [(row["session_id"], row["block_id"]) for row in eligible]
    references = [normalize_text(reference_only(row)["reference_text"]).split() for row in eligible]
    lengths = np.asarray([len(row) for row in references], dtype=np.int64)
    summary = {}
    for name, path in predictions.items():
        indexed = _by_id(read_jsonl(path), name)
        if any(trial not in indexed for trial in ids):
            raise ValueError(f"{name} does not cover every matching tuning ID")
        counts, by_session, by_corpus = Counter(), defaultdict(Counter), defaultdict(Counter)
        errors = []
        for trial, row, words in zip(ids, eligible, references, strict=True):
            prediction = indexed[trial]
            text = prediction.get("text", "") if prediction.get("status", "ok") == "ok" else ""
            edits = edit_counts(words, normalize_text(text).split())
            if prediction.get("counts") is not None and edits != prediction["counts"]:
                raise ValueError(f"Saved word edits disagree for {name}, {trial}")
            errors.append(edits["errors"])
            counts.update(edits)
            counts.update(trials=1, empty=int(not text))
            by_session[row["session_id"]].update(edits)
            by_corpus[row.get("corpus_id") or "unknown"].update(edits)
        errors = np.asarray(errors, dtype=np.int64)
        result = {"counts": counts, "wer": counts["errors"] / counts["reference_length"],
                  "prediction_sha256": sha256(path), "by_session": by_session, "by_corpus": by_corpus}
        if not summary:
            control = errors
        else:
            result["paired_vs_first"] = {"delta": float((errors - control).sum() / lengths.sum()),
                "fewer_edits_trials": int((errors < control).sum()), "more_edits_trials": int((errors > control).sum()),
                "session_block_bootstrap": _bootstrap(control, errors, lengths, blocks,
                                                       np.random.default_rng(20261127), 2000)}
        summary[name] = result
    write_json(output, {"score_split": "dev_tune", "split_hash": split["manifest_hash"],
                        "pilot_count": pilot_count, "results": summary})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-count", type=int)
    parser.add_argument("--corpus-metadata", type=Path)
    parser.add_argument("--prediction", action="append", type=lambda pair: pair.split("=", 1), required=True)
    cli = parser.parse_args()
    paths = {name: Path(path) for name, path in cli.prediction}
    if len(paths) != len(cli.prediction):
        parser.error("Each comparison name must be unique")
    compare(cli.config, cli.output, paths, cli.pilot_count, cli.corpus_metadata)
