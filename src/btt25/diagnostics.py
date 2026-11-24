"""Reference diagnostics and fixed examples from authorized scored artifacts.

No dataset reference loader is accepted or called. Reporting authorization is
checked in metrics.json before cached reference text is read. OOV statistics
use normalized reference words; failed predictions retain their references.
"""

from collections import Counter
from html import escape
import json
from pathlib import Path

from .assets import sha256
from .comparison import _load_run
from .data import object_hash
from .evaluation import normalize_text
from .reporting import _group_value, _write_json


_TRACE = ("score_split", "split_manifest_hash", "freeze_manifest_hash")
_COUNTS = ("substitutions", "deletions", "insertions", "errors", "reference_length")


def _authorized_run(directory):
    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("score_split") not in {"dev_tune", "dev_report", "official_val"}:
        raise ValueError("Diagnostics require an explicit scored split")
    if not metrics.get("split_manifest_hash"):
        raise ValueError("Diagnostics require a split manifest hash")
    if metrics["score_split"] != "dev_tune" and not metrics.get("freeze_manifest_hash"):
        raise ValueError("Reporting diagnostics require a recorded joint freeze hash")
    loaded, rows, edits, _ = _load_run(directory)
    if any(metrics.get(key) != loaded.get(key) for key in _TRACE):
        raise ValueError("Scoring identity changed while reading authorized artifacts")
    return loaded, rows, edits


def _text_available(metrics):
    available = metrics.get("text_available", metrics.get("metadata", {}).get("text_available", True))
    if not isinstance(available, bool):
        raise ValueError("text_available must be boolean")
    return available


def _lexicon_words(path):
    words = set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith((";;;", "#")):
                continue
            fields = line.split()
            normalized = normalize_text(fields[0]).split()
            if len(fields) < 2 or len(normalized) != 1:
                raise ValueError(f"Invalid lexicon entry at line {number}")
            words.add(normalized[0])
    if not words:
        raise ValueError("Lexicon contains no normalized word types")
    return words


def _oov(rows, lexicon):
    sentences = [normalize_text(row["reference_text"]).split() for row in rows]
    counts = Counter(word for sentence in sentences for word in sentence)
    unknown = {word: counts[word] for word in sorted(counts) if word not in lexicon}
    tokens, types = sum(counts.values()), len(counts)
    oov_tokens, oov_types = sum(unknown.values()), len(unknown)
    sentence_oov = sum(any(word not in lexicon for word in sentence) for sentence in sentences)
    return {
        "reference_token_count": tokens, "reference_type_count": types,
        "in_lexicon_token_count": tokens - oov_tokens, "in_lexicon_type_count": types - oov_types,
        "oov_token_count": oov_tokens, "oov_type_count": oov_types,
        "token_coverage": (tokens - oov_tokens) / tokens if tokens else None,
        "type_coverage": (types - oov_types) / types if types else None,
        "oov_token_rate": oov_tokens / tokens if tokens else None,
        "oov_type_rate": oov_types / types if types else None,
        "sentence_count": len(sentences), "sentences_with_oov": sentence_oov,
        "sentence_oov_rate": sentence_oov / len(sentences) if sentences else None,
        "empty_reference_count": sum(not sentence for sentence in sentences),
        "oov_counts": unknown,
    }


def _summary(rows, edits, ids, lexicon, text_available, lm_words=None):
    selected = [rows[trial_id] for trial_id in ids]
    totals = {kind: {key: sum(edits[trial_id][kind][key] for trial_id in ids) for key in _COUNTS}
              for kind in ("word", "phoneme")}
    for kind, measured in (("word", text_available), ("phoneme", True)):
        counts = totals[kind]
        counts["error_rate"] = counts["errors"] / counts["reference_length"] if measured and counts["reference_length"] else None
    result = {
        "trial_count": len(ids),
        "failed_count": sum(row.get("status", "ok") != "ok" for row in selected),
        "missing_count": sum(row.get("status") == "missing" for row in selected),
        "phoneme_failed_count": sum(row.get("phoneme_status", row.get("status", "ok")) != "ok" for row in selected),
        **totals, "word_metrics_measured": text_available,
        "wer": totals["word"]["error_rate"], "per": totals["phoneme"]["error_rate"],
        "lexicon_oov": _oov(selected, lexicon),
    }
    if lm_words is not None:
        result["lm_oov"] = _oov(selected, lm_words)
    return result


def write_diagnostics(scored_run, lexicon_path, output_dir=None, lm_vocabulary=None) -> dict:
    """Write diagnostics.json and return corpus/session/OOV/edit summaries.

    lexicon_path is the decoder's word-plus-pronunciation text file. Alternate
    pronunciations count once after official text normalization. Token/type
    coverage has a zero denominator represented by None. Sentence OOV rate
    includes empty-reference trials as sentences without an OOV word. Corpus
    tags come exclusively from saved metadata, with missing values as unknown.
    """
    metrics, rows, edits = _authorized_run(scored_run)
    lexicon = _lexicon_words(lexicon_path)
    lm_words = None
    if lm_vocabulary is not None:
        entries = [line for line in Path(lm_vocabulary).read_text(encoding="utf-8").splitlines() if line]
        if not entries or any(normalize_text(word)!=word or len(word.split())!=1 for word in entries):
            raise ValueError("Language model vocabulary requires one normalized word per line")
        lm_words=set(entries)
    available = _text_available(metrics)
    groups = {}
    for field, name in (("session_id", "by_session"), ("corpus_id", "by_corpus")):
        memberships = {}
        for trial_id, row in rows.items():
            memberships.setdefault(_group_value(row, field), []).append(trial_id)
        groups[name] = {group: _summary(rows, edits, ids, lexicon, available, lm_words)
                        for group, ids in sorted(memberships.items())}
    result = {
        **{key: metrics.get(key) for key in _TRACE},
        "text_available": available, "lexicon_sha256": sha256(lexicon_path),
        "lexicon_type_count": len(lexicon), "oov_basis": "normalized authorized reference words",
        "scored_ids_hash": object_hash(sorted(rows)),
        "scored_artifact_hashes": {name: sha256(Path(scored_run) / name) for name in ("metrics.json", "predictions.jsonl", "per_trial.jsonl")},
        "overall": _summary(rows, edits, list(rows), lexicon, available, lm_words), **groups,
        "unusual_random_corpora": {name: summary for name, summary in groups["by_corpus"].items()
                                   if "random" in name.casefold() or "unusual" in name.casefold()},
    }
    if lm_vocabulary is not None:
        result["lm_vocabulary_sha256"]=sha256(lm_vocabulary)
    output = Path(scored_run if output_dir is None else output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "diagnostics.json", result)
    return result


def _cell(value):
    return escape(str(value)).replace("|", "\\|").replace("`", "\\`").replace("\n", "<br>")


def fixed_examples(scored_runs, ids, output_dir=None) -> dict:
    """Build identical predeclared-ID examples across scored condition paths.

    Preserve the caller's ID and condition order. Require identical full scored
    allowlists, scoring identity, and exact reference text/phones across runs.
    No edit count or output quality chooses examples. ids_hash is object_hash(ids)
    so it can be recorded from IDs alone before labels. Optional output writes
    JSON and Markdown.
    """
    if not isinstance(ids, list) or any(not isinstance(trial_id, str) or not trial_id for trial_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Predeclared example IDs must be a unique list of nonempty strings")
    if not scored_runs or any(not isinstance(name, str) or not name for name in scored_runs):
        raise ValueError("Examples require named scored conditions")
    loaded = {name: _authorized_run(path) for name, path in scored_runs.items()}
    first_metrics, first_rows, _ = next(iter(loaded.values()))
    for name, (metrics, rows, _) in loaded.items():
        if any(first_metrics.get(key) != metrics.get(key) for key in _TRACE):
            raise ValueError(f"Scoring split/freeze mismatch for {name}")
        if set(first_rows) != set(rows):
            raise ValueError(f"Scored allowlist mismatch for {name}")
        for trial_id, reference in first_rows.items():
            if any(reference[field] != rows[trial_id][field] for field in ("reference_text", "reference_phoneme_ids")):
                raise ValueError(f"Scored reference mismatch for {name}/{trial_id}")
    if set(ids) - first_rows.keys():
        raise ValueError("Requested example IDs are outside the authorized scored allowlist")
    available = {name: _text_available(metrics) for name, (metrics, _, _) in loaded.items()}
    rows = []
    for trial_id in ids:
        reference = first_rows[trial_id]
        predictions = {}
        for name, (_, condition_rows, edits) in loaded.items():
            row = condition_rows[trial_id]
            predictions[name] = {
                "text": row.get("text", "") if available[name] and row.get("status", "ok") == "ok" else "" if available[name] else None,
                "phoneme_ids": row.get("phoneme_ids", []), "status": row.get("status", "ok"),
                "phoneme_status": row.get("phoneme_status", row.get("status", "ok")),
                "word_edit_counts": edits[trial_id]["word"] if available[name] else None,
                "phoneme_edit_counts": edits[trial_id]["phoneme"],
            }
        rows.append({"trial_id": trial_id, "reference_text": reference["reference_text"],
                     "reference_phoneme_ids": reference["reference_phoneme_ids"], "predictions": predictions})
    show_phones = not all(available.values())
    headers = ["Trial ID", "Reference"] + (["Reference phones"] if show_phones else [])
    headers += [name if available[name] else name + " (phonemes)" for name in loaded]
    table = ["| " + " | ".join(_cell(header) for header in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [row["trial_id"], row["reference_text"]] + ([row["reference_phoneme_ids"]] if show_phones else [])
        for name, prediction in row["predictions"].items():
            value = prediction["text"] if available[name] else prediction["phoneme_ids"]
            status = prediction["status"] if available[name] else prediction["phoneme_status"]
            cells.append(str(value) + (f" [status: {status}]" if status != "ok" else ""))
        table.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    markdown = "\n".join(table) + "\n"
    result = {
        **{key: first_metrics.get(key) for key in _TRACE},
        "ids": list(ids), "ids_hash": object_hash(ids), "allowed_ids_hash": object_hash(sorted(first_rows)),
        "condition_order": list(loaded), "text_available": available,
        "selection_rule": "caller-supplied IDs in their declared order", "rows": rows, "markdown": markdown,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "fixed_examples.json", result)
        (output / "fixed_examples.md").write_text(markdown, encoding="utf-8")
    return result
