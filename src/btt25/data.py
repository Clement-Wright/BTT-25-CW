"""Lazy official-release HDF5 access. Manifest construction never reads references."""
from collections import defaultdict
import hashlib
import json
import csv
from pathlib import Path

import h5py
import numpy as np

TOKENS = ["BLANK", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL"]


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def load_split_manifest(path,records=None):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("manifest_hash")!=object_hash({k:v for k,v in value.items() if k!="manifest_hash"}):
        raise ValueError("Frozen split content hash mismatch")
    tune,report=value.get("dev_tune"),value.get("dev_report")
    if not isinstance(tune,list) or not isinstance(report,list) or not tune or not report:
        raise ValueError("Both frozen tune/report ID lists are required")
    if len(set(tune))!=len(tune) or len(set(report))!=len(report) or set(tune)&set(report):
        raise ValueError("Frozen split membership duplicates or leaks")
    if records is not None and set(tune)|set(report)!={r["trial_id"] for r in records if r["split"]=="val"}:
        raise ValueError("Frozen split does not cover the current validation partition")
    return value


def attach_corpus_metadata(records, path):
    mapping={}
    with Path(path).open(newline="",encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key=("t15."+row["Date"].replace("-","."),int(row["Block number"]))
            if key in mapping and mapping[key]!=row["Corpus"]:
                raise ValueError("Conflicting official corpus metadata")
            mapping[key]=row["Corpus"]
    return [{**r,"corpus_id":mapping.get((r["session_id"],r["block_id"]))} for r in records]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_manifest(data_root, sessions):
    root = Path(data_root).resolve()
    present = {p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("t15.")}
    if present - set(sessions):
        raise ValueError(f"Unknown session(s) not in checkpoint: {sorted(present - set(sessions))}")
    records = []
    for split in ("train", "val", "test"):
        order = 0
        for day, session in enumerate(sessions):
            path = root / session / f"data_{split}.hdf5"
            if not path.is_file():
                # Some official sessions genuinely have no val/test partition.
                continue
            with h5py.File(path, "r") as file:
                for key in list(file.keys()):
                    group = file[key]
                    shape = group["input_features"].shape
                    length = int(group.attrs["n_time_steps"])
                    if len(shape) != 2 or shape[1] != 512:
                        raise ValueError(f"Expected T x 512 features: {path}/{key}: {shape}")
                    if not 0 < length <= shape[0]:
                        raise ValueError(f"Invalid input length: {path}/{key}")
                    actual_session = group.attrs.get("session", session)
                    if isinstance(actual_session, bytes):
                        actual_session = actual_session.decode()
                    if str(actual_session) != session:
                        raise ValueError(f"Session metadata mismatch: {path}/{key}")
                    has_labels = "seq_class_ids" in group and "seq_len" in group.attrs and "sentence_label" in group.attrs
                    if split != "test" and not has_labels:
                        raise ValueError(f"Missing required labels: {path}/{key}")
                    record = {"trial_id": f"{session}/{split}/{key}", "split": split, "session_id": session, "day_index": day, "source_file": str(path), "source_group": key, "official_order_index": order, "input_length": length, "feature_shape": list(shape), "feature_dtype": str(group["input_features"].dtype), "target_length": int(group.attrs["seq_len"]) if has_labels else None, "has_labels": has_labels, "block_id": int(group.attrs["block_num"]) if "block_num" in group.attrs else None, "corpus_id": None}
                    records.append(record)
                    order += 1
    if not any(r["split"] == "train" for r in records) or not any(r["split"] == "val" for r in records):
        raise ValueError("Both train and validation partitions are required; no train fallback")
    return records


def read_trial(record, include_labels=False):
    if include_labels and (record["split"] == "test" or not record["has_labels"]):
        raise ValueError("Cannot request labels from an unlabeled test trial")
    with h5py.File(record["source_file"], "r") as file:
        group = file[record["source_group"]]
        result = {"features": np.asarray(group["input_features"][:record["input_length"]], dtype=np.float32)}
        if not np.isfinite(result["features"]).all():
            raise ValueError(f"Nonfinite neural features: {record['trial_id']}")
        if include_labels:
            n = record["target_length"]
            target = np.asarray(group["seq_class_ids"][:n], dtype=np.int64)
            if len(target) != n or not np.all((target >= 1) & (target <= 40)):
                raise ValueError(f"Invalid target token or blank: {record['trial_id']}")
            text = group.attrs["sentence_label"]
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            result.update(phoneme_ids=target.tolist(), reference_text=str(text))
    return result


def freeze_split(records, seed=20261116):
    groups = defaultdict(list)
    for record in records:
        if record["split"] == "val":
            groups[(record["session_id"], record["block_id"])].append(record["trial_id"])
    if len(groups) < 2:
        raise ValueError("At least two validation session/block groups required")
    # Hash rank is stable across OS, Python versions and input order.
    keys = sorted(groups, key=lambda key: object_hash([seed, *key]))
    target = sum(len(ids) for ids in groups.values()) / 3
    report_groups, count = [], 0
    for key in keys[:-1]:
        if count >= target:
            break
        report_groups.append(key)
        count += len(groups[key])
    report = {trial for key in report_groups for trial in groups[key]}
    result = {"version": 1, "seed": seed, "grouping": "session_id + block_id; deterministic hash-ranked approximately one-third report", "dev_tune": [r["trial_id"] for r in records if r["split"] == "val" and r["trial_id"] not in report], "dev_report": [r["trial_id"] for r in records if r["trial_id"] in report], "prior_exposure": "Official pretrained checkpoint used all known sessions; historical validation selection may include this local holdout. No unseen-day claim."}
    result["manifest_hash"] = object_hash(result)
    return result


def reference_only(record):
    # Keep reference selection in the evaluator, not in inference or manifests.
    if record["split"] == "test" or not record["has_labels"]:
        raise ValueError("Cannot request labels from an unlabeled test trial")
    with h5py.File(record["source_file"], "r") as file:
        group = file[record["source_group"]]
        n = record["target_length"]
        target = np.asarray(group["seq_class_ids"][:n],dtype=np.int64)
        if len(target)!=n or not np.all((target>=1)&(target<=40)):
            raise ValueError(f"Invalid target token or blank: {record['trial_id']}")
        text=group.attrs["sentence_label"]
        if isinstance(text,bytes): text=text.decode("utf-8")
        return {"reference_text":str(text),"phoneme_ids":target.tolist()}
