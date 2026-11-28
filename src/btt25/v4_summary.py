"""Verify measured V4 trials and reduce private runs to public aggregate evidence."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .assets import sha256
from .comparison import _bootstrap
from .data import load_split_manifest, write_json
from .evaluation import _by_id, _selection
from .inference import load_config, load_json, read_jsonl


def distribution(values):
    return dict(zip(("p50", "p90", "p95", "p99", "max"),
                    np.percentile(values, [50, 90, 95, 99, 100]).tolist(), strict=True))


def summarize(args):
    cfg = load_config(args.config)
    records = load_json(cfg["data_manifest"])
    split = load_split_manifest(cfg["split_manifest"], records)
    tune = _selection(_by_id(records, "trial"), split, "dev_tune")
    ids = [r["trial_id"] for r in tune]
    groups = [(r["session_id"], r["block_id"]) for r in tune]
    sources = dict(line.split("=", 1) for line in
                   (args.benchmark / "sources.txt").read_text().splitlines())
    clock = load_json(args.clock / "metrics.json")
    if clock["status"] != "complete":
        raise ValueError("Incomplete clock diagnostic")
    public = {"split": "dev_tune", "split_hash": split["manifest_hash"], "models": {},
              "recurrent_clock": clock["counts"], "benchmark_job": args.benchmark.name}
    missing = args.benchmark / "unavailable.txt"
    public["unavailable"] = missing.read_text().splitlines() if missing.exists() else []
    all_counts = {}
    timing_ids = None
    membership = set()
    gpu_uuids = set()
    for name, source in sources.items():
        base = Path(source)
        endpoint = base if name == "official" else base / name
        m = load_json(endpoint / "metrics.json")
        pred = read_jsonl(endpoint / "predictions.jsonl")
        rows = _by_id(pred, "prediction")
        if m["status"] != "complete" or set(rows) != set(ids):
            raise ValueError(f"Incomplete scored cohort for {name}")
        counts = [rows[i]["counts"] for i in ids]
        total = sum(map(Counter, counts), Counter())
        for k in ("word_errors", "words", "phone_errors", "phones"):
            if total[k] != m["counts"][k]:
                raise ValueError(f"Aggregate differs from per-trial {k} for {name}")
        all_counts[name] = counts
        result = {"wer": total["word_errors"]/total["words"], "per": total["phone_errors"]/total["phones"],
                  "counts": total, "updates": m.get("updates", 0),
                  "trainable": m.get("trainable_parameters", 0), "prediction_sha256": sha256(endpoint/"predictions.jsonl"),
                  "by_corpus": m["by_corpus"], "training_job": base.parent.name if name != "official" else None}
        if name != "official":
            membership.add(load_json(endpoint / "config.resolved.yaml")["membership_hash"])
            result.update({k: m[k] for k in ("contract", "checkpoint_sha256", "by_session")})
            res = [load_json(p) for p in [base/"setup/resources.json",
                   endpoint/"pilot/resources.json", endpoint/"resources.json"]]
            result["training"] = {"peak_cgroup_bytes": max(r["peak_cgroup_bytes"] for r in res),
                "peak_torch_allocated_bytes": max(r["peak_cuda_allocated_bytes"] for r in res),
                "gpu": res[-1]["hardware"]["gpu"][0]["name"], "admission": load_json(base/"admission.json"),
                "pilot_seconds": load_json(endpoint/"pilot/metrics.json")["sustained_seconds"],
                "phase_wall_s": dict(zip(("setup", "pilot", "remaining_training_and_scoring"),
                    [r["elapsed_seconds"] for r in res], strict=True)),
                "phase_cpu": dict(zip(("setup", "pilot", "remaining_training_and_scoring"),
                    [r["cpu"] for r in res], strict=True)),
                "final_phase_device_utilization": res[-1]["gpu_summary"][0].get("utilization_percent")}
        warm, cold, resources, charged = [], [], [], []
        for pass_index in range(3):
            root = args.benchmark / "benchmark" / name / str(pass_index)
            state = load_json(root / "metrics.json")
            if state["status"] != "complete":
                raise ValueError(f"Incomplete timing pass {name}/{pass_index}")
            observations = [r for r in read_jsonl(root/"predictions.jsonl") if not r["warmup"]]
            cohort = {r["trial_id"] for r in observations}
            if len(cohort) != 24 or len(observations) != 24 or (timing_ids is not None and cohort != timing_ids):
                raise ValueError("The timing cohort differs")
            timing_ids = cohort
            warm.extend(observations)
            cold.append(state["cold_model_verify_and_load_s"] + state["decoder_startup"]["process_ready_s"])
            resources.append(load_json(root / "resources.json"))
            charged.extend(s["cgroup_current_bytes"] for s in read_jsonl(root/"resources.samples.jsonl")
                           if s.get("cgroup_current_bytes") is not None)
        hw = resources[-1]["hardware"]["gpu"][0]
        gpu_uuids.update(r["hardware"]["gpu"][0]["uuid"] for r in resources)
        seconds = [r["elapsed_s"] for r in warm]
        raw = np.array([r["raw_frames"]*.020 for r in warm])
        quartile = np.searchsorted(np.percentile(raw, [25, 50, 75]), raw, side="right")
        timing = {"warm_requests": len(warm), "gpu": hw["name"], "allocated_cpus": resources[-1]["allocated_cpus"],
                  "cold_load_and_decoder_s": cold, "seconds": distribution(seconds),
                  "rtf": distribution([r["rtf"] for r in warm]),
                  "pooled_rtf": sum(seconds)/sum(raw), "requests_per_second": len(warm)/sum(seconds),
                  "different_text_replays": sum(not r["matches_training_run_replay"] for r in warm),
                  "empty_outputs": sum(not r["text"] for r in warm),
                  "length_quartiles": [distribution(np.asarray(seconds)[quartile == q]) for q in range(4)],
                  "length_quartile_edges_s": np.percentile(raw, [25, 50, 75]).tolist()}
        timing["max_sampled_cgroup_current_bytes"] = max(charged, default=None)
        timing["shared_job_cgroup_high_water_bytes"] = max(r["peak_cgroup_bytes"] for r in resources)
        for key in ("peak_tree_pss_bytes", "peak_tree_rss_bytes",
                    "peak_device_used_bytes", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
            timing[key] = max((r[key] for r in resources if r[key] is not None), default=None)
        result["timing"] = timing
        public["models"][name] = result
    if len(membership) > 1 or len(gpu_uuids) != 1:
        raise ValueError("Matched training IDs or timing GPU differ across arms")
    public["train_membership_hash"] = next(iter(membership), None)
    for name, counts in all_counts.items():
        changes = {}
        for control in ("official", "gru80", "gru40"):
            if control not in all_counts:
                continue
            for unit in ("word", "phone"):
                a = np.array([r[unit + "_errors"] for r in all_counts[control]])
                b = np.array([r[unit + "_errors"] for r in counts])
                n = np.array([r[unit + "s"] for r in counts])
                if not np.array_equal(n, [r[unit + "s"] for r in all_counts[control]]):
                    raise ValueError("Reference lengths differ")
                ci = _bootstrap(a, b, n, groups, np.random.default_rng(20261127), 2000)
                changes[control+"_"+unit] = {"delta": float((b-a).sum()/n.sum()), **ci}
        public["models"][name]["paired"] = changes
    write_json(args.output, public)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "benchmark", "clock", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    summarize(parser.parse_args())
