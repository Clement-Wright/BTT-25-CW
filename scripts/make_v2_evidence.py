"""Export allowlisted aggregate evidence for the public continuation report."""
import argparse
import csv
import json
from pathlib import Path
import re


SEARCH_FIELDS = ("reference_words", "trials", "successful", "selected_errors", "selected_wer", "substitutions",
                 "deletions", "insertions", "oracle", "candidate_count_histogram", "lexicon_oov_tokens",
                 "lm_oov_tokens", "chosen_weight", "tuning_weight_grid")
RESOURCE_FIELDS = ("elapsed_seconds", "host_ceiling_bytes", "slurm_requested_bytes", "cgroup_effective_limit_bytes",
                   "peak_accounted_host_bytes", "peak_cgroup_bytes", "peak_tree_rss_bytes", "peak_tree_pss_bytes",
                   "peak_tree_uss_bytes", "peak_rss_bytes", "peak_device_used_bytes", "peak_cuda_allocated_bytes",
                   "peak_cuda_reserved_bytes", "cpu", "sampler_seconds", "method", "outcome", "cgroup_events_delta")
GPU_FIELDS = ("name", "capability", "total_vram_bytes", "device_wide_samples", "used_mib", "utilization_percent",
              "temperature_c", "power_w", "sm_clock_mhz", "memory_clock_mhz", "peak_sampled_owned_processes_mib")
SCHEDULER_FIELDS = ("JobID", "State", "ElapsedRaw", "TotalCPU", "UserCPU", "SystemCPU", "MaxRSS", "MaxVMSize",
                    "ReqMem", "AllocCPUS", "AllocTRES", "ExitCode")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fields(data, keys):
    return {k: data[k] for k in keys if k in data}


def resource_summary(path):
    source = read(path)
    result = fields(source, RESOURCE_FIELDS)
    if "gpu_summary" in source:
        result["gpu_summary"] = [fields(row, GPU_FIELDS) for row in source["gpu_summary"]]
    if "hardware" in source:
        result["runtime"] = fields(source["hardware"], ("torch", "cuda_runtime"))
    return result


def condition(spec):
    name, path = spec.split("=", 1)
    if not re.fullmatch(r"v2-[a-z0-9-]+", name):
        raise ValueError("Expected a registered v2 condition ID")
    directory = Path(path)
    tune = directory / "dev-tune" if (directory / "dev-tune").is_dir() else directory
    raw = read(tune / "metrics.json")
    if raw["score_split"] != "dev_tune" or raw["selected_errors"] != sum(raw[k] for k in ("substitutions", "deletions", "insertions")):
        raise ValueError("Unexpected tuning scope or edit count")
    if not raw["reference_words"] or abs(raw["selected_wer"] - raw["selected_errors"] / raw["reference_words"]) > 1e-10:
        raise ValueError("Inconsistent tuning WER")
    metrics = fields(raw, SEARCH_FIELDS)
    resources = directory / "decode/resources.json"
    if resources.is_file():
        metrics["decode_resources"] = resource_summary(resources)
    for field in ("by_corpus", "by_session"):
        metrics[field] = {key: fields(value, SEARCH_FIELDS) for key, value in raw.get(field, {}).items()}
    if raw.get("model"):
        metrics["model"] = fields(raw["model"], ("model_id", "revision", "precision", "dtype", "candidates", "tokens", "timing"))
    return name, metrics


def report_results(directory):
    selected = ("wer", "per", "word", "phoneme", "prediction_coverage", "evaluated_count", "failed_count")
    freeze = read(directory / "freeze.json")
    result = {"freeze": freeze["manifest_hash"], "selection": freeze["selected_condition"], "splits": {}}
    for name in ("dev_report", "official_val"):
        data = read(directory / name / "metrics.json")
        if data["freeze_manifest_hash"] != result["freeze"] or data["evaluation_status"] != "exploratory_after_v1_report_exposure":
            raise ValueError("Unexpected report status or freeze")
        conditions = {}
        for key, val in data["conditions"].items():
            if not re.fullmatch(r"v2-[a-z0-9-]+", key):
                raise ValueError("Unexpected frozen condition ID")
            entry = fields(val, selected)
            for filename in ("by_corpus", "by_session", "length_bins"):
                grouped = directory / name / key / f"{filename}.csv"
                with grouped.open(encoding="utf-8", newline="") as stream:
                    entry[filename] = list(csv.DictReader(stream))
            conditions[key] = entry
        result["splits"][name] = {
            "conditions": conditions,
            "paired_wer": data["paired_wer"], "paired_by_corpus": data["paired_by_corpus"],
        }
    return result


def scheduler(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|")
        if tuple(reader.fieldnames or ()) != SCHEDULER_FIELDS:
            raise ValueError("Use the documented sacct fields for the public scheduler receipt")
        rows = list(reader)
    if any(not re.fullmatch(r"[0-9]+(?:\.(?:batch|extern|[0-9]+))?", row["JobID"]) for row in rows):
        raise ValueError("Expected numeric Slurm job and step IDs")
    return rows


def latency(path):
    metrics = read(path / "metrics.json")
    names = metrics.get("by_condition", metrics.get("by_beam"))
    return {**fields(metrics, ("state", "mode", "streaming", "completed_passes", "attempted_requests", "expected_requests", "memory_scope")),
            "common_encoder_cold_s": metrics["cold_setup"].get("encoder_s"),
            "sustained_warm_s": metrics["cold_setup"].get("sustained_warm_s"),
            "conditions": {k: {**v, "setup": fields(metrics["cold_setup"].get(k, {}),
                 ("mode", "native_setup_s", "process_ready_s", "warmup_s", "encoder_setup_s", "neural_setup_s"))}
                           for k, v in names.items()}, "resources": resource_summary(path / "resources.json")}


def lightbeam(path):
    from btt25.benchmark import timing_summary

    summary = fields(read(path / "metrics.json"), ("status", "admitted_as", "passes", "warm_seconds",
                     "scope", "rows", "cold_setup_s"))
    resource = path / "resources.json"
    if resource.is_file():
        summary["resources"] = resource_summary(resource)
    matched = path / "matched_tune_subset.json"
    if matched.is_file():
        summary["matched_tuning_subset"] = read(matched)
    trace = path / "timing.jsonl"
    if trace.is_file():
        buckets = {}
        with trace.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                buckets.setdefault(row["condition"], []).append(row)
        summary["timing_after_first_pass"] = {name: {
            "first_request_s": values[0]["latency_s"],
            **timing_summary([row for row in values if row["pass"] > 1]),
        } for name, values in buckets.items()}
    return summary


def encoder_study(path):
    summary = {}
    metrics = ("status", "updates", "sustained_seconds", "trials", "wer", "per", "excluded_infeasible",
               "trainable_parameters", "total_parameters", "checkpoint_sha256")
    for stage in ("pilot", "train"):
        directory = path / stage
        if (directory / "metrics.json").is_file():
            summary[stage] = fields(read(directory / "metrics.json"), metrics)
            if (directory / "resources.json").is_file():
                summary[stage]["resources"] = resource_summary(directory / "resources.json")
    admission = path / "pilot/admission.json"
    if admission.is_file():
        summary["admission"] = fields(read(admission), ("forecast_seconds", "admitted_updates", "resume_pilot",
                                                       "registration_sha256"))
    return summary


def fixed_study(path):
    summary = fields(read(path / "metrics.json"), ("state", "cold_model_setup_s", "pilot_passes", "pilot_seconds",
                                                "forecast_seconds"))
    for filename in ("resources", "tuning-resources"):
        if (path / f"{filename}.json").is_file():
            summary[filename] = resource_summary(path / f"{filename}.json")
    return summary


def figures(evidence, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    rows = {k: v for k, v in evidence["tuning"].items() if k.startswith("v2-s")}
    if rows:
        fig, ax = plt.subplots(figsize=(9, max(3, len(rows) * .45 + 1.2)), layout="constrained")
        names = [key[3:] for key in rows]
        ax.barh(names, [100 * v["selected_wer"] for v in rows.values()], color="#1c778b", alpha=.9, label="Selected")
        ax.scatter([100 * v["oracle"]["50"]["wer"] for v in rows.values()], names, color="#f3aa3c", zorder=3, label="Oracle at up to 50")
        ax.axvline(10, color="#99595a", ls="--", label="10% target")
        ax.invert_yaxis()
        ax.set_xlabel("Corpus word error rate (%)")
        ax.set_title("Search coverage on the 926-trial tuning partition", loc="left", weight="bold")
        ax.legend(loc="best")
        fig.savefig(output / "v2_candidate_coverage.png", dpi=160)
        plt.close(fig)
    if evidence.get("reports") and evidence.get("matched_timing"):
        report = evidence["reports"]["splits"]["dev_report"]["conditions"]
        points = [(k, v) for k, v in evidence["matched_timing"]["conditions"].items() if k in report and v["successes"]]
        if points:
            fig, ax = plt.subplots(figsize=(8.6, 4), layout="constrained")
            for name, row in points:
                ax.scatter(row["latency_seconds"]["p95"] * 1000, report[name]["wer"] * 100, s=80, label=name[3:])
            ax.set(xlabel="Feature-to-text latency p95 (ms)", ylabel="Exploratory dev_report WER (%)")
            ax.set_title("Measured accuracy and complete-pipeline latency", loc="left", weight="bold")
            ax.legend()
            fig.savefig(output / "v2_accuracy_latency.png", dpi=160)
            plt.close(fig)
    if evidence.get("gilbreth_search_timing"):
        timing = evidence["gilbreth_search_timing"]["conditions"]
        labels = {"v2-s0-archived": "Original beam 50", "v2-s3-original-250": "Original beam 250",
                  "v2-s6-repaired-250": "Repaired beam 250"}
        fig, ax = plt.subplots(figsize=(8.3, 3.4), layout="constrained")
        for key, row in timing.items():
            ms = {name: 1000 * val for name, val in row["latency_seconds"].items()}
            y = labels.get(key, key)
            ax.plot([ms["p50"], ms["p99"]], [y, y], color="#86b1bd", linewidth=4, solid_capstyle="round")
            ax.scatter(ms["p95"], y, color="#1c778b", s=65, zorder=3)
            ax.text(ms["p95"], y, f"{ms['p95']:.0f}", ha="center", va="bottom", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Batch-one response time in milliseconds · p50 to p99, dot at p95")
        ax.set_title("Complete feature-to-text requests on one A100", loc="left", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(output / "v2_search_latency.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="append", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--laptop-benchmark", type=Path)
    parser.add_argument("--search-benchmark", type=Path)
    parser.add_argument("--lightbeam", type=Path)
    parser.add_argument("--scheduler-receipt", type=Path)
    parser.add_argument("--search-pilot", action="append", default=[], metavar="JOB_ID=DIRECTORY")
    parser.add_argument("--encoder-run", action="append", default=[], metavar="CONDITION=DIRECTORY")
    parser.add_argument("--fixed-run", action="append", default=[], metavar="JOB_ID=DIRECTORY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {"tuning": dict(map(condition, args.tune))}
    if args.report:
        evidence["reports"] = report_results(args.report)
        benchmark = args.report / "feature-benchmark"
        if (benchmark / "metrics.json").is_file() and (benchmark / "resources.json").is_file():
            evidence["matched_timing"] = latency(benchmark)
        else:
            evidence["matched_timing_status"] = "No complete timing artifacts in the report run"
    if args.laptop_benchmark:
        evidence["laptop_zerolm"] = latency(args.laptop_benchmark)
    if args.search_benchmark:
        evidence["gilbreth_search_timing"] = latency(args.search_benchmark)
    if args.lightbeam:
        evidence["lightbeam"] = lightbeam(args.lightbeam)
    if args.scheduler_receipt:
        evidence["scheduler_steps"] = scheduler(args.scheduler_receipt)
    for spec in args.search_pilot:
        job, directory = spec.split("=", 1)
        if not re.fullmatch(r"[0-9]+", job):
            raise ValueError("Expected a Slurm job ID for the search pilot")
        evidence.setdefault("search_pilots", {})[job] = latency(Path(directory))
    for spec in args.encoder_run:
        name, directory = spec.split("=", 1)
        if not re.fullmatch(r"v2-e[1-3]-[a-z0-9-]+", name):
            raise ValueError("Expected a registered encoder condition ID")
        evidence.setdefault("encoder_studies", {})[name] = encoder_study(Path(directory))
    for spec in args.fixed_run:
        job, directory = spec.split("=", 1)
        if not re.fullmatch(r"[0-9]+", job):
            raise ValueError("Expected a Slurm job ID for the fixed-candidate study")
        evidence.setdefault("fixed_candidate_studies", {})[job] = fixed_study(Path(directory))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "continuation.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figures(evidence, args.output / "figures")


if __name__ == "__main__":
    main()
