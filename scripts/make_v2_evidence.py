"""Export allowlisted aggregate evidence for the public continuation report."""
import argparse
import json
from pathlib import Path
import re


SEARCH_FIELDS = ("reference_words", "trials", "successful", "selected_errors", "selected_wer", "substitutions",
                 "deletions", "insertions", "oracle", "candidate_count_histogram", "lexicon_oov_tokens",
                 "lm_oov_tokens", "chosen_weight")
RESOURCE_FIELDS = ("elapsed_seconds", "host_ceiling_bytes", "slurm_requested_bytes", "cgroup_effective_limit_bytes",
                   "peak_accounted_host_bytes", "peak_cgroup_bytes", "peak_tree_rss_bytes", "peak_tree_pss_bytes",
                   "peak_tree_uss_bytes", "peak_rss_bytes", "peak_device_used_bytes", "peak_cuda_allocated_bytes",
                   "peak_cuda_reserved_bytes", "cpu", "sampler_seconds", "method", "outcome", "cgroup_events_delta")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fields(data, keys):
    return {k: data[k] for k in keys if k in data}


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
        metrics["decode_resources"] = fields(read(resources), RESOURCE_FIELDS)
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
        result["splits"][name] = {
            "conditions": {key: fields(val, selected) for key, val in data["conditions"].items()},
            "paired_wer": data["paired_wer"], "paired_by_corpus": data["paired_by_corpus"],
        }
    return result


def latency(path):
    metrics = read(path / "metrics.json")
    names = metrics.get("by_condition", metrics.get("by_beam"))
    return {**fields(metrics, ("state", "mode", "streaming", "completed_passes", "attempted_requests", "memory_scope")),
            "common_encoder_cold_s": metrics["cold_setup"].get("encoder_s"),
            "conditions": {k: {**v, "setup": fields(metrics["cold_setup"].get(k, {}),
                 ("mode", "native_setup_s", "process_ready_s", "warmup_s", "encoder_setup_s", "neural_setup_s"))}
                           for k, v in names.items()}, "resources": fields(read(path / "resources.json"), RESOURCE_FIELDS)}


def lightbeam(path):
    from btt25.benchmark import timing_summary

    summary = fields(read(path / "metrics.json"), ("status", "admitted_as", "passes", "warm_seconds",
                     "scope", "rows", "cold_setup_s"))
    resource = path / "resources.json"
    if resource.is_file():
        summary["resources"] = fields(read(resource), RESOURCE_FIELDS)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="append", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--laptop-benchmark", type=Path)
    parser.add_argument("--search-benchmark", type=Path)
    parser.add_argument("--lightbeam", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {"tuning": dict(map(condition, args.tune))}
    if args.report:
        evidence["reports"] = report_results(args.report)
        evidence["matched_timing"] = latency(args.report / "feature-benchmark")
    if args.laptop_benchmark:
        evidence["laptop_zerolm"] = latency(args.laptop_benchmark)
    if args.search_benchmark:
        evidence["gilbreth_search_timing"] = latency(args.search_benchmark)
    if args.lightbeam:
        evidence["lightbeam"] = lightbeam(args.lightbeam)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "continuation.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figures(evidence, args.output / "figures")


if __name__ == "__main__":
    main()
