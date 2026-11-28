"""Build public aggregate CPU and streaming evidence from completed local runs."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from btt25.assets import sha256
from btt25.data import write_json
from render_public_figures import BLUE, ORANGE, GRID, STYLE, save


PRECISIONS = ("fp32", "int8")
LABELS = {"fp32": "FP32", "int8": "Dynamic INT8"}


def data(path):
    return json.loads(path.read_text(encoding="utf8"))


def lines(path):
    with path.open(encoding="utf8") as file:
        return [json.loads(line) for line in file if line.strip()]


def compact_resource(path):
    summary = data(path / "resources.json")
    if summary["outcome"] != "complete":
        raise ValueError(f"Incomplete run: {path}")
    reported = {
        k: summary[k]
        for k in (
            "elapsed_seconds",
            "sample_count",
            "peak_tree_rss_bytes",
            "peak_tree_uss_bytes",
            "peak_tree_pss_bytes",
            "peak_accounted_host_bytes",
            "minimum_system_available_bytes",
            "host_ceiling_bytes",
            "method",
            "cpu",
            "sampler_seconds",
        )
    }
    setup = data(path / "setup.json")
    if "measured_requests_start_elapsed_s" in setup:
        steady = [
            s
            for s in lines(path / "resources.samples.jsonl")
            if setup["measured_requests_start_elapsed_s"]
            <= s["elapsed_seconds"]
            <= setup["measured_requests_end_elapsed_s"]
        ]
        reported["warm_request_peak_tree_rss_bytes"] = max(s["tree_rss_bytes"] for s in steady)
    return reported


def paired(by_precision, metrics):
    sessions = sorted(metrics["fp32"]["per_session"])
    rng = np.random.default_rng(4369)
    draws = rng.integers(0, len(sessions), size=(10000, len(sessions)))
    paired_results = {}
    for metric in ("word", "phoneme"):
        trials = {
            p: {r["trial_id"]:r[metric]["errors"] for r in metrics[p]["trial_edits"]} for p in PRECISIONS
        }
        if trials["fp32"].keys() != trials["int8"].keys():
            raise ValueError("Precision arms differ in scored trial IDs")
        trial_deltas = [trials["int8"][key] - errors for key,errors in trials["fp32"].items()]
        denom = np.array([metrics["fp32"]["per_session"][s][metric]["reference_length"] for s in sessions])
        edits = np.array(
            [
                metrics["int8"]["per_session"][s][metric]["errors"]
                - metrics["fp32"]["per_session"][s][metric]["errors"]
                for s in sessions
            ]
        )
        interval = np.percentile(100 * edits[draws].sum(axis=1) / denom[draws].sum(axis=1), [2.5, 97.5])
        paired_results[metric] = {
            "fewer_errors_on_int8_trials":sum(delta < 0 for delta in trial_deltas),
            "more_errors_on_int8_trials":sum(delta > 0 for delta in trial_deltas),
            "equal_error_count_trials":sum(delta == 0 for delta in trial_deltas),
            "int8_minus_fp32_errors": int(edits.sum()),
            "percentage_points": 100 * edits.sum() / denom.sum(),
            "session_bootstrap_95_pp": interval.tolist(),
        }
    fp, quant = by_precision["fp32"], by_precision["int8"]
    if set(fp) != set(quant):
        raise ValueError("Precision arms differ in membership")
    paired_results["sentence_predictions_changed"] = sum(
        fp[trial]["text"] != quant[trial]["text"] for trial in fp
    )
    return paired_results


def draft_cadence(trace, input_seconds):
    previous, last_change, intervals = {}, {}, []
    changes = 0
    for row in trace:
        trial, words = row["trial_id"], row["draft"]
        if words != previous.get(trial, []):
            changes += 1
            if trial in last_change:
                intervals.append(row["elapsed_s"] - last_change[trial])
            last_change[trial] = row["elapsed_s"]
        previous[trial] = words
    return {
        "decoder_updates":len(trace), "changed_drafts":changes,
        "changed_drafts_per_recorded_second":changes/input_seconds,
        "median_interval_between_changes_within_trial_s":float(np.median(intervals)) if intervals else None,
    }


def compile_public(pilot, campaign):
    if data(campaign / "state.json")["status"] != "complete":
        raise RuntimeError("Campaign has outstanding work")
    by_precision, metrics, arms = {}, {}, {}
    pilot_setup = data(pilot / "setup.json")
    registry = data(campaign / "conditions.json") if (campaign / "conditions.json").is_file() else {}

    def run_path(name):
        return campaign / registry.get(name, name)

    timing = data(run_path("timing") / "metrics.json")
    timing_rows = lines(run_path("timing") / "predictions.jsonl")
    hashes = {}
    for precision in PRECISIONS:
        path = run_path(f"score-{precision}")
        metrics[precision] = data(path / "metrics.json")[precision]
        by_precision[precision] = {r["trial_id"]: r for r in lines(path / "predictions.jsonl")}
        replay = run_path(f"replay-{precision}")
        arms[precision] = {
            "accuracy": {k: metrics[precision][k] for k in ("overall", "per_session", "per_corpus")},
            "serialized_state_bytes": pilot_setup[precision]["serialized_bytes"],
            "state_sha256": pilot_setup[precision]["sha256"],
            "score_resources": compact_resource(path),
            "setup": data(path / "setup.json"),
            "warm_latency": timing[precision],
            "streaming": data(replay / "metrics.json"),
            "replay_resources": compact_resource(replay),
            "runtime": yaml.safe_load((path / "config.resolved.yaml").read_text(encoding="utf8")),
        }
        arms[precision]["streaming"]["replay_clock"] = [
            {
                key: r[key]
                for key in (
                    "input_seconds",
                    "first_phone_update_s",
                    "first_partial_s",
                    "first_stable_word_s",
                    "finalization_after_last_scheduled_bin_s",
                    "draft_revision_events",
                )
            }
            for r in lines(replay / "predictions.jsonl")
        ]
        arms[precision]["streaming"]["draft_cadence"] = draft_cadence(
            lines(replay / "drafts.jsonl"), arms[precision]["streaming"]["input_seconds"]
        )
        repeated = defaultdict(set)
        for r in timing_rows:
            if r["precision"] == precision:
                repeated[r["trial_id"]].add(r["text"])
        arms[precision]["warm_timing_text_repeatability"] = {
            "trials": len(repeated),
            "three_passes_all_agree": sum(len(texts) == 1 for texts in repeated.values()),
            "at_least_two_final_texts": sum(len(texts) > 1 for texts in repeated.values()),
        }
    for path in [
        pilot,
        *(run_path(name) for name in ("score-fp32", "score-int8", "timing", "replay-fp32", "replay-int8")),
    ]:
        hashes[path.name] = {
            name: sha256(path / name)
            for name in (
                "config.resolved.yaml",
                "resources.json",
                "resources.samples.jsonl",
                "setup.json",
                "predictions.jsonl",
                "metrics.json",
                "report.md",
            )
        }
    runtime = arms["fp32"]["runtime"]
    witness = None
    vm_path = pilot / "windows-vm.jsonl"
    if vm_path.is_file():
        phases = defaultdict(list)
        for entry in lines(vm_path):
            if entry["phase"] and entry["all_wsl_vm_working_set_bytes"] is not None:
                phases[entry["phase"]].append(entry["all_wsl_vm_working_set_bytes"])
        witness = {
            "scope": "Windows working set of the entire WSL VM, including other WSL activity and resident caches",
            "interval_seconds": 2,
            "observations": {k: {"samples":len(v), "peak_bytes":max(v)} for k,v in phases.items()},
            "private_trace_sha256": sha256(vm_path),
        }
    return {
        "scope": "dev_tune, T15 copy-task intracortical release",
        "cpu": runtime["cpu"],
        "threads": runtime["cpu_threads"],
        "gpu_used": False,
        "windows_wsl_vm_observer": witness,
        "cases": arms,
        "paired": paired(by_precision, metrics),
        "paired_interval": "10,000 paired bootstrap draws of entire observed tuning sessions; seed 4369",
        "timing": "3 interleaved passes over the same 96 length-stratified tuning trials after warmup, one request at a time",
        "pilot": compact_resource(pilot),
        "admission": data(campaign / "admission.json"),
        "artifacts": hashes,
    }


def figure(result, output):
    with plt.rc_context(STYLE):
        fig, (latency, accuracy) = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
        fig.suptitle("Quantizing the laptop GRU", x=0.02, ha="left", weight="bold", fontsize=17)
        quantiles = ["p50", "p90", "p95", "p99"]
        for precision, color, offset in zip(PRECISIONS, (BLUE, ORANGE), (-0.09, 0.09)):
            case = result["cases"][precision]
            ys = [1000 * case["warm_latency"]["latency_seconds"][key] for key in quantiles]
            latency.plot(range(4), ys, color=color, marker="o", label=LABELS[precision], lw=2)
            q = case["accuracy"]["overall"]
            accuracy.scatter(np.arange(2) + offset, [100 * q["per"], 100 * q["wer"]], color=color, s=50)
            for i, value in enumerate((100 * q["per"], 100 * q["wer"])):
                accuracy.annotate(
                    f"{value:.2f}%",
                    (i + offset, value),
                    xytext=(-5 if offset < 0 else 5, 9 if offset < 0 else -14),
                    textcoords="offset points",
                    color=color,
                    ha="center",
                    fontsize=9,
                )
        latency.set(
            xticks=range(4),
            xticklabels=["p50", "p90", "p95", "p99"],
            ylabel="Feature-to-text latency (ms)",
            title="Batch-one request latency",
        )
        accuracy.set(
            xticks=[0, 1],
            xticklabels=["Phoneme error", "Word error"],
            ylabel="Corpus error rate (%)",
            xlim=(-0.55, 1.55),
            title="926 tuning sentences",
        )
        for ax in (latency, accuracy):
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
            ax.grid(axis="y", color=GRID)
            ax.set_axisbelow(True)
        latency.legend(frameon=False, loc="upper left", fontsize=9)
        save(fig, output, "consumer_v4_cpu")

        fig, ax = plt.subplots(figsize=(9.5, 4.4), layout="constrained")
        clock = sorted(
            result["cases"]["int8"]["streaming"]["replay_clock"], key=lambda r: r["input_seconds"]
        )
        y = np.arange(len(clock))
        ax.hlines(y, 0, [r["input_seconds"] for r in clock], color=GRID, lw=2)
        for key, label, color, marker in (
            ("first_partial_s", "First draft", BLUE, "o"),
            ("first_stable_word_s", "First word settles", ORANGE, "|"),
            ("input_seconds", "Last input bin", "#465463", "s"),
        ):
            ax.scatter(
                [np.nan if r[key] is None else r[key] for r in clock],
                y,
                color=color,
                s=105 if marker == "|" else 35,
                marker=marker,
                linewidths=2,
                label=label,
                zorder=4 if marker == "|" else 3,
            )
        ax.axvline(0.44, color="#697986", ls=":", lw=1.2, label="First eligible phone")
        ax.set(
            yticks=y,
            yticklabels=[f"{r['input_seconds']:.1f} s" for r in clock],
            ylabel="Recorded trial duration",
            xlabel="Seconds since first bin interval began",
            xlim=(0, None),
        )
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID)
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, fontsize=9)
        fig.suptitle(
            "When the first word appears during INT8 replay", x=0.02, ha="left", weight="bold", fontsize=17
        )
        save(fig, output, "consumer_v4_replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--aggregate", type=Path, help="Redraw both figures from public JSON")
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--windows-pilot", type=Path)
    parser.add_argument("--windows-campaign", type=Path)
    parser.add_argument("--two-rate-replay", type=Path, help="Completed trained 40 ms deployment check")
    args = parser.parse_args()
    if args.aggregate:
        if any((args.pilot, args.campaign, args.windows_pilot, args.windows_campaign, args.two_rate_replay)):
            parser.error("Use an aggregate or private run directories")
        figure(data(args.aggregate), args.reports / "figures")
        return
    if not args.pilot or not args.campaign:
        parser.error("Supply --pilot and --campaign to assemble a new report")
    result = compile_public(args.pilot, args.campaign)
    if bool(args.windows_pilot) != bool(args.windows_campaign):
        raise ValueError("Supply both Windows directories")
    if args.windows_pilot:
        result["windows_zerolm"] = compile_public(args.windows_pilot, args.windows_campaign)
    if args.two_rate_replay:
        path = args.two_rate_replay
        replay = data(path / "metrics.json")
        cfg = yaml.safe_load((path / "config.resolved.yaml").read_text())
        if cfg["encoder"] != "two_rate" or cfg["phone_step_ms"] != 40:
            raise ValueError("Invalid two-rate deployment condition")
        registry = data(args.campaign/"conditions.json")
        control_cfg = yaml.safe_load((args.campaign / registry["replay-fp32"] / "config.resolved.yaml").read_text())
        if any(control_cfg[k] != cfg[k] for k in ("id_hash","checkpoint","lexicon","lm_sha256","cpu_threads")):
            raise ValueError("The replay subset or decoder differs from the FP32 control")
        result["two_rate_replay"] = {
            "metrics":replay,"delta_sha256":cfg["delta_sha256"],"id_hash":cfg["id_hash"],
            "resources":compact_resource(path),"draft_cadence":draft_cadence(lines(path/"drafts.jsonl"),replay["input_seconds"]),
            "hashes":{name:sha256(path/name) for name in ("config.resolved.yaml","metrics.json","resources.json","predictions.jsonl","drafts.jsonl")},
        }
    write_json(args.reports / "consumer_v4.json", result)
    figure(result, args.reports / "figures")


if __name__ == "__main__":
    main()
