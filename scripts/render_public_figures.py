"""Redraw public figures from aggregate JSON. No model or dataset is required."""
from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

INK, MUTED, GRID = "#172a39", "#486173", "#e0e7ed"
BLUE, ORANGE, GRAY = "#136d85", "#a74b1c", "#687987"
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#becbd4", "svg.hashsalt": "btt-public-figures",
}
V1_NAMES = {
    "A0": "Compact", "A1": "Score every trial", "A2": "Confidence gate",
    "B1": "Head only", "B2": "Head + session bias",
}
LAPTOP_NAMES = {
    "baseline_inference": "Laptop encoder", "lexicon_decode": "Laptop lexicon decoder",
    "session_bias_pilot": "Laptop session-bias pilot", "head_pilot": "Laptop head pilot",
    "head_training": "Laptop head training", "session_bias_training": "Laptop session-bias training",
}
V2_NAMES = {
    "v2-s0-archived": "Archived beam 50 (up to 10)",
    **{f"v2-s{i}-{variant}-{beam}": f"{variant.title()} beam {beam}"
       for i, (variant, beam) in enumerate(
           [(variant, beam) for variant in ("original", "repaired") for beam in (50, 100, 250)], 1)},
    "v2-f1-distilgpt2": "Repaired + DistilGPT2", "v2-e2-monophone": "Monophone control",
}


def v1_name(key):
    if key.startswith("A3_seed"):
        return f"Random gate ({key.removeprefix('A3_seed')})"
    return V1_NAMES[key]


def panels(title, *, count=1, rows=4, width=9):
    fig, axes = plt.subplots(1, count, figsize=(width, 2.05 + .40 * rows),
                             sharey=True, layout="constrained", squeeze=False)
    fig.suptitle(title, x=.015, ha="left", fontsize=17, weight="bold")
    for ax in axes[0]:
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=.8)
        ax.tick_params(axis="y", length=0, pad=8)
    return fig, axes[0]


def rows_on(ax, labels):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .45, -.55)


def save(fig, directory, name, *, vector=False):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=180, facecolor="white", bbox_inches="tight",
                metadata={"Software": "BTT-25-CW"})
    if vector:
        path = directory / f"{name}.svg"
        fig.savefig(path, facecolor="white", bbox_inches="tight", metadata={"Date": None})
        svg = path.read_text(encoding="utf-8")
        path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def dot(ax, x, y, label, *, color=BLUE, marker="o"):
    if x is None:
        ax.text(.02, y, "Unavailable", transform=ax.get_yaxis_transform(), va="center", color=MUTED, fontsize=9)
        return
    ax.scatter(x, y, marker=marker, color=color, s=45, zorder=3)
    if label:
        ax.annotate(label, (x, y), xytext=(7, 0), textcoords="offset points",
                    va="center", color=color, fontsize=9)


def legend(ax):
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=3, borderaxespad=0)


def render_v1(data, output):
    with plt.rc_context(STYLE):
        score = data["results"]
        names = list(score["dev_report"])
        fig, (ax,) = panels("V1 word errors across both validation splits", rows=len(names))
        rows_on(ax, list(map(v1_name, names)))
        for i, key in enumerate(names):
            a, b = (score[split][key]["wer"] for split in ("dev_report", "official_val"))
            if a is not None and b is not None:
                ax.plot([100*a, 100*b], [i, i], color=GRID, lw=3)
            dot(ax, None if a is None else 100*a, i, "", color=BLUE)
            dot(ax, None if b is None else 100*b, i, "", color=ORANGE, marker="D")
        ax.scatter([], [], color=BLUE, label="Held-back report")
        ax.scatter([], [], color=ORANGE, marker="D", label="Full validation")
        legend(ax)
        ax.set_xlabel("Corpus word error rate (%)")
        save(fig, output, "error_rates", vector=True)

        pairs = list(data["comparisons"])
        fig, axes = panels("Paired error changes in the v1 report", count=2, rows=len(pairs), width=11)
        rows_on(axes[0], [key.replace("-minus-", " − ").replace("_seed", " ") for key in pairs])
        for ax, metric, title in zip(axes, ("wer", "per"), ("Word errors", "Phone errors")):
            for i, key in enumerate(pairs):
                value = data["comparisons"][key][metric]
                delta = value.get("delta_b_minus_a")
                if not value.get("measured") or delta is None:
                    continue
                ci = value["session_cluster_bootstrap"]
                if ci["lower"] is not None:
                    ax.hlines(i, 100*ci["lower"], 100*ci["upper"], color=BLUE, lw=2)
                dot(ax, 100*delta, i, "")
            ax.axvline(0, color=GRAY, ls="--", lw=1)
            ax.set_title(title, loc="left", color=INK, fontsize=12, weight="bold")
            ax.set_xlabel("Change (percentage points)")
        save(fig, output, "paired_changes", vector=True)

        policies = data["confidence"]["conditions"]
        saved = policies["A1"]["lm_calls"] - policies["A2"]["lm_calls"]
        title = f"The confidence gate cut {saved:,} scorer calls" if saved > 0 else "Scorer calls and v1 word errors"
        fig, (ax,) = panels(title, rows=len(policies))
        offsets = {
            "A0": (8, 0), "A1": (-8, 0), "A2": (8, -9),
            "A3_seed20261116": (8, -13), "A3_seed20261117": (8, 6), "A3_seed20261118": (8, 24),
        }
        for key, row in policies.items():
            wer = score["dev_report"][key]["wer"]
            if wer is None:
                continue
            x, y = row["lm_calls"], 100*wer
            color = BLUE if key == "A2" else GRAY
            ax.scatter(x, y, color=color, s=48, zorder=3)
            dx, dy = offsets.get(key, (8, 0))
            label = v1_name(key)
            ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                        va="center", ha="left" if dx > 0 else "right", fontsize=9, color=color,
                        arrowprops={"arrowstyle": "-", "color": color, "lw": .6} if key.startswith("A3") else None)
        ax.margins(x=.15, y=.24)
        ax.set_xlabel("Additional language-model calls, full validation")
        ax.set_ylabel("Report WER (%)")
        save(fig, output, "rescoring_cost", vector=True)

        runs = [(LAPTOP_NAMES[key], row) for key, row in data["resources"]["laptop"].items()]
        if "encoder_parity" in data:
            runs.append(("A100 encoder", data["encoder_parity"]["resources"]))
        runs += [(v1_name(key), row) for key, row in data["resources"]["study"].items()]
        fig, axes = panels("V1 memory by stage", count=2, rows=len(runs), width=10)
        rows_on(axes[0], [name for name, _ in runs])
        for ax, title in zip(axes, ("Host RSS (GB)", "Device-wide GPU use (GB)")):
            for i, (_, row) in enumerate(runs):
                value = (row.get("peak_tree_rss_bytes", row.get("peak_rss_bytes")) if ax is axes[0]
                         else row.get("peak_device_used_bytes"))
                x = None if value is None else value / 1e9
                dot(ax, x, i, "" if x is None else f"{x:.2f}")
            ax.set_xlim(left=0)
            ax.margins(x=.25)
            ax.set_xlabel(title)
        save(fig, output, "resource_footprint", vector=True)


def latency_marks(ax, row, i):
    p = {key: 1000*value for key, value in row["latency_seconds"].items()}
    ax.hlines(i, p["p50"], p["p99"], color="#97b3c1", lw=2.5)
    dot(ax, p["p50"], i, "", color=GRAY, marker="|")
    dot(ax, p["p95"], i, "", marker="D")
    dot(ax, p["p99"], i, "", color=GRAY, marker="|")
    ax.annotate(f"{p['p95']:.0f}", (p["p95"], i), xytext=(0, 8), textcoords="offset points",
                ha="center", color=BLUE, fontsize=9)


def render_v2(data, output):
    with plt.rc_context(STYLE):
        search = [(key, row) for key, row in data["tuning"].items() if key.startswith("v2-s")]
        if search:
            fig, (ax,) = panels("Repaired search narrowed the candidate gap", rows=len(search))
            rows_on(ax, [V2_NAMES.get(key, key) for key, _ in search])
            for i, (_, row) in enumerate(search):
                oracle, selected = 100*row["oracle"]["50"]["wer"], 100*row["selected_wer"]
                ax.hlines(i, oracle, selected, color=GRID, lw=4)
                dot(ax, oracle, i, "", marker="D")
                dot(ax, selected, i, f"{selected:.1f}%", color=ORANGE)
                ax.annotate(f"{oracle:.1f}%", (oracle, i), xytext=(0, 9), textcoords="offset points",
                            ha="center", color=BLUE, fontsize=9)
            ax.axvline(10, color=MUTED, ls="--", lw=1, label="10% target")
            ax.scatter([], [], marker="D", color=BLUE, label="Oracle, up to 50")
            ax.scatter([], [], color=ORANGE, label="Selected text")
            legend(ax)
            ax.set_xlabel("Tuning word error rate (%)")
            ax.margins(x=.18)
            save(fig, output, "v2_candidate_coverage")

        if data.get("reports") and data.get("matched_timing"):
            score = data["reports"]["splits"]["dev_report"]["conditions"]
            timing = data["matched_timing"]["conditions"]
            order = ("v2-s1-original-50", "v2-s6-repaired-250", "v2-f1-distilgpt2", "v2-e2-monophone")
            ordered_keys = (*order, *(key for key in timing if key not in order))
            names = [key for key in ordered_keys if key in score and key in timing and timing[key]["successes"]]
            if names:
                fig, (err, lat) = panels("Fewer word errors, longer responses", count=2, rows=len(names), width=10)
                rows_on(err, [V2_NAMES.get(key, key) for key in names])
                for i, key in enumerate(names):
                    color = BLUE if key == "v2-f1-distilgpt2" else GRAY
                    x = 100*score[key]["wer"]
                    dot(err, x, i, f"{x:.2f}%", color=color)
                    latency_marks(lat, timing[key], i)
                err.margins(x=.25)
                lat.margins(x=.13)
                err.set_xlabel("Exploratory report WER (%)")
                lat.set_xlabel("Latency (ms) · p50—p99, diamond at p95")
                save(fig, output, "v2_accuracy_latency")

        if data.get("gilbreth_search_timing"):
            timing = data["gilbreth_search_timing"]["conditions"]
            fig, (ax,) = panels("Wider search raised p95 latency", rows=len(timing))
            rows_on(ax, ["Original beam 50" if key == "v2-s0-archived" else V2_NAMES.get(key, key) for key in timing])
            for i, row in enumerate(timing.values()):
                if row["successes"]:
                    latency_marks(ax, row, i)
            ax.set_xlabel("Feature-to-text latency (ms) · p50—p99, diamond at p95")
            save(fig, output, "v2_search_latency")

        fixed = data.get("fixed_candidate_studies", {})
        jobs = (
            ("Full feature benchmark", data.get("matched_timing", {}).get("resources")),
            ("DistilGPT2 + Qwen 1.7B", fixed.get("12523725", {}).get("resources")),
            ("Qwen 4B", fixed.get("12523757", {}).get("resources")),
            ("LightBeam pilot", data.get("lightbeam", {}).get("resources")),
            ("Transformer resume", data.get("transformer_resume", {}).get("train", {}).get("resources")),
        )
        jobs = [(name, row) for name, row in jobs if row]
        if len(jobs) >= 2:
            fig, (host, gpu) = panels("File cache drove the Transformer’s host peak", count=2, rows=len(jobs), width=10)
            rows_on(host, [name for name, _ in jobs])
            for i, (_, row) in enumerate(jobs):
                for key, offset, color, marker in (("peak_cgroup_bytes", -.14, BLUE, "o"),
                                                   ("peak_tree_rss_bytes", .14, ORANGE, "D")):
                    value = row.get(key)
                    if value is not None:
                        dot(host, value/1e9, i+offset, "", color=color, marker=marker)
                value = row.get("peak_device_used_bytes")
                dot(gpu, None if value is None else value/1e9, i, "" if value is None else f"{value/1e9:.2f}")
            host.axvline(12, ls="--", lw=1, color=MUTED, label="12 GB target")
            host.scatter([], [], color=BLUE, label="Cgroup")
            host.scatter([], [], color=ORANGE, marker="D", label="RSS")
            legend(host)
            host.margins(x=.10)
            host.set_xlim(left=0)
            gpu.set_xlim(left=0)
            gpu.margins(x=.25)
            host.set_xlabel("Host memory (GB)")
            gpu.set_xlabel("Device-wide GPU memory (GB)")
            host.xaxis.set_major_locator(MaxNLocator(5))
            save(fig, output, "v2_memory_accounting")


def render_temporal(data, output):
    conditions = data["conditions"]
    labels = {
        "baseline-s4": "Official Gaussian",
        "raw-s4": "Unsmoothed input",
        "gaussian-1-s4": "Lighter Gaussian",
        "box-5-s4": "Five-bin moving average",
        "coarse-40-s4": "40 ms input averages",
        "coarse-80-s4": "80 ms input averages",
        "baseline-s2": "40 ms GRU output step",
        "baseline-s1": "20 ms GRU output step",
    }
    with plt.rc_context(STYLE):
        fig, (ax,) = panels("The frozen GRU lost accuracy at faster output steps", rows=len(labels), width=9)
        rows_on(ax, list(labels.values()))
        for i, key in enumerate(labels):
            row = conditions[key]
            ax.hlines(i, 100 * row["per"], 100 * row["wer"], color=GRID, lw=2)
            dot(ax, 100 * row["per"], i, "", color=BLUE)
            ax.annotate(f"{row['per']:.1%}", (100 * row["per"], i), xytext=(-7, 0),
                        textcoords="offset points", ha="right", va="center", color=BLUE, fontsize=9)
            dot(ax, 100 * row["wer"], i, f"{row['wer']:.1%}", color=ORANGE, marker="D")
        ax.scatter([], [], color=BLUE, label="Phoneme error rate")
        ax.scatter([], [], color=ORANGE, marker="D", label="Word error rate")
        legend(ax)
        ax.set_xlim(0, 85)
        ax.set_xlabel("Corpus error rate on 926 tuning trials (%)")
        save(fig, output, "temporal_screen")

        phone_input = data.get("reference_phone", {}).get("repaired_kenlm_spike")
        if phone_input is None:
            return
        counts = phone_input["overall"]
        fig, (ax,) = panels("Word errors drop when the decoder gets reference phones", rows=2, width=9)
        rows_on(ax, ["Flashlight + KenLM", "Exact dictionary lookup"])
        for y, (actual, reference) in enumerate((
            ("saved_errors", "reference_phone_errors"),
            ("neural_exact_lexicon_errors", "exact_lexicon_errors"),
        )):
            low, high = [100 * counts[k] / counts["reference_words"] for k in (reference, actual)]
            ax.hlines(y, low, high, color=GRID, lw=2)
            dot(ax, low, y, f"{low:.2f}%", color=BLUE)
            dot(ax, high, y, f"{high:.2f}%", color=ORANGE, marker="D")
        ax.scatter([], [], color=ORANGE, marker="D", label="GRU phone evidence")
        ax.scatter([], [], color=BLUE, label="Reference phones and boundaries")
        legend(ax)
        ax.set_xlim(0, 26)
        ax.set_xlabel("Word error rate on 926 tuning trials (%)")
        save(fig, output, "reference_phone_diagnostic")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    for stem, render in (("results", render_v1), ("continuation", render_v2)):
        data = json.loads((args.reports / f"{stem}.json").read_text(encoding="utf-8"))
        render(data, args.reports / "figures")
    temporal = args.reports / "temporal.json"
    if temporal.is_file():
        render_temporal(json.loads(temporal.read_text(encoding="utf-8")), args.reports / "figures")


if __name__ == "__main__":
    main()
