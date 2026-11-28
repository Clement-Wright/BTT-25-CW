"""Draw the rate-study figures from public aggregate counts."""
from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from render_public_figures import BLUE, GRID, INK, MUTED, ORANGE, STYLE, dot, panels, rows_on, save


LABELS = {"official": "Official GRU · 80 ms", "gru80": "Trained GRU · 80 ms",
          "gru40": "Trained GRU · 40 ms", "gru20": "Trained GRU · 20 ms", "two_rate": "Two-rate GRU · 80 / 40 ms",
          "patch10": "10-bin patch · 200 ms", "patch20": "20-bin patch · 400 ms",
          "bin40": "40 ms bins · paired means", "bin10_hold": "10 ms grid · held samples",
          "adaptive_filter": "Adaptive channel filter", "activation_only": "Learned activation",
          "transform_only": "Low-rank input transform", "input_transform": "Transform + activation",
          "causal_conv": "Causal convolution"}


def clock_figure(counts, directory):
    positions = [["native80", "repeat80_on40"], ["fast40_on80", "native40"]]
    palette = LinearSegmentedColormap.from_list("errors", ["#f3f6f8", "#c1d8e0", "#478596", "#154a5e"])
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.7), layout="constrained")
    for ax, unit, title in zip(axes, ["phone", "word"], ["Phone error rate", "Word error rate"], strict=True):
        values = np.array([[100*counts[k][unit+"_errors"]/counts[k][unit+"_reference_length"]
                            for k in row] for row in positions])
        ax.imshow(values, vmin=0, vmax=45, cmap=palette, aspect="auto")
        for (i,j), value in np.ndenumerate(values):
            color = "white" if value >= 28 else INK
            ax.text(j, i-.07, f"{value:.2f}%", ha="center", va="center", fontsize=19, weight="bold", color=color)
            errors = counts[positions[i][j]][unit+"_errors"]
            ax.text(j, i+.16, f"{errors:,} edits", ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks([0, 1], ["80 ms", "40 ms"])
        ax.set_yticks([0, 1], ["80 ms", "40 ms"])
        ax.set_xlabel("Emission interval")
        ax.set_xticks([.5], minor=True)
        ax.set_yticks([.5], minor=True)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, color=INK, loc="left", weight="bold", fontsize=13, pad=8)
    axes[0].set_ylabel("Recurrent update interval")
    fig.suptitle("The frozen GRU and word search run on different clocks",
                 x=.015, ha="left", fontsize=16, weight="bold")
    save(fig, directory, "v4_clock")


def model_figures(models, directory):
    names = [name for name in LABELS if name in models]
    fig, axes = panels("Input timing and encoder accuracy", count=2, rows=len(names), width=10)
    central = {"official", "gru20", "gru40", "gru80", "two_rate"}
    for ax, metric, title in zip(axes, ("per", "wer"), ("Phone error rate (%)", "Word error rate (%)"), strict=True):
        maxval = max(models[name][metric]*100 for name in names)
        for i, name in enumerate(names):
            value = models[name][metric]*100
            dot(ax, value, i, f"{value:.2f}", color=BLUE if name in central else MUTED)
        rows_on(ax, [LABELS[n] for n in names])
        ax.set_xlim(0, maxval*1.19)
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
    save(fig, directory, "v4_rate_models")

    names = [n for n in ("patch10", "patch20", "bin40", "bin10_hold", "adaptive_filter",
             "activation_only", "transform_only", "input_transform") if n in models and "gru40" in models]
    fig, (ax,) = panels("Which input changes helped the 40 ms GRU?", rows=len(names), width=8)
    ax.axvline(0., color=INK, linewidth=1.1)
    for i, name in enumerate(names):
        paired = models[name]["paired"]["gru40_word"]
        lo, delta, hi = [100*paired[k] for k in ("lower", "delta", "upper")]
        color = BLUE if delta <= 0 else ORANGE
        ax.hlines(i, lo, hi, color=color, linewidth=2)
        ax.scatter(delta, i, color=color, s=36, zorder=3)
    rows_on(ax, [LABELS[n] for n in names])
    ax.set_xlabel("Change in word error rate (percentage points) · lower is better")
    ax.set_title("Paired 95% intervals by recording session and block", loc="left", fontsize=11, color=MUTED)
    save(fig, directory, "v4_input_differences")

    names = [n for n in ("official", "gru80", "gru40", "two_rate", "causal_conv") if n in models]
    fig, axes = panels("One trial from released features to final text", count=3, rows=len(names), width=11)
    for ax, label, measure, color, fmt in [
            (axes[0], "P95 latency (ms)", lambda t: t["seconds"]["p95"]*1000, BLUE, ".0f"),
            (axes[1], "Peak process-tree RSS (GB)", lambda t: t["peak_tree_rss_bytes"]/1e9, MUTED, ".2f"),
            (axes[2], "Peak PyTorch allocation (GB)", lambda t: t["peak_cuda_allocated_bytes"]/1e9, ORANGE, ".2f")]:
        values = [measure(models[name]["timing"]) for name in names]
        for i, value in enumerate(values):
            dot(ax, value, i, format(value, fmt), color=color)
        rows_on(ax, [LABELS[n] for n in names])
        ax.set_xlim(0, max(values)*1.38)
        ax.set_title(label, loc="left", fontsize=10.5, weight="bold")
    save(fig, directory, "v4_rate_cost")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=Path("reports/rate_v4.json"))
    p.add_argument("--clock", type=Path)
    p.add_argument("--output", type=Path, default=Path("reports/figures"))
    args = p.parse_args()
    with plt.rc_context(STYLE):
        if args.clock:
            clock_figure(json.loads(args.clock.read_text())["counts"], args.output)
        else:
            summary = json.loads(args.summary.read_text())
            clock_figure(summary["recurrent_clock"], args.output)
            model_figures(summary["models"], args.output)
