"""Draw cached-decoder accuracy and request latency from the published rate study."""
import argparse
import json
from pathlib import Path

from render_public_figures import BLUE, ORANGE, STYLE, plt


def plot(source, output):
    data = json.loads(source.read_text(encoding="utf-8"))
    conditions = data["conditions"]
    warm = data["warm_timing"]["conditions"]
    methods = ("native", "temperature2", "pair_average", "merge_agree")
    labels = ("Every frame", "Temperature 2", "Average every pair", "Merge agreeing pairs")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=True)
        for name, label, color, mark, shift in (
            ("gru40", "40 ms GRU", BLUE, "o", -.12),
            ("two_rate", "80 ms GRU + 40 ms midpoint", ORANGE, "s", .12),
        ):
            ids = [f"{name}/{method}" for method in methods]
            wer = [100 * conditions[key]["wer"] for key in ids]
            matched_ms = [1000 * warm[key]["p95"] for key in ids]
            for ax, x in zip(axes, (wer, matched_ms), strict=True):
                y = [i + shift for i in range(len(methods))]
                ax.scatter(x, y, c=color, marker=mark, s=48, label=label, zorder=3)
                for xx, yy in zip(x, y, strict=True):
                    ax.annotate(f"{xx:.1f}", (xx, yy), xytext=(7, 0), textcoords="offset points",
                                va="center", fontsize=9, color=color)
        for ax, label in zip(axes, ("Tuning WER (%)", "Matched warm cached-logit-to-text p95 (ms)"), strict=True):
            ax.set_xlabel(label)
            ax.grid(axis="x", color="#e0e7ed")
            ax.set_axisbelow(True)
            left, right = ax.get_xlim()
            ax.set_xlim(left, right + max((right-left)*.24, .1))
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].invert_yaxis()
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, frameon=False, loc="upper center", bbox_to_anchor=(.61,.98), ncol=2)
        axes[1].tick_params(axis="y", length=0)
        fig.subplots_adjust(left=.20, right=.98, bottom=.18, top=.82, wspace=.17)
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    plot(args.source, args.output)
