"""Redraw the held-session scalp EEG confusion matrices from aggregate counts."""
import argparse
import json
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from render_public_figures import BLUE, INK, STYLE, plt, save


def draw(source, directory):
    data = json.loads(source.read_text(encoding="utf8"))
    colors = LinearSegmentedColormap.from_list("command_count", ["#f4f8fa", BLUE])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.9, 4.4), constrained_layout=True)
        vmax = max(max(row) for c in data["conditions"].values() for row in c["confusion_true_by_predicted"])
        for ax, (key, name) in zip(axes, (("action", "During inner speech"), ("precue", "Before the cue")), strict=True):
            c = data["conditions"][key]
            cm = c["confusion_true_by_predicted"]
            ax.imshow(cm, cmap=colors, vmin=0, vmax=vmax)
            for i, row in enumerate(cm):
                ax.add_patch(Rectangle((i-.48, i-.48), .96, .96, fill=False, edgecolor=INK, lw=1.35))
                for j, count in enumerate(row):
                    ax.text(j, i, str(count), ha="center", va="center", fontsize=14,
                            color="white" if count>=.73*vmax else INK)
            ax.set_title(f"{name} · {c['correct']}/{c['total']} correct", pad=10, fontsize=12)
            ax.set_xticks(range(4), c["labels"], rotation=24, ha="right")
            ax.set_yticks(range(4), c["labels"])
            ax.tick_params(length=0)
            ax.set_xlabel("Predicted direction")
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0].set_ylabel("Cued direction")
        save(fig, directory, "scalp_commands")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    draw(args.source, args.output)
