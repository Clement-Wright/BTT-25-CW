"""Draw development phone errors by transcript density from saved rate diagnostics."""
import argparse
import json
from pathlib import Path

from render_public_figures import BLUE, ORANGE, GRAY, GRID, INK, STYLE, plt, save


def draw(source, directory):
    data=json.loads(source.read_text(encoding="utf8"))
    cuts=data["quantile_cuts"]["target_phones_per_trial_second"]
    labels=[f"Below {cuts[0]:.2f}",f"{cuts[0]:.2f}–{cuts[1]:.2f}",
            f"{cuts[1]:.2f}–{cuts[2]:.2f}",f"{cuts[2]:.2f} or more"]
    styles={"gru80":("GRU · 80 ms",BLUE,"o"),"gru40":("GRU · 40 ms",ORANGE,"s"),
            "gru20":("GRU · 20 ms",GRAY,"^"),"two_rate":("Two-rate GRU · 40 ms",INK,"D")}
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(8.1,4.1),constrained_layout=True)
        for name,(label,color,mark) in styles.items():
            if name in data["counts"]:
                strata=data["counts"][name]["strata"]
                y=[100*strata[f"target_phones_per_trial_second_Q{i}"]["per"] for i in range(1,5)]
                ax.plot(range(4),y,label=label,marker=mark,color=color,lw=2,ms=6)
        ax.set_xticks(range(4),labels)
        ax.set_xlabel("Reference phones per second of trial duration")
        ax.set_ylabel("Tuning phoneme error rate (%)")
        ax.grid(axis="y",color=GRID)
        ax.legend(loc="lower center",bbox_to_anchor=(.5,1.),frameon=False,ncol=2)
        save(fig,directory,"rate_phone_density")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("directory",type=Path)
    args=p.parse_args()
    draw(args.source,args.directory)
