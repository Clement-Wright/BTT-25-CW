"""Export aggregate decoder evidence and render paired development comparisons."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


INPUTS = ["neural", "neural_hard_frames", "neural_spikes", "reference_spikes"]
LABELS = ["GRU probabilities", "Hard GRU frame labels", "GRU phones as spikes", "Perfect phones as spikes"]
RULES = ["native", "native_mbr", "alignment_sum", "alignment_sum_mbr"]
NAMES = ["Native top-1", "Minimum word risk", "CTC alignment sum", "Alignment sum + word risk"]


def compact_resources(source):
    fields = ["elapsed_seconds", "peak_tree_rss_bytes", "peak_tree_pss_bytes", "peak_tree_uss_bytes",
              "peak_accounted_host_bytes", "peak_cuda_allocated_bytes", "allocated_cpus", "cpu",
              "method", "slurm_requested_bytes", "host_ceiling_bytes", "peak_cgroup_bytes"]
    return {k:source[k] for k in fields}


def summarize(study, priors, proposals, output, job):
    data = json.loads((study / "metrics.json").read_text())
    resources = json.loads((study / "resources.json").read_text())
    if data["status"] != "complete" or resources["outcome"] != "complete":
        raise ValueError("Expected complete run evidence")
    rows = defaultdict(list)
    for line in (study / "predictions.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["input"]].append(r)
    ids = [r["trial_id"] for r in rows["neural"]]
    if len(set(ids)) != len(ids) or len(ids) != data["trials"]:
        raise ValueError("Missing or duplicate study predictions")
    sampling = np.random.default_rng(20261127).integers(len(ids), size=(2000, len(ids)))
    for kind, cases in rows.items():
        if [r["trial_id"] for r in cases] != ids:
            raise ValueError("Input trials are unpaired")
        denom = np.array([r["edits"]["native"]["reference_length"] for r in cases])
        for rule in RULES:
            delta = np.array([r["edits"][rule]["errors"] - r["edits"]["native"]["errors"] for r in cases])
            conf = np.quantile(delta[sampling].sum(1) / denom[sampling].sum(1), [.025,.975])
            data["results"][kind]["overall"][rule].update(
                delta_edits_vs_own_native=int(delta.sum()), paired_95_vs_own_native=conf.tolist())
        data["results"][kind]["latency_seconds"] = {
            key: {q: float(np.quantile([r[key] for r in cases], pct))
                  for q, pct in [("p50",.5),("p95",.95)]}
            for key in ("search_seconds", "all_four_selections_seconds")
        }
    data.pop("decoder")
    explicit=json.loads((priors/"explicit-compact-metrics.json").read_text())
    followup=json.loads((priors/"metrics.json").read_text())
    if followup["status"]!="complete" or followup["trials"]!=data["trials"]:
        raise ValueError("Incomplete cached-prior experiment")
    data["explicit_compact"]=explicit
    data["cached_neural_priors"]=followup
    expanded=json.loads((proposals/"metrics.json").read_text())
    if expanded["status"]!="complete":
        raise ValueError("Incomplete one-edit proposal comparison")
    data["expanded_lists"]=expanded["results"]
    raw_diffs=[]
    generation=defaultdict(list)
    for line in (proposals/"predictions.jsonl").read_text().splitlines():
        packet=json.loads(line)
        raw_diffs.append(packet["max_abs_native_lm_disagreement_nats"])
        generation[packet["input"]].append(packet["seconds"])
    data["expanded_search_metadata"]={
        "max_abs_kenlm_weighted_score_difference_in_nats":max(raw_diffs),
        "p95_generation_and_scoring_seconds":{kind:float(np.quantile(v,.95)) for kind,v in generation.items()}}
    data["kenlm_audit"]=json.loads((proposals/"kenlm-audit.json").read_text())
    data["score_conventions"]={
        "results.alignment_sum":"CTC phone log-mass plus native score minus native path acoustic, including native merge surplus.",
        "explicit_compact.alignment_sum":"CTC phone log-mass plus weighted natural-log KenLM, with no native merge surplus.",
        "cached_neural_priors.alignment_sum":"CTC phone log-mass plus weighted saved neural language score. KenLM only proposes candidates.",
        "risk":"Expected pairwise word edit count over returned unique candidates; softmax temperature 2."}
    data.update(job_id=job, hardware="Gilbreth CPU decoder, one Slurm step",
                config=json.loads((study / "config.resolved.yaml").read_text())["rules"],
                resources=compact_resources(resources),
                priors_resources=compact_resources(json.loads((priors/"resources.json").read_text())),
                proposals_resources=compact_resources(json.loads((proposals/"resources.json").read_text())),
                priors_provenance=json.loads((priors/"provenance.json").read_text()),
                proposals_provenance=json.loads((proposals/"provenance.json").read_text()),
                provenance=json.loads((study/"provenance.json").read_text()))
    output.write_text(json.dumps(data,indent=2) + "\n",encoding="utf-8")


def chart(data, path):
    fig, (a,b) = plt.subplots(1,2,figsize=(11.3,3.7),gridspec_kw={"width_ratios":[1.12,1]})
    fig.subplots_adjust(left=.17,right=.985,bottom=.20,top=.75,wspace=.57)
    y = np.arange(4)
    baseline = [data["results"][kind]["overall"]["native"] for kind in INPUTS]
    rates = np.array([v["errors"]/v["reference_length"] for v in baseline]) * 100
    oracle = np.array([v["oracle_errors"]/v["reference_length"] for v in baseline]) * 100
    for i in y:
        a.hlines(i,oracle[i],rates[i],color="#b5c6cf",lw=2,zorder=2)
    a.scatter(rates,y,s=47,color="#145879",label="Selected sentence",zorder=3)
    a.scatter(oracle,y,marker="D",s=32,facecolor="white",edgecolor="#ab5529",linewidth=1.5,
              label="Best available candidate",zorder=3)
    for i,rate in enumerate(rates):
        a.annotate(f"{rate:.2f}%",(rate,i),xytext=(5,0),textcoords="offset points",
                   va="center",fontsize=8.3,color="#183d51")
    a.set(xlim=(-2, max(rates)*1.22),yticks=y,yticklabels=LABELS,ylim=(3.5,-.5),xlabel="Word error rate")
    a.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    a.legend(loc="upper left",bbox_to_anchor=(0,1.18),ncol=2,frameon=False,fontsize=7.2,
             handletextpad=.3,columnspacing=.8,borderaxespad=0)
    a.set_title("What the decoder receives",loc="left",pad=34,fontweight="bold",fontsize=12)
    stats = data["results"]["neural"]["overall"]
    for j,rule in enumerate(RULES):
        v = (data["explicit_compact"]["neural"]["overall"][rule] if rule.startswith("alignment_sum")
             else stats[rule])
        val = 100 * v["delta_edits_vs_own_native"] / v["reference_length"]
        lo,hi = np.array(v["paired_95_vs_own_native"])*100
        b.hlines(j,lo,hi,color="#6190a2",lw=2)
        b.scatter(val,j,color="#145879",s=38,zorder=3)
    b.axvline(0,lw=1,color="#849197",linestyle="--")
    b.set(yticks=y,yticklabels=NAMES,ylim=(3.5,-.5),xlabel="Change in WER (percentage points)")
    b.set_title("Selection on GRU probabilities",loc="left",pad=34,fontweight="bold",fontsize=12)
    b.text(0,1.09,"Paired 95% trial-bootstrap intervals",transform=b.transAxes,fontsize=8,color="#4b575c")
    for ax in (a,b):
        ax.xaxis.grid(True,color="#e3e9ec",linewidth=.7); ax.set_axisbelow(True)
        ax.spines[["top","right","left"]].set_visible(False); ax.spines["bottom"].set_color("#92a1a8")
        ax.tick_params(axis="both",length=0,pad=5,labelsize=8)
        ax.xaxis.label.set_size(9)
    fig.savefig(path,dpi=200,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path)
    p.add_argument("--priors",type=Path)
    p.add_argument("--proposals",type=Path)
    p.add_argument("--job")
    p.add_argument("--output",type=Path,default=Path("reports/phone_robustness.json"))
    p.add_argument("--figure",type=Path,default=Path("reports/figures/phone_robustness.png"))
    args=p.parse_args()
    if args.study:
        summarize(args.study,args.priors,args.proposals,args.output,args.job)
    chart(json.loads(args.output.read_text()),args.figure)
